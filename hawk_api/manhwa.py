"""Manhwa chapter cataloguer: fetch a chapter page, follow its "next" links, and have the vision role describe it.

A run starts from one chapter URL. Each chapter's ``<img>`` panels are downloaded in reading order (with the
chapter as Referer, since most readers refuse hotlinks), stored as image assets under
``manhwa/<series>/<chapter>``, cut into tiles a vision model can actually read -- a webtoon strip is often
690 x 15000 px, which every model downsamples into mush -- and sent a few tiles per call with the running
character roster. What comes back (scene, characters, dialogue, tags) goes into jobs.sqlite3 with an FTS5
index, so "which chapter has the rooftop fight" is one query.

Runs are kept in memory like Drive imports (``library.ImportManager``); the catalogue itself is persistent,
and a chapter URL already catalogued is skipped, so re-running a series resumes where it stopped.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import random
import re
import sqlite3
import threading
import time
import uuid
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse

import httpx

from . import models_llm
from .atlas import AtlasError
from .jobs import NotFound, RequestError
from .schemas import PlanRequest, ReferenceIn

log = logging.getLogger("hawk_api.manhwa")

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0 Safari/537.36")
MAX_CHAPTERS = 50
DEFAULT_CHAPTERS = 5
MAX_IMAGES = 300  # per chapter; a page with more is a listing, not a chapter
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MIN_IMAGE_BYTES = 20 * 1024  # logos, avatars, spacer gifs and most ad pixels are smaller
MIN_IMAGE_WIDTH = 300
TILE_WIDTH = 1024
TILE_HEIGHT = 1600
TILE_OVERLAP = 120
TILES_PER_CALL = 4
CATALOG_MAX_TOKENS = 8000  # reasoning models spend part of this before the JSON
CHAPTER_DELAY = (3.0, 5.0)  # seconds between chapters, so a run never hammers a site

# Panels, for turning a chapter into photographs. A tile is a fixed-height cut that can split a panel in two;
# the panels are found from the gutters instead: rows of one flat colour running the full width.
GUTTER_SPREAD = 14  # max - min grey level across a row that still counts as gutter
MIN_GUTTER = 16  # px; a flat band thinner than this is part of the art (a sky, a wall)
MIN_PANEL = 200  # px; shorter runs are caption boxes and sound effects, not scenes
MAX_PANEL = 2400  # px; a taller run is cut into equal parts (a 1024 x 2400 frame is as tall as Qwen 2.1 edits well)
MAX_CONVERT_PANELS = 300
BUSY_RETRIES = 10  # ComfyUI busy with a video render: wait a minute and try again, this many times
BUSY_WAIT = 60.0
PLAN_PICTURES = 9  # a render takes at most nine pictures, so a chapter is planned in parts this size
PLAN_SEGMENT_SECONDS = 10.0

#: Images whose URL says what they are, and it is not a panel.
NOT_PANEL = re.compile(r"logo|avatar|icon|banner|sprite|emoji|badge|button|loading|placeholder|/ads?/|adserver|pixel",
                       re.I)
#: An anime-to-real LoRA's file name: "Qwen Image2.1_Anime2Real", "Anime2Real_LAtest", "Anything2RealCharacters".
TO_REAL = re.compile(r"(?:anime|anything|cartoon|toon)[ _-]?(?:2|to)[ _-]?real", re.I)
NEXT_TEXT = re.compile(r"\bnext\b|다음|次|下一|›|»|→", re.I)
#: "Next page" inside a paged reader moves within the chapter, which is not what a run follows.
NOT_NEXT = re.compile(r"\bpage\b|\bprev|\bprevious\b|이전", re.I)
#: An <img> whose class or id says it is page furniture: the other episodes' thumbnails, avatars, logos.
NOT_PANEL_CLASS = re.compile(r"thumb|avatar|logo|icon|banner|profile|cover", re.I)
#: Query parameters that hold the chapter number (Webtoon: ?title_no=95&episode_no=1), checked before the slug,
#: whose "ep-0" is the title's own numbering and not the episode's.
CHAPTER_PARAMS = ("episode_no", "episode", "ep", "chapter_no", "chapter", "ch", "no")
CHAPTER_NUMBER = re.compile(r"(?<![a-z])(?:chapter|chap|ch|episode|ep)[-_/ .=]?(\d+(?:\.\d+)?)", re.I)

CATALOG_PROMPT = """\
You catalogue manhwa (Korean webtoon) chapters. The images are consecutive tiles of one chapter, in reading
order, top to bottom; neighbouring tiles overlap slightly, so do not describe the same panel twice.

For each tile, in the order given, write:
- scene: 1-3 sentences, what happens in this tile.
- setting: where it is (or "" if unclear). mood: one or two words.
- characters: everyone visible. Use a name from the roster when the person matches its description; use a name
  the text gives you; otherwise "unknown-1", "unknown-2", ... (keep the same label for the same person across
  tiles). appearance: short and distinctive (hair, clothes, build). action: what they do here.
- dialogue: every speech bubble, caption and narration box in reading order. speaker: a character name, or
  "narration". text: exactly as written. lang: ISO code. text_en: English translation ("" when already English).
- sfx: sound-effect lettering, transliterated. tags: 2-6 short lowercase keywords.

Roster so far (name: appearance):
{roster}

Story so far: {story}

Reply with only JSON:
{{"tiles": [{{"scene": "", "setting": "", "mood": "", "characters": [{{"name": "", "appearance": "", "action": ""}}],
"dialogue": [{{"speaker": "", "text": "", "lang": "", "text_en": ""}}], "sfx": [], "tags": []}}],
"summary": "one sentence: what happened in these tiles"}}
"""

REAL_PROMPT = ("Turn this manhwa panel into a realistic photograph of real adult people. Keep the composition, "
               "camera angle, poses, expressions, clothing and setting exactly as drawn. Remove all speech bubbles, "
               "captions and sound-effect lettering.")
REAL_NEGATIVE = ("anime, cartoon, illustration, drawing, comic, manga, cel shading, line art, speech bubble, text, "
                 "letters, watermark, childlike, young-looking")

WRAPUP_PROMPT = """\
Below is a scene-by-scene catalogue of one manhwa chapter. Write the chapter summary and settle the cast.

Known roster (name: appearance):
{roster}

Catalogue:
{catalogue}

Reply with only JSON:
{{"summary": "3-6 sentences", "rename": {{"unknown-1": "Name or a stable descriptive label"}},
"characters": [{{"name": "", "aliases": [], "appearance": ""}}]}}
"rename" maps every unknown-N label to a roster name when it is clearly that person, else to a short stable
descriptive label such as "the tall manager". "characters" lists everyone in this chapter, after renaming.
"""


# --------------------------------------------------------------------------- page parsing


class _ChapterParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images: list[str] = []
        self.links: list[tuple[int, str]] = []  # (rank, href): 0 rel=next, 1 a "next" button
        self.title = ""
        self._in_title = False
        self._anchor: dict | None = None

    def handle_starttag(self, tag, attrs):
        attrs = {k.lower(): (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "img":
            if NOT_PANEL_CLASS.search(f"{attrs.get('class', '')} {attrs.get('id', '')}"):
                return
            declared = re.match(r"\d+", attrs.get("width", ""))
            if declared and int(declared.group()) < MIN_IMAGE_WIDTH:
                return
            src = (attrs.get("data-src") or attrs.get("data-lazy-src") or attrs.get("data-original")
                   or attrs.get("data-url") or _largest_srcset(attrs.get("srcset") or attrs.get("data-srcset", ""))
                   or attrs.get("src", ""))
            if src.strip():
                self.images.append(src.strip())
        elif tag == "link" and "next" in attrs.get("rel", "").lower().split() and attrs.get("href"):
            self.links.append((0, attrs["href"]))
        elif tag == "a" and attrs.get("href"):
            self._anchor = {"href": attrs["href"], "rel": attrs.get("rel", ""), "text": "",
                            "label": " ".join(attrs.get(k, "") for k in ("class", "id", "aria-label", "title"))}

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._anchor is not None:
            self._anchor["text"] += data

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._anchor is not None:
            anchor, self._anchor = self._anchor, None
            if "next" in anchor["rel"].lower().split():
                self.links.append((0, anchor["href"]))
                return
            words = f"{anchor['text']} {anchor['label']}"
            if NEXT_TEXT.search(words) and not NOT_NEXT.search(words):
                self.links.append((1, anchor["href"]))


def _largest_srcset(srcset: str) -> str:
    """The last candidate of a srcset, which by convention is the largest."""
    parts = [p.strip().split()[0] for p in srcset.split(",") if p.strip()]
    return parts[-1] if parts else ""


def parse_chapter(html: str, base_url: str) -> dict:
    """The chapter's panel URLs in reading order, its next-chapter link (or None) and its title."""
    parser = _ChapterParser()
    parser.feed(html)
    seen: dict[str, None] = {}
    for src in parser.images:
        if src.startswith("data:"):
            continue
        url = urljoin(base_url, src)
        if urlparse(url).scheme not in ("http", "https") or NOT_PANEL.search(url):
            continue
        if re.search(r"\.svg(\?|$)", url, re.I):
            continue
        seen.setdefault(url, None)
    here = base_url.split("#")[0].rstrip("/")
    next_url = None
    for _rank, href in sorted(parser.links, key=lambda item: item[0]):
        if href.startswith(("#", "javascript:", "mailto:")):
            continue
        url = urljoin(base_url, href).split("#")[0]
        if url.rstrip("/") != here and urlparse(url).scheme in ("http", "https"):
            next_url = url
            break
    return {"images": list(seen)[:MAX_IMAGES], "next_url": next_url, "title": " ".join(parser.title.split())}


def _chapter_param(url: str) -> tuple[str, str] | None:
    """(name, value) of the query parameter that numbers the chapter, if the URL has one."""
    params = dict(parse_qsl(urlparse(url).query))
    for name in CHAPTER_PARAMS:
        if re.fullmatch(r"\d+(?:\.\d+)?", params.get(name, "")):
            return name, params[name]
    return None


def chapter_number(url: str, title: str = "") -> str:
    """The chapter number as written in the URL or title ("12", "12.5"), or ""."""
    param = _chapter_param(url)
    if param:
        return param[1]
    for text in (urlparse(url).path + "?" + urlparse(url).query, title):
        match = CHAPTER_NUMBER.search(text or "")
        if match:
            return match.group(1)
    numbers = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", urlparse(url).path)
    return numbers[-1] if numbers else ""


def increment_url(url: str) -> str | None:
    """The URL with its chapter number plus one: the guess used when a page has no "next" link."""
    parsed = urlparse(url)
    param = _chapter_param(url)
    if param:
        pairs = [(k, str(int(float(v)) + 1) if k == param[0] else v) for k, v in parse_qsl(parsed.query)]
        return parsed._replace(query=urlencode(pairs)).geturl()
    target = parsed.path + ("?" + parsed.query if parsed.query else "")
    match = CHAPTER_NUMBER.search(target)
    if match:
        start, end = match.span(1)
    else:
        matches = list(re.finditer(r"(?<![\w.])(\d+)(?![\w.])", parsed.path))
        if not matches:
            return None
        start, end = matches[-1].span(1)
    number = target[start:end]
    if "." in number:
        number = number.split(".")[0]
        end = start + len(number)
    bumped = target[:start] + str(int(number) + 1) + target[end:]
    path, _, query = bumped.partition("?")
    return parsed._replace(path=path, query=query).geturl()


# --------------------------------------------------------------------------- images


def slice_image(data: bytes) -> list[bytes]:
    """JPEG tiles a vision model can read: at most TILE_WIDTH wide, TILE_HEIGHT tall, overlapping by
    TILE_OVERLAP so a bubble cut by one edge is whole in the next tile."""
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")
    if image.width > TILE_WIDTH:
        image = image.resize((TILE_WIDTH, max(1, round(image.height * TILE_WIDTH / image.width))))
    tiles: list[bytes] = []
    top = 0
    while True:
        bottom = min(image.height, top + TILE_HEIGHT)
        buffer = io.BytesIO()
        image.crop((0, top, image.width, bottom)).save(buffer, format="JPEG", quality=85)
        tiles.append(buffer.getvalue())
        if bottom >= image.height:
            return tiles
        top = bottom - TILE_OVERLAP


def image_width(data: bytes) -> int:
    from PIL import Image

    try:
        return Image.open(io.BytesIO(data)).width
    except Exception:
        return 0


def find_panels(data: bytes) -> tuple[list[tuple[int, int]], int]:
    """(top, bottom) of each panel in the page as slice_image scales it (TILE_WIDTH wide), and how many
    tiles slice_image cuts it into -- so a panel can be matched to the tile that describes it."""
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("L")
    width = min(image.width, TILE_WIDTH)
    height = max(1, round(image.height * width / image.width))
    columns = 48
    raw = image.resize((columns, height)).tobytes()
    flat = [max(raw[y * columns:(y + 1) * columns]) - min(raw[y * columns:(y + 1) * columns]) <= GUTTER_SPREAD
            for y in range(height)]
    runs, start, gap = [], None, 0
    for y, is_gutter in enumerate(flat + [True] * MIN_GUTTER):
        if not is_gutter:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= MIN_GUTTER:
                runs.append((start, y - gap + 1))
                start, gap = None, 0
    panels = []
    for top, bottom in runs:
        if bottom - top < MIN_PANEL:
            continue
        parts = -(-(bottom - top) // MAX_PANEL)
        step = (bottom - top) / parts
        panels += [(round(top + i * step), round(top + (i + 1) * step)) for i in range(parts)]
    tiles, top = 1, 0
    while top + TILE_HEIGHT < height:
        top += TILE_HEIGHT - TILE_OVERLAP
        tiles += 1
    return panels, tiles


def cut_panels(data: bytes) -> list[tuple[int, bytes]]:
    """Each panel as a JPEG, with the number of the tile (as slice_image cuts the page) it sits in."""
    from PIL import Image

    panels, tiles = find_panels(data)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if image.width > TILE_WIDTH:
        image = image.resize((TILE_WIDTH, max(1, round(image.height * TILE_WIDTH / image.width))))
    out = []
    for top, bottom in panels:
        buffer = io.BytesIO()
        image.crop((0, top, image.width, bottom)).save(buffer, format="JPEG", quality=92)
        part = min(tiles - 1, ((top + bottom) // 2) // (TILE_HEIGHT - TILE_OVERLAP))
        out.append((part, buffer.getvalue()))
    return out


def real_prompt(tile: dict | None, extra: str = "") -> str:
    """REAL_PROMPT plus what the catalogue says is in the panel, so the people keep their looks."""
    parts = [REAL_PROMPT]
    if tile and tile.get("status") == "ok":
        if tile.get("scene"):
            parts.append(f"Scene: {tile['scene']}")
        if tile.get("setting"):
            parts.append(f"Setting: {tile['setting']}.")
        people = [f"{p.get('name')}: {p.get('appearance')}" for p in tile.get("characters") or []
                  if p.get("appearance")]
        if people:
            parts.append("People: " + "; ".join(people) + ".")
    if extra.strip():
        parts.append(extra.strip())
    return " ".join(parts)[:1500]


def plan_brief(chapter: dict, shots: list[dict], part: int, parts: int) -> str:
    """The story planner's brief for one part of a chapter: one segment per photograph, in reading order,
    with the panel's dialogue as the exact lines to speak."""
    lines = [f"A live-action film of {chapter['series']}, chapter {chapter.get('chapter') or '?'}"
             + (f", part {part} of {parts}" if parts > 1 else "") + ".",
             "The attached pictures are photographs made from the comic's panels, in reading order. Write exactly "
             "one segment per picture, in the same order: segment N starts from <Picture N>, uses only that picture "
             "(\"pictures\": [N]) and plays out what the panel shows, with the people looking as they do in it.",
             "Characters speak the dialogue below word for word, in English, in the order given; narration is a "
             "voice-over. Keep segments with no dialogue free of speech. Leave references_to_make empty.",
             "All characters are adults."]
    if chapter.get("summary"):
        lines.append(f"Chapter so far: {chapter['summary']}")
    for number, shot in enumerate(shots, 1):
        lines.append(f"\n<Picture {number}>: {shot.get('scene') or 'see the picture'}")
        people = [f"{p['name']} ({p.get('appearance', '')})" for p in shot.get("characters") or [] if p.get("name")]
        if people:
            lines.append("People: " + "; ".join(people))
        for said in shot.get("dialogue") or []:
            text = (said.get("text_en") or said.get("text") or "").strip()
            if not text:
                continue
            who = said.get("speaker") or "narration"
            lines.append(f'{"Narration (voice-over)" if who.lower() == "narration" else who}: "{text}"')
    return "\n".join(lines)


# --------------------------------------------------------------------------- store


class ManhwaStore:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS manhwa_chapters (
        id TEXT PRIMARY KEY, series TEXT NOT NULL, chapter TEXT NOT NULL, url TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS manhwa_chapters_series ON manhwa_chapters(series, created_at);
    CREATE TABLE IF NOT EXISTS manhwa_pages (
        id TEXT PRIMARY KEY, chapter_id TEXT NOT NULL, idx INTEGER NOT NULL, asset_id TEXT, data TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS manhwa_pages_chapter ON manhwa_pages(chapter_id, idx);
    CREATE TABLE IF NOT EXISTS manhwa_characters (
        series TEXT NOT NULL, name TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (series, name));
    CREATE VIRTUAL TABLE IF NOT EXISTS manhwa_fts USING fts5(page_id UNINDEXED, series UNINDEXED, body);
    """

    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(self._SCHEMA)
            self._db.commit()

    def chapter_by_url(self, url: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT id, series, chapter, url, status, data, created_at FROM manhwa_chapters "
                                   "WHERE url = ?", (url,)).fetchone()
        return _chapter_row(row) if row else None

    def get_chapter(self, chapter_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT id, series, chapter, url, status, data, created_at FROM manhwa_chapters "
                                   "WHERE id = ?", (chapter_id,)).fetchone()
        return _chapter_row(row) if row else None

    def save_chapter(self, chapter: dict) -> None:
        extra = {k: v for k, v in chapter.items() if k not in ("id", "series", "chapter", "url", "status", "created_at")}
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO manhwa_chapters VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (chapter["id"], chapter["series"], chapter["chapter"], chapter["url"], chapter["status"],
                              json.dumps(extra, ensure_ascii=False), chapter["created_at"]))
            self._db.commit()

    def list_chapters(self, series: str = "") -> list[dict]:
        sql = "SELECT id, series, chapter, url, status, data, created_at FROM manhwa_chapters"
        args: tuple = ()
        if series:
            sql, args = sql + " WHERE series = ?", (series,)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY series, created_at", args).fetchall()
        return [_chapter_row(row) for row in rows]

    def save_pages(self, chapter_id: str, series: str, pages: list[dict]) -> None:
        with self._lock:
            old = [r[0] for r in self._db.execute("SELECT id FROM manhwa_pages WHERE chapter_id = ?", (chapter_id,))]
            self._db.executemany("DELETE FROM manhwa_fts WHERE page_id = ?", [(i,) for i in old])
            self._db.execute("DELETE FROM manhwa_pages WHERE chapter_id = ?", (chapter_id,))
            for page in pages:
                self._db.execute("INSERT INTO manhwa_pages VALUES (?, ?, ?, ?, ?)",
                                 (page["id"], chapter_id, page["idx"], page.get("asset_id"),
                                  json.dumps(page, ensure_ascii=False)))
                self._db.execute("INSERT INTO manhwa_fts VALUES (?, ?, ?)", (page["id"], series, page_text(page)))
            self._db.commit()

    def pages(self, chapter_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM manhwa_pages WHERE chapter_id = ? ORDER BY idx",
                                    (chapter_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def search(self, query: str, series: str = "", limit: int = 30) -> list[dict]:
        terms = [t for t in re.findall(r"[\w'’-]+", query or "", re.U) if t]
        if not terms:
            return []
        match = " ".join('"' + t.replace('"', '""') + '"' for t in terms)
        sql = ("SELECT p.data, c.id, c.series, c.chapter, c.url FROM manhwa_fts f "
               "JOIN manhwa_pages p ON p.id = f.page_id JOIN manhwa_chapters c ON c.id = p.chapter_id "
               "WHERE manhwa_fts MATCH ?")
        args: list = [match]
        if series:
            sql += " AND f.series = ?"
            args.append(series)
        sql += " ORDER BY bm25(manhwa_fts) LIMIT ?"
        args.append(max(1, min(limit, 200)))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(json.loads(data), chapter_id=cid, series=s, chapter=ch, chapter_url=url)
                for data, cid, s, ch, url in rows]

    def characters(self, series: str) -> dict[str, dict]:
        with self._lock:
            rows = self._db.execute("SELECT name, data FROM manhwa_characters WHERE series = ?", (series,)).fetchall()
        return {name: json.loads(data) for name, data in rows}

    def save_characters(self, series: str, characters: dict[str, dict]) -> None:
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO manhwa_characters VALUES (?, ?, ?)",
                                 [(series, name, json.dumps(data, ensure_ascii=False))
                                  for name, data in characters.items()])
            self._db.commit()

    def delete_chapters(self, chapter_ids: list[str]) -> list[str]:
        """Drop chapters with their tiles and search rows. Returns the asset ids their tiles named."""
        assets: list[str] = []
        with self._lock:
            for chapter_id in chapter_ids:
                rows = self._db.execute("SELECT id, asset_id FROM manhwa_pages WHERE chapter_id = ?",
                                        (chapter_id,)).fetchall()
                self._db.executemany("DELETE FROM manhwa_fts WHERE page_id = ?", [(r[0],) for r in rows])
                assets += [r[1] for r in rows if r[1]]
                self._db.execute("DELETE FROM manhwa_pages WHERE chapter_id = ?", (chapter_id,))
                self._db.execute("DELETE FROM manhwa_chapters WHERE id = ?", (chapter_id,))
            self._db.commit()
        return list(dict.fromkeys(assets))

    def delete_characters(self, series: str, names: list[str] | None = None) -> None:
        with self._lock:
            if names is None:
                self._db.execute("DELETE FROM manhwa_characters WHERE series = ?", (series,))
            else:
                self._db.executemany("DELETE FROM manhwa_characters WHERE series = ? AND name = ?",
                                     [(series, n) for n in names])
            self._db.commit()

    def series(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT series, COUNT(*), SUM(status = 'done') FROM manhwa_chapters "
                                    "GROUP BY series ORDER BY series").fetchall()
        return [{"series": s, "chapters": n, "done": d or 0} for s, n, d in rows]


def _chapter_row(row) -> dict:
    chapter_id, series, chapter, url, status, data, created_at = row
    return dict(json.loads(data), id=chapter_id, series=series, chapter=chapter, url=url, status=status,
                created_at=created_at)


def page_text(page: dict) -> str:
    """Everything a search should find a tile by."""
    parts = [page.get("scene", ""), page.get("setting", ""), page.get("mood", "")]
    for person in page.get("characters") or []:
        parts += [person.get("name", ""), person.get("appearance", ""), person.get("action", "")]
    for line in page.get("dialogue") or []:
        parts += [line.get("speaker", ""), line.get("text", ""), line.get("text_en", "")]
    parts += list(page.get("sfx") or []) + list(page.get("tags") or [])
    return "\n".join(str(p) for p in parts if p)


def parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------- runs


class ManhwaManager:
    """Catalogue runs with progress, kept in memory for the session; results persist in ManhwaStore."""

    def __init__(self, service, store: ManhwaStore | None = None, *, transport: httpx.AsyncBaseTransport | None = None,
                 delay: tuple[float, float] = CHAPTER_DELAY):
        self.service = service
        self.store = store or ManhwaStore(service.settings.db_path)
        self.runs: dict[str, dict] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._transport = transport
        self.delay = delay
        self.conversions: dict[str, dict] = {}
        self._convert_tasks: dict[str, asyncio.Task] = {}
        self.busy_wait = BUSY_WAIT

    # ----------------------------------------------------------------- API

    async def start(self, url: str, series: str, chapter: str = "", *, cookies: str = "", follow_next: bool = False,
                    max_chapters: int = DEFAULT_CHAPTERS) -> dict:
        url = (url or "").strip()
        if not re.match(r"^https?://", url):
            raise RequestError("Give the chapter's http(s) URL.")
        series = " ".join((series or "").split())
        if not series:
            raise RequestError("Give the series name: it groups chapters and their characters.")
        key = series_key(series)
        series = next((s["series"] for s in self.store.series() if series_key(s["series"]) == key), series)
        for other in self.runs.values():
            if other["status"] != "running":
                continue
            if series_key(other["series"]) == key or url in (other["start_url"], (other["current"] or {}).get("url")):
                # Two runs on one chapter download every panel twice and overwrite each other's chapter row.
                raise RequestError(f"Run {other['id']} is already cataloguing {other['series']!r}. Cancel it or wait "
                                   "for it to finish.", {"run_id": other["id"]})
        limit = max(1, min(int(max_chapters or 1), MAX_CHAPTERS)) if follow_next else 1
        state = {
            "id": uuid.uuid4().hex[:12],
            "status": "running",
            "series": series,
            "start_url": url,
            "follow_next": bool(follow_next),
            "max_chapters": limit,
            "chapters_done": 0,
            "current": None,  # {url, chapter, step, tiles_done, tiles_total}
            "chapter_ids": [],
            "skipped": [],
            "errors": [],
            "stop_reason": "",
            "cancel_requested": False,
            "created_at": time.time(),
            "finished_at": None,
        }
        self.runs[state["id"]] = state
        self._tasks[state["id"]] = asyncio.create_task(
            self._run(state, url, (chapter or "").strip(), (cookies or "").strip()))
        return state

    def get(self, run_id: str) -> dict:
        state = self.runs.get(run_id)
        if state is None:
            raise NotFound(f"No catalogue run {run_id!r} (runs are forgotten when the API restarts; "
                           "the catalogue itself is kept).")
        return state

    def recent(self) -> list[dict]:
        return sorted(self.runs.values(), key=lambda s: s["created_at"], reverse=True)[:20]

    def cancel(self, run_id: str) -> dict:
        """Stop a run now. The chapter it was on keeps the tiles already described, marked cancelled."""
        state = self.get(run_id)
        if state["status"] == "running":
            state["cancel_requested"] = True
            task = self._tasks.get(run_id)
            if task is not None:
                task.cancel()
        return state

    async def cancel_and_wait(self, run_id: str, seconds: float = 30) -> dict:
        self.cancel(run_id)
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait([task], timeout=seconds)
        return self.get(run_id)

    async def delete(self, series: str, chapter_id: str = "", *, images: bool = True) -> dict:
        """Delete one chapter, or a whole series, from the catalogue; with images, its downloaded panels too.
        Runs working on that series are cancelled first, so nothing writes the rows back afterwards."""
        key = series_key(series)
        series = next((s["series"] for s in self.store.series() if series_key(s["series"]) == key), series)
        cancelled = []
        for run in list(self.runs.values()):
            if run["status"] == "running" and series_key(run["series"]) == key:
                await self.cancel_and_wait(run["id"])
                cancelled.append(run["id"])
        chapters = self.store.list_chapters(series)
        if chapter_id:
            chapters = [c for c in chapters if c["id"] == chapter_id]
            if not chapters:
                raise NotFound(f"No catalogued chapter {chapter_id!r} in {series!r}.")
        named = self.store.delete_chapters([c["id"] for c in chapters])
        if chapter_id:
            self._forget_chapter(series, chapter_id)
        else:
            self.store.delete_characters(series)
        deleted_images = 0
        if images:
            prefixes = [f"manhwa/{series}/{c['chapter'] or 'chapter'}" for c in chapters] if chapter_id \
                else [f"manhwa/{series}/"]
            ids = set(named)
            for asset in self.service.store.all_assets():
                collection = str(asset.get("collection") or "")
                if (asset.get("source") or {}).get("type") == "manhwa" and any(
                        collection == p or (p.endswith("/") and collection.startswith(p)) for p in prefixes):
                    ids.add(asset["id"])
            for asset_id in ids:
                try:
                    self.service.delete_asset(asset_id)
                    deleted_images += 1
                except NotFound:
                    pass
        return {"series": series, "chapters_deleted": len(chapters), "images_deleted": deleted_images,
                "runs_cancelled": cancelled}

    def _forget_chapter(self, series: str, chapter_id: str) -> None:
        roster = self.store.characters(series)
        gone = []
        for name, data in roster.items():
            ids = [i for i in data.get("chapter_ids") or [] if i != chapter_id]
            if not ids:
                gone.append(name)
            data["chapter_ids"] = ids
        self.store.delete_characters(series, gone)
        self.store.save_characters(series, {n: d for n, d in roster.items() if n not in gone})

    async def wait(self, run_id: str, seconds: float) -> dict:
        deadline = time.monotonic() + seconds
        while self.get(run_id)["status"] == "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        return self.get(run_id)

    async def stop(self) -> None:
        tasks = list(self._tasks.values()) + list(self._convert_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ----------------------------------------------------------------- photographs from panels

    async def real_loras(self, realism: float = 0.5, age_slider: float = 1.0) -> list[dict]:
        """The LoRAs a panel is converted with, read from what is in models/loras now: the newest anime-to-real
        LoRA (a file named "latest" wins), the photoreal one, and the age slider so drawn faces read as the
        adults they are. Qwen 2.1's repair LoRA and its adult default are added by the engine itself."""
        items = [i for i in await self.service.local_images.catalogue("qwen21") if i.installed]
        to_real = [i for i in items if TO_REAL.search(i.file)]
        if not to_real:
            raise RequestError("No anime-to-real LoRA in models/loras/qwen21 (e.g. Anime2Real_LAtest.safetensors). "
                               "Download one there, or name the LoRAs to use with loras.")
        best = max(to_real, key=lambda i: ("latest" in i.file.lower(), to_real.index(i)))
        chosen = [{"name": best.file, "strength": best.strength or 1.0}]
        for pattern, strength in ((r"lenovo_qwen21", realism), (r"age_slider", age_slider)):
            found = next((i for i in items if re.search(pattern, i.file, re.I)), None)
            if found and strength > 0:
                chosen.append({"name": found.file, "strength": float(strength)})
        return chosen

    async def convert(self, chapter_id: str, *, tile_ids: list[str] | None = None, loras: list[dict] | None = None,
                      prompt: str = "", realism: float = 0.5, age_slider: float = 1.0,
                      max_panels: int = MAX_CONVERT_PANELS, plan_after: bool = True) -> dict:
        """Turn a catalogued chapter's panels into photographs with Qwen Image 2.1 on this GPU. Runs in the
        background, one panel at a time; poll conversion(). tile_ids limits it to the panels in those tiles."""
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        for other in self.conversions.values():
            if other["status"] == "running":
                # one GPU: two conversions would only queue behind each other and time out
                raise RequestError(f"Conversion {other['id']} is still running. Cancel it or wait for it to finish.",
                                   {"conversion_id": other["id"]})
        chosen = [dict(l) for l in loras] if loras else await self.real_loras(realism, age_slider)
        state = {
            "id": uuid.uuid4().hex[:12], "status": "running", "chapter_id": chapter_id, "series": chapter["series"],
            "chapter": chapter["chapter"], "loras": chosen, "tile_ids": list(tile_ids or []),
            "panels_total": 0, "panels_done": 0, "converted": 0, "refused": 0, "failed": 0, "results": [],
            "errors": [], "cancel_requested": False, "created_at": time.time(), "finished_at": None,
            "collection": f"manhwa/{chapter['series']}/{chapter['chapter'] or 'chapter'}/real",
            "plan_after": bool(plan_after), "plans": [],
        }
        self.conversions[state["id"]] = state
        self._convert_tasks[state["id"]] = asyncio.create_task(
            self._convert(state, prompt, max(1, min(int(max_panels or 1), MAX_CONVERT_PANELS))))
        return state

    def conversion(self, conversion_id: str) -> dict:
        state = self.conversions.get(conversion_id)
        if state is None:
            raise NotFound(f"No conversion {conversion_id!r} (forgotten when the API restarts; the images are kept).")
        return state

    def recent_conversions(self) -> list[dict]:
        return sorted(self.conversions.values(), key=lambda s: s["created_at"], reverse=True)[:20]

    def cancel_conversion(self, conversion_id: str) -> dict:
        state = self.conversion(conversion_id)
        if state["status"] == "running":
            state["cancel_requested"] = True
            task = self._convert_tasks.get(conversion_id)
            if task is not None:
                task.cancel()
        return state

    async def wait_conversion(self, conversion_id: str, seconds: float) -> dict:
        deadline = time.monotonic() + seconds
        while self.conversion(conversion_id)["status"] == "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        return self.conversion(conversion_id)

    async def _convert(self, state: dict, extra: str, max_panels: int) -> None:
        series, chapter_id = state["series"], state["chapter_id"]
        pages = self.store.pages(chapter_id)
        tiles = {(p.get("asset_id"), p.get("part")): p for p in pages}
        wanted = set(state["tile_ids"])
        try:
            work: list[tuple[dict, int, bytes, dict | None]] = []
            for asset_id in dict.fromkeys(p.get("asset_id") for p in pages if p.get("asset_id")):
                if wanted and not any(p["id"] in wanted for p in pages if p.get("asset_id") == asset_id):
                    continue
                asset = self.service.store.get_asset(asset_id)
                if asset is None:
                    state["errors"].append(f"Page image {asset_id} is no longer in the library.")
                    continue
                try:
                    panels = await asyncio.to_thread(cut_panels, await self.service.asset_bytes(asset))
                except Exception as exc:
                    state["errors"].append(f"{asset.get('filename')}: {exc}")
                    continue
                for number, (part, jpeg) in enumerate(panels, 1):
                    tile = tiles.get((asset_id, part))
                    if wanted and (tile or {}).get("id") not in wanted:
                        continue
                    work.append((asset, number, jpeg, tile))
            work = work[:max_panels]
            state["panels_total"] = len(work)
            for asset, number, jpeg, tile in work:
                stem = (asset.get("filename") or "page").rsplit(".", 1)[0]
                panel = await self.service.add_asset(
                    f"{stem}_panel{number:02d}.jpg", io.BytesIO(jpeg), "image/jpeg", len(jpeg),
                    collection=state["collection"].rsplit("/", 1)[0] + "/panels", tags=["manhwa", series, "panel"],
                    source={"type": "manhwa", "from_asset": asset["id"], "panel": number,
                            "tile_id": (tile or {}).get("id", "")})
                row = {"tile_id": (tile or {}).get("id", ""), "panel_asset_id": panel["id"], "page": asset.get("filename")}
                try:
                    result = await self._convert_panel(panel["id"], real_prompt(tile, extra), state["loras"])
                except Exception as exc:  # one panel failing (a refusal, ComfyUI down) does not end the run
                    refused = isinstance(exc, RequestError) and str(exc).startswith("Refused")
                    row.update(status="refused" if refused else "failed", error=str(exc)[:400])
                    state["refused" if refused else "failed"] += 1
                else:
                    ids = [a["id"] for a in result.get("assets") or []]
                    for image_id in ids:
                        self.service.update_asset(image_id, collection=state["collection"],
                                                  add_tags=["manhwa", series, "real"])
                    row.update(status="ok", asset_ids=ids,
                               thumb_url=(result.get("assets") or [{}])[0].get("thumb_url"))
                    state["converted"] += 1
                    if tile is not None:
                        tile.setdefault("real", []).append({"panel_asset_id": panel["id"], "asset_ids": ids})
                        self.store.save_pages(chapter_id, series, pages)
                state["results"].append(row)
                state["panels_done"] += 1
            if state["plan_after"] and state["converted"]:
                try:
                    state["plans"] = (await self.plan_chapter(chapter_id))["plans"]
                except Exception as exc:  # the photographs are made; a plan can be asked for again
                    state["errors"].append(f"Video plan: {exc}")
            state["status"] = "done"
        except asyncio.CancelledError:
            state["status"] = "cancelled"
            raise
        except Exception as exc:
            log.exception("manhwa conversion %s failed", state["id"])
            state["status"] = "failed"
            state["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            state["finished_at"] = time.time()
            self._convert_tasks.pop(state["id"], None)

    def shots(self, chapter_id: str) -> list[dict]:
        """The chapter's photographs in reading order, one per panel (the latest made from it), each with its
        tile's scene, characters and dialogue -- the dialogue only on the first panel of a tile, so a line is
        not spoken twice."""
        shots, spoken = [], set()
        for tile in self.store.pages(chapter_id):
            latest: dict[str, str] = {}
            for made in tile.get("real") or []:
                ids = [i for i in made.get("asset_ids") or [] if self.service.store.get_asset(i) is not None]
                if ids:
                    latest[made.get("panel_asset_id") or ids[0]] = ids[0]
            for asset_id in latest.values():
                shot = {"asset_id": asset_id, "tile_id": tile["id"], "scene": tile.get("scene", ""),
                        "characters": tile.get("characters") or [], "dialogue": []}
                if tile["id"] not in spoken:
                    shot["dialogue"] = tile.get("dialogue") or []
                    spoken.add(tile["id"])
                shots.append(shot)
        return shots

    async def plan_chapter(self, chapter_id: str, *, aspect_ratio: str = "9:16",
                           segment_seconds: float = PLAN_SEGMENT_SECONDS, model: str | None = None) -> dict:
        """Start story-planner jobs that turn the chapter's photographs into a film: one plan per nine
        photographs (a render's picture limit), one segment per photograph, the tiles' dialogue spoken word
        for word. Render each finished plan with plan_job_id."""
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        shots = self.shots(chapter_id)
        if not shots:
            raise RequestError("This chapter has no photographs yet: convert its panels first.")
        parts = [shots[i:i + PLAN_PICTURES] for i in range(0, len(shots), PLAN_PICTURES)]
        plans = []
        for number, part in enumerate(parts, 1):
            try:
                request = self._plan_request(chapter, part, number, len(parts), aspect_ratio, segment_seconds, model)
            except ValueError as exc:  # pydantic: an aspect ratio or length the planner does not take
                raise RequestError(str(exc).splitlines()[0] if str(exc) else "Bad plan settings.") from None
            job = await self.service.create_plan(request)
            plans.append({"part": number, "job_id": job["id"], "pictures": [s["asset_id"] for s in part],
                          "segments": len(part)})
        chapter["plans"] = plans
        self.store.save_chapter(chapter)
        return {"chapter_id": chapter_id, "photographs": len(shots), "plans": plans}

    @staticmethod
    def _plan_request(chapter, part, number, parts, aspect_ratio, segment_seconds, model) -> PlanRequest:
        return PlanRequest(
            story=plan_brief(chapter, part, number, parts), segment_count=len(part),
            segment_seconds=segment_seconds, aspect_ratio=aspect_ratio, model=model,
            references=[ReferenceIn(asset_id=shot["asset_id"], role="picture",
                                    label=(shot["scene"] or f"panel photo {k}")[:200])
                        for k, shot in enumerate(part, 1)])

    async def _convert_panel(self, panel_id: str, prompt: str, loras: list[dict]) -> dict:
        for attempt in range(BUSY_RETRIES + 1):
            try:
                return await self.service.generate_images(prompt, reference_asset_ids=[panel_id], engine="qwen21",
                                                          loras=loras, negative=REAL_NEGATIVE)
            except RequestError as exc:
                if "busy" not in str(exc).lower() or attempt == BUSY_RETRIES:
                    raise
            await asyncio.sleep(self.busy_wait)
        raise RequestError("ComfyUI stayed busy.")

    def chapter_view(self, chapter_id: str) -> dict:
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        return dict(chapter, pages=[self._with_thumb(p) for p in self.store.pages(chapter_id)])

    def search(self, query: str, series: str = "", limit: int = 30) -> list[dict]:
        return [self._with_thumb(hit) for hit in self.store.search(query, series, limit)]

    def characters(self, series: str) -> list[dict]:
        roster = self.store.characters(series)
        return sorted((dict(data, name=name) for name, data in roster.items()),
                      key=lambda c: (-len(c.get("chapters") or []), c["name"].lower()))

    def _with_thumb(self, page: dict) -> dict:
        asset = self.service.store.get_asset(page["asset_id"]) if page.get("asset_id") else None
        if asset is None:
            return page
        view = self.service.asset_view(asset)
        real = []  # the photographs made from this tile's panels, still in the library
        for made in page.get("real") or []:
            for image_id in made.get("asset_ids") or []:
                image = self.service.store.get_asset(image_id)
                if image is not None:
                    shown = self.service.asset_view(image)
                    real.append({"id": image_id, "thumb_url": shown.get("thumb_url"), "file_url": shown.get("file_url")})
        return dict(page, thumb_url=view.get("thumb_url"), file_url=view.get("file_url"), real=real)

    # ----------------------------------------------------------------- the run

    def _client(self, cookies: str) -> httpx.AsyncClient:
        headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9,ko;q=0.8"}
        if cookies:
            headers["Cookie"] = cookies
        return httpx.AsyncClient(follow_redirects=True, headers=headers, transport=self._transport,
                                 timeout=httpx.Timeout(30.0, read=120.0))

    async def _run(self, state: dict, url: str, chapter: str, cookies: str) -> None:
        previous_images: list[str] | None = None
        previous_number = ""
        try:
            async with self._client(cookies) as http:
                while True:
                    known = self.store.chapter_by_url(url)
                    page = await self._fetch_page(http, url)
                    if page is None:
                        state["stop_reason"] = f"{url} did not load."
                        break
                    if previous_images is not None and page["images"] == previous_images:
                        state["stop_reason"] = "The next page shows the same images: this is the last chapter."
                        break
                    number = chapter or chapter_number(url, page["title"]) or _bump(previous_number)
                    chapter = ""  # only the first chapter takes the number the user typed
                    if known and known["status"] == "done":
                        state["skipped"].append({"url": url, "chapter": known["chapter"], "id": known["id"]})
                    else:
                        state["current"] = {"url": url, "chapter": number, "step": "downloading",
                                            "tiles_done": 0, "tiles_total": 0}
                        chapter_id = await self._catalog_chapter(state, http, url, number, page, known)
                        state["chapter_ids"].append(chapter_id)
                        state["chapters_done"] += 1
                    previous_images, previous_number = page["images"], number

                    if state["cancel_requested"]:
                        state["stop_reason"] = "Cancelled."
                        break
                    if not state["follow_next"] or state["chapters_done"] + len(state["skipped"]) >= state["max_chapters"]:
                        state["stop_reason"] = state["stop_reason"] or "Reached the chapter limit."
                        break
                    next_url = page["next_url"] or increment_url(url)
                    if not next_url or next_url == url:
                        state["stop_reason"] = "No next chapter link."
                        break
                    url = next_url
                    state["current"] = {"url": url, "chapter": "", "step": "waiting", "tiles_done": 0, "tiles_total": 0}
                    await asyncio.sleep(random.uniform(*self.delay) if self.delay[1] > 0 else 0)
            state["status"] = "cancelled" if state["cancel_requested"] else "done"
        except asyncio.CancelledError:
            state["status"] = "cancelled"
            state["stop_reason"] = "Cancelled."
            raise
        except Exception as exc:  # a run that dies says why instead of hanging at "running"
            log.exception("manhwa run %s failed", state["id"])
            state["status"] = "failed"
            state["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            state["current"] = None
            state["finished_at"] = time.time()
            self._tasks.pop(state["id"], None)

    async def _fetch_page(self, http: httpx.AsyncClient, url: str) -> dict | None:
        try:
            response = await http.get(url)
        except httpx.HTTPError as exc:
            log.warning("manhwa: could not fetch %s: %s", url, exc)
            return None
        if response.status_code != 200 or "html" not in response.headers.get("content-type", "text/html"):
            return None
        page = parse_chapter(response.text, str(response.url))
        return page if page["images"] else None

    async def _download_images(self, state: dict, http: httpx.AsyncClient, url: str, number: str,
                               images: list[str]) -> list[tuple[dict, bytes]]:
        collection = f"manhwa/{state['series']}/{number or 'chapter'}"
        kept: list[tuple[dict, bytes]] = []
        for src in images:
            try:
                response = await http.get(src, headers={"Referer": url})
            except httpx.HTTPError as exc:
                state["errors"].append(f"{src}: {exc}")
                continue
            data = response.content
            if response.status_code != 200 or len(data) < MIN_IMAGE_BYTES or len(data) > MAX_IMAGE_BYTES:
                continue
            if image_width(data) < MIN_IMAGE_WIDTH:
                continue
            ext = {"image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(
                response.headers.get("content-type", "").split(";")[0], ".jpg")
            asset = await self.service.add_asset(
                f"p{len(kept) + 1:03d}{ext}", io.BytesIO(data), None, len(data), collection=collection,
                tags=["manhwa", state["series"]], source={"type": "manhwa", "url": src, "chapter_url": url,
                                                          "index": len(kept)})
            kept.append((asset, data))
        return kept

    async def _catalog_chapter(self, state: dict, http: httpx.AsyncClient, url: str, number: str, page: dict,
                               known: dict | None) -> str:
        series = state["series"]
        chapter = {"id": (known or {}).get("id") or uuid.uuid4().hex[:12], "series": series, "chapter": number,
                   "url": url, "status": "downloading", "title": page["title"], "created_at": time.time(),
                   "summary": "", "tiles": 0, "failed_tiles": 0, "models": []}
        self.store.save_chapter(chapter)

        pages: list[dict] = []
        try:
            return await self._catalog_body(state, http, url, number, page, chapter, pages)
        except asyncio.CancelledError:
            # Cancelled mid-chapter: keep the tiles already described (they were paid for) and say so.
            if pages:
                self.store.save_pages(chapter["id"], series, pages)
            chapter.update(status="cancelled", tiles=len(pages))
            self.store.save_chapter(chapter)
            state["chapter_ids"].append(chapter["id"])
            raise

    async def _catalog_body(self, state: dict, http: httpx.AsyncClient, url: str, number: str, page: dict,
                            chapter: dict, pages: list[dict]) -> str:
        series = state["series"]
        downloaded = await self._download_images(state, http, url, number, page["images"])
        tiles: list[tuple[str, int, bytes]] = []  # (asset_id, tile number within the image, jpeg)
        for asset, data in downloaded:
            try:
                parts = await asyncio.to_thread(slice_image, data)
            except Exception as exc:
                state["errors"].append(f"{asset['filename']}: {exc}")
                continue
            tiles += [(asset["id"], n, jpeg) for n, jpeg in enumerate(parts)]
        if not tiles:
            chapter["status"] = "failed"
            chapter["error"] = "No panel images could be downloaded from the page."
            self.store.save_chapter(chapter)
            state["errors"].append(f"{url}: {chapter['error']}")
            return chapter["id"]

        current = state["current"]
        current.update(step="cataloguing", tiles_total=len(tiles))
        chapter["status"] = "cataloguing"
        self.store.save_chapter(chapter)

        roster = self.store.characters(series)
        models = await self._vision_models()
        story = ""
        for start in range(0, len(tiles), TILES_PER_CALL):
            batch = tiles[start:start + TILES_PER_CALL]
            result, model, error = await self._describe(batch, roster, story, models)
            described = (result or {}).get("tiles") if result else None
            for offset, (asset_id, part, _jpeg) in enumerate(batch):
                entry = {"id": f"{chapter['id']}-{start + offset:04d}", "idx": start + offset, "asset_id": asset_id,
                         "part": part}
                item = described[offset] if isinstance(described, list) and offset < len(described) else None
                if isinstance(item, dict):
                    entry.update(_clean_tile(item))
                    entry["model"] = model
                else:
                    entry.update(status="failed", error=error or "The model returned fewer tiles than it was sent.")
                    chapter["failed_tiles"] += 1
                pages.append(entry)
            if result and isinstance(result.get("summary"), str):
                story = (story + " " + result["summary"]).strip()[-1500:]
            if model and model not in chapter["models"]:
                chapter["models"].append(model)
            current["tiles_done"] = min(len(tiles), start + len(batch))

        current["step"] = "summarising"
        wrapup = await self._wrapup(pages, roster)
        rename = wrapup.get("rename") if isinstance(wrapup.get("rename"), dict) else {}
        for entry in pages:
            for person in entry.get("characters") or []:
                person["name"] = str(rename.get(person.get("name"), person.get("name")) or "")
            for line in entry.get("dialogue") or []:
                line["speaker"] = str(rename.get(line.get("speaker"), line.get("speaker")) or "")
        self._merge_roster(roster, pages, wrapup.get("characters"), number, chapter["id"])
        self.store.save_characters(series, roster)
        self.store.save_pages(chapter["id"], series, pages)

        chapter.update(status="done", summary=str(wrapup.get("summary") or story), tiles=len(pages))
        self.store.save_chapter(chapter)
        return chapter["id"]

    async def _vision_models(self) -> list[str]:
        llm = self.service.llm()
        configured = llm.agent_vision_model_override or self.service.settings.agent_vision_model
        return models_llm.resolve_many("vision", configured, await self.service.model_catalogue())

    async def _describe(self, batch, roster: dict, story: str, models: list[str]) -> tuple[dict | None, str, str]:
        prompt = CATALOG_PROMPT.format(roster=_roster_text(roster), story=story or "(this is the start)")
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for n, (_asset, _part, jpeg) in enumerate(batch, 1):
            parts.append({"type": "text", "text": f"Tile {n}:"})
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}})
        failures = []
        for model in models:
            try:
                text, _usage = await self.service.atlas.chat(model, [{"role": "user", "content": parts}], json_mode=True,
                                                             max_tokens=CATALOG_MAX_TOKENS, temperature=0.2,
                                                             max_retries=1, purpose="manhwa catalogue")
            except AtlasError as exc:
                failures.append(f"{model}: {exc}"[:300])
                continue
            result = parse_json(text)
            if result and isinstance(result.get("tiles"), list) and result["tiles"]:
                return result, model, ""
            failures.append(f"{model}: no usable JSON ({(text or '').strip()[:120]!r})")
        return None, "", "; ".join(failures) or "No vision model is configured."

    async def _wrapup(self, pages: list[dict], roster: dict) -> dict:
        lines = []
        for entry in pages:
            if entry.get("status") == "failed":
                continue
            who = ", ".join(f"{p.get('name')} ({p.get('appearance', '')})" for p in entry.get("characters") or [])
            said = " / ".join(f"{d.get('speaker')}: {d.get('text_en') or d.get('text')}" for d in entry.get("dialogue") or [])
            lines.append(f"[{entry['idx']}] {entry.get('scene', '')} | {who} | {said}")
        if not lines:
            return {}
        prompt = WRAPUP_PROMPT.format(roster=_roster_text(roster), catalogue="\n".join(lines)[:60000])
        try:
            model = await self.service.model_for("summary")
            text, _usage = await self.service.atlas.chat(model, [{"role": "user", "content": prompt}], json_mode=True,
                                                         max_tokens=CATALOG_MAX_TOKENS, temperature=0.2,
                                                         max_retries=1, purpose="manhwa summary")
        except Exception as exc:  # the catalogue is still worth keeping without a summary
            log.warning("manhwa: chapter summary failed: %s", exc)
            return {}
        return parse_json(text) or {}

    @staticmethod
    def _merge_roster(roster: dict, pages: list[dict], listed, number: str, chapter_id: str) -> None:
        counts: dict[str, int] = {}
        looks: dict[str, str] = {}
        for entry in pages:
            for person in entry.get("characters") or []:
                name = (person.get("name") or "").strip()
                if not name or re.fullmatch(r"unknown-\d+", name, re.I):
                    continue
                counts[name] = counts.get(name, 0) + 1
                looks.setdefault(name, person.get("appearance") or "")
        for person in listed if isinstance(listed, list) else []:
            if isinstance(person, dict) and person.get("name") in counts:
                looks[person["name"]] = person.get("appearance") or looks.get(person["name"], "")
                aliases = [str(a) for a in person.get("aliases") or [] if a]
                if aliases:
                    known = roster.setdefault(person["name"], {})
                    known["aliases"] = sorted(set(known.get("aliases") or []) | set(aliases))
        for name, count in counts.items():
            known = roster.setdefault(name, {})
            known["appearance"] = looks.get(name) or known.get("appearance", "")
            chapters = known.setdefault("chapters", [])
            if number not in chapters:
                chapters.append(number)
            known.setdefault("first_chapter", number)
            known["last_chapter"] = number
            known["tiles"] = int(known.get("tiles") or 0) + count
            known.setdefault("chapter_ids", [])
            if chapter_id not in known["chapter_ids"]:
                known["chapter_ids"].append(chapter_id)


def series_key(name: str) -> str:
    """"Warrior of the Soulblade" and "warrior-of-the-soulblade" are one series."""
    return re.sub(r"[^0-9a-z]+", "", str(name or "").lower())


def _clean_tile(item: dict) -> dict:
    def text(value) -> str:
        return " ".join(str(value or "").split())

    return {
        "status": "ok",
        "scene": text(item.get("scene")),
        "setting": text(item.get("setting")),
        "mood": text(item.get("mood")),
        "characters": [{"name": text(p.get("name")), "appearance": text(p.get("appearance")),
                        "action": text(p.get("action"))}
                       for p in item.get("characters") or [] if isinstance(p, dict)],
        "dialogue": [{"speaker": text(d.get("speaker")), "text": str(d.get("text") or "").strip(),
                      "lang": text(d.get("lang")), "text_en": str(d.get("text_en") or "").strip()}
                     for d in item.get("dialogue") or [] if isinstance(d, dict)],
        "sfx": [text(s) for s in item.get("sfx") or [] if s],
        "tags": [text(t).lower() for t in item.get("tags") or [] if t],
    }


def _roster_text(roster: dict) -> str:
    if not roster:
        return "(empty)"
    lines = []
    for name, data in list(roster.items())[:60]:
        alias = f" (also {', '.join(data.get('aliases') or [])})" if data.get("aliases") else ""
        lines.append(f"- {name}{alias}: {data.get('appearance', '')}")
    return "\n".join(lines)


def _bump(number: str) -> str:
    try:
        return str(int(float(number)) + 1) if number else ""
    except ValueError:
        return ""
