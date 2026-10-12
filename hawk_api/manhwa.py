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
import os
import random
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse

import httpx

from . import models_llm
from .atlas import AtlasError
from hawk_h3.script import ScriptError, max_segment_seconds, parse_script

from .jobs import NotFound, RequestError, Unavailable
from .schemas import PlanRequest, ReferenceIn, RenderSettings, VideoRequest

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
#: The Manhwa page's own Qwen Image 2.1 settings, kept apart from the Images page's: DATA_DIR/manhwa_settings.json.
#: loras None means "pick them from the folder" (real_loras); steps / cfg None mean the engine's own.
SETTINGS_DEFAULTS = {
    "loras": None, "realism": 0.5, "age_slider": 1.0, "steps": None, "cfg": None, "negative": "", "prompt": "",
    "analyze": True, "prompt_model": "xai/grok-4.3", "plan_after": True, "aspect_ratio": "9:16",
    "segment_seconds": 10.0,
    # The chapter's video: the planner model ("" = the server's) and RenderSettings fields for Render chapter
    # (base model, steps, sampler, continuity, video LoRAs...); anything left out is the server's default.
    "planner_model": "", "render": {},
    # Quality check: a vision model compares each photograph with its panel and, below qa_min_score, rewrites the
    # prompt and has Qwen try again, up to qa_retries more times; the best attempt is kept.
    "qa": True, "qa_model": "deepseek-ai/deepseek-v4.1-flash", "qa_retries": 2, "qa_min_score": 7,
    # The autopilot drafts a chapter's scene context from its catalogue when none was given.
    "auto_context": True,
    # Cast sheet: one photoreal portrait per character, given to Qwen as an extra reference on every panel that
    # character is in, so a face stays the same across panels and chapters. The autopilot makes missing ones.
    "cast_refs": True, "auto_cast": True, "cast_size": 8,
    # Panel triage: text-only and blank panels, and near-duplicates of the panel before, are not converted.
    "triage": True,
    # Video plans group neighbouring panels of one tile into a shot (up to 3 pictures) and time each shot by its
    # dialogue (about 2.5 words a second), instead of one fixed-length segment per panel.
    "group_shots": True,
    # Chapter films get subtitles from the catalogue's dialogue: "file" (an .srt beside the film), "burn" (drawn
    # into the picture), or "off".
    "subtitles": "file",
}
SHOT_PICTURES = 3
WORDS_PER_SECOND = 2.5
SILENT_SHOT_SECONDS = 5.0
MIN_SHOT_SECONDS = 4.0
#: Render settings the Manhwa page sets elsewhere: aspect_ratio and segment_seconds above.
RENDER_OWN_KEYS = ("aspect_ratio", "default_seconds")
#: Autopilot: how long it waits on the slow steps before giving up on them, and how often it looks.
PIPELINE_POLL = 5.0
PLAN_WAIT = 30 * 60
RENDER_WAIT = 12 * 3600
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

#: The opening the anime-to-real LoRAs were trained and published with (Anime to Realism, Civitai 2783657),
#: then what this pipeline needs on top: adults, and no lettering.
CONVERT_SENTENCE = ("Convert the anime image into a natural photorealistic live-action photograph while preserving the "
                    "exact subject count, age category, identity-defining features, pose, composition, clothing, "
                    "objects, background, lighting, and colors.")
REAL_PROMPT = (CONVERT_SENTENCE + " Every person is an adult. Remove all speech bubbles, captions and sound-effect "
               "lettering.")
REAL_NEGATIVE = ("anime, cartoon, illustration, drawing, comic, manga, cel shading, line art, speech bubble, text, "
                 "letters, watermark, childlike, young-looking")

#: The model that looks at each panel and writes its photograph prompt; the vision chain follows it.
#: deepseek-ai/deepseek-v4.1-flash also reads images, at about a quarter of the price.
PANEL_PROMPT_MODEL = "xai/grok-4.3"
PANEL_PROMPT_MAX_TOKENS = 3000  # a reasoning model spends part of this before the JSON

PANEL_PROMPT = """\
You write the instruction for an image-editing model (Qwen Image 2.1 Edit with an anime-to-real LoRA). It
receives the attached manhwa panel as its canvas and must turn it into a realistic photograph of the same
moment. Look at the panel closely and describe exactly what is there, so nothing is lost or invented.

Write "prompt" (80-180 words, plain English, no lists) that:
- starts with exactly this sentence: "{convert}"
- for each person, left to right: an adult age (20s, 30s...), gender, hair, face and expression, clothing (or
  what they wear or don't), pose, what they are doing, and any contact with another person. Use their roster
  names and looks below when they match.
- the setting, time of day, lighting, camera angle and framing (close-up, medium, wide), depth of field.
- says to keep the composition, poses and framing exactly, and to remove every speech bubble, caption,
  sound-effect lettering and drawn effect (hearts, speed lines, sparkles, sweat drops).
Write "negative": 10-25 comma-separated things to keep out (drawn styles, text, artefacts, anatomy errors).

Also answer these about the panel itself:
- "people": the roster names of everyone visible in it (only names from the roster; [] when no one listed is there).
- "content": "people" when it shows a person or part of one, "scenery" for a place or object with no one in it,
  "text_only" when it is only speech bubbles, captions or sound-effect lettering, "blank" when there is nothing
  to see.
- "drawn": true when it is a drawing, illustration or comic art; false when it is a photograph or photoreal.
- "identifiable": true when it shows a real, recognisable person (a photo of someone, or a likeness of a real
  celebrity); false for fictional drawn characters.

Every character in this series is an adult; describe them as adults. If anyone in the panel looks like a child
or a teenager, do not write a prompt: reply {{"skip": "why"}} instead.

Roster (name: appearance):
{roster}
What the catalogue says about this panel: {scene}
What happens in the chapter: {summary}
{context}
Reply with only JSON: {{"drawn": true, "identifiable": false, "people": [], "content": "people", "prompt": "",
"negative": ""}}
"""

QA_PROMPT = """\
You check an anime-to-photograph conversion. Image 1 is the original manhwa panel. Image 2 is the photograph an
image model made from it with this prompt:
{prompt}

Score image 2 from 0 to 10 as a realistic photograph of exactly the moment in image 1:
- pose and body positions match: limbs, hands, head turn, who is where, contact between people, and the sexual
  position when there is one
- camera angle, framing and crop match
- the same number of people, with matching hair, clothing (or nudity) and expressions
- anatomy is right: no extra, missing or fused limbs, fingers or faces, no melted bodies
- it looks photographic, with no anime or drawn look left, and no speech bubbles or lettering
A 7 is a usable photograph with small faults; below 7 something above is clearly wrong.

List the concrete problems. Then write "revised_prompt": the whole prompt rewritten to fix them, describing the
exact pose, limb placement, body orientation, camera angle and framing seen in image 1, 80-200 words, starting with
"{convert}" Describe everyone as adults.

Every character is an adult. If anyone in either image looks like a child or a teenager, set "minor": true.

Reply with only JSON: {{"score": 0, "problems": [], "minor": false, "revised_prompt": ""}}
"""

CAST_PROMPT = ("Make a photorealistic head-and-shoulders portrait photograph of {name}, the person in <image1>: a real "
               "adult, {appearance}. Keep their face shape, eyes, hair style and colour, and skin tone exactly as drawn. "
               "Facing the camera, calm expression, fully clothed in their usual clothes, plain light-grey studio "
               "background, soft even light, 85mm lens. Only {name} in the picture, no text.")
CAST_REF_LINE = "<image{n}> is a photograph of {name}: give {name} exactly that face and hair."

CONTEXT_PROMPT = """\
From this catalogue of one manhwa chapter, write the scene context an image model and a film planner will follow.

Series: {series}, chapter {chapter}
Chapter summary: {summary}
Character roster (name: appearance):
{roster}
Scenes in reading order:
{scenes}

Reply with only JSON:
{{"characters": "each person in this chapter: name, adult age range, build, face, hair, usual clothing -- one line each",
"setting": "where and when it happens, the light and the mood",
"action": "what happens across the chapter, in 2-4 sentences",
"one_scene": true when the whole chapter happens in one place and one continuous stretch of time, else false}}
Every character is an adult; describe them as adults.
"""

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


CONTEXT_FIELDS = ("characters", "setting", "action")
CONTEXT_MAX = 2000


def context_text(context: dict | None) -> str:
    """The user's scene context for a chapter as prompt lines, or "" when there is none."""
    context = context or {}
    lines = [f"{key.capitalize()}: {' '.join(str(context.get(key) or '').split())}"
             for key in CONTEXT_FIELDS if str(context.get(key) or "").strip()]
    if not lines:
        return ""
    head = ("The whole chapter is one continuous scene" if context.get("one_scene") else "Chapter context")
    return f"{head} (from the user; follow it for who the people are, how they look, where and what happens):\n" \
        + "\n".join(lines)


def panel_fingerprint(jpeg: bytes) -> tuple[int, float]:
    """(16x16 average hash, grey-level spread) of a panel: the hash finds a near-duplicate of the panel before,
    the spread an empty one."""
    from PIL import Image, ImageStat

    grey = Image.open(io.BytesIO(jpeg)).convert("L")
    # the spread at a readable size: a 16 x 16 thumbnail averages any texture into flat grey
    detail = grey.resize((256, max(1, round(grey.height * 256 / max(1, grey.width)))))
    pixels = list(grey.resize((16, 16)).getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | (value > mean)
    return bits, ImageStat.Stat(detail).stddev[0]


def near_duplicate(a: int, b: int, bits: int = 12) -> bool:
    return bin(a ^ b).count("1") <= bits


def real_prompt(tile: dict | None, extra: str = "", context: dict | None = None) -> str:
    """REAL_PROMPT plus what the catalogue says is in the panel, so the people keep their looks."""
    parts = [REAL_PROMPT]
    given = context or {}
    for key in CONTEXT_FIELDS:  # the user's word on the people and the place comes before the catalogue's guess
        if str(given.get(key) or "").strip():
            parts.append(f"{key.capitalize()}: {' '.join(str(given[key]).split())}.")
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
    return " ".join(parts)[:2000]


def _spoken(shot: dict) -> list[dict]:
    lines = []
    for said in shot.get("dialogue") or []:
        text = (said.get("text_en") or said.get("text") or "").strip()
        if text:
            lines.append({"speaker": said.get("speaker") or "narration", "text": text})
    return lines


def shot_seconds(lines: list[dict], cap: float) -> float:
    """How long a shot runs: long enough to say its lines at a natural pace, never past the segment cap."""
    words = sum(len(line["text"].split()) for line in lines)
    if not words:
        return min(cap, SILENT_SHOT_SECONDS)
    return round(max(MIN_SHOT_SECONDS, min(cap, 2.0 + words / WORDS_PER_SECOND)), 1)


def group_shots(shots: list[dict], cap: float, together: bool = True, fixed: float | None = None) -> list[dict]:
    """Shots for the film: neighbouring photographs of one tile (one moment) share a segment, up to SHOT_PICTURES,
    unless their lines would run past the cap. Each group: {shots, lines, duration}."""
    groups: list[dict] = []
    for shot in shots:
        lines = _spoken(shot)
        last = groups[-1] if groups else None
        if (together and last and last["shots"][-1].get("tile_id") == shot.get("tile_id")
                and len(last["shots"]) < SHOT_PICTURES
                and shot_seconds(last["lines"] + lines, 10_000) <= cap):
            last["shots"].append(shot)
            last["lines"] += lines
        else:
            groups.append({"shots": [shot], "lines": lines})
    for group in groups:
        group["duration"] = fixed if fixed else shot_seconds(group["lines"], cap)
    return groups


def pack_groups(groups: list[dict], pictures: int = PLAN_PICTURES) -> list[list[dict]]:
    """Plans of whole groups, each with at most ``pictures`` photographs (a render's limit)."""
    parts: list[list[dict]] = []
    for group in groups:
        if parts and sum(len(g["shots"]) for g in parts[-1]) + len(group["shots"]) <= pictures:
            parts[-1].append(group)
        else:
            parts.append([group])
    return parts


def plan_brief(chapter: dict, shots: list[dict], part: int, parts: int, groups: list[dict] | None = None) -> str:
    """The story planner's brief for one part of a chapter: one segment per shot, in reading order, each starting
    from its first photograph, timed by its dialogue, with the lines to speak word for word."""
    if groups is None:  # one shot per photograph
        groups = [{"shots": [shot], "lines": _spoken(shot), "duration": None} for shot in shots]
    lines = [f"A live-action film of {chapter['series']}, chapter {chapter.get('chapter') or '?'}"
             + (f", part {part} of {parts}" if parts > 1 else "") + ".",
             f"The attached pictures are photographs made from the comic's panels, in reading order. Write exactly "
             f"{len(groups)} segment{'s' if len(groups) != 1 else ''}, one per shot below, in the same order. A shot "
             "names its pictures: the segment starts from the first, moves through the others in order, sets "
             "\"pictures\" to exactly those numbers and \"duration\" to the seconds given, and plays out what the "
             "panels show, with the people looking as they do in them.",
             "Characters speak the dialogue below word for word, in English, in the order given; narration is a "
             "voice-over. Keep segments with no dialogue free of speech. Leave references_to_make empty.",
             "All characters are adults."]
    context = chapter.get("context") or {}
    if context_text(context):
        lines.append(context_text(context))
        if context.get("one_scene"):
            lines.append("Keep one place, the same people and one continuous action across every segment: no cuts "
                         "to other locations or times, each segment picking up where the last ended (continuity "
                         "\"inherit\").")
    if chapter.get("summary"):
        lines.append(f"Chapter so far: {chapter['summary']}")
    number = 0
    for index, group in enumerate(groups, 1):
        first = number + 1
        number += len(group["shots"])
        pictures = ", ".join(str(n) for n in range(first, number + 1))
        timing = f", {group['duration']:g} s" if group.get("duration") else ""
        lines.append(f"\nShot {index} -- pictures [{pictures}]{timing}:")
        for offset, shot in enumerate(group["shots"]):
            lines.append(f"<Picture {first + offset}>: {shot.get('scene') or 'see the picture'}")
            people = [f"{p['name']} ({p.get('appearance', '')})" for p in shot.get("characters") or [] if p.get("name")]
            if people and offset == 0:
                lines.append("People: " + "; ".join(people))
        for said in group["lines"]:
            who = said["speaker"]
            lines.append(f'{"Narration (voice-over)" if who.lower() == "narration" else who}: "{said["text"]}"')
    return "\n".join(lines)


def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def build_srt(parts: list[dict]) -> str:
    """Subtitles for a chapter film. Each part: {"duration": its real length in seconds, "segments": [{"seconds":
    planned length, "lines": [{speaker, text}]}]}. A part's segments are stretched to its real length (renders
    trim and blend at the joins), and each segment's lines share its time by word count."""
    cues, offset = [], 0.0
    for part in parts:
        segments = part["segments"]
        planned = sum(max(0.1, s["seconds"]) for s in segments) or 1.0
        scale = part["duration"] / planned if part.get("duration") else 1.0
        start = offset
        for segment in segments:
            length = max(0.1, segment["seconds"]) * scale
            lines = [l for l in segment.get("lines") or [] if l.get("text")]
            words = sum(max(1, len(l["text"].split())) for l in lines) or 1
            at = start + 0.2
            usable = max(0.5, length - 0.4)
            for line in lines:
                span = usable * max(1, len(line["text"].split())) / words
                text = line["text"] if line.get("speaker", "").lower() != "narration" else f"<i>{line['text']}</i>"
                cues.append((at, at + span, text))
                at += span
            start += length
        offset += part["duration"] if part.get("duration") else planned
    return "".join(f"{n}\n{_srt_time(a)} --> {_srt_time(b)}\n{text}\n\n" for n, (a, b, text) in enumerate(cues, 1))


def media_seconds(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def burn_subtitles(video: str, srt: str, out: str) -> None:
    """Draw the subtitles into the picture (re-encodes the video; audio is copied)."""
    escaped = srt.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    result = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", video, "-vf",
                             f"subtitles='{escaped}':force_style='Fontsize=14,Outline=1,MarginV=40'",
                             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                             "-c:a", "copy", "-movflags", "+faststart", out], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not burn the subtitles: {result.stderr.strip()[-300:]}")


class SkipPanel(Exception):
    """A panel that must not be converted (someone reads as a minor)."""


def _video_signature(path: str) -> tuple:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return tuple(sorted(tuple(sorted(s.items())) for s in json.loads(out).get("streams", [])))


def concat_videos(paths: list[str], out: str) -> bool:
    """Join clips end to end into out. True when it was a stream copy (identical codecs, size, frame rate and
    audio layout -- the parts of one chapter), False when the clips had to be re-encoded to match."""
    listing = out + ".txt"
    with open(listing, "w", encoding="utf-8") as handle:
        for path in paths:
            handle.write("file '" + os.path.abspath(path).replace("'", "'\\''") + "'\n")
    same = len({_video_signature(p) for p in paths}) == 1
    if same:
        cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing,
               "-c", "copy", "-movflags", "+faststart", out]
    else:
        cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing,
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not join the parts: {result.stderr.strip()[-400:]}")
    return same


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
    CREATE TABLE IF NOT EXISTS manhwa_pipelines (
        id TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL, inputs TEXT NOT NULL, created_at REAL NOT NULL);
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

    def save_pipeline(self, state: dict, inputs: dict) -> None:
        """An autopilot's progress, and what it needs to start again (inputs: chapter, cookies, context -- kept
        out of the state the API shows)."""
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO manhwa_pipelines VALUES (?, ?, ?, ?, ?)",
                             (state["id"], state["status"], json.dumps(state, ensure_ascii=False),
                              json.dumps(inputs, ensure_ascii=False), state["created_at"]))
            self._db.commit()

    def pipelines(self, statuses: tuple[str, ...] = ()) -> list[tuple[dict, dict]]:
        sql, args = "SELECT data, inputs FROM manhwa_pipelines", ()
        if statuses:
            sql, args = sql + f" WHERE status IN ({','.join('?' * len(statuses))})", statuses
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY created_at DESC LIMIT 50", args).fetchall()
        return [(json.loads(data), json.loads(inputs)) for data, inputs in rows]

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
        self.settings_path = os.path.join(os.path.dirname(os.path.abspath(self.store_path(service))),
                                          "manhwa_settings.json")
        self.conversions: dict[str, dict] = {}
        self.pipelines: dict[str, dict] = {}
        self.cast_runs: dict[str, dict] = {}
        self._pipeline_inputs: dict[str, dict] = {}
        self._cast_tasks: dict[str, asyncio.Task] = {}
        self._pipeline_tasks: dict[str, asyncio.Task] = {}
        self.poll = PIPELINE_POLL
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
        A run working on what is deleted is cancelled first, so nothing writes the rows back afterwards: for a
        series, every run on it; for one chapter, only a run cataloguing that chapter right now -- deleting
        chapter 55 must not stop the run that is busy with chapter 86."""
        key = series_key(series)
        series = next((s["series"] for s in self.store.series() if series_key(s["series"]) == key), series)
        target = self.store.get_chapter(chapter_id) if chapter_id else None
        cancelled = []
        for run in list(self.runs.values()):
            if run["status"] != "running" or series_key(run["series"]) != key:
                continue
            if chapter_id and not (target and (run.get("current") or {}).get("url") == target["url"]):
                continue
            await self.cancel_and_wait(run["id"])
            cancelled.append(run["id"])
        for state in list(self.conversions.values()):  # or it would write photographs back into a deleted chapter
            if state["status"] == "running" and series_key(state["series"]) == key and (
                    not chapter_id or state["chapter_id"] == chapter_id):
                self.cancel_conversion(state["id"])
                task = self._convert_tasks.get(state["id"])
                if task is not None:
                    await asyncio.wait([task], timeout=30)
                cancelled.append(state["id"])
        chapters = self.store.list_chapters(series)
        if chapter_id:
            chapters = [c for c in chapters if c["id"] == chapter_id]
            if not chapters:
                raise NotFound(f"No catalogued chapter {chapter_id!r} in {series!r}.")
        # Photographs and cut panels made from these chapters, wherever they were moved to since
        made: set[str] = set()
        for chapter in chapters:
            for page in self.store.pages(chapter["id"]):
                for real in page.get("real") or []:
                    made.update(real.get("asset_ids") or [])
                    if real.get("panel_asset_id"):
                        made.add(real["panel_asset_id"])
        named = self.store.delete_chapters([c["id"] for c in chapters])
        if chapter_id:
            self._forget_chapter(series, chapter_id)
        else:
            self.store.delete_characters(series)
        deleted_images = 0
        if images:
            prefixes = [f"manhwa/{series}/{c['chapter'] or 'chapter'}" for c in chapters] if chapter_id \
                else [f"manhwa/{series}/"]
            ids = set(named) | made
            for asset in self.service.store.all_assets():
                collection = str(asset.get("collection") or "")
                # pages, panels (.../panels) and photographs (.../real); rendered videos are kept
                if asset.get("kind", "image") == "image" and any(
                        collection == p.rstrip("/") or collection.startswith(p.rstrip("/") + "/") for p in prefixes):
                    ids.add(asset["id"])
            ids = {i for i in ids if (self.service.store.get_asset(i) or {}).get("kind", "image") == "image"}
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
        tasks = (list(self._tasks.values()) + list(self._convert_tasks.values()) + list(self._pipeline_tasks.values())
                 + list(self._cast_tasks.values()))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ----------------------------------------------------------------- photographs from panels

    # ----------------------------------------------------------------- autopilot: the whole pipeline

    async def start_pipeline(self, url: str, series: str, chapter: str = "", *, cookies: str = "",
                             follow_next: bool = False, max_chapters: int = 1, context: dict | None = None,
                             render: bool = True) -> dict:
        """Catalogue the chapter(s) at url, make their photographs, plan their videos and render them, with the
        Manhwa page's saved Photograph and Video settings and no further input. Poll pipeline()."""
        for other in self.pipelines.values():
            if other["status"] == "running":
                raise RequestError(f"Autopilot {other['id']} is still running ({other['series']}). Cancel it or "
                                   "wait for it to finish.", {"pipeline_id": other["id"]})
        if not re.match(r"^https?://", (url or "").strip()):
            raise RequestError("Give the chapter's http(s) URL.")
        if not " ".join((series or "").split()):
            raise RequestError("Give the series name.")
        state = {
            "id": uuid.uuid4().hex[:12], "status": "running", "series": series.strip(), "url": url.strip(),
            "follow_next": bool(follow_next), "max_chapters": max(1, min(int(max_chapters or 1), MAX_CHAPTERS)),
            "render": bool(render), "stage": "cataloguing", "run_id": "", "chapters": [], "errors": [],
            "cancel_requested": False, "created_at": time.time(), "finished_at": None,
        }
        self.pipelines[state["id"]] = state
        self._launch_pipeline(state, {"chapter": (chapter or "").strip(), "cookies": (cookies or "").strip(),
                                      "context": context or {}})
        return state

    def _launch_pipeline(self, state: dict, inputs: dict) -> None:
        self._pipeline_inputs[state["id"]] = inputs
        self._persist(state)
        self._pipeline_tasks[state["id"]] = asyncio.create_task(
            self._pipeline(state, inputs.get("chapter", ""), inputs.get("cookies", ""), inputs.get("context") or {}))

    def _persist(self, state: dict) -> None:
        """Save the autopilot's progress, so a runtime that restarts carries on from it."""
        try:
            # Cookies are a login to someone's site: kept in memory only, never in the database (or its Drive copy).
            inputs = {k: v for k, v in self._pipeline_inputs.get(state["id"], {}).items() if k != "cookies"}
            self.store.save_pipeline(state, inputs)
        except Exception as exc:  # never fail the run over its bookkeeping
            log.warning("manhwa: could not save autopilot %s: %s", state["id"], exc)

    async def resume_pipelines(self, wait_for_gpu: float = 600.0) -> list[str]:
        """Start again every autopilot a restart interrupted (saved as running or interrupted), once ComfyUI
        answers. Each step skips what is already done: catalogued chapters, converted panels, written plans,
        rendered parts, merged films."""
        saved = [(state, inputs) for state, inputs in self.store.pipelines(("running", "interrupted"))
                 if state["id"] not in self.pipelines]
        if not saved:
            return []
        deadline = time.monotonic() + wait_for_gpu
        while time.monotonic() < deadline:
            try:
                if await self.service.comfy.object_info("KSampler"):
                    break
            except Exception:
                pass
            await asyncio.sleep(10)
        resumed = []
        for state, inputs in saved[:1]:  # one autopilot at a time, as when started by hand
            state.update(status="running", resumed_at=time.time(), finished_at=None, cancel_requested=False)
            state["errors"] = state.get("errors", [])[-5:]
            self.pipelines[state["id"]] = state
            self._launch_pipeline(state, inputs)
            resumed.append(state["id"])
        for state, inputs in saved[1:]:
            state["status"] = "interrupted"
            self.store.save_pipeline(state, inputs)
        return resumed

    def pipeline(self, pipeline_id: str) -> dict:
        state = self.pipelines.get(pipeline_id) or next(
            (saved for saved, _ in self.store.pipelines() if saved["id"] == pipeline_id), None)
        if state is None:
            raise NotFound(f"No autopilot {pipeline_id!r} (forgotten when the API restarts; its chapters, "
                           "photographs and videos are kept).")
        return state

    def recent_pipelines(self) -> list[dict]:
        known = dict(self.pipelines)
        for state, _ in self.store.pipelines():  # earlier runtimes' autopilots too
            known.setdefault(state["id"], state)
        return sorted(known.values(), key=lambda s: s["created_at"], reverse=True)[:20]

    async def cancel_pipeline(self, pipeline_id: str) -> dict:
        """Stop the autopilot and whatever step it is on. Renders already queued keep going (cancel them in Jobs);
        everything made so far is kept."""
        state = self.pipeline(pipeline_id)
        if state["status"] == "running":
            state["cancel_requested"] = True
            task = self._pipeline_tasks.get(pipeline_id)
            if task is not None:
                task.cancel()
                await asyncio.wait([task], timeout=30)
        return state

    async def _pipeline(self, state: dict, chapter: str, cookies: str, context: dict) -> None:
        try:
            # 1. catalogue every chapter first: it needs no GPU, and the list of chapters is known after it
            run = await self.start(state["url"], state["series"], chapter, cookies=cookies,
                                   follow_next=state["follow_next"], max_chapters=state["max_chapters"])
            state["run_id"], state["series"] = run["id"], run["series"]
            while run["status"] == "running":
                await asyncio.sleep(self.poll)
            if run["errors"]:
                state["errors"] += [f"catalogue: {e}" for e in run["errors"][-3:]]
            ids = list(dict.fromkeys(run["chapter_ids"] + [s["id"] for s in run["skipped"]]))
            state["chapters"] = []  # rebuilt on a resume: chapters done before come back as skipped by the run
            for chapter_id in ids:
                row = self.store.get_chapter(chapter_id) or {}
                state["chapters"].append({"chapter_id": chapter_id, "chapter": row.get("chapter", ""),
                                          "stage": "catalogued" if row.get("status") == "done" else "skipped",
                                          "conversion_id": "", "plans": [], "renders": [], "error":
                                          "" if row.get("status") == "done" else f"catalogue {row.get('status')}"})
            work = [c for c in state["chapters"] if c["stage"] == "catalogued"]
            self._persist(state)
            if not work:
                raise RequestError(run["stop_reason"] or "No chapter was catalogued.")

            # 2. the cast sheet first, so every chapter's photographs share one face per character
            conf = self.settings()
            if conf["cast_refs"] and conf["auto_cast"]:
                state["stage"] = "cast portraits"
                cast = {"id": "", "series": state["series"], "done": [], "skipped": [], "errors": []}
                try:
                    await self._cast(cast, None, False)
                except Exception as exc:  # no portraits: the photographs are made without them
                    cast["errors"].append(str(exc))
                state["cast"] = {"made": [c["name"] for c in cast["done"]], "errors": cast["errors"][:3]}

            # 3. photographs, then the plans (LLM only) while the next chapter converts on the GPU
            for item in work:
                state["stage"] = f"photographs: ch {item['chapter']}"
                has_context = bool((self.store.get_chapter(item["chapter_id"]) or {}).get("context"))
                if context and not has_context:
                    self.set_context(item["chapter_id"], context)
                elif not has_context and self.settings()["auto_context"]:
                    try:  # the catalogue's summary, cast and scenes, as the context every prompt and plan follows
                        item["context"] = "drafted"
                        self.set_context(item["chapter_id"], await self.draft_context(item["chapter_id"]))
                    except Exception as exc:
                        item["context"] = f"not drafted: {exc}"[:200]
                item["stage"] = "photographs"
                conversion = await self.convert(item["chapter_id"], plan_after=False, skip_done=True)
                item["conversion_id"] = conversion["id"]
                while conversion["status"] == "running":
                    await asyncio.sleep(self.poll)
                if not self.shots(item["chapter_id"]):
                    item.update(stage="failed", error="No photographs were made: " + "; ".join(
                        [r.get("error", "") for r in conversion["results"] if r.get("error")][:2]
                        + conversion["errors"][:1])[:400])
                    continue
                existing = (self.store.get_chapter(item["chapter_id"]) or {}).get("plans") or []
                usable = [p for p in existing if (self.service.store.get_job(p["job_id"]) or {}).get("status")
                          in ("planning", "queued", "done")]
                if existing and len(usable) == len(existing):
                    item["plans"] = existing
                else:
                    item["plans"] = (await self.plan_chapter(item["chapter_id"]))["plans"]
                item["stage"] = "planning"
                self._persist(state)

            # 3. render each chapter as soon as all its plans are written
            if state["render"]:
                for item in work:
                    if item["stage"] != "planning":
                        continue
                    state["stage"] = f"waiting for plans: ch {item['chapter']}"
                    if not await self._wait_plans(item):
                        continue
                    state["stage"] = f"rendering: ch {item['chapter']}"
                    out = await self.render_chapter(item["chapter_id"])
                    item["renders"] = [r["job_id"] for r in out["started"] + out["already"]]
                    item["stage"] = "rendering" if item["renders"] else "failed"
                    if not item["renders"]:
                        item["error"] = "No plan could be rendered."
                    self._persist(state)

                # 4. watch the videos until they are finished
                state["stage"] = "waiting for videos"
                deadline = time.monotonic() + RENDER_WAIT
                while time.monotonic() < deadline:
                    pending = False
                    for item in work:
                        if item["stage"] != "rendering":
                            continue
                        jobs = [self.service.store.get_job(j) or {} for j in item["renders"]]
                        if any(j.get("status") in ("queued", "rendering", "planning") for j in jobs):
                            pending = True
                            continue
                        failed = [j.get("id", "?") for j in jobs if j.get("status") != "done"]
                        if failed:
                            item.update(stage="failed", error=f"Video job(s) {', '.join(failed)} did not finish.")
                            continue
                        item["stage"] = "merging"
                        try:  # one film per chapter
                            film = (await self.merge_chapter(item["chapter_id"]))["film"]
                            item.update(stage="done", film_asset_id=film["asset_id"])
                        except Exception as exc:  # the parts are rendered; the film can be merged again by hand
                            item.update(stage="failed", error=f"Merge: {exc}"[:400])
                    self._persist(state)
                    if not pending:
                        break
                    await asyncio.sleep(self.poll * 6)
            else:
                for item in work:
                    if item["stage"] == "planning" and await self._wait_plans(item):
                        item["stage"] = "planned"
            state["stage"] = "finished"
            state["status"] = "done"
        except asyncio.CancelledError:
            if not state["cancel_requested"]:  # the API is shutting down: carry on after the restart
                state["status"] = "interrupted"
                raise
            state["status"] = "cancelled"
            state["stage"] = "cancelled"
            for run_id in [state.get("run_id")]:
                if run_id and self.runs.get(run_id, {}).get("status") == "running":
                    self.cancel(run_id)
            for item in state["chapters"]:
                conversion = self.conversions.get(item.get("conversion_id"))
                if conversion and conversion["status"] == "running":
                    self.cancel_conversion(conversion["id"])
            raise
        except Exception as exc:
            log.exception("manhwa autopilot %s failed", state["id"])
            state["status"] = "failed"
            state["errors"].append(f"{type(exc).__name__}: {exc}"[:500])
        finally:
            state["finished_at"] = time.time()
            self._pipeline_tasks.pop(state["id"], None)
            self._persist(state)

    async def _wait_plans(self, item: dict) -> bool:
        """Wait for a chapter's plan jobs; a failed one is retried once. False when none can be rendered."""
        retried: set[str] = set()
        deadline = time.monotonic() + PLAN_WAIT
        while time.monotonic() < deadline:
            statuses = {}
            for plan in item["plans"]:
                job = self.service.store.get_job(plan["job_id"]) or {}
                status = job.get("status", "missing")
                if status in ("failed", "cancelled") and plan["job_id"] not in retried:
                    retried.add(plan["job_id"])
                    try:
                        await self.service.retry(plan["job_id"])
                        status = "planning"
                    except Exception as exc:
                        item["error"] = f"plan retry: {exc}"[:300]
                statuses[plan["job_id"]] = status
            if all(s in ("done", "failed", "cancelled", "missing") for s in statuses.values()):
                done = [j for j, s in statuses.items() if s == "done"]
                if len(done) < len(statuses):
                    item["error"] = f"{len(statuses) - len(done)} of {len(statuses)} plan(s) failed after a retry."
                if not done:
                    item["stage"] = "failed"
                return bool(done)
            await asyncio.sleep(self.poll)
        item.update(stage="failed", error="The plans were not written within 30 minutes.")
        return False

    @staticmethod
    def store_path(service) -> str:
        return service.settings.db_path

    def settings(self) -> dict:
        """The stored Manhwa photograph settings over the defaults."""
        try:
            with open(self.settings_path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError):
            stored = {}
        stored = stored if isinstance(stored, dict) else {}
        return {key: stored.get(key, default) for key, default in SETTINGS_DEFAULTS.items()}

    def save_settings(self, changes: dict) -> dict:
        """Merge ``changes`` (already validated by ManhwaSettingsIn) into the stored settings. loras=[] goes back
        to picking them from the folder."""
        current = self.settings()
        for key, value in changes.items():
            if key in SETTINGS_DEFAULTS:
                current[key] = value
        if current.get("loras") == []:
            current["loras"] = None
        current["render"] = {k: v for k, v in (current.get("render") or {}).items()
                             if k not in RENDER_OWN_KEYS and v not in (None, "")}
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        tmp = f"{self.settings_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, self.settings_path)
        return current

    def set_context(self, chapter_id: str, context: dict) -> dict:
        """The user's scene context for one chapter -- characters, setting, action, and whether the chapter is one
        continuous scene. It steers every panel prompt and the chapter's video plans."""
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        clean = {key: str(context.get(key) or "").strip()[:CONTEXT_MAX] for key in CONTEXT_FIELDS}
        clean["one_scene"] = bool(context.get("one_scene"))
        chapter["context"] = clean if any(clean[key] for key in CONTEXT_FIELDS) or clean["one_scene"] else {}
        self.store.save_chapter(chapter)
        return {"chapter_id": chapter_id, "context": chapter["context"]}

    async def settings_view(self) -> dict:
        """The settings, what "automatic" LoRAs would be right now, and every installed Qwen 2.1 LoRA to pick from."""
        current = self.settings()
        try:
            auto = await self.real_loras(current["realism"], current["age_slider"])
            auto_error = ""
        except Exception as exc:  # no anime-to-real LoRA, or ComfyUI down: the settings still load and save
            auto, auto_error = [], str(exc) or type(exc).__name__
        try:
            items = [i for i in await self.service.local_images.catalogue("qwen21") if i.installed]
        except Exception:
            items = []
        available = [{"file": i.file, "label": i.label or i.file.rsplit("/", 1)[-1], "kind": i.kind,
                      "strength": i.strength, "range": list(i.range), "notes": i.notes, "trigger": i.trigger}
                     for i in items]
        return {"settings": current, "auto_loras": auto, "auto_error": auto_error, "available": available,
                "prompt_models": ["xai/grok-4.3", "deepseek-ai/deepseek-v4.1-flash"],
                "render_defaults": RenderSettings().model_dump(exclude={"loras"})}

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
                      prompt: str | None = None, realism: float | None = None, age_slider: float | None = None,
                      max_panels: int = MAX_CONVERT_PANELS, plan_after: bool | None = None,
                      analyze: bool | None = None, prompt_model: str | None = None, steps: int | None = None,
                      cfg: float | None = None, negative: str | None = None, qa: bool | None = None,
                      qa_retries: int | None = None, skip_done: bool = False) -> dict:
        """Turn a catalogued chapter's panels into photographs with Qwen Image 2.1 on this GPU. Runs in the
        background, one panel at a time; poll conversion(). tile_ids limits it to the panels in those tiles.
        Anything left as None comes from the Manhwa page's settings."""
        conf = self.settings()

        def pick(value, key):
            return conf[key] if value is None else value
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        for other in self.conversions.values():
            if other["status"] == "running":
                # one GPU: two conversions would only queue behind each other and time out
                raise RequestError(f"Conversion {other['id']} is still running. Cancel it or wait for it to finish.",
                                   {"conversion_id": other["id"]})
        if loras:
            chosen = [dict(l) for l in loras]
        elif conf["loras"]:
            chosen = [dict(l) for l in conf["loras"]]
        else:
            chosen = await self.real_loras(pick(realism, "realism"), pick(age_slider, "age_slider"))
        state = {
            "id": uuid.uuid4().hex[:12], "status": "running", "chapter_id": chapter_id, "series": chapter["series"],
            "chapter": chapter["chapter"], "loras": chosen, "tile_ids": list(tile_ids or []),
            "panels_total": 0, "panels_done": 0, "converted": 0, "refused": 0, "failed": 0, "results": [],
            "errors": [], "cancel_requested": False, "created_at": time.time(), "finished_at": None,
            "collection": f"manhwa/{chapter['series']}/{chapter['chapter'] or 'chapter'}/real",
            "plan_after": bool(pick(plan_after, "plan_after")), "plans": [], "analyze": bool(pick(analyze, "analyze")),
            "skipped": 0, "prompt_model": (pick(prompt_model, "prompt_model") or "").strip() or PANEL_PROMPT_MODEL,
            "steps": pick(steps, "steps"), "cfg": pick(cfg, "cfg"), "negative": pick(negative, "negative") or "",
            "qa": bool(pick(qa, "qa")), "qa_model": conf["qa_model"] or "deepseek-ai/deepseek-v4.1-flash",
            "qa_retries": max(0, min(int(pick(qa_retries, "qa_retries") or 0), 5)),
            "qa_min_score": float(conf["qa_min_score"] or 7), "retried": 0,
            "cast_refs": bool(conf["cast_refs"]), "triage": bool(conf["triage"]), "skip_done": bool(skip_done),
        }
        self.conversions[state["id"]] = state
        self._convert_tasks[state["id"]] = asyncio.create_task(
            self._convert(state, pick(prompt, "prompt") or "", max(1, min(int(max_panels or 1), MAX_CONVERT_PANELS))))
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
        roster = self.store.characters(series) if state["analyze"] else {}
        chapter_row = self.store.get_chapter(chapter_id) or {}
        context = chapter_row.get("context") or {}
        summary = chapter_row.get("summary") or ""
        models = await self._panel_prompt_models(state["prompt_model"]) if state["analyze"] else []
        qa_models = await self._panel_prompt_models(state["qa_model"]) if state["qa"] else []
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
            if state["skip_done"]:  # resuming: panels already turned into photographs are left as they are
                done = self._converted_panels(pages)
                work = [w for w in work if (w[0]["id"], w[1]) not in done]
            work = work[:max_panels]
            state["panels_total"] = len(work)
            portraits = self._portraits(series) if state["cast_refs"] else {}
            previous_hash = None
            for asset, number, jpeg, tile in work:
                if state["triage"]:
                    fingerprint, spread = await asyncio.to_thread(panel_fingerprint, jpeg)
                    reason = ("blank panel" if spread < 6 else
                              "same picture as the panel before" if previous_hash is not None
                              and near_duplicate(previous_hash, fingerprint) else "")
                    previous_hash = fingerprint
                    if reason:
                        state["results"].append({"tile_id": (tile or {}).get("id", ""), "page": asset.get("filename"),
                                                 "panel": number, "status": "skipped", "error": f"Triage: {reason}"})
                        state["skipped"] += 1
                        state["panels_done"] += 1
                        self._mark_skipped(tile, asset["id"], number, reason, pages, chapter_id, series)
                        continue
                stem = (asset.get("filename") or "page").rsplit(".", 1)[0]
                panel = await self.service.add_asset(
                    f"{stem}_panel{number:02d}.jpg", io.BytesIO(jpeg), "image/jpeg", len(jpeg),
                    collection=state["collection"].rsplit("/", 1)[0] + "/panels", tags=["manhwa", series, "panel"],
                    source={"type": "manhwa", "from_asset": asset["id"], "panel": number,
                            "tile_id": (tile or {}).get("id", "")})
                row = {"tile_id": (tile or {}).get("id", ""), "panel_asset_id": panel["id"], "page": asset.get("filename"),
                       "panel": number}
                refs: list[tuple[str, str]] = []
                prompt, negative = real_prompt(tile, extra, context), REAL_NEGATIVE
                if models:
                    written = await self._panel_prompt(jpeg, tile, roster, models, context, summary)
                    if written.get("skip"):
                        row.update(status="skipped", error=f"Not converted: {written['skip']}"[:400])
                        state["skipped"] += 1
                        state["results"].append(row)
                        state["panels_done"] += 1
                        continue
                    if state["triage"] and written.get("content") in ("text_only", "blank"):
                        self._drop(panel["id"])
                        self._mark_skipped(tile, asset["id"], number, written["content"], pages, chapter_id, series)
                        row.update(status="skipped", error=f"Triage: {written['content'].replace('_', ' ')} panel")
                        state["skipped"] += 1
                        state["results"].append(row)
                        state["panels_done"] += 1
                        continue
                    refs = [(portraits[n], n) for n in written.get("people") or [] if n in portraits][:3]
                    if refs:
                        row["cast"] = [n for _, n in refs]
                    if written.get("prompt"):
                        self._record_person_check(panel["id"], written)
                        prompt = (written["prompt"] + (" " + extra.strip() if extra.strip() else ""))[:2000]
                        negative = written.get("negative") or REAL_NEGATIVE
                        row["prompt_by"] = written["model"]
                    elif written.get("error"):
                        row["prompt_note"] = f"Template prompt used: {written['error']}"[:300]
                row["prompt"] = prompt
                try:
                    if state["negative"].strip():
                        negative = f"{negative}, {state['negative'].strip()}"
                    ids = await self._make_photo(state, panel["id"], jpeg, prompt, negative, row, qa_models,
                                                 refs if models else [])
                except SkipPanel as exc:
                    row.update(status="skipped", error=f"Not converted: {exc}"[:400])
                    state["skipped"] += 1
                    self._mark_skipped(tile, asset["id"], number, str(exc), pages, chapter_id, series)
                except Exception as exc:  # one panel failing (a refusal, ComfyUI down) does not end the run
                    refused = isinstance(exc, RequestError) and str(exc).startswith("Refused")
                    row.update(status="refused" if refused else "failed", error=str(exc)[:400])
                    state["refused" if refused else "failed"] += 1
                else:
                    for image_id in ids:
                        self.service.update_asset(image_id, collection=state["collection"],
                                                  add_tags=["manhwa", series, "real"])
                    shown = self.service.store.get_asset(ids[0]) if ids else None
                    row.update(status="ok", asset_ids=ids,
                               thumb_url=self.service.asset_view(shown).get("thumb_url") if shown else None)
                    state["converted"] += 1
                    if tile is not None:
                        tile.setdefault("real", []).append({"panel_asset_id": panel["id"], "asset_ids": ids,
                                                            "page_asset_id": asset["id"], "panel": number})
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

    async def plan_chapter(self, chapter_id: str, *, aspect_ratio: str | None = None,
                           segment_seconds: float | None = None, model: str | None = None) -> dict:
        """Start story-planner jobs that turn the chapter's photographs into a film: one plan per nine
        photographs (a render's picture limit), one segment per photograph, the tiles' dialogue spoken word
        for word. Render each finished plan with plan_job_id."""
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        conf = self.settings()
        model = model or conf["planner_model"] or None
        aspect_ratio = aspect_ratio or conf["aspect_ratio"] or "9:16"
        segment_seconds = segment_seconds or conf["segment_seconds"] or PLAN_SEGMENT_SECONDS
        shots = self.shots(chapter_id)
        if not shots:
            raise RequestError("This chapter has no photographs yet: convert its panels first.")
        cap = min(max_segment_seconds(), 15.0)
        groups = group_shots(shots, cap, together=bool(conf["group_shots"]),
                             fixed=None if conf["group_shots"] else min(cap, float(segment_seconds)))
        parts = pack_groups(groups)
        plans = []
        for number, part in enumerate(parts, 1):
            pictures = [shot for group in part for shot in group["shots"]]
            try:
                request = self._plan_request(chapter, pictures, number, len(parts), aspect_ratio, segment_seconds,
                                             model, part)
            except ValueError as exc:  # pydantic: an aspect ratio or length the planner does not take
                raise RequestError(str(exc).splitlines()[0] if str(exc) else "Bad plan settings.") from None
            job = await self.service.create_plan(request)
            plans.append({"part": number, "job_id": job["id"], "pictures": [s["asset_id"] for s in pictures],
                          "segments": len(part),
                          # kept for the chapter film's subtitles: what is said in each segment, and for how long
                          "shots": [{"duration": g["duration"], "lines": g["lines"]} for g in part]})
        chapter["plans"] = plans
        self.store.save_chapter(chapter)
        return {"chapter_id": chapter_id, "photographs": len(shots), "plans": plans}

    async def render_chapter(self, chapter_id: str, *, parts: list[int] | None = None, force: bool = False,
                             settings: dict | None = None) -> dict:
        """Start a video render for each finished plan of the chapter. Plans still being written are listed in
        not_ready (render again when they are done); a part already rendering or rendered is left alone unless
        force. settings are RenderSettings fields over the Manhwa defaults (aspect ratio, segment length)."""
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        plans = chapter.get("plans") or []
        if not plans:
            raise RequestError("This chapter has no video plans yet: Plan video first.")
        conf = self.settings()
        try:
            render = RenderSettings(**{**(conf["render"] or {}), "aspect_ratio": conf["aspect_ratio"],
                                       "default_seconds": conf["segment_seconds"], **(settings or {})})
        except ValueError as exc:
            raise RequestError(str(exc).splitlines()[0] if str(exc) else "Bad render settings.") from None
        done = {r["plan_job_id"]: r for r in chapter.get("renders") or []}
        started, not_ready, kept = [], [], []
        for plan in plans:
            if parts and plan["part"] not in parts:
                continue
            job = self.service.store.get_job(plan["job_id"])
            status = (job or {}).get("status", "missing")
            if status != "done":
                not_ready.append({"part": plan["part"], "plan_job_id": plan["job_id"], "status": status,
                                  **({"error": job.get("error")} if job and job.get("error") else {})})
                continue
            prior = done.get(plan["job_id"])
            prior_job = self.service.store.get_job(prior["job_id"]) if prior else None
            if prior_job and prior_job["status"] not in ("failed", "cancelled") and not force:
                kept.append({**prior, "status": prior_job["status"]})
                continue
            video = await self.service.create_video(VideoRequest(plan_job_id=plan["job_id"], settings=render))
            row = {"part": plan["part"], "plan_job_id": plan["job_id"], "job_id": video["id"]}
            done[plan["job_id"]] = row
            started.append(row)
        chapter["renders"] = sorted(done.values(), key=lambda r: r["part"])
        self.store.save_chapter(chapter)
        return {"chapter_id": chapter_id, "started": started, "already": kept, "not_ready": not_ready}

    # ----------------------------------------------------------------- one film per chapter

    def merge_status(self, chapter_id: str) -> dict:
        """Whether the chapter's videos can be joined: every render finished, and what film exists already."""
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        renders = sorted(chapter.get("renders") or [], key=lambda r: r["part"])
        statuses = [(r, (self.service.store.get_job(r["job_id"]) or {}).get("status", "missing")) for r in renders]
        return {"chapter": chapter, "renders": renders,
                "pending": [{"part": r["part"], "job_id": r["job_id"], "status": st} for r, st in statuses if st != "done"]}

    async def merge_chapter(self, chapter_id: str, *, force: bool = False) -> dict:
        """Join the chapter's rendered parts, in part order, into one video in Media (manhwa/<series>/<chapter>),
        copied to Drive like other renders. Stream copy -- instant and lossless -- when the parts match, as parts
        rendered with the same settings do; re-encoded otherwise. The same parts are not joined twice unless force."""
        status = self.merge_status(chapter_id)
        chapter, renders = status["chapter"], status["renders"]
        if not renders:
            raise RequestError("This chapter has no rendered videos yet: Render chapter first.")
        if status["pending"]:
            raise RequestError("Not every part has finished rendering: " + ", ".join(
                f"part {p['part']} is {p['status']}" for p in status["pending"]) + ".", {"pending": status["pending"]})
        jobs = [r["job_id"] for r in renders]
        film = chapter.get("film") or {}
        if not force and film.get("jobs") == jobs and self.service.store.get_asset(film.get("asset_id") or ""):
            return {"chapter_id": chapter_id, "film": film, "merged": False}
        if not shutil.which("ffmpeg"):
            raise Unavailable("ffmpeg is not installed on the server; run the notebook's ffmpeg cell.")
        work = tempfile.mkdtemp(prefix="hawk_film_")
        try:
            parts = [await self._part_file(job_id, os.path.join(work, f"part_{n:03d}.mp4"))
                     for n, job_id in enumerate(jobs, 1)]
            out = os.path.join(work, "film.mp4")
            copied = await asyncio.to_thread(concat_videos, parts, out)
            mode = str(self.settings().get("subtitles") or "off")
            srt, burned = "", False
            if mode in ("file", "burn"):
                srt = await self._chapter_srt(chapter, renders, parts)
            if srt and mode == "burn":
                srt_path = os.path.join(work, "film.srt")
                with open(srt_path, "w", encoding="utf-8") as handle:
                    handle.write(srt)
                try:
                    burned_path = os.path.join(work, "film_subtitled.mp4")
                    await asyncio.to_thread(burn_subtitles, out, srt_path, burned_path)
                    out, burned = burned_path, True
                except Exception as exc:  # no libass in this ffmpeg: the .srt still goes beside the film
                    log.warning("manhwa: subtitles not burned: %s", exc)
            name = f"{re.sub(r'[^0-9A-Za-z]+', '-', chapter['series']).strip('-').lower()}_ch{chapter['chapter'] or 'x'}.mp4"
            with open(out, "rb") as handle:
                asset = await self.service.add_asset(
                    name, handle, "video/mp4", os.path.getsize(out),
                    collection=f"manhwa/{chapter['series']}/{chapter['chapter'] or 'chapter'}",
                    tags=["manhwa", chapter["series"], "film"], local_path=out, dedupe=False,
                    source={"type": "generated", "engine": "ffmpeg", "generator": "chapter film",
                            "chapter_id": chapter_id, "jobs": jobs})
            drive = await self._film_to_drive(out, name)
            srt_drive = ""
            if srt:
                srt_file = self._srt_path(chapter_id)
                os.makedirs(os.path.dirname(srt_file), exist_ok=True)
                with open(srt_file, "w", encoding="utf-8") as handle:
                    handle.write(srt)
                if drive:
                    srt_drive = await self._film_to_drive(srt_file, name.rsplit(".", 1)[0] + ".srt")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        chapter["film"] = {"asset_id": asset["id"], "jobs": jobs, "parts": len(jobs), "created_at": time.time(),
                           "stream_copy": copied, "drive": drive,
                           "subtitles": ("burned" if burned else "file") if srt else "", "srt_drive": srt_drive}
        self.store.save_chapter(chapter)
        return {"chapter_id": chapter_id, "film": chapter["film"], "merged": True}

    def _srt_path(self, chapter_id: str) -> str:
        return os.path.join(os.path.dirname(self.settings_path), "manhwa_subtitles", f"{chapter_id}.srt")

    def subtitles(self, chapter_id: str) -> str:
        path = self._srt_path(chapter_id)
        if not os.path.isfile(path):
            raise NotFound("This chapter's film has no subtitles yet: merge it with subtitles on.")
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    async def _chapter_srt(self, chapter: dict, renders: list[dict], files: list[str]) -> str:
        """Subtitles from what each plan says is spoken in each segment, timed by the render's own script."""
        plans = {p["job_id"]: p for p in chapter.get("plans") or []}
        parts = []
        for render, path in zip(renders, files):
            plan = plans.get(render["plan_job_id"]) or {}
            shots = plan.get("shots") or []
            job = self.service.store.get_job(render["job_id"]) or {}
            plan_job = self.service.store.get_job(render["plan_job_id"]) or {}
            seconds: list[float] = []
            try:
                script = parse_script(job.get("script") or plan_job.get("script") or "")
                default = float(((job.get("request") or {}).get("settings") or {}).get("default_seconds") or 10)
                seconds = [float(seg.duration or default) for seg in script.segments]
            except (ScriptError, TypeError, ValueError):
                pass
            if len(seconds) != len(shots):  # the planner wrote a different number of segments: spread the lines
                lines = [line for shot in shots for line in shot.get("lines") or []]
                total = sum(seconds) or sum(float(s.get("duration") or 5) for s in shots) or 5.0
                segments = [{"seconds": total, "lines": lines}]
            else:
                segments = [{"seconds": sec, "lines": shot.get("lines") or []} for sec, shot in zip(seconds, shots)]
            parts.append({"duration": await asyncio.to_thread(media_seconds, path), "segments": segments})
        return build_srt(parts)

    async def _part_file(self, job_id: str, target: str) -> str:
        """A rendered part on disk: ComfyUI's own file when its output folder is here, else downloaded."""
        location = self.service.video_location(job_id)
        local = self.service.local_output(*location)
        if local:
            return local
        opened = await self.service.open_remote(*location, None)
        with open(target, "wb") as handle:
            async for chunk in opened["body"]:
                handle.write(chunk)
        return target

    async def _film_to_drive(self, path: str, name: str) -> str:
        """Copy the film into the Drive video folder when render export is on. Never fails the merge."""
        exporter = getattr(self.service, "drive_exporter", None)
        try:
            if exporter is None or not exporter.browser.available or not exporter.settings().get("enabled"):
                return ""
            day = time.strftime("%Y-%m-%d")
            folder = f"{exporter.settings()['folder']}/{day}"
            target = os.path.join(exporter.browser.root, folder, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, path, target)
            return f"{folder}/{name}"
        except Exception as exc:  # Drive full or unmounted: the film is still in Media
            log.warning("manhwa: film not copied to Drive: %s", exc)
            return ""

    @staticmethod
    def _plan_request(chapter, part, number, parts, aspect_ratio, segment_seconds, model,
                      groups: list[dict] | None = None) -> PlanRequest:
        return PlanRequest(
            story=plan_brief(chapter, part, number, parts, groups), segment_count=len(groups or part),
            segment_seconds=segment_seconds, aspect_ratio=aspect_ratio, model=model,
            references=[ReferenceIn(asset_id=shot["asset_id"], role="picture",
                                    label=(shot["scene"] or f"panel photo {k}")[:200])
                        for k, shot in enumerate(part, 1)])

    async def _panel_prompt_models(self, first: str = PANEL_PROMPT_MODEL) -> list[str]:
        llm = self.service.llm()
        configured = llm.agent_vision_model_override or self.service.settings.agent_vision_model
        return models_llm.resolve_many("vision", f"{first}, {configured or ''}",
                                       await self.service.model_catalogue())

    async def _panel_prompt(self, jpeg: bytes, tile: dict | None, roster: dict, models: list[str],
                            context: dict | None = None, summary: str = "") -> dict:
        """Grok (then the vision chain) looks at the panel and writes its photograph prompt.
        Returns {prompt, negative, model}, {skip} when someone in it reads as a minor, or {error}."""
        scene = (tile or {}).get("scene") or "(not catalogued)"
        given = context_text(context)
        text = PANEL_PROMPT.format(roster=_roster_text(roster), scene=scene, convert=CONVERT_SENTENCE,
                                   summary=" ".join((summary or "(not summarised)").split())[:1200],
                                   context=f"\n{given}\nWhere the panel shows it, use these looks and this place.\n"
                                   if given else "")
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}}]
        failures = []
        for model in models:
            try:
                reply, _usage = await self.service.atlas.chat(model, [{"role": "user", "content": content}],
                                                              json_mode=True, max_tokens=PANEL_PROMPT_MAX_TOKENS,
                                                              temperature=0.3, max_retries=1, purpose="panel prompt")
            except AtlasError as exc:
                failures.append(f"{model}: {exc}"[:200])
                continue
            answer = parse_json(reply) or {}
            if isinstance(answer.get("skip"), str) and answer["skip"].strip():
                return {"skip": answer["skip"].strip()[:300], "model": model}
            prompt = " ".join(str(answer.get("prompt") or "").split())
            if len(prompt) >= 40:
                negative = " ".join(str(answer.get("negative") or "").split())[:600]
                return {"prompt": prompt[:1800], "negative": negative, "model": model,
                        "drawn": answer.get("drawn"), "identifiable": answer.get("identifiable"),
                        "people": [str(n) for n in answer.get("people") or [] if isinstance(n, str)][:6],
                        "content": str(answer.get("content") or "people")}
            failures.append(f"{model}: no usable prompt")
        return {"error": "; ".join(failures) or "No vision model is configured."}

    async def _make_photo(self, state: dict, panel_id: str, jpeg: bytes, prompt: str, negative: str, row: dict,
                          qa_models: list[str], refs: list[tuple[str, str]] | None = None) -> list[str]:
        """Make the panel's photograph; with the quality check on, a vision model scores it against the panel and,
        below the bar, rewrites the prompt (pose, limbs, camera) for another try. The best attempt is kept, the
        others are deleted. Raises SkipPanel when someone reads as a minor, deleting what was made."""
        tries = 1 + (state["qa_retries"] if state["qa"] and qa_models else 0)
        best: tuple[float, list[str], str] | None = None
        attempts = []
        refs = refs or []
        # the cast portraits ride along as <image2>.. on every try, whatever the checker rewrote
        cast = " ".join(CAST_REF_LINE.format(n=k, name=name) for k, (_, name) in enumerate(refs, 2))
        for attempt in range(1, tries + 1):
            result = await self._convert_panel(panel_id, f"{prompt} {cast}".strip()[:2400], state["loras"], negative,
                                               steps=state["steps"], cfg=state["cfg"],
                                               refs=[asset_id for asset_id, _ in refs])
            ids = [a["id"] for a in result.get("assets") or []]
            if not (state["qa"] and qa_models and ids):
                row["prompt"] = prompt
                return ids
            verdict = await self._quality_check(jpeg, ids[0], prompt, qa_models)
            if verdict.get("minor"):
                for doomed in ids + (best[1] if best else []):
                    self._drop(doomed)
                raise SkipPanel("the quality check saw someone who looks under 18")
            if verdict.get("error"):  # no verdict: keep what we have rather than guess
                row["qa_note"] = f"Quality check unavailable: {verdict['error']}"[:300]
                if best is not None:
                    for doomed in ids:
                        self._drop(doomed)
                    break
                row["prompt"] = prompt
                return ids
            score = float(verdict.get("score") or 0)
            attempts.append({"attempt": attempt, "score": score, "problems": verdict.get("problems", [])[:6],
                             "by": verdict.get("model", "")})
            if best is None or score > best[0]:
                for doomed in (best[1] if best else []):
                    self._drop(doomed)
                best = (score, ids, prompt)
            else:
                for doomed in ids:
                    self._drop(doomed)
            if score >= state["qa_min_score"] or attempt == tries:
                break
            revised = " ".join(str(verdict.get("revised_prompt") or "").split())
            if len(revised) >= 40:
                prompt = revised[:2000]
            state["retried"] += 1
        row.update(qa=attempts, qa_score=best[0] if best else None, prompt=best[2] if best else prompt)
        return best[1] if best else []

    def _drop(self, asset_id: str) -> None:
        try:
            self.service.delete_asset(asset_id)
        except Exception:  # already gone
            pass

    async def _quality_check(self, panel_jpeg: bytes, photo_id: str, prompt: str, models: list[str]) -> dict:
        """{score, problems, minor, revised_prompt, model}, or {error} when no model gives a usable verdict."""
        try:
            data, mime = await self.service.asset_file(photo_id, 768)
        except Exception:
            asset = self.service.store.get_asset(photo_id)
            data, mime = (await self.service.asset_bytes(asset), "image/png") if asset else (b"", "image/png")
        content = [{"type": "text", "text": QA_PROMPT.format(prompt=prompt[:1800], convert=CONVERT_SENTENCE)},
                   {"type": "text", "text": "Image 1 (the panel):"},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(panel_jpeg).decode("ascii")}},
                   {"type": "text", "text": "Image 2 (the photograph):"},
                   {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}}]
        failures = []
        for model in models:
            try:
                reply, _usage = await self.service.atlas.chat(model, [{"role": "user", "content": content}],
                                                              json_mode=True, max_tokens=PANEL_PROMPT_MAX_TOKENS,
                                                              temperature=0.2, max_retries=1, purpose="photo check")
            except AtlasError as exc:
                failures.append(f"{model}: {exc}"[:200])
                continue
            answer = parse_json(reply) or {}
            if isinstance(answer.get("score"), (int, float)):
                return {"score": max(0.0, min(10.0, float(answer["score"]))), "model": model,
                        "problems": [str(p)[:200] for p in answer.get("problems") or [] if p],
                        "minor": answer.get("minor") is True,
                        "revised_prompt": str(answer.get("revised_prompt") or "")}
            failures.append(f"{model}: no score")
        return {"error": "; ".join(failures) or "No vision model is configured."}

    async def draft_context(self, chapter_id: str) -> dict:
        """A scene context drafted from the chapter's catalogue (summary, roster, scenes) by the summary model.
        Not saved: the caller decides, and the user can edit it first."""
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        pages = [p for p in self.store.pages(chapter_id) if p.get("status") == "ok"]
        if not pages and not chapter.get("summary"):
            raise RequestError("This chapter has no catalogue to draft from.")
        scenes = "\n".join(f"- {p.get('scene', '')}" + (f" ({p['setting']})" if p.get("setting") else "")
                           for p in pages[:60])
        text = CONTEXT_PROMPT.format(series=chapter["series"], chapter=chapter.get("chapter") or "?",
                                     summary=chapter.get("summary") or "(none)",
                                     roster=_roster_text(self.store.characters(chapter["series"])), scenes=scenes[:12000])
        try:
            model = await self.service.model_for("summary")
            reply, _usage = await self.service.atlas.chat(model, [{"role": "user", "content": text}], json_mode=True,
                                                          max_tokens=CATALOG_MAX_TOKENS, temperature=0.3,
                                                          max_retries=1, purpose="manhwa context")
        except AtlasError as exc:
            raise Unavailable(f"Could not draft the context: {exc}") from None
        answer = parse_json(reply) or {}
        draft = {key: " ".join(str(answer.get(key) or "").split())[:CONTEXT_MAX] for key in CONTEXT_FIELDS}
        if not any(draft.values()):
            raise Unavailable("The model gave no usable context.")
        draft["one_scene"] = answer.get("one_scene") is True
        return draft

    # ----------------------------------------------------------------- cast sheet

    def _portraits(self, series: str) -> dict[str, str]:
        """Character name -> portrait asset id, for the portraits still in the library."""
        return {name: data["portrait_asset_id"] for name, data in self.store.characters(series).items()
                if data.get("portrait_asset_id") and self.service.store.get_asset(data["portrait_asset_id"])}

    @staticmethod
    def _converted_panels(pages: list[dict]) -> set[tuple[str, int]]:
        """(page asset id, panel number) of every panel already dealt with: given a photograph, or skipped on
        purpose (triage, someone who reads as a minor) -- a resumed run must not try those again either."""
        done = set()
        for page in pages:
            for made in (page.get("real") or []) + (page.get("skipped_panels") or []):
                if (made.get("asset_ids") or made.get("why")) and made.get("page_asset_id") and made.get("panel") is not None:
                    done.add((made["page_asset_id"], int(made["panel"])))
        return done

    def _mark_skipped(self, tile: dict | None, page_asset_id: str, panel: int, why: str, pages: list[dict],
                      chapter_id: str, series: str) -> None:
        if tile is None:
            return
        tile.setdefault("skipped_panels", []).append({"page_asset_id": page_asset_id, "panel": panel,
                                                      "why": why[:200] or "skipped"})
        self.store.save_pages(chapter_id, series, pages)

    def _cast_source(self, series: str, name: str) -> tuple[str, int] | None:
        """The tile that shows this character most plainly: the first one with the fewest other people in it."""
        best = None
        for chapter in self.store.list_chapters(series):
            for page in self.store.pages(chapter["id"]):
                people = [p.get("name") for p in page.get("characters") or []]
                if name in people and page.get("asset_id"):
                    key = (len(people), chapter["created_at"], page["idx"])
                    if best is None or key < best[0]:
                        best = (key, page["asset_id"], int(page.get("part") or 0))
        return (best[1], best[2]) if best else None

    async def make_cast(self, series: str, *, names: list[str] | None = None, force: bool = False) -> dict:
        """Make the photoreal portraits of a series' main characters in the background. Poll cast_run()."""
        key = series_key(series)
        series = next((s["series"] for s in self.store.series() if series_key(s["series"]) == key), series)
        if not self.store.characters(series):
            raise RequestError(f"No characters catalogued for {series!r} yet.")
        for other in self.cast_runs.values():
            if other["status"] == "running":
                raise RequestError(f"Cast run {other['id']} is still running.", {"cast_run_id": other["id"]})
        state = {"id": uuid.uuid4().hex[:12], "status": "running", "series": series, "done": [], "skipped": [],
                 "errors": [], "created_at": time.time(), "finished_at": None}
        self.cast_runs[state["id"]] = state
        self._cast_tasks[state["id"]] = asyncio.create_task(self._cast_task(state, names, force))
        return state

    def cast_run(self, run_id: str) -> dict:
        state = self.cast_runs.get(run_id)
        if state is None:
            raise NotFound(f"No cast run {run_id!r}.")
        return state

    async def _cast_task(self, state: dict, names, force: bool) -> None:
        try:
            await self._cast(state, names, force)
            state["status"] = "done"
        except asyncio.CancelledError:
            state["status"] = "cancelled"
            raise
        except Exception as exc:
            log.exception("manhwa cast %s failed", state["id"])
            state["status"] = "failed"
            state["errors"].append(f"{type(exc).__name__}: {exc}"[:400])
        finally:
            state["finished_at"] = time.time()
            self._cast_tasks.pop(state["id"], None)

    async def _cast(self, state: dict, names: list[str] | None, force: bool) -> None:
        series, conf = state["series"], self.settings()
        roster = self.store.characters(series)
        named = [n for n in roster if not re.fullmatch(r"unknown-\d+", n, re.I)]
        wanted = [n for n in (names or sorted(named, key=lambda n: -int(roster[n].get("tiles") or 0)))
                  if n in roster][:max(1, int(conf["cast_size"] or 8)) if not names else None]
        loras = [dict(l) for l in conf["loras"]] if conf["loras"] else await self.real_loras(conf["realism"],
                                                                                              conf["age_slider"])
        have = self._portraits(series)
        for name in wanted:
            if name in have and not force:
                state["skipped"].append({"name": name, "why": "already has a portrait"})
                continue
            source = self._cast_source(series, name)
            page = self.service.store.get_asset(source[0]) if source else None
            if page is None:
                state["skipped"].append({"name": name, "why": "no panel shows them"})
                continue
            tiles = await asyncio.to_thread(slice_image, await self.service.asset_bytes(page))
            crop = tiles[min(source[1], len(tiles) - 1)]
            ref = await self.service.add_asset(
                f"cast_{series_key(name)[:30] or 'person'}_source.jpg", io.BytesIO(crop), "image/jpeg", len(crop),
                collection=f"manhwa/{series}/cast/source", tags=["manhwa", series, "cast"],
                source={"type": "manhwa", "from_asset": page["id"], "cast_of": name})
            appearance = " ".join(str(roster[name].get("appearance") or "as drawn").split())[:400]
            try:
                result = await self._convert_panel(ref["id"], CAST_PROMPT.format(name=name, appearance=appearance),
                                                   loras, REAL_NEGATIVE, steps=conf["steps"], cfg=conf["cfg"])
            except Exception as exc:  # a refusal or a busy GPU: the next character still gets theirs
                state["errors"].append(f"{name}: {exc}"[:300])
                continue
            ids = [a["id"] for a in result.get("assets") or []]
            if not ids:
                state["errors"].append(f"{name}: no image came back")
                continue
            self.service.update_asset(ids[0], collection=f"manhwa/{series}/cast", add_tags=["manhwa", series, "cast", name])
            self.set_portrait(series, name, ids[0])
            state["done"].append({"name": name, "asset_id": ids[0]})

    def set_portrait(self, series: str, name: str, asset_id: str | None) -> dict:
        """Use this image as the character's portrait (None removes it)."""
        roster = self.store.characters(series)
        if name not in roster:
            raise NotFound(f"No character {name!r} in {series!r}.")
        if asset_id and self.service.store.get_asset(asset_id) is None:
            raise NotFound(f"No asset {asset_id!r}.")
        data = roster[name]
        if asset_id:
            data["portrait_asset_id"] = asset_id
        else:
            data.pop("portrait_asset_id", None)
        self.store.save_characters(series, {name: data})
        return dict(data, name=name)

    def _record_person_check(self, panel_id: str, written: dict) -> None:
        """The panel-prompt call already looked at the panel, so its answer is the panel's person check and the
        separate check (another vision call, often a 429 on Atlas) never runs. A drawing with no real person in it
        is cleared; a real, recognisable person keeps the upload rules on. An unclear answer records nothing, and
        the usual check runs before the edit."""
        drawn, identifiable = written.get("drawn"), written.get("identifiable")
        if identifiable is True:
            verdict = {"identifiable": True, "reason": "The panel-prompt model saw a real, recognisable person."}
        elif drawn is True and identifiable is False:
            verdict = {"identifiable": False, "reason": "A drawn panel with no real, recognisable person."}
        else:
            return
        asset = self.service.store.get_asset(panel_id)
        if asset is None:
            return
        asset["source"] = {**(asset.get("source") or {}),
                           "person_check": {**verdict, "model": written.get("model", ""), "checked_at": time.time(),
                                            "by": "panel prompt"}}
        save = getattr(self.service.store, "add_asset", None)
        if save is not None:
            save(asset)

    async def _convert_panel(self, panel_id: str, prompt: str, loras: list[dict], negative: str = REAL_NEGATIVE, *,
                             steps: int | None = None, cfg: float | None = None, refs: list[str] | None = None) -> dict:
        for attempt in range(BUSY_RETRIES + 1):
            try:
                return await self.service.generate_images(prompt, reference_asset_ids=[panel_id, *(refs or [])],
                                                          engine="qwen21",
                                                          loras=loras, negative=negative, steps=steps, cfg=cfg)
            except RequestError as exc:
                if "busy" not in str(exc).lower() or attempt == BUSY_RETRIES:
                    raise
            await asyncio.sleep(self.busy_wait)
        raise RequestError("ComfyUI stayed busy.")

    def chapter_view(self, chapter_id: str) -> dict:
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None:
            raise NotFound(f"No catalogued chapter {chapter_id!r}.")
        film = dict(chapter.get("film") or {})
        asset = self.service.store.get_asset(film.get("asset_id") or "") if film else None
        if asset is not None:
            shown = self.service.asset_view(asset)
            film.update(file_url=shown.get("file_url"), thumb_url=shown.get("thumb_url"))
        return dict(chapter, film=film if asset is not None else None,
                    pages=[self._with_thumb(p) for p in self.store.pages(chapter_id)])

    def search(self, query: str, series: str = "", limit: int = 30) -> list[dict]:
        return [self._with_thumb(hit) for hit in self.store.search(query, series, limit)]

    def characters(self, series: str) -> list[dict]:
        roster = self.store.characters(series)
        out = []
        for name, data in roster.items():
            row = dict(data, name=name)
            asset = self.service.store.get_asset(data.get("portrait_asset_id") or "") if data.get("portrait_asset_id") else None
            if asset is not None:
                shown = self.service.asset_view(asset)
                row.update(portrait_thumb=shown.get("thumb_url"), portrait_url=shown.get("file_url"))
            out.append(row)
        return sorted(out, key=lambda c: (-len(c.get("chapters") or []), c["name"].lower()))

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
