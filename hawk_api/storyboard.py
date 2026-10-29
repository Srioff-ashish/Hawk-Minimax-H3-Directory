"""Storyboard sheets: many shots from one cheap image, each panel the first frame of a video segment.

One Seedream Flash image ($0.018, any size) holds a grid of up to eight panels of the same character in one style,
which costs an eighth of eight separate images and keeps the face, body and light consistent across the shots
because they are one generation. The sheet is cut into its panels, and each panel opens its segment of the film
(H3's keyframe completion), with the character's own identity picture beside it in every segment.

Eight panels at most: with the identity picture that is nine, a render's picture limit.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid

from .jobs import NotFound, RequestError, Unavailable
from .schemas import PlannerOptions, ReferenceIn, RenderSettings, VideoRequest

log = logging.getLogger(__name__)

SHEET_ENGINE = "seedream-flash"
MAX_PANELS = 8
SPLIT_MAX = 40  # panels a sheet the user brings may have (a board's shot limit)
RENDER_PICTURES = 9  # a render's picture limit
#: Seedream Flash presets: one price at every size, so the sheet takes whichever fits its grid best.
SHEET_SIZES = ((2048, 2048), (2368, 1776), (1776, 2368), (2816, 1584), (1584, 2816), (2496, 1664), (1664, 2496),
               (3136, 1344), (2352, 1008))
#: Grey level below which a pixel counts as the sheet's black border.
BORDER_LEVEL = 40
#: A run of bright columns or rows narrower than this share of the sheet is noise, not a panel.
MIN_PANEL_SHARE = 0.06
#: Trimmed off every edge of a cut panel, so no border line rides into the video's first frame.
TRIM_SHARE = 0.012

ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9, "9:21": 9 / 21}


class StoryboardError(RequestError):
    """A sheet that can't be planned or cut as asked (an HTTP 422, like any bad request)."""


def grid_for(panels: int, aspect_ratio: str) -> tuple[int, int, tuple[int, int]]:
    """(columns, rows, sheet size) whose panels come closest to the video's aspect ratio."""
    if not 1 <= panels <= MAX_PANELS:
        raise StoryboardError(f"A sheet holds 1 to {MAX_PANELS} panels, not {panels}.")
    target = ASPECTS.get(aspect_ratio)
    if target is None:
        raise StoryboardError(f"aspect_ratio should be one of: {', '.join(ASPECTS)}.")
    best = None
    for columns in range(1, panels + 1):
        if panels % columns:
            continue
        rows = panels // columns
        for width, height in SHEET_SIZES:
            off = abs(math.log((width / columns) / (height / rows) / target))
            # ties go to the bigger panel, then to the squarer grid
            key = (round(off, 2), -(width * height) / panels, abs(columns - rows))
            if best is None or key < best[0]:
                best = (key, columns, rows, (width, height))
    _, columns, rows, size = best
    return columns, rows, size


def sheet_prompt(beats: list[str], character: str, columns: int, rows: int, style: str = "") -> str:
    """The Seedream prompt for one sheet: the grid, the rules that keep it cuttable, the character, the beats."""
    panels = len(beats)
    layout = (f"exactly {panels} photo panel{'s' if panels > 1 else ''} arranged in a grid of {columns} "
              f"column{'s' if columns > 1 else ''} and {rows} row{'s' if rows > 1 else ''}, all the same size")
    return (
        f"A storyboard contact sheet: {layout}, separated by thin solid black borders, on a black background. "
        "No text, no numbers, no labels, no captions, no logos anywhere. "
        f"Every panel shows the same person from the reference image -- keep the face, hair, skin tone and exact body. "
        f"{character.strip()} "
        f"{(style.strip() + ' ') if style.strip() else ''}"
        "One continuous story in panel order, left to right, top to bottom: "
        + " ".join(f"Panel {number}: {beat.strip().rstrip('.')}." for number, beat in enumerate(beats, 1))
    )


#: A column or row is border when fewer than this share (of 255) of its pixels are brighter than the border:
#: borders are solid black edge to edge, while even a night scene has some light somewhere along its length.
BORDER_BRIGHT_SHARE = 20


def _runs(profile: list[int], minimum: int) -> list[tuple[int, int]]:
    """(start, end) of each stretch that isn't border, skipping stretches shorter than minimum."""
    runs, start = [], None
    for index, value in enumerate(profile + [0]):
        bright = value > BORDER_BRIGHT_SHARE
        if bright and start is None:
            start = index
        elif not bright and start is not None:
            if index - start >= minimum:
                runs.append((start, index))
            start = None
    return runs


def _merge_to(runs: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    """Join runs across their narrowest gaps until there are count of them.

    A dark band inside the panels (a bed frame, a night sky lining up across one row) splits a panel in two; the
    sheet's real borders are the widest gaps, so the narrow ones are the ones to close."""
    runs = list(runs)
    while len(runs) > count:
        gap = min(range(len(runs) - 1), key=lambda i: runs[i + 1][0] - runs[i][1])
        runs[gap:gap + 2] = [(runs[gap][0], runs[gap + 1][1])]
    return runs


def _uniform_count(runs: list[tuple[int, int]]) -> int:
    """How many panels the runs really are: the most whose sizes agree. A band that only looks like a border (a
    bed frame in every panel of a row) splits a panel unevenly, so merging it back is what makes the sizes agree."""
    for count in range(len(runs), 1, -1):
        sizes = [end - start for start, end in _merge_to(runs, count)]
        mean = sum(sizes) / len(sizes)
        spread = (sum((size - mean) ** 2 for size in sizes) / len(sizes)) ** 0.5 / mean
        if spread <= 0.12:
            return count
    return 1


def find_grid(data: bytes, columns: int, rows: int) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Panel boxes (left, top, right, bottom) in reading order, and whether the borders were found.

    The borders are found from how much of each column and row is brighter than the black border. When they
    can't be (a dark night scene reads as border too), the sheet is cut as the even grid it was asked to be."""
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("L")
    width, height = image.size
    mask = image.point(lambda level: 255 if level > BORDER_LEVEL else 0)
    across = list(mask.resize((width, 1), Image.BOX).tobytes())
    down = list(mask.resize((1, height), Image.BOX).tobytes())
    xs = _runs(across, max(1, round(width * MIN_PANEL_SHARE * (0.5 if columns <= 0 else 1))))
    ys = _runs(down, max(1, round(height * MIN_PANEL_SHARE * (0.5 if rows <= 0 else 1))))
    if columns <= 0 or rows <= 0:  # a sheet whose grid nobody wrote down: take what the borders show
        if not xs or not ys or len(xs) * len(ys) > SPLIT_MAX:
            raise StoryboardError("Couldn't find the sheet's panels from its borders: give columns and rows.")
        columns, rows = _uniform_count(xs), _uniform_count(ys)
    xs, ys = _merge_to(xs, columns), _merge_to(ys, rows)
    found = len(xs) == columns and len(ys) == rows
    if not found:
        xs = [(round(width * c / columns), round(width * (c + 1) / columns)) for c in range(columns)]
        ys = [(round(height * r / rows), round(height * (r + 1) / rows)) for r in range(rows)]
    boxes = []
    for top, bottom in ys:
        for left, right in xs:
            trim_x = max(2, round((right - left) * TRIM_SHARE))
            trim_y = max(2, round((bottom - top) * TRIM_SHARE))
            boxes.append((left + trim_x, top + trim_y, right - trim_x, bottom - trim_y))
    return boxes, found


def cut_grid(data: bytes, columns: int, rows: int) -> tuple[list[bytes], bool]:
    """Each panel as a PNG, in reading order, and whether the borders were found (else an even grid)."""
    from PIL import Image

    boxes, found = find_grid(data, columns, rows)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    panels = []
    for box in boxes:
        buffer = io.BytesIO()
        image.crop(box).save(buffer, format="PNG")
        panels.append(buffer.getvalue())
    return panels, found


def film_brief(beats: list[str], character: str, mood: str = "", music: str = "", identities: int = 1) -> str:
    """The planner's brief: every segment opens on its panel and keeps the identity picture(s) beside it."""
    panels = len(beats)
    tags = [f"<Picture {panels + n}>" for n in range(1, identities + 1)]
    who = (f"{tags[0]} is the identity reference" if identities == 1
           else f"{', '.join(tags[:-1])} and {tags[-1]} are the identity references (one per person)")
    return (
        f"{character.strip()} {who} (face and body) for every segment. "
        f"{(mood.strip() + ' ') if mood.strip() else ''}"
        f"{panels} segments in story order. Each segment N starts exactly on its storyboard frame <Picture N> -- that "
        "picture is the segment's first frame (keyframe completion) -- and animates it: keep the pose, framing, "
        "lighting, setting and clothing (or nudity) of that frame, and add only the motion described, natural and "
        "realistic, with real weight. Nobody speaks unless a segment says so. "
        f"{('Music: ' + music.strip() + ' ') if music.strip() else ''}"
        + " ".join(f"Segment {number}: <Picture {number}> -- {beat.strip().rstrip('.')}." for number, beat in enumerate(beats, 1))
    )


async def make_sheet(service, beats: list[str], character: str, identity_asset_ids: list[str],
                     aspect_ratio: str = "9:16", style: str = "", engine: str = SHEET_ENGINE) -> dict:
    """One sheet image, with the grid it was drawn on so split_sheet can cut it the same way."""
    beats = [beat for beat in (b.strip() for b in beats or []) if beat]
    if not beats:
        raise StoryboardError("Give one beat (a short shot description) per panel.")
    if not identity_asset_ids:
        raise StoryboardError("Give the character's identity picture (reference_asset_ids) so every panel shows them.")
    columns, rows, (width, height) = grid_for(len(beats), aspect_ratio)
    prompt = sheet_prompt(beats, character, columns, rows, style)
    result = await service.generate_images(prompt, engine=engine, reference_asset_ids=list(identity_asset_ids),
                                           size=f"{width}x{height}", n=1)
    sheet = result["assets"][0]
    grid = {"columns": columns, "rows": rows, "panels": len(beats), "aspect_ratio": aspect_ratio}
    service.update_asset(sheet["id"], add_tags=["storyboard-sheet"])
    return {"sheet": sheet, "grid": grid, "beats": beats, "cost_usd": result.get("cost_usd"),
            "model": result.get("model"), "note": result.get("note", "")}


async def split_sheet(service, sheet_asset_id: str, columns: int = 0, rows: int = 0, panels: int | None = None,
                      collection: str | None = None, tags: list[str] | None = None) -> dict:
    """The sheet cut into its panels, each stored as an asset that remembers its sheet and position.

    columns and rows 0: the grid a merged sheet recorded when it was made, else whatever its borders show -- so any
    sheet works, one drawn here, one merged from the user's images, or one the user made elsewhere."""
    from PIL import Image

    asset = service.store.get_asset(sheet_asset_id)
    if asset is None or asset.get("kind") != "image":
        raise StoryboardError(f"No image asset {sheet_asset_id!r}.")
    if columns <= 0 or rows <= 0:
        recorded = (asset.get("source") or {}).get("grid")
        if isinstance(recorded, list) and len(recorded) == 2:
            columns, rows = int(recorded[0]), int(recorded[1])
    if columns * rows > SPLIT_MAX:
        raise StoryboardError(f"columns x rows should be at most {SPLIT_MAX} panels.")
    data = await service.asset_bytes(asset)
    boxes, found = find_grid(data, columns, rows)
    if columns <= 0 or rows <= 0:  # detected from the borders
        found = True
    image = Image.open(io.BytesIO(data)).convert("RGB")
    images = []
    for box in boxes:
        buffer = io.BytesIO()
        image.crop(box).save(buffer, format="PNG")
        images.append(buffer.getvalue())
    recorded = (asset.get("source") or {}).get("images") or []
    count = panels or (len(recorded) if recorded and len(recorded) <= len(images) else len(images))
    images = images[:count]  # a merged sheet's empty last cells are not panels
    stem = (asset.get("filename") or "sheet").rsplit(".", 1)[0][:40]
    out = []
    for number, cut in enumerate(images, 1):
        stored = await service.add_asset(
            f"{stem}_panel_{number}.png", io.BytesIO(cut), "image/png", len(cut),
            collection=collection or asset.get("collection") or "Generated",
            tags=["storyboard-panel", *(tags or [])],
            source={"type": "storyboard", "sheet": sheet_asset_id, "panel": number},
        )
        out.append(service.asset_view(stored))
    note = "" if found else ("The sheet's borders weren't clear, so it was cut as an even grid; check the panel edges.")
    return {"panels": out, "borders_found": found, "note": note}


async def render_film(service, panel_asset_ids: list[str], identity_asset_ids, beats: list[str], character: str,
                      settings: RenderSettings | None = None, segment_seconds: float = 6.0, mood: str = "",
                      music: str = "") -> dict:
    """One render of the panels in order, planned in the H3 reference format, each segment opening on its panel."""
    identity_asset_ids = [identity_asset_ids] if isinstance(identity_asset_ids, str) else list(identity_asset_ids)
    if not panel_asset_ids:
        raise StoryboardError("No panels to animate.")
    if not identity_asset_ids:
        raise StoryboardError("Give the identity picture the panels keep the face and body of.")
    if len(panel_asset_ids) + len(identity_asset_ids) > RENDER_PICTURES:
        raise StoryboardError(f"{len(panel_asset_ids)} panels and {len(identity_asset_ids)} identity pictures are more "
                              f"than a render's {RENDER_PICTURES}; use a board, which splits the film into parts.")
    if len(beats) != len(panel_asset_ids):
        raise StoryboardError(f"{len(panel_asset_ids)} panels need {len(panel_asset_ids)} beats, one each; got {len(beats)}.")
    settings = settings or RenderSettings(aspect_ratio="9:16", continuity="off")
    references = [ReferenceIn(asset_id=asset_id, role="picture", label=f"storyboard frame {number}")
                  for number, asset_id in enumerate(panel_asset_ids, 1)]
    references += [ReferenceIn(asset_id=asset_id, role="picture", label="identity (face and body)")
                   for asset_id in identity_asset_ids]
    request = VideoRequest(
        references=references,
        story=PlannerOptions(story=film_brief(beats, character, mood, music, len(identity_asset_ids)),
                             segment_count=len(beats),
                             segment_seconds=segment_seconds, aspect_ratio=settings.aspect_ratio),
        settings=settings,
    )
    return await service.create_video(request)


# --------------------------------------------------------------------------- one panel, identity packs


def panel_size(aspect_ratio: str) -> str:
    """The Flash preset closest to one panel's shape, for a panel drawn on its own."""
    target = ASPECTS.get(aspect_ratio, 9 / 16)
    singles = SHEET_SIZES + ((1536, 1536), (1792, 1344), (1344, 1792), (2048, 1152), (1152, 2048), (1872, 1248), (1248, 1872))
    width, height = min(singles, key=lambda size: (abs(math.log(size[0] / size[1] / target)), -size[0] * size[1]))
    return f"{width}x{height}"


async def redo_panel(service, beat: str, character: str, identity_asset_ids: list[str], aspect_ratio: str = "9:16",
                     style: str = "", previous_asset_id: str | None = None, sheet_asset_id: str | None = None,
                     number: int = 0, engine: str = SHEET_ENGINE, collection: str | None = None) -> dict:
    """One shot drawn again on its own ($0.018): the same person, and the old panel's setting when there is one."""
    refs = list(identity_asset_ids)
    keep = ""
    if previous_asset_id and service.store.get_asset(previous_asset_id):
        refs.append(previous_asset_id)
        keep = (f" The last reference image is an earlier take of this shot: keep its setting, light and framing, "
                "and fix whatever is wrong in it.")
    prompt = (f"A single photo, full frame, no borders, no text. The person from the first reference image -- keep "
              f"the face, hair, skin tone and exact body. {character.strip()} {(style.strip() + ' ') if style.strip() else ''}"
              f"{beat.strip().rstrip('.')}.{keep}")
    result = await service.generate_images(prompt, engine=engine, reference_asset_ids=refs, size=panel_size(aspect_ratio), n=1)
    made = result["assets"][0]
    asset = service.store.get_asset(made["id"]) or made
    service.update_asset(made["id"], add_tags=["storyboard-panel"], collection=collection or asset.get("collection"))
    return {"panel": made, "cost_usd": result.get("cost_usd"), "sheet": sheet_asset_id, "number": number}


#: The views of an identity pack, in panel order. The anatomy views are for adult platforms and only on request.
PACK_VIEWS = [
    "full body from the front, standing straight, arms relaxed at her sides, neutral expression, plain light grey studio backdrop",
    "full body from behind, standing straight, plain light grey studio backdrop",
    "full body in left profile, standing straight, plain light grey studio backdrop",
    "head and shoulders close-up, looking into the camera, soft even light, plain light grey studio backdrop",
]
PACK_ANATOMY_VIEWS = [
    "fully nude, full body from the front, standing straight, plain light grey studio backdrop, realistic natural anatomy",
    "fully nude, close-up from behind at hip height, bent slightly forward, plain light grey studio backdrop, "
    "realistic, anatomically correct rear anatomy in sharp focus",
]


async def identity_pack(service, name: str, description: str, identity_asset_ids: list[str], style: str = "",
                        anatomy: bool = False, collection: str | None = None) -> dict:
    """One sheet of a character's reference views (front, back, side, face; nude views only when asked), cut into
    panels tagged identity-pack -- pictures to give every later image and video of them."""
    views = PACK_VIEWS + (PACK_ANATOMY_VIEWS if anatomy else [])
    character = (f"{name.strip()}: {description.strip()} " if description.strip() else f"{name.strip()}. ") + \
        "A character reference sheet: the same person in every panel, the same light, nothing else in frame."
    made = await make_sheet(service, views, character, identity_asset_ids, "3:4", style)
    grid = made["grid"]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "character"
    cut = await split_sheet(service, made["sheet"]["id"], grid["columns"], grid["rows"], grid["panels"],
                            collection=collection, tags=["identity-pack", f"pack:{slug}"])
    labels = ["front", "back", "side", "face"] + (["nude front", "nude rear close-up"] if anatomy else [])
    panels = [{"view": label, **panel} for label, panel in zip(labels, cut["panels"])]
    return {"name": name, "sheet": made["sheet"], "panels": panels, "cost_usd": made["cost_usd"],
            "borders_found": cut["borders_found"], "note": cut["note"]}


# --------------------------------------------------------------------------- boards

#: A board while it is being worked on; anything else can be edited.
BUSY = ("drawing", "rendering", "joining")


class StoryboardStore:
    """``DATA_DIR/storyboards.json``: every board, small enough to keep in one file."""

    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "storyboards.json")
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def all(self) -> list[dict]:
        return sorted(self._load().values(), key=lambda board: -board.get("updated_at", 0))

    def get(self, board_id: str) -> dict | None:
        return self._load().get(board_id)

    def save(self, board: dict) -> dict:
        with self._lock:
            data = self._load()
            board["updated_at"] = time.time()
            data[board["id"]] = board
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=1)
            os.replace(tmp, self.path)
        return board

    def delete(self, board_id: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(board_id, None)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=1)
            os.replace(tmp, self.path)


class StoryboardManager:
    """Boards: a character, an ordered shot list, its sheet(s) and panels, and the film made from them.

    More than eight shots are drawn as several sheets at once. The film is rendered in parts small enough for a
    render's nine pictures (the panels plus the identity pictures) and joined into one video."""

    poll = 5.0
    RENDER_WAIT = 6 * 3600.0

    def __init__(self, service):
        self.service = service
        self.store = StoryboardStore(service.settings.data_dir)
        self._tasks: dict[str, asyncio.Task] = {}

    # ---- reading
    def list(self) -> list[dict]:
        return [self.view(board) for board in self.store.all()]

    def board(self, board_id: str) -> dict:
        board = self.store.get(board_id)
        if board is None:
            raise NotFound(f"No storyboard {board_id!r}.")
        return board

    def view(self, board: dict | str) -> dict:
        board = self.board(board) if isinstance(board, str) else board
        assets = self.service.store.get_asset
        out = dict(board)
        out["panel_views"] = [self.service.asset_view(assets(a)) if a and assets(a) else None for a in board["panels"]]
        out["identity_views"] = [self.service.asset_view(assets(a)) for a in board["identity_asset_ids"] if assets(a)]
        out["sheet_views"] = [dict(s, asset=self.service.asset_view(assets(s["asset_id"])) if assets(s["asset_id"]) else None)
                              for s in board.get("sheets") or []]
        film = (board.get("render") or {}).get("film_asset_id")
        if film and assets(film):
            out["film"] = self.service.asset_view(assets(film))
        out["jobs"] = [self.service.job_summary(job) for job in
                       (self.service.store.get_job(j) for j in (board.get("render") or {}).get("jobs", [])) if job]
        return out

    # ---- writing
    def create(self, *, title: str, character: str, identity_asset_ids: list[str], shots: list[str],
               aspect_ratio: str = "9:16", style: str = "", motions: list[str] | None = None, draw: bool = True,
               link: dict | None = None, then_render: dict | None = None,
               panel_asset_ids: list[str] | None = None, sheet_asset_id: str | None = None,
               sheet_columns: int = 0, sheet_rows: int = 0) -> dict:
        """panel_asset_ids: the user's own images as the panels, in shot order -- nothing is drawn (no model call),
        the board is ready at once, and shots default to "shot N" (say what moves in motions).

        sheet_asset_id: a ready-made sheet instead. One merged from the user's images gives back those images
        (sharper than cut-outs); any other sheet is cut into its panels (grid from columns/rows or its borders)."""
        shots = [s.strip() for s in shots or [] if s and s.strip()]
        from_sheet = None
        if sheet_asset_id:
            sheet = self.service.store.get_asset(sheet_asset_id)
            if sheet is None or sheet.get("kind") != "image":
                raise StoryboardError(f"No image asset {sheet_asset_id!r}.")
            source = sheet.get("source") or {}
            if source.get("type") == "composed" and source.get("images"):
                panel_asset_ids = [i for i in source["images"] if self.service.store.get_asset(i)]
            else:
                from_sheet = {"asset_id": sheet_asset_id, "columns": sheet_columns, "rows": sheet_rows}
        if panel_asset_ids:
            for asset_id in panel_asset_ids:
                asset = self.service.store.get_asset(asset_id)
                if asset is None or asset.get("kind") != "image":
                    raise StoryboardError(f"No image asset {asset_id!r}.")
            if shots and len(shots) != len(panel_asset_ids):
                raise StoryboardError(f"{len(panel_asset_ids)} images need {len(panel_asset_ids)} shots, one each (or none).")
            shots = shots or [f"shot {n}: animate this frame with natural, realistic motion"
                              for n in range(1, len(panel_asset_ids) + 1)]
        if not shots and not from_sheet:
            raise StoryboardError("Give at least one shot.")
        if len(shots) > 40:
            raise StoryboardError("At most 40 shots on one board.")
        if not identity_asset_ids:
            raise StoryboardError("Give the character's identity picture.")
        if len(identity_asset_ids) > 3:
            raise StoryboardError("At most 3 identity pictures (each one is a picture every render part carries).")
        for asset_id in identity_asset_ids:
            asset = self.service.store.get_asset(asset_id)
            if asset is None or asset.get("kind") != "image":
                raise StoryboardError(f"No image asset {asset_id!r}.")
        grid_for(max(1, min(len(shots), MAX_PANELS)), aspect_ratio)  # validates the aspect ratio
        board = {"id": uuid.uuid4().hex[:12], "title": (title or "Storyboard").strip()[:120],
                 "character": character.strip(), "identity_asset_ids": list(identity_asset_ids),
                 "aspect_ratio": aspect_ratio, "style": style.strip(), "shots": shots,
                 "motions": [m.strip() for m in motions] if motions else [], "sheets": [], "panels": [None] * len(shots),
                 "status": "draft", "error": "", "cost_usd": 0.0, "render": {}, "link": link or {},
                 # {settings, segment_seconds, mood, music}: render as soon as the drawing is done (one-call films)
                 "then_render": then_render, "from_sheet": from_sheet, "created_at": time.time()}
        if from_sheet:  # cut in the background, like a drawing; the shots come from the panel count
            self.store.save(board)
            return self.start_draw(board["id"])
        if panel_asset_ids:
            board.update(panels=list(panel_asset_ids), status="ready", then_render=None, own_images=True)
            self.store.save(board)
            if then_render:
                return self.start_render(board["id"], RenderSettings(**then_render["settings"]) if then_render.get("settings") else None,
                                         then_render.get("segment_seconds", 6.0), then_render.get("mood", ""),
                                         then_render.get("music", ""))
            return self.view(board)
        self.store.save(board)
        if draw:
            return self.start_draw(board["id"])
        return self.view(board)

    def update(self, board_id: str, **fields) -> dict:
        board = self.board(board_id)
        if board["status"] in BUSY:
            raise StoryboardError(f"The board is {board['status']}; wait for it to finish.")
        for key in ("title", "character", "style", "aspect_ratio"):
            if fields.get(key) is not None:
                board[key] = str(fields[key]).strip()
        if fields.get("shots") is not None:
            shots = [s.strip() for s in fields["shots"] if s and s.strip()]
            if not shots:
                raise StoryboardError("Give at least one shot.")
            panels = board["panels"] + [None] * max(0, len(shots) - len(board["panels"]))
            board["shots"], board["panels"] = shots, panels[: len(shots)]
        if fields.get("motions") is not None:
            board["motions"] = [m.strip() for m in fields["motions"]]
        grid_for(min(len(board["shots"]), MAX_PANELS), board["aspect_ratio"])
        return self.view(self.store.save(board))

    def delete(self, board_id: str) -> dict:
        board = self.board(board_id)
        task = self._tasks.pop(board_id, None)
        if task:
            task.cancel()
        self.store.delete(board_id)
        return {"deleted": board_id, "note": "Its sheets, panels and film stay in Media."}

    def _set(self, board_id: str, **fields) -> dict:
        board = self.board(board_id)
        board.update(fields)
        return self.store.save(board)

    def _launch(self, board_id: str, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks[board_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(board_id, None) if self._tasks.get(board_id) is _t else None)

    # ---- drawing
    def start_draw(self, board_id: str, only_missing: bool = False) -> dict:
        board = self.board(board_id)
        if board["status"] in BUSY:
            raise StoryboardError(f"The board is {board['status']}; wait for it to finish.")
        board = self._set(board_id, status="drawing", error="")
        self._launch(board_id, self._draw(board_id, only_missing))
        return self.view(board)

    async def _cut(self, board_id: str) -> None:
        """A board made from a ready-made sheet: its panels become the shots."""
        board = self.board(board_id)
        sheet = board["from_sheet"]
        cut = await split_sheet(self.service, sheet["asset_id"], sheet.get("columns", 0), sheet.get("rows", 0),
                                collection=f"storyboard/{board['title']}", tags=[f"board:{board_id}"])
        panels = [p["id"] for p in cut["panels"]]
        board = self.board(board_id)
        if board["shots"] and len(board["shots"]) != len(panels):
            raise StoryboardError(f"The sheet has {len(panels)} panels but {len(board['shots'])} shots were given: "
                                  "give one shot per panel, or none.")
        shots = board["shots"] or [f"shot {n}: animate this frame with natural, realistic motion" for n in range(1, len(panels) + 1)]
        board.update(shots=shots, panels=panels, from_sheet=None, status="ready", error="", note=cut["note"],
                     sheets=[{"asset_id": sheet["asset_id"], "grid": None, "shots": list(range(len(panels))),
                              "borders_found": cut["borders_found"]}])
        self.store.save(board)

    async def _draw(self, board_id: str, only_missing: bool) -> None:
        try:
            board = self.board(board_id)
            if board.get("from_sheet"):
                await self._cut(board_id)
                board = self.board(board_id)
                follow = board.pop("then_render", None)
                if follow:
                    self.store.save(board)
                    self.start_render(board_id, RenderSettings(**follow["settings"]) if follow.get("settings") else None,
                                      follow.get("segment_seconds", 6.0), follow.get("mood", ""), follow.get("music", ""))
                return
            todo = [i for i, panel in enumerate(board["panels"]) if not (only_missing and panel)]
            chunks = [todo[i:i + MAX_PANELS] for i in range(0, len(todo), MAX_PANELS)]
            collection = f"storyboard/{board['title']}"

            async def one(indices: list[int]) -> tuple[list[int], dict, dict]:
                made = await make_sheet(self.service, [board["shots"][i] for i in indices], board["character"],
                                        board["identity_asset_ids"], board["aspect_ratio"], board["style"])
                grid = made["grid"]
                cut = await split_sheet(self.service, made["sheet"]["id"], grid["columns"], grid["rows"], grid["panels"],
                                        collection=collection, tags=[f"board:{board_id}"])
                return indices, made, cut

            results = await asyncio.gather(*(one(c) for c in chunks))
            board = self.board(board_id)
            notes = []
            for indices, made, cut in results:
                board["sheets"].append({"asset_id": made["sheet"]["id"], "grid": made["grid"], "shots": indices,
                                        "borders_found": cut["borders_found"]})
                for index, panel in zip(indices, cut["panels"]):
                    board["panels"][index] = panel["id"]
                board["cost_usd"] = round(board.get("cost_usd", 0) + (made.get("cost_usd") or 0), 4)
                if cut["note"]:
                    notes.append(cut["note"])
            board.update(status="ready", error="", note=" ".join(dict.fromkeys(notes)))
            self.store.save(board)
            follow = board.pop("then_render", None)
            if follow:
                self.store.save(board)
                self.start_render(board_id, RenderSettings(**follow["settings"]) if follow.get("settings") else None,
                                  follow.get("segment_seconds", 6.0), follow.get("mood", ""), follow.get("music", ""))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the board says what went wrong; nothing else is waiting on it
            log.warning("storyboard %s: drawing failed: %s", board_id, exc)
            self._set(board_id, status="failed", error=f"Drawing failed: {exc}")

    async def redo_panel(self, board_id: str, number: int, shot: str | None = None, keep_setting: bool = True) -> dict:
        """Shot `number` (1-based) drawn again on its own; the new panel takes its place."""
        board = self.board(board_id)
        if board["status"] in BUSY:
            raise StoryboardError(f"The board is {board['status']}; wait for it to finish.")
        if not 1 <= number <= len(board["shots"]):
            raise StoryboardError(f"Shot {number} is not on this board (1 to {len(board['shots'])}).")
        if shot is not None and shot.strip():
            board["shots"][number - 1] = shot.strip()
        previous = board["panels"][number - 1] if keep_setting else None
        made = await redo_panel(self.service, board["shots"][number - 1], board["character"], board["identity_asset_ids"],
                                board["aspect_ratio"], board["style"], previous, number=number,
                                collection=f"storyboard/{board['title']}")
        text = board["shots"][number - 1]
        board = self.board(board_id)  # fresh: the drawing took a while
        board["shots"][number - 1] = text
        board["panels"][number - 1] = made["panel"]["id"]
        board["cost_usd"] = round(board.get("cost_usd", 0) + (made.get("cost_usd") or 0), 4)
        if board["status"] in ("draft", "failed") and all(board["panels"]):
            board["status"] = "ready"
        return self.view(self.store.save(board))

    def set_panel(self, board_id: str, number: int, asset_id: str) -> dict:
        """Any image from Media as shot `number`'s panel."""
        board = self.board(board_id)
        asset = self.service.store.get_asset(asset_id)
        if asset is None or asset.get("kind") != "image":
            raise StoryboardError(f"No image asset {asset_id!r}.")
        if not 1 <= number <= len(board["shots"]):
            raise StoryboardError(f"Shot {number} is not on this board (1 to {len(board['shots'])}).")
        board["panels"][number - 1] = asset_id
        if all(board["panels"]) and board["status"] in ("draft", "failed"):
            board["status"] = "ready"
        return self.view(self.store.save(board))

    # ---- the film
    def start_render(self, board_id: str, settings: RenderSettings | None = None, segment_seconds: float = 6.0,
                     mood: str = "", music: str = "") -> dict:
        board = self.board(board_id)
        if board["status"] in BUSY:
            raise StoryboardError(f"The board is {board['status']}; wait for it to finish.")
        missing = [str(i + 1) for i, panel in enumerate(board["panels"]) if not panel or not self.service.store.get_asset(panel)]
        if missing:
            raise StoryboardError(f"Shot{'s' if len(missing) > 1 else ''} {', '.join(missing)} ha{'ve' if len(missing) > 1 else 's'} "
                                  "no panel yet: draw the board, or redo those panels.")
        settings = (settings or RenderSettings(continuity="off")).model_copy(update={"aspect_ratio": board["aspect_ratio"]})
        board["render"] = {"jobs": [], "settings": settings.model_dump(), "segment_seconds": segment_seconds,
                           "mood": mood, "music": music, "film_asset_id": None, "drive": ""}
        board.update(status="rendering", error="")
        self.store.save(board)
        self._launch(board_id, self._render(board_id))
        return self.view(board)

    def _parts(self, board: dict) -> list[list[int]]:
        size = RENDER_PICTURES - len(board["identity_asset_ids"])
        indices = list(range(len(board["shots"])))
        return [indices[i:i + size] for i in range(0, len(indices), size)]

    async def _wait(self, job_id: str) -> dict:
        deadline = time.monotonic() + self.RENDER_WAIT
        while time.monotonic() < deadline:
            job = self.service.store.get_job(job_id) or {}
            if not job or job.get("status") in ("done", "failed", "cancelled"):
                return job
            await asyncio.sleep(self.poll)
        return {"status": "timeout"}

    async def _render(self, board_id: str) -> None:
        try:
            board = self.board(board_id)
            render = board["render"]
            settings = RenderSettings(**render["settings"])
            parts = self._parts(board)
            # resumable: parts already submitted keep their jobs
            for number, indices in enumerate(parts):
                if number < len(render["jobs"]):
                    continue
                motions = [(board["motions"][i] if i < len(board["motions"]) and board["motions"][i] else board["shots"][i])
                           for i in indices]
                job = await render_film(self.service, [board["panels"][i] for i in indices], board["identity_asset_ids"],
                                        motions, board["character"], settings, render["segment_seconds"],
                                        render["mood"], render["music"])
                board = self.board(board_id)
                board["render"]["jobs"].append(job["id"])
                self.store.save(board)
            board = self.board(board_id)
            for job_id in board["render"]["jobs"]:
                done = await self._wait(job_id)
                if done.get("status") != "done":
                    raise RequestError(f"Render {job_id[:8]} did not finish: {done.get('error') or done.get('status')}")
            self._set(board_id, status="joining")
            await self._join(board_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("storyboard %s: render failed: %s", board_id, exc)
            self._set(board_id, status="failed", error=str(exc))

    async def _join(self, board_id: str) -> None:
        from .manhwa import concat_videos, film_to_drive, rendered_file

        board = self.board(board_id)
        jobs = board["render"]["jobs"]
        work = tempfile.mkdtemp(prefix="hawk_board_")
        try:
            files = [await rendered_file(self.service, job_id, os.path.join(work, f"part_{n:02d}.mp4"))
                     for n, job_id in enumerate(jobs, 1)]
            out = os.path.join(work, "film.mp4")
            if len(files) == 1:
                shutil.copyfile(files[0], out)
            else:
                if not shutil.which("ffmpeg"):
                    raise Unavailable("ffmpeg is not installed on the server; run the notebook's ffmpeg cell.")
                await asyncio.to_thread(concat_videos, files, out)
            slug = re.sub(r"[^0-9A-Za-z]+", "-", board["title"]).strip("-").lower() or "storyboard"
            name = f"{slug}_{len(board['shots'])}-shots.mp4"
            with open(out, "rb") as handle:
                asset = await self.service.add_asset(
                    name, handle, "video/mp4", os.path.getsize(out), collection=f"storyboard/{board['title']}",
                    tags=["storyboard", "film"], local_path=out, dedupe=False,
                    source={"type": "generated", "engine": "ffmpeg", "generator": "storyboard film",
                            "board": board_id, "jobs": jobs})
            drive = await film_to_drive(self.service, out, name)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        board = self.board(board_id)
        board["render"].update(film_asset_id=asset["id"], drive=drive)
        board.update(status="done", error="")
        self.store.save(board)

    async def resume(self) -> list[str]:
        """After a restart: renders carry on (finished parts are reused); a drawing in flight is marked failed."""
        resumed = []
        for board in self.store.all():
            if board.get("status") == "drawing":
                self._set(board["id"], status="failed", error="The server restarted while drawing; draw again.")
            elif board.get("status") in ("rendering", "joining"):
                self._set(board["id"], status="rendering")
                self._launch(board["id"], self._render(board["id"]))
                resumed.append(board["id"])
        return resumed


# --------------------------------------------------------------------------- sheets from existing images

COMPOSE_MAX = 16
COMPOSE_LONG_SIDE = 4096  # the sheet's longest side; cells shrink to fit


def compose_grid(images: list[bytes], columns: int = 0, cell_aspect: str = "", fit: str = "contain",
                 border: int = 24, background: str = "black") -> tuple[bytes, dict]:
    """Images laid out as one sheet (PNG) and the grid used -- Pillow only, no model call.

    contain fits each image whole inside its cell (bars where shapes differ); cover fills the cell and crops the
    overflow. The cell shape defaults to the first image's."""
    from PIL import Image, ImageOps

    if not 1 <= len(images) <= COMPOSE_MAX:
        raise StoryboardError(f"A sheet takes 1 to {COMPOSE_MAX} images, not {len(images)}.")
    if fit not in ("contain", "cover"):
        raise StoryboardError("fit should be contain or cover.")
    opened = [ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB") for data in images]
    count = len(opened)
    if cell_aspect:
        ratio = ASPECTS.get(cell_aspect)
        if ratio is None:
            raise StoryboardError(f"cell_aspect should be one of: {', '.join(ASPECTS)}.")
    else:
        ratio = opened[0].width / opened[0].height
    if columns <= 0:  # the most square sheet, preferring full rows: each empty cell costs a little squareness
        columns = min(range(1, count + 1),
                      key=lambda c: abs(math.log((c * ratio) / math.ceil(count / c))) + 0.25 * (c * math.ceil(count / c) - count))
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    border = max(0, min(200, border))
    # each cell as big as the source allows, then shrunk so the sheet's long side stays within the cap
    cell_h = max(min(image.height, round(image.width / ratio)) for image in opened)
    cell_w = round(cell_h * ratio)
    width, height = columns * cell_w + (columns + 1) * border, rows * cell_h + (rows + 1) * border
    scale = min(1.0, COMPOSE_LONG_SIDE / max(width, height))
    cell_w, cell_h, border = max(16, math.floor(cell_w * scale)), max(16, math.floor(cell_h * scale)), math.floor(border * scale)
    width, height = columns * cell_w + (columns + 1) * border, rows * cell_h + (rows + 1) * border
    fill = {"black": (0, 0, 0), "white": (255, 255, 255), "grey": (128, 128, 128)}.get(background, (0, 0, 0))
    sheet = Image.new("RGB", (width, height), fill)
    for index, image in enumerate(opened):
        row, column = divmod(index, columns)
        if fit == "cover":
            cell = ImageOps.fit(image, (cell_w, cell_h), Image.LANCZOS)
            left, top = 0, 0
        else:
            cell = ImageOps.contain(image, (cell_w, cell_h), Image.LANCZOS)
            left, top = (cell_w - cell.width) // 2, (cell_h - cell.height) // 2
        sheet.paste(cell, (border + column * (cell_w + border) + left, border + row * (cell_h + border) + top))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue(), {"columns": columns, "rows": rows, "panels": count, "cell": [cell_w, cell_h],
                               "size": [width, height]}


async def compose_sheet(service, asset_ids: list[str], columns: int = 0, cell_aspect: str = "", fit: str = "contain",
                        border: int = 24, background: str = "black", title: str = "", collection: str | None = None) -> dict:
    """Existing images merged into one sheet asset ($0, no model call): a contact sheet to share, or one reference
    picture that carries several views in a single slot of a render's nine."""
    assets = []
    for asset_id in asset_ids or []:
        asset = service.store.get_asset(asset_id)
        if asset is None or asset.get("kind") != "image":
            raise StoryboardError(f"No image asset {asset_id!r}.")
        assets.append(asset)
    if not assets:
        raise StoryboardError("Pick the images to merge.")
    images = [await service.asset_bytes(asset) for asset in assets]
    try:
        data, grid = await asyncio.to_thread(compose_grid, images, columns, cell_aspect, fit, border, background)
    except StoryboardError:
        raise
    except Exception as exc:  # an unreadable file
        raise StoryboardError(f"Could not read one of the images: {exc}") from None
    slug = re.sub(r"[^a-z0-9]+", "_", (title or "sheet").lower()).strip("_")[:40] or "sheet"
    stored = await service.add_asset(
        f"{slug}_{len(assets)}.png", io.BytesIO(data), "image/png", len(data),
        collection=collection or assets[0].get("collection") or "Generated", tags=["composed-sheet"],
        source={"type": "composed", "images": [a["id"] for a in assets], "grid": [grid["columns"], grid["rows"]]},
    )
    return {"sheet": service.asset_view(stored), "grid": grid, "cost_usd": 0.0}


# --------------------------------------------------------------------------- the shot writer

#: Who writes the shots: vision models first, so the reference images are actually looked at.
WRITER_MODELS = "google/gemma-4-31b-it, deepseek-ai/deepseek-v4.1-flash, moonshotai/kimi-k2.6"
WRITER_IMAGES = 4


async def suggest_shots(service, idea: str, reference_asset_ids: list[str] | None = None, count: int = 8,
                        aspect_ratio: str = "9:16", models: str | None = None) -> dict:
    """A storyboard written by an LLM from the idea and the reference images: the character as the images show
    them, the look, one shot line and one motion line per shot. The system prompt is the editable "storyboard"
    prompt (Prompts page), with the platform rules after it."""
    import base64

    from . import models_llm
    from .atlas import AtlasError
    from .manhwa import parse_json
    from .prompts import PLATFORM_RULES

    idea = (idea or "").strip()
    if not idea:
        raise StoryboardError("Say in a line what the storyboard is about.")
    if not 1 <= count <= 40:
        raise StoryboardError("Ask for 1 to 40 shots.")
    parts = []
    for asset_id in (reference_asset_ids or [])[:WRITER_IMAGES]:
        asset = service.store.get_asset(asset_id)
        if asset is None or asset.get("kind") != "image":
            raise StoryboardError(f"No image asset {asset_id!r}.")
        data, mime = await service.asset_file(asset_id, 768)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
    chain = models_llm.resolve_many("planner", models or WRITER_MODELS, await service.model_catalogue(), vision=bool(parts))
    if not chain:
        raise Unavailable("No model is set up to write the shots.")
    ask = (f"Idea: {idea}\nShots: exactly {count}\nVideo shape: {aspect_ratio}\n"
           + (f"The first {len(parts)} attached image{'s are' if len(parts) > 1 else ' is'} the character's reference."
              if parts else "No reference image: describe the character from the idea."))
    messages = [{"role": "system", "content": f"{service.prompts.get('storyboard').rstrip()}\n\n{PLATFORM_RULES}"},
                {"role": "user", "content": [{"type": "text", "text": ask}, *parts] if parts else ask}]
    failures = []
    for model in chain:
        try:
            reply, _usage = await service.atlas.chat(model, messages, json_mode=True, max_tokens=4000, temperature=0.8,
                                                     max_retries=1, purpose="storyboard shots")
        except AtlasError as exc:
            failures.append(f"{model}: {exc}"[:240])
            continue
        data = parse_json(reply or "")
        shots = [str(s).strip() for s in (data or {}).get("shots") or [] if str(s).strip()] if isinstance(data, dict) else []
        if len(shots) < count:
            failures.append(f"{model}: wrote {len(shots)} of {count} shots")
            continue
        motions = [str(m).strip() for m in data.get("motions") or []][:count]
        return {"character": str(data.get("character") or "").strip(), "style": str(data.get("style") or "").strip(),
                "shots": shots[:count], "motions": motions + [""] * (count - len(motions)), "model": model}
    raise Unavailable("No model wrote the shots: " + (" | ".join(failures) or "none answered."))
