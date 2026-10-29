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


def _runs(profile: list[int], minimum: int) -> list[tuple[int, int]]:
    """(start, end) of each stretch where the profile is mostly bright, skipping stretches shorter than minimum."""
    runs, start = [], None
    for index, value in enumerate(profile + [0]):
        bright = value > 127
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
    xs = _merge_to(_runs(across, max(1, round(width * MIN_PANEL_SHARE))), columns)
    ys = _merge_to(_runs(down, max(1, round(height * MIN_PANEL_SHARE))), rows)
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


async def split_sheet(service, sheet_asset_id: str, columns: int, rows: int, panels: int | None = None,
                      collection: str | None = None, tags: list[str] | None = None) -> dict:
    """The sheet cut into its panels, each stored as an asset that remembers its sheet and position."""
    asset = service.store.get_asset(sheet_asset_id)
    if asset is None or asset.get("kind") != "image":
        raise StoryboardError(f"No image asset {sheet_asset_id!r}.")
    if columns < 1 or rows < 1 or columns * rows > MAX_PANELS:
        raise StoryboardError(f"columns x rows should be 1 to {MAX_PANELS} panels.")
    data = await service.asset_bytes(asset)
    images, found = cut_grid(data, columns, rows)
    images = images[: panels or len(images)]
    stem = (asset.get("filename") or "sheet").rsplit(".", 1)[0][:40]
    out = []
    for number, image in enumerate(images, 1):
        stored = await service.add_asset(
            f"{stem}_panel_{number}.png", io.BytesIO(image), "image/png", len(image),
            collection=collection or asset.get("collection") or "Generated",
            tags=["storyboard-panel", *(tags or [])],
            source={"type": "storyboard", "sheet": sheet_asset_id, "panel": number, "grid": [columns, rows]},
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
               link: dict | None = None, then_render: dict | None = None) -> dict:
        shots = [s.strip() for s in shots or [] if s and s.strip()]
        if not shots:
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
        grid_for(min(len(shots), MAX_PANELS), aspect_ratio)  # validates the aspect ratio
        board = {"id": uuid.uuid4().hex[:12], "title": (title or "Storyboard").strip()[:120],
                 "character": character.strip(), "identity_asset_ids": list(identity_asset_ids),
                 "aspect_ratio": aspect_ratio, "style": style.strip(), "shots": shots,
                 "motions": [m.strip() for m in motions] if motions else [], "sheets": [], "panels": [None] * len(shots),
                 "status": "draft", "error": "", "cost_usd": 0.0, "render": {}, "link": link or {},
                 # {settings, segment_seconds, mood, music}: render as soon as the drawing is done (one-call films)
                 "then_render": then_render, "created_at": time.time()}
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

    async def _draw(self, board_id: str, only_missing: bool) -> None:
        try:
            board = self.board(board_id)
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
