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


MODES = ("frame", "look", "none")
#: What a panel is to its segment. frame: the literal first frame. look: the place, light, outfit and person --
#: the video model composes the action, the position and the framing itself (what H3 does better than an image
#: model for anything physical). none: no panel of its own, only the identity pictures.
MODE_WORDING = {
    "frame": "{tag} is this segment's first frame (keyframe completion): start exactly on it -- its pose, framing, "
             "setting and clothing (or nudity) -- and animate from there",
    "look": "{tag} is only a look reference for this segment -- its place, light, outfit (or nudity) and the person -- "
            "NOT its first frame: compose the action, the body position and the camera framing freely from the "
            "description below, with natural, physically real movement",
    "none": "this segment has no picture of its own: build it from the description, keeping the person from the "
            "identity pictures",
}


def segment_pictures(modes: list[str]) -> list[int | None]:
    """The picture number each segment's panel gets in one render (None for a segment without a panel)."""
    numbers, count = [], 0
    for mode in modes:
        if mode == "none":
            numbers.append(None)
        else:
            count += 1
            numbers.append(count)
    return numbers


def film_brief(beats: list[str], character: str, mood: str = "", music: str = "", identities: int = 1,
               modes: list[str] | None = None, dialogue: list[list[str]] | None = None,
               music_over: bool = False) -> str:
    """The planner's brief for one render: what each segment's picture is to it (by mode), the motion, any lines."""
    modes = [m if m in MODES else "frame" for m in (modes or [])] + ["frame"] * max(0, len(beats) - len(modes or []))
    numbers = segment_pictures(modes[: len(beats)])
    panels = sum(1 for n in numbers if n)
    tags = [f"<Picture {panels + n}>" for n in range(1, identities + 1)]
    who = (f"{tags[0]} is the identity reference" if identities == 1
           else f"{', '.join(tags[:-1])} and {tags[-1]} are the identity references (one per person)")
    sound = ("No music in any segment: only the sounds of the scene (one music track is laid over the whole film "
             "afterwards). " if music_over else (f"Music: {music.strip().rstrip('.')}. " if music.strip() else ""))
    lines = []
    for index, (beat, mode, number) in enumerate(zip(beats, modes, numbers), 1):
        wording = MODE_WORDING[mode].format(tag=f"<Picture {number}>" if number else "")
        said = [line.strip() for line in (dialogue[index - 1] if dialogue and index - 1 < len(dialogue) else []) if line.strip()]
        spoken = (" Lines, spoken word for word in this order: " + " / ".join(said) + ".") if said else ""
        lines.append(f"Segment {index}: {wording}. {beat.strip().rstrip('.')}.{spoken}")
    return (
        f"{character.strip()} {who} (face and body) for every segment. "
        f"{(mood.strip() + ' ') if mood.strip() else ''}"
        f"{len(beats)} segments in story order. Natural, realistic motion with real weight. Nobody speaks unless a "
        f"segment gives lines. {sound}" + " ".join(lines)
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


async def render_film(service, panel_asset_ids: list, identity_asset_ids, beats: list[str], character: str,
                      settings: RenderSettings | None = None, segment_seconds: float = 6.0, mood: str = "",
                      music: str = "", modes: list[str] | None = None, dialogue: list[list[str]] | None = None,
                      music_over: bool = False) -> dict:
    """One render of the shots in order, planned in the H3 reference format; each panel is to its segment what its
    mode says (first frame, look reference, or none)."""
    identity_asset_ids = [identity_asset_ids] if isinstance(identity_asset_ids, str) else list(identity_asset_ids)
    if not beats:
        raise StoryboardError("No shots to animate.")
    if not identity_asset_ids:
        raise StoryboardError("Give the identity picture the panels keep the face and body of.")
    modes = [m if m in MODES else "frame" for m in (modes or ["frame"] * len(beats))][: len(beats)]
    modes += ["frame"] * (len(beats) - len(modes))
    if len(panel_asset_ids) != len(beats):
        raise StoryboardError(f"{len(beats)} shots need {len(beats)} panel slots (None where a shot has no panel).")
    panels = [p for p, mode in zip(panel_asset_ids, modes) if mode != "none"]
    if any(not p for p in panels):
        raise StoryboardError("A shot with a panel mode of frame or look has no panel.")
    if len(panels) + len(identity_asset_ids) > RENDER_PICTURES:
        raise StoryboardError(f"{len(panels)} panels and {len(identity_asset_ids)} identity pictures are more "
                              f"than a render's {RENDER_PICTURES}; use a board, which splits the film into parts.")
    settings = settings or RenderSettings(aspect_ratio="9:16", continuity="off")
    numbers = segment_pictures(modes)
    references = [ReferenceIn(asset_id=p, role="picture",
                              label=f"{'first frame' if m == 'frame' else 'look reference'} for segment {i}")
                  for i, (p, m) in enumerate(zip(panel_asset_ids, modes), 1) if m != "none"]
    references += [ReferenceIn(asset_id=asset_id, role="picture", label="identity (face and body)")
                   for asset_id in identity_asset_ids]
    del numbers
    request = VideoRequest(
        references=references,
        story=PlannerOptions(story=film_brief(beats, character, mood, music, len(identity_asset_ids), modes, dialogue,
                                              music_over),
                             segment_count=len(beats), segment_seconds=segment_seconds,
                             aspect_ratio=settings.aspect_ratio),
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


# --------------------------------------------------------------------------- engine rules

PLATFORMS = ("general", "instagram", "fanvue")
#: Added to every sheet and panel prompt, and checked on every panel, for a platform with rules.
PLATFORM_RULE = {
    "instagram": "Everyone is fully and modestly clothed in every panel: no nudity, no see-through or wet "
                 "transparent fabric, no exposed nipples, no underwear-only looks.",
}
#: Draft renders fast to check the film; final is the one to post.
QUALITY = {"draft": 0.6, "final": 0.98}
REVIEWS = ("auto", "ask")
CHECK_ROUNDS = 2
CHECK_BATCH = 3
MUSIC_MODEL = "minimax/music-3.0"
MUSIC_COST = 0.15


def platform_style(style: str, platform: str) -> str:
    rule = PLATFORM_RULE.get(platform or "general", "")
    return f"{style.strip()} {rule}".strip() if rule else style.strip()


def pick_models(files: list[str]) -> dict:
    """Which base model renders which kind of shot. Faces held to a first frame go to a reference-faithful hybrid
    (10Eros beta5), physical action and positions to the newer one (beta7); either falls back to the server's
    default ("") when it isn't installed."""
    names = [f for f in files or []]
    frame = next((f for f in names if "beta5" in f.lower()), "")
    action = next((f for f in names if "beta7" in f.lower()), "")
    return {"frame": frame or action, "action": action or frame}


def plan_parts(modes: list[str], identities: int, models: dict) -> list[dict]:
    """The renders a film is cut into, in order: a new part when the picture limit is reached or when the next
    shot's kind needs another base model. [{"shots": [indices], "kind": "frame" | "action", "unet": file}]."""
    room = RENDER_PICTURES - identities
    parts: list[dict] = []
    for index, mode in enumerate(modes):
        kind = "frame" if mode == "frame" else "action"
        unet = models.get(kind, "")
        needs = 0 if mode == "none" else 1
        part = parts[-1] if parts else None
        used = sum(1 for i in part["shots"] if modes[i] != "none") if part else 0
        if part is None or part["unet"] != unet or used + needs > room or len(part["shots"]) >= 12:
            parts.append({"shots": [index], "kind": kind, "unet": unet})
        else:
            part["shots"].append(index)
    return parts


CHECK_PROMPT = """You check storyboard panels before they are animated. The FIRST image is the identity reference:
the person every panel must show. The images after it are panels, in the order listed. For each panel decide
whether it is usable, and say briefly what is wrong if it is not. A panel fails when:
- it does not show the same person as the identity reference (a different face, hair, skin tone or a clearly
  different body and proportions);
- anatomy is broken: extra or missing fingers, limbs or nipples, twisted joints, a melted or doubled face;
- there are artifacts: garbled text, a watermark, a duplicated person, a cut-off head where a face matters;
- it breaks the platform rule given below, when one is given.
Do not fail a panel for its style, its nudity (unless the platform rule forbids it) or the shot it chose.
Reply with JSON only: {"panels": [{"shot": <number>, "ok": true or false, "issues": "<short, empty when ok>"}]}"""


async def check_panels(service, board: dict, shots: list[int]) -> dict[int, str]:
    """{shot index: what is wrong} for the panels a vision model rejects; {} when every panel passes or no vision
    model answers (the check is a safety net, never a reason the film can't go ahead)."""
    import base64

    from . import models_llm
    from .atlas import AtlasError
    from .manhwa import parse_json

    try:
        identity, mime = await service.asset_file(board["identity_asset_ids"][0], 640)
    except Exception:
        return {}
    chain = models_llm.resolve_many("vision", None, await service.model_catalogue(), vision=True)
    rule = PLATFORM_RULE.get(board.get("platform") or "general", "")
    failed: dict[int, str] = {}
    for start in range(0, len(shots), CHECK_BATCH):
        batch = [i for i in shots[start:start + CHECK_BATCH] if board["panels"][i]]
        if not batch:
            continue
        parts = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(identity).decode()}}]
        for index in batch:
            try:
                data, kind = await service.asset_file(board["panels"][index], 640)
            except Exception:
                continue
            parts.append({"type": "image_url", "image_url": {"url": f"data:{kind};base64," + base64.b64encode(data).decode()}})
        ask = ("Panels, in order: " + ", ".join(f"shot {i + 1} ({board['shots'][i][:120]})" for i in batch) + "."
               + (f"\nPlatform rule: {rule}" if rule else "\nNo platform rule."))
        messages = [{"role": "system", "content": CHECK_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": ask}, *parts]}]
        for model in chain:
            try:
                reply, _ = await service.atlas.chat(model, messages, json_mode=True, max_tokens=1200, temperature=0,
                                                    max_retries=1, purpose="storyboard check")
            except AtlasError:
                continue
            verdicts = (parse_json(reply or "") or {}).get("panels")
            if not isinstance(verdicts, list):
                continue
            for verdict in verdicts:
                try:
                    shot = int(verdict.get("shot")) - 1
                except (TypeError, ValueError, AttributeError):
                    continue
                if shot in batch and verdict.get("ok") is False:
                    failed[shot] = str(verdict.get("issues") or "rejected").strip()[:300]
            break
    return failed


async def make_music(service, prompt: str) -> dict:
    """One instrumental track for a whole film from MiniMax Music 3.0 on Atlas (about $0.15), stored as an audio asset."""
    data = await service.image_atlas.generate_media(
        {"model": MUSIC_MODEL, "prompt": prompt.strip()[:2000], "is_instrumental": True, "format": "mp3"},
        endpoint="generateAudio", what="Music")
    asset = await service.add_asset("film_music.mp3", io.BytesIO(data[0]), "audio/mpeg", len(data[0]),
                                    collection="Music", tags=["music", "generated"],
                                    source={"type": "generated", "engine": "atlas", "generator": MUSIC_MODEL,
                                            "prompt": prompt.strip()[:500], "cost_usd": MUSIC_COST})
    return asset


def lay_music(film: str, music: str, out: str, seconds: float) -> bool:
    """The track under the whole film: looped if short, faded out at the end, the scene's own sound kept low."""
    import subprocess

    fade = max(0.0, seconds - 2.5)
    graph = (f"[1:a]volume=0.9,afade=t=out:st={fade:.2f}:d=2.5[m];"
             "[0:a]volume=0.35[s];[s][m]amix=inputs=2:duration=first:dropout_transition=0[a]")
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", film, "-stream_loop", "-1", "-i", music,
               "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", out]
    if subprocess.run(command).returncode == 0:
        return True
    # a film without an audio stream: the track alone
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", film, "-stream_loop", "-1", "-i", music,
               "-filter_complex", f"[1:a]afade=t=out:st={fade:.2f}:d=2.5[a]", "-map", "0:v", "-map", "[a]",
               "-c:v", "copy", "-c:a", "aac", "-shortest", out]
    return subprocess.run(command).returncode == 0


# --------------------------------------------------------------------------- boards

#: A board while it is being worked on; anything else can be edited.
BUSY = ("drawing", "checking", "rendering", "joining")


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
    """The film engine. A board is one film: its people, an ordered shot list (each with a mode, a motion and any
    lines), the panels (drawn on Seedream Flash sheets, the user's own images, or cut from ready-made sheets), an
    automatic check, an optional review stop, and the film -- rendered in parts (by base model and the nine-picture
    limit), joined, with one music track over the whole of it.

    Story, Manhwa, MCP, the agent and the Storyboard page all make films through this, so every one of them gets the
    same steps and the same rules."""

    poll = 5.0
    RENDER_WAIT = 6 * 3600.0

    def __init__(self, service):
        self.service = service
        self.store = StoryboardStore(service.settings.data_dir)
        self._tasks: dict[str, asyncio.Task] = {}
        self.characters = None  # the CharacterManager; create_app sets it
        #: called with the board when its film is done -- Story uses it to take a chapter's film
        self.done_hooks: list = []

    # ---- reading
    def list(self) -> list[dict]:
        return [self.view(board) for board in self.store.all()]

    def board(self, board_id: str) -> dict:
        board = self.store.get(board_id)
        if board is None:
            raise NotFound(f"No storyboard {board_id!r}.")
        return board

    @staticmethod
    def modes(board: dict) -> list[str]:
        modes = [m if m in MODES else "frame" for m in board.get("modes") or []]
        return (modes + ["frame"] * len(board["shots"]))[: len(board["shots"])]

    def view(self, board: dict | str) -> dict:
        board = self.board(board) if isinstance(board, str) else board
        assets = self.service.store.get_asset
        out = dict(board)
        out["modes"] = self.modes(board)
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
    def create(self, *, title: str, character: str = "", identity_asset_ids: list[str] | None = None,
               shots: list[str] | None = None, aspect_ratio: str = "9:16", style: str = "",
               motions: list[str] | None = None, draw: bool = True, link: dict | None = None,
               then_render: dict | None = None, panel_asset_ids: list[str] | None = None,
               sheet_asset_id: str | None = None, sheet_columns: int = 0, sheet_rows: int = 0,
               characters: list[str] | None = None, modes: list[str] | None = None,
               dialogue: list[list[str]] | None = None, platform: str = "", quality: str = "final",
               review: str = "auto", check: bool = True, music_asset_id: str | None = None, music_prompt: str = "",
               from_sheets: list[dict] | None = None) -> dict:
        """A board, drawn (and, with then_render, filmed) in the background.

        characters: names from the library -- their facts become the character text (character adds to it) and
        their identity pictures the identity; their platform is the default. panel_asset_ids: the user's own
        images as the panels (nothing drawn). sheet_asset_id / from_sheets: ready-made sheets cut into panels (a
        sheet merged from the user's images gives those images back). modes: frame, look or none per shot.
        review: auto films as soon as the panels pass the check; ask stops at status review for approve()."""
        shots = [s.strip() for s in shots or [] if s and s.strip()]
        identity_asset_ids = list(identity_asset_ids or [])
        names: list[str] = []
        if characters:
            if self.characters is None:
                raise StoryboardError("The character library isn't available on this server.")
            people = self.characters.resolve(characters)
            names = people["names"]
            character = f"{people['character']} {character.strip()}".strip()
            identity_asset_ids = identity_asset_ids or people["identity_asset_ids"]
            platform = platform or people["platform"]
        platform = platform or "general"
        if platform not in PLATFORMS:
            raise StoryboardError(f"platform should be one of: {', '.join(PLATFORMS)}.")
        if quality not in QUALITY:
            raise StoryboardError(f"quality should be one of: {', '.join(QUALITY)}.")
        if review not in REVIEWS:
            raise StoryboardError(f"review should be one of: {', '.join(REVIEWS)}.")
        sheets_to_cut = list(from_sheets or [])
        if sheet_asset_id:
            sheet = self.service.store.get_asset(sheet_asset_id)
            if sheet is None or sheet.get("kind") != "image":
                raise StoryboardError(f"No image asset {sheet_asset_id!r}.")
            source = sheet.get("source") or {}
            if source.get("type") == "composed" and source.get("images"):
                panel_asset_ids = [i for i in source["images"] if self.service.store.get_asset(i)]
            else:
                sheets_to_cut.append({"asset_id": sheet_asset_id, "columns": sheet_columns, "rows": sheet_rows})
        for sheet in sheets_to_cut:
            asset = self.service.store.get_asset(sheet.get("asset_id"))
            if asset is None or asset.get("kind") != "image":
                raise StoryboardError(f"No image asset {sheet.get('asset_id')!r}.")
        if panel_asset_ids:
            for asset_id in panel_asset_ids:
                asset = self.service.store.get_asset(asset_id)
                if asset is None or asset.get("kind") != "image":
                    raise StoryboardError(f"No image asset {asset_id!r}.")
            if shots and len(shots) != len(panel_asset_ids):
                raise StoryboardError(f"{len(panel_asset_ids)} images need {len(panel_asset_ids)} shots, one each (or none).")
            shots = shots or [f"shot {n}: animate this frame with natural, realistic motion"
                              for n in range(1, len(panel_asset_ids) + 1)]
        if not shots and not sheets_to_cut:
            raise StoryboardError("Give at least one shot.")
        if len(shots) > 40:
            raise StoryboardError("At most 40 shots on one board.")
        if not identity_asset_ids:
            raise StoryboardError("Give the character's identity picture, or a character from the library.")
        if len(identity_asset_ids) > 3:
            raise StoryboardError("At most 3 identity pictures (each one is a picture every render part carries).")
        for asset_id in identity_asset_ids:
            asset = self.service.store.get_asset(asset_id)
            if asset is None or asset.get("kind") != "image":
                raise StoryboardError(f"No image asset {asset_id!r}.")
        if music_asset_id:
            music = self.service.store.get_asset(music_asset_id)
            if music is None or music.get("kind") != "audio":
                raise StoryboardError(f"music_asset_id must be an audio file in the library; {music_asset_id!r} isn't.")
        grid_for(max(1, min(len(shots), MAX_PANELS)), aspect_ratio)  # validates the aspect ratio
        modes = [m if m in MODES else "frame" for m in (modes or [])]
        board = {"id": uuid.uuid4().hex[:12], "title": (title or "Storyboard").strip()[:120],
                 "character": character.strip(), "characters": names, "identity_asset_ids": identity_asset_ids,
                 "aspect_ratio": aspect_ratio, "style": style.strip(), "shots": shots,
                 "motions": [m.strip() for m in motions] if motions else [], "modes": modes,
                 "dialogue": [list(d or []) for d in dialogue] if dialogue else [],
                 "sheets": [], "panels": [None] * len(shots), "platform": platform, "quality": quality,
                 "review": review, "check": bool(check), "checks": {},
                 "music": {"asset_id": music_asset_id or None, "prompt": music_prompt.strip()},
                 "status": "draft", "error": "", "cost_usd": 0.0, "render": {}, "link": link or {},
                 # {settings, segment_seconds, mood, music}: film as soon as the panels are ready (one-call films)
                 "then_render": then_render, "from_sheets": sheets_to_cut or None, "created_at": time.time()}
        if sheets_to_cut:  # cut in the background, like a drawing; the shots come from the panel count
            self.store.save(board)
            return self.start_draw(board["id"])
        if panel_asset_ids:
            board.update(panels=list(panel_asset_ids), own_images=True)
            self.store.save(board)
            return self.start_draw(board["id"]) if (check or then_render) else self._ready(board["id"])
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
        for key, allowed in (("platform", PLATFORMS), ("quality", tuple(QUALITY)), ("review", REVIEWS)):
            if fields.get(key) is not None:
                if fields[key] not in allowed:
                    raise StoryboardError(f"{key} should be one of: {', '.join(allowed)}.")
                board[key] = fields[key]
        if fields.get("shots") is not None:
            shots = [s.strip() for s in fields["shots"] if s and s.strip()]
            if not shots:
                raise StoryboardError("Give at least one shot.")
            panels = board["panels"] + [None] * max(0, len(shots) - len(board["panels"]))
            board["shots"], board["panels"] = shots, panels[: len(shots)]
        if fields.get("motions") is not None:
            board["motions"] = [m.strip() for m in fields["motions"]]
        if fields.get("modes") is not None:
            bad = [m for m in fields["modes"] if m not in MODES]
            if bad:
                raise StoryboardError(f"modes are frame, look or none, not {bad[0]!r}.")
            board["modes"] = list(fields["modes"])
        if fields.get("music_asset_id") is not None or fields.get("music_prompt") is not None:
            music = board.get("music") or {}
            if fields.get("music_asset_id") is not None:
                music["asset_id"] = fields["music_asset_id"] or None
            if fields.get("music_prompt") is not None:
                music["prompt"] = fields["music_prompt"].strip()
            board["music"] = music
        grid_for(max(1, min(len(board["shots"]), MAX_PANELS)), board["aspect_ratio"])
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

    # ---- drawing, checking, review
    def start_draw(self, board_id: str, only_missing: bool = False) -> dict:
        board = self.board(board_id)
        if board["status"] in BUSY:
            raise StoryboardError(f"The board is {board['status']}; wait for it to finish.")
        board = self._set(board_id, status="drawing", error="")
        self._launch(board_id, self._draw(board_id, only_missing))
        return self.view(board)

    def _platform_style(self, board: dict) -> str:
        return platform_style(board.get("style") or "", board.get("platform") or "general")

    async def _cut(self, board_id: str) -> None:
        """Panels from ready-made sheets, in order: they become the shots."""
        board = self.board(board_id)
        panels, sheets, notes = [], [], []
        for sheet in board.get("from_sheets") or []:
            cut = await split_sheet(self.service, sheet["asset_id"], sheet.get("columns", 0), sheet.get("rows", 0),
                                    sheet.get("panels"), collection=f"storyboard/{board['title']}", tags=[f"board:{board_id}"])
            first = len(panels)
            panels += [p["id"] for p in cut["panels"]]
            sheets.append({"asset_id": sheet["asset_id"], "grid": None, "shots": list(range(first, len(panels))),
                           "borders_found": cut["borders_found"]})
            if cut["note"]:
                notes.append(cut["note"])
        board = self.board(board_id)
        if board["shots"] and len(board["shots"]) != len(panels):
            raise StoryboardError(f"The sheets have {len(panels)} panels but {len(board['shots'])} shots were given: "
                                  "give one shot per panel, or none.")
        shots = board["shots"] or [f"shot {n}: animate this frame with natural, realistic motion" for n in range(1, len(panels) + 1)]
        board.update(shots=shots, panels=panels, from_sheets=None, sheets=sheets, note=" ".join(dict.fromkeys(notes)))
        self.store.save(board)

    async def _draw(self, board_id: str, only_missing: bool) -> None:
        try:
            board = self.board(board_id)
            if board.get("from_sheets") or board.get("from_sheet"):
                if board.get("from_sheet"):  # a board saved by an earlier version
                    board["from_sheets"] = [board.pop("from_sheet")]
                    self.store.save(board)
                await self._cut(board_id)
            elif not board.get("own_images"):
                await self._draw_sheets(board_id, only_missing)
            board = self.board(board_id)
            if board.get("check"):  # boards from before the check have none
                await self._check(board_id)
            self._ready(board_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the board says what went wrong; nothing else is waiting on it
            log.warning("storyboard %s: drawing failed: %s", board_id, exc)
            self._set(board_id, status="failed", error=f"Drawing failed: {exc}")

    async def _draw_sheets(self, board_id: str, only_missing: bool) -> None:
        board = self.board(board_id)
        modes = self.modes(board)
        todo = [i for i, panel in enumerate(board["panels"]) if modes[i] != "none" and not (only_missing and panel)]
        chunks = [todo[i:i + MAX_PANELS] for i in range(0, len(todo), MAX_PANELS)]
        collection = f"storyboard/{board['title']}"
        style = self._platform_style(board)

        async def one(indices: list[int]) -> tuple[list[int], dict, dict]:
            made = await make_sheet(self.service, [board["shots"][i] for i in indices], board["character"],
                                    board["identity_asset_ids"], board["aspect_ratio"], style)
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
        board["note"] = " ".join(dict.fromkeys(notes))
        self.store.save(board)

    async def _check(self, board_id: str) -> None:
        """A vision model looks at every panel; rejected drawn panels are drawn again (at most CHECK_ROUNDS times).
        The user's own images are only reported on, never replaced."""
        board = self._set(board_id, status="checking")
        modes = self.modes(board)
        todo = [i for i, panel in enumerate(board["panels"]) if panel and modes[i] != "none"]
        checks: dict = {}
        for round_number in range(CHECK_ROUNDS + 1):
            board = self.board(board_id)
            failed = await check_panels(self.service, board, todo)
            for index in todo:
                checks[str(index)] = {"ok": index not in failed, "issues": failed.get(index, ""), "round": round_number}
            self._set(board_id, checks=dict(checks))
            if not failed or board.get("own_images") or round_number == CHECK_ROUNDS:
                break
            for index, issues in failed.items():
                board = self.board(board_id)
                made = await redo_panel(self.service, f"{board['shots'][index]} Fix: {issues}", board["character"],
                                        board["identity_asset_ids"], board["aspect_ratio"], self._platform_style(board),
                                        board["panels"][index], number=index + 1,
                                        collection=f"storyboard/{board['title']}")
                board = self.board(board_id)
                board["panels"][index] = made["panel"]["id"]
                board["cost_usd"] = round(board.get("cost_usd", 0) + (made.get("cost_usd") or 0), 4)
                self.store.save(board)
            todo = list(failed)

    def _ready(self, board_id: str) -> dict:
        """After drawing and checking: film straight away (review auto with a film asked for), stop for review
        (review ask), or wait (ready)."""
        board = self.board(board_id)
        follow = board.get("then_render")
        if follow and board.get("review", "auto") == "auto":
            board = self._set(board_id, then_render=None, status="ready")
            return self.start_render(board_id, RenderSettings(**follow["settings"]) if follow.get("settings") else None,
                                     follow.get("segment_seconds", 6.0), follow.get("mood", ""), follow.get("music", ""))
        return self.view(self._set(board_id, status="review" if follow else "ready", error=""))

    def approve(self, board_id: str, settings: RenderSettings | None = None, segment_seconds: float | None = None,
                mood: str | None = None, music: str | None = None) -> dict:
        """The user has looked at the panels: make the film (with what the one-call request asked for, unless
        overridden here)."""
        board = self.board(board_id)
        if board["status"] not in ("review", "ready"):
            raise StoryboardError(f"The board is {board['status']}: there is nothing to approve yet.")
        follow = board.get("then_render") or {}
        self._set(board_id, then_render=None, status="ready")
        base = settings or (RenderSettings(**follow["settings"]) if follow.get("settings") else None)
        return self.start_render(board_id, base, segment_seconds if segment_seconds is not None else follow.get("segment_seconds", 6.0),
                                 mood if mood is not None else follow.get("mood", ""),
                                 music if music is not None else follow.get("music", ""))

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
                                board["aspect_ratio"], self._platform_style(board), previous, number=number,
                                collection=f"storyboard/{board['title']}")
        text = board["shots"][number - 1]
        board = self.board(board_id)  # fresh: the drawing took a while
        board["shots"][number - 1] = text
        board["panels"][number - 1] = made["panel"]["id"]
        board["cost_usd"] = round(board.get("cost_usd", 0) + (made.get("cost_usd") or 0), 4)
        (board.get("checks") or {}).pop(str(number - 1), None)
        if board["status"] in ("draft", "failed") and self._missing(board) == []:
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
        (board.get("checks") or {}).pop(str(number - 1), None)
        if self._missing(board) == [] and board["status"] in ("draft", "failed"):
            board["status"] = "ready"
        return self.view(self.store.save(board))

    # ---- the film
    def _missing(self, board: dict) -> list[str]:
        modes = self.modes(board)
        return [str(i + 1) for i, panel in enumerate(board["panels"])
                if modes[i] != "none" and (not panel or not self.service.store.get_asset(panel))]

    def start_render(self, board_id: str, settings: RenderSettings | None = None, segment_seconds: float = 6.0,
                     mood: str = "", music: str = "") -> dict:
        board = self.board(board_id)
        if board["status"] in BUSY:
            raise StoryboardError(f"The board is {board['status']}; wait for it to finish.")
        missing = self._missing(board)
        if missing:
            raise StoryboardError(f"Shot{'s' if len(missing) > 1 else ''} {', '.join(missing)} ha{'ve' if len(missing) > 1 else 's'} "
                                  "no panel yet: draw the board, or redo those panels.")
        quality = board.get("quality") or "final"
        settings = settings or RenderSettings(continuity="off")
        update = {"aspect_ratio": board["aspect_ratio"]}
        if "megapixels" not in settings.model_fields_set:  # the quality preset decides unless a size was asked for
            update["megapixels"] = QUALITY.get(quality, 0.98)
        settings = settings.model_copy(update=update)
        board["render"] = {"jobs": [], "settings": settings.model_dump(), "segment_seconds": segment_seconds,
                           "mood": mood, "music": music, "film_asset_id": None, "drive": "",
                           "explicit_unet": bool(settings.unet_name)}
        board.update(status="rendering", error="")
        self.store.save(board)
        self._launch(board_id, self._render(board_id))
        return self.view(board)

    async def _models(self, board: dict) -> dict:
        try:
            files = await self.service.available_models("diffusion_models")
        except Exception:
            files = []
        return pick_models(files)

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
            base = RenderSettings(**render["settings"])
            music = board.get("music") or {}
            if music.get("prompt") and not music.get("asset_id"):  # one track for the whole film, made once
                track = await make_music(self.service, music["prompt"])
                board = self.board(board_id)
                board["music"]["asset_id"] = track["id"]
                board["cost_usd"] = round(board.get("cost_usd", 0) + MUSIC_COST, 4)
                self.store.save(board)
            music_over = bool((board.get("music") or {}).get("asset_id"))
            modes = self.modes(board)
            if "parts" not in render:
                models = {"frame": "", "action": ""} if render.get("explicit_unet") else await self._models(board)
                render["parts"] = plan_parts(modes, len(board["identity_asset_ids"]), models)
                board = self.board(board_id)
                board["render"]["parts"] = render["parts"]
                self.store.save(board)
            # resumable: parts already submitted keep their jobs
            for number, part in enumerate(render["parts"]):
                board = self.board(board_id)
                if number < len(board["render"]["jobs"]):
                    continue
                indices = part["shots"]
                motions = [(board["motions"][i] if i < len(board["motions"]) and board["motions"][i] else board["shots"][i])
                           for i in indices]
                lines = [(board.get("dialogue") or [])[i] if i < len(board.get("dialogue") or []) else [] for i in indices]
                settings = base.model_copy(update={"unet_name": part["unet"]}) if part.get("unet") else base
                job = await render_film(self.service, [board["panels"][i] for i in indices], board["identity_asset_ids"],
                                        motions, board["character"], settings, render["segment_seconds"],
                                        render["mood"], render["music"], [modes[i] for i in indices], lines, music_over)
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
        from .manhwa import concat_videos, film_to_drive, media_seconds, rendered_file

        board = self.board(board_id)
        jobs = board["render"]["jobs"]
        work = tempfile.mkdtemp(prefix="hawk_board_")
        note = ""
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
            track_id = (board.get("music") or {}).get("asset_id")
            track = self.service.store.get_asset(track_id) if track_id else None
            if track and shutil.which("ffmpeg"):
                music_path = os.path.join(work, "music" + (os.path.splitext(track.get("filename") or "")[1] or ".mp3"))
                with open(music_path, "wb") as handle:
                    handle.write(await self.service.asset_bytes(track))
                mixed = os.path.join(work, "film_music.mp4")
                seconds = await asyncio.to_thread(media_seconds, out) or 0.0
                if await asyncio.to_thread(lay_music, out, music_path, mixed, seconds):
                    out = mixed
                else:
                    note = "The music track could not be laid over the film; it has each part's own sound."
            slug = re.sub(r"[^0-9A-Za-z]+", "-", board["title"]).strip("-").lower() or "storyboard"
            name = f"{slug}_{len(board['shots'])}-shots.mp4"
            with open(out, "rb") as handle:
                asset = await self.service.add_asset(
                    name, handle, "video/mp4", os.path.getsize(out), collection=f"storyboard/{board['title']}",
                    tags=["storyboard", "film"], local_path=out, dedupe=False,
                    source={"type": "generated", "engine": "ffmpeg", "generator": "storyboard film",
                            "board": board_id, "jobs": jobs, "music": track_id})
            drive = await film_to_drive(self.service, out, name)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        board = self.board(board_id)
        board["render"].update(film_asset_id=asset["id"], drive=drive)
        board.update(status="done", error="", note=note or board.get("note", ""))
        self.store.save(board)
        for hook in self.done_hooks:
            try:
                result = hook(board)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # a hook failing (a story chapter gone) never undoes the film
                log.warning("storyboard %s: done hook failed: %s", board_id, exc)

    async def resume(self) -> list[str]:
        """After a restart: renders carry on (finished parts are reused); a drawing in flight is marked failed."""
        resumed = []
        for board in self.store.all():
            if board.get("status") in ("drawing", "checking"):
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
                        aspect_ratio: str = "9:16", models: str | None = None, character: str = "",
                        platform: str = "general") -> dict:
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
    ask = (f"Idea: {idea}\nShots: exactly {count}\nVideo shape: {aspect_ratio}\nPlatform: {platform or 'general'}\n"
           + (f"The character, as the user describes them -- these facts are fixed; keep them and add only what the "
              f"pictures show beyond them: {character.strip()}\n" if character.strip() else "")
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
        modes = [str(m).strip().lower() for m in data.get("modes") or []][:count]
        modes = [m if m in MODES else "frame" for m in modes] + ["frame"] * (count - len(modes))
        return {"character": str(data.get("character") or "").strip(), "style": str(data.get("style") or "").strip(),
                "shots": shots[:count], "motions": motions + [""] * (count - len(motions)), "modes": modes,
                "model": model}
    raise Unavailable("No model wrote the shots: " + (" | ".join(failures) or "none answered."))


# --------------------------------------------------------------------------- one call for a film

async def make_film(boards: "StoryboardManager", *, characters: list[str] | None = None,
                    identity_asset_ids: list[str] | None = None, character: str = "", idea: str = "",
                    shots: list[str] | None = None, motions: list[str] | None = None, modes: list[str] | None = None,
                    dialogue: list[list[str]] | None = None, shot_count: int = 8,
                    panel_asset_ids: list[str] | None = None, sheet_asset_id: str | None = None,
                    from_sheets: list[dict] | None = None, title: str = "", aspect_ratio: str = "9:16",
                    style: str = "", platform: str = "", quality: str = "final", review: str = "ask",
                    music_asset_id: str | None = None, music_prompt: str = "", segment_seconds: float = 6.0,
                    mood: str = "", settings: RenderSettings | None = None, link: dict | None = None) -> dict:
    """Who, what, where it goes -- and the engine does the rest: the shots are written when only an idea is given
    (with the people's facts fixed and a mode chosen per shot), drawn or taken from the user's images or sheets,
    checked, and filmed -- at once (review auto) or after approve (review ask)."""
    service = boards.service
    facts, identity, people_platform = character, list(identity_asset_ids or []), ""
    if characters:
        if boards.characters is None:
            raise StoryboardError("The character library isn't available on this server.")
        people = boards.characters.resolve(characters)
        facts = f"{people['character']} {character.strip()}".strip()
        identity = identity or people["identity_asset_ids"]
        people_platform = people["platform"]
    platform = platform or people_platform or "general"
    written = None
    if not shots and not panel_asset_ids and not sheet_asset_id and not from_sheets:
        if not idea.strip():
            raise StoryboardError("Give an idea (the shots are written for you), shots, your own images or a sheet.")
        written = await suggest_shots(service, idea, identity, shot_count, aspect_ratio, character=facts, platform=platform)
        shots, motions, modes = written["shots"], motions or written["motions"], modes or written["modes"]
        style = style or written["style"]
    follow = {"settings": settings.model_dump() if settings else None, "segment_seconds": segment_seconds,
              "mood": mood, "music": ""}
    board = boards.create(title=title or (idea.strip()[:80] if idea else "Film"),
                          character="" if characters else (facts or (written or {}).get("character", "")),
                          identity_asset_ids=identity if not characters else (identity_asset_ids or None),
                          characters=characters, shots=shots or [], motions=motions, modes=modes, dialogue=dialogue,
                          aspect_ratio=aspect_ratio, style=style, platform=platform, quality=quality, review=review,
                          music_asset_id=music_asset_id, music_prompt=music_prompt, then_render=follow,
                          panel_asset_ids=panel_asset_ids, sheet_asset_id=sheet_asset_id, from_sheets=from_sheets,
                          link=link)
    if written:
        board["written_by"] = written.get("model")
    return board
