"""Storyboard sheets: many shots from one cheap image, each panel the first frame of a video segment.

One Seedream Flash image ($0.018, any size) holds a grid of up to eight panels of the same character in one style,
which costs an eighth of eight separate images and keeps the face, body and light consistent across the shots
because they are one generation. The sheet is cut into its panels, and each panel opens its segment of the film
(H3's keyframe completion), with the character's own identity picture beside it in every segment.

Eight panels at most: with the identity picture that is nine, a render's picture limit.
"""

from __future__ import annotations

import io
import math

from .schemas import PlannerOptions, ReferenceIn, RenderSettings, VideoRequest

SHEET_ENGINE = "seedream-flash"
MAX_PANELS = 8
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


class StoryboardError(ValueError):
    """A sheet that can't be planned or cut as asked."""


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


def film_brief(beats: list[str], character: str, mood: str = "", music: str = "") -> str:
    """The planner's brief: every segment opens on its panel and keeps the identity picture beside it."""
    panels = len(beats)
    identity = panels + 1
    return (
        f"{character.strip()} <Picture {identity}> is the identity reference (face and body) for every segment. "
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


async def render_film(service, panel_asset_ids: list[str], identity_asset_id: str, beats: list[str], character: str,
                      settings: RenderSettings | None = None, segment_seconds: float = 6.0, mood: str = "",
                      music: str = "") -> dict:
    """A film of the panels in order, planned in the H3 reference format, each segment opening on its panel."""
    if not panel_asset_ids:
        raise StoryboardError("No panels to animate.")
    if len(panel_asset_ids) > MAX_PANELS:
        raise StoryboardError(f"At most {MAX_PANELS} panels per film (plus the identity picture is a render's 9).")
    if len(beats) != len(panel_asset_ids):
        raise StoryboardError(f"{len(panel_asset_ids)} panels need {len(panel_asset_ids)} beats, one each; got {len(beats)}.")
    settings = settings or RenderSettings(aspect_ratio="9:16", continuity="off")
    references = [ReferenceIn(asset_id=asset_id, role="picture", label=f"storyboard frame {number}")
                  for number, asset_id in enumerate(panel_asset_ids, 1)]
    references.append(ReferenceIn(asset_id=identity_asset_id, role="picture", label="identity (face and body)"))
    request = VideoRequest(
        references=references,
        story=PlannerOptions(story=film_brief(beats, character, mood, music), segment_count=len(beats),
                             segment_seconds=segment_seconds, aspect_ratio=settings.aspect_ratio),
        settings=settings,
    )
    return await service.create_video(request)
