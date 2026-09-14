"""Director scripts: parse a plan into segments, then resolve each into a render job.

A script is either JSON (what HawkH3StoryPlanner emits, or hand-written) or
plain-text blocks separated by a line of ``---``. Everything in this module is
pure Python so ``tests/test_script.py`` runs without ComfyUI or torch.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

FPS = 24
#: 362 frames -- the top of H3's trained range.
MAX_SECONDS = 15.1
MIN_FRAMES = 5
#: Per-kind caps. Pictures and poses both travel as H3 reference images, so a
#: segment may send at most MAX_IMAGES of them combined.
LIMITS = {"Picture": 9, "Pose": 9, "Video": 3, "Audio": 3}
MAX_IMAGES = 9

#: Appended to a segment that sends pose references; {tags} becomes their picture tags.
DEFAULT_POSE_INSTRUCTION = (
    "Pose reference {tags}: take only the body pose, limb and hand positions, head angle and "
    "framing. Do not take identity, face, hair, clothing, colours, lighting, style or background "
    "from any pose reference."
)

#: continuity mode -> trailing frames of the previous segment re-anchored at frame 0.
#: Multi-frame guides must sit on H3's 17k+5 grid, hence 5 / 22 / 39.
CONTINUITY_FRAMES = {"off": 0, "last_frame": 1, "tail_5": 5, "tail_22": 22, "tail_39": 39}
CONTINUITY_MODES = list(CONTINUITY_FRAMES)
MAX_TAIL_FRAMES = max(CONTINUITY_FRAMES.values())

_CONTINUITY_ALIASES = {
    "none": "off",
    "cut": "off",
    "hard_cut": "off",
    "hardcut": "off",
    "last": "last_frame",
    "frame": "last_frame",
    "tail": "tail_22",
    "5": "tail_5",
    "22": "tail_22",
    "39": "tail_39",
}


class ScriptError(ValueError):
    """A script problem, worded for the node's error popup."""


@dataclass
class Segment:
    prompt: str
    title: str = ""
    duration: float | None = None
    #: 1-based numbers of the connected references; None means all of them.
    pictures: list[int] | None = None
    videos: list[int] | None = None
    audios: list[int] | None = None
    #: None means the poses the prompt mentions.
    poses: list[int] | None = None
    seed: int | None = None
    #: None inherits the Director's continuity setting.
    continuity: str | None = None


@dataclass
class Script:
    segments: list[Segment]
    style: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {"style": self.style, "segments": [asdict(s) for s in self.segments]},
            indent=2,
            ensure_ascii=False,
        )


@dataclass
class Job:
    """One segment, fully resolved: exactly what gets encoded and sampled."""

    index: int
    title: str
    #: Final encoder text: style + prompt (+ pose instruction), tags renumbered for this segment.
    prompt: str
    seconds: float
    frames: int
    pictures: list[int]
    videos: list[int]
    audios: list[int]
    #: Sent as reference images after the pictures.
    poses: list[int]
    seed: int
    #: Frames of the previous segment anchored at frame 0 (0 for the first segment).
    tail_frames: int
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------- frames


def align_frame_count(n: int) -> int:
    """Snap up to H3's 17k+5 frame grid (124 = ~5s, 362 = ~15s at 24 fps)."""
    n = max(MIN_FRAMES, int(n))
    return n + (5 - n % 17) % 17


def frames_for_seconds(seconds: float) -> int:
    return align_frame_count(round(float(seconds) * FPS))


# ---------------------------------------------------------------------- parsing


def parse_script(text: str) -> Script:
    text = (text or "").strip()
    if not text:
        raise ScriptError(
            "The script is empty. Write one prompt per segment separated by a line "
            "of ---, or connect a Hawk H3 Story Planner."
        )
    data = _load_json(text)
    script = _from_json(data) if data is not None else _from_text(text)
    if not script.segments:
        raise ScriptError("The script has no segment with prompt text.")
    return script


_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def _load_json(text: str):
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if text[:1] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScriptError(f"The script looks like JSON but does not parse: {exc}") from None

    # An LLM reply may wrap the JSON in a sentence. Only accept an embedded object
    # that is clearly a script, so plain prompts containing braces stay text.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and ("segments" in data or "shots" in data) else None


def _from_json(data) -> Script:
    style = ""
    if isinstance(data, dict):
        style = str(data.get("style") or data.get("style_bible") or "").strip()
        items = data.get("segments", data.get("shots"))
        if items is None:
            raise ScriptError(
                'A JSON script needs a "segments" list, e.g. {"segments": [{"prompt": "..."}]}.'
            )
    elif isinstance(data, list):
        items = data
    else:
        raise ScriptError("A JSON script must be an object with a segments list, or a list.")
    if not isinstance(items, list):
        raise ScriptError('"segments" must be a list.')

    segments = []
    for number, item in enumerate(items, 1):
        where = f"Segment {number}"
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise ScriptError(f"{where} must be an object or a string.")
        prompt = str(item.get("prompt") or item.get("text") or "").strip()
        if not prompt:
            raise ScriptError(f"{where} has no prompt.")
        segments.append(
            Segment(
                prompt=prompt,
                title=str(item.get("title") or "").strip(),
                duration=_parse_seconds(item.get("duration", item.get("seconds")), where),
                pictures=_parse_indices(item.get("pictures", item.get("images")), "Picture", where),
                videos=_parse_indices(item.get("videos"), "Video", where),
                audios=_parse_indices(item.get("audios"), "Audio", where),
                poses=_parse_indices(item.get("poses"), "Pose", where),
                seed=_parse_seed(item.get("seed"), where),
                continuity=_parse_continuity(item.get("continuity"), where),
            )
        )
    return Script(segments, style)


_SEPARATOR = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)
_HEADER = re.compile(
    r"^\s*(title|duration|seconds|pictures|images|videos|audios|poses|seed|continuity|style)\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)


def _from_text(text: str) -> Script:
    style_parts: list[str] = []
    segments: list[Segment] = []

    for block in _SEPARATOR.split(text):
        lines = block.strip().splitlines()
        fields: dict[str, str] = {}
        index = 0
        while index < len(lines):
            match = _HEADER.match(lines[index])
            if not match:
                break
            key = match.group(1).lower()
            if key == "style":
                # A style header takes the rest of its block.
                style_parts.append("\n".join([match.group(2)] + lines[index + 1 :]).strip())
                index = len(lines)
                break
            fields[key] = match.group(2)
            index += 1

        prompt = "\n".join(lines[index:]).strip()
        if not prompt:
            continue

        where = f"Segment {len(segments) + 1}"
        segments.append(
            Segment(
                prompt=prompt,
                title=fields.get("title", "").strip(),
                duration=_parse_seconds(fields.get("duration", fields.get("seconds")), where),
                pictures=_parse_indices(fields.get("pictures", fields.get("images")), "Picture", where),
                videos=_parse_indices(fields.get("videos"), "Video", where),
                audios=_parse_indices(fields.get("audios"), "Audio", where),
                poses=_parse_indices(fields.get("poses"), "Pose", where),
                seed=_parse_seed(fields.get("seed"), where),
                continuity=_parse_continuity(fields.get("continuity"), where),
            )
        )

    return Script(segments, "\n\n".join(part for part in style_parts if part))


def _parse_seconds(value, where: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s*(seconds|second|secs|sec|s)$", "", text)
    try:
        seconds = float(text)
    except ValueError:
        raise ScriptError(f"{where}: duration {value!r} is not a number of seconds.") from None
    if seconds <= 0:
        raise ScriptError(f"{where}: duration must be above 0 seconds.")
    return seconds


def _parse_seed(value, where: str) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        seed = int(str(value).strip())
    except ValueError:
        raise ScriptError(f"{where}: seed {value!r} is not a whole number.") from None
    if seed < 0:
        raise ScriptError(f"{where}: seed must be 0 or more.")
    return seed


def _parse_continuity(value, where: str) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[\s\-]+", "_", str(value).strip().lower())
    if text in ("", "inherit", "default"):
        return None
    text = _CONTINUITY_ALIASES.get(text, text)
    if text not in CONTINUITY_FRAMES:
        raise ScriptError(
            f"{where}: continuity {value!r} is not one of {', '.join(CONTINUITY_MODES)}."
        )
    return text


def _parse_indices(value, kind: str, where: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScriptError(f"{where}: {kind.lower()}s must be numbers, 'all' or 'none'.")
    if isinstance(value, (int, float)):
        items: list = [value]
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "all", "*"):
            return None
        if text in ("none", "no", "-"):
            return []
        items = re.findall(r"\d+", text)
    elif isinstance(value, (list, tuple)):
        items = []
        for entry in value:
            items.extend(re.findall(r"\d+", entry) if isinstance(entry, str) else [entry])
    else:
        raise ScriptError(f"{where}: {kind.lower()}s must be a list of numbers.")

    try:
        indices = sorted({int(item) for item in items})
    except (TypeError, ValueError):
        raise ScriptError(f"{where}: {kind.lower()}s must be a list of numbers.") from None
    limit = LIMITS[kind]
    bad = [i for i in indices if i < 1 or i > limit]
    if bad:
        raise ScriptError(f"{where}: {kind.lower()} numbers run 1-{limit}; got {bad}.")
    return indices


# ------------------------------------------------------------------------- tags

_TAG = re.compile(
    r"<\s*(picture|image|pose|video|audio)\s*_?(\d{1,2})\s*>"  # <Picture 1>, <image_1>, <pose_1>
    r"|@(picture|image|pose|video|audio)\s*_?(\d{1,2})\b"  # @image1, @pose2
    r"|\b(picture|image|pose|video|audio)\s+(\d{1,2})\b",  # Image 1, Pose 2
    re.IGNORECASE,
)
_KIND = {"picture": "Picture", "image": "Picture", "pose": "Pose", "video": "Video", "audio": "Audio"}


def find_tags(text: str) -> list[tuple[str, int]]:
    tags = []
    for match in _TAG.finditer(text):
        word, number = [g for g in match.groups() if g is not None]
        tags.append((_KIND[word.lower()], int(number)))
    return tags


def remap_tags(
    text: str,
    mapping: dict[str, dict[int, tuple[str, int]]],
    available: dict[str, int],
    where: str,
) -> str:
    """Normalise every reference mention to the tag H3 expects and renumber it for
    this segment. Scripts number references as they are connected; each segment only
    sends the ones it uses, so H3 sees them renumbered from 1 -- and poses, which
    travel as extra pictures, become ``<Picture k>``."""

    def replace(match: re.Match) -> str:
        word, number = [g for g in match.groups() if g is not None]
        kind, number = _KIND[word.lower()], int(number)
        target = mapping[kind].get(number)
        if target is None:
            count = available.get(kind, 0)
            noun = kind.lower()
            if number < 1 or number > count:
                raise ScriptError(
                    f"{where} mentions <{kind} {number}> but only {count} {noun}(s) are connected."
                )
            raise ScriptError(
                f"{where} mentions <{kind} {number}> but its {noun}s list leaves it out. "
                f"Add {number} to the list or remove the mention."
            )
        return "<%s %d>" % target

    return _TAG.sub(replace, text)


def _join_tags(tags: list[str]) -> str:
    return tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]


# ------------------------------------------------------------------------- jobs


def build_jobs(
    script: Script,
    *,
    available: dict[str, int],
    video_has_audio: list[bool],
    default_seconds: float,
    continuity: str,
    base_seed: int,
    seed_mode: str = "increment",
    pose_instruction: str = DEFAULT_POSE_INSTRUCTION,
) -> list[Job]:
    """Validate the whole script up front, so a long run never dies at segment 7."""
    if continuity not in CONTINUITY_FRAMES:
        raise ScriptError(f"continuity must be one of {', '.join(CONTINUITY_MODES)}.")

    jobs: list[Job] = []
    for index, segment in enumerate(script.segments):
        where = f"Segment {index + 1}" + (f" ({segment.title})" if segment.title else "")
        warnings: list[str] = []

        style = script.style.strip()
        text = f"{style}\n\n{segment.prompt.strip()}" if style else segment.prompt.strip()

        selected: dict[str, list[int]] = {}
        for kind, chosen in (
            ("Picture", segment.pictures),
            ("Video", segment.videos),
            ("Audio", segment.audios),
            ("Pose", segment.poses),
        ):
            count = available.get(kind, 0)
            if chosen is None:
                if kind == "Pose":
                    # Poses are moment-specific: send only the ones this segment names.
                    chosen = sorted({n for k, n in find_tags(text) if k == "Pose" and 1 <= n <= count})
                else:
                    chosen = list(range(1, count + 1))
            chosen = list(chosen)
            missing = [n for n in chosen if n > count]
            if missing:
                raise ScriptError(
                    f"{where} asks for {kind.lower()} {missing} but only {count} "
                    f"{kind.lower()}(s) are connected."
                )
            selected[kind] = chosen

        images = len(selected["Picture"]) + len(selected["Pose"])
        if images > MAX_IMAGES:
            raise ScriptError(
                f"{where} sends {len(selected['Picture'])} picture(s) and {len(selected['Pose'])} "
                f"pose(s); H3 takes at most {MAX_IMAGES} images per segment. List fewer with "
                f"'pictures:' or 'poses:'."
            )

        # H3 numbers audio labels across both kinds: each selected video's
        # soundtrack takes an <Audio j> first, then the standalone clips follow.
        # Poses are sent as pictures after the regular ones.
        soundtracks = sum(
            1 for n in selected["Video"] if n <= len(video_has_audio) and video_has_audio[n - 1]
        )
        picture_count = len(selected["Picture"])
        mapping = {
            "Picture": {n: ("Picture", slot + 1) for slot, n in enumerate(selected["Picture"])},
            "Pose": {n: ("Picture", picture_count + slot + 1) for slot, n in enumerate(selected["Pose"])},
            "Video": {n: ("Video", slot + 1) for slot, n in enumerate(selected["Video"])},
            "Audio": {n: ("Audio", soundtracks + slot + 1) for slot, n in enumerate(selected["Audio"])},
        }
        prompt = remap_tags(text, mapping, available, where)
        if selected["Pose"] and pose_instruction.strip():
            tags = [f"<Picture {mapping['Pose'][n][1]}>" for n in selected["Pose"]]
            prompt = f"{prompt}\n\n{pose_instruction.strip().replace('{tags}', _join_tags(tags))}"

        seconds = segment.duration if segment.duration is not None else float(default_seconds)
        if seconds > MAX_SECONDS:
            warnings.append(f"{where}: {seconds:g}s is past H3's ~15s trained range; clamped to 15s.")
            seconds = 15.0
        frames = frames_for_seconds(seconds)

        mode = segment.continuity or continuity
        tail = CONTINUITY_FRAMES[mode] if index > 0 else 0
        if tail:
            limit = min(frames - 1, jobs[-1].frames)
            if tail > limit:
                smaller = sorted((n for n in CONTINUITY_FRAMES.values() if 0 < n <= limit), reverse=True)
                new_tail = smaller[0] if smaller else 0
                warnings.append(
                    f"{where}: a {tail}-frame continuity guide does not fit a {frames}-frame "
                    f"segment; using {new_tail}."
                )
                tail = new_tail

        seed = segment.seed if segment.seed is not None else base_seed + (index if seed_mode == "increment" else 0)

        files = sum(len(v) for v in selected.values()) + soundtracks
        if files > 12:
            warnings.append(f"{where}: {files} reference files; H3 is documented for at most 12.")
        if len(prompt) > 7000:
            warnings.append(f"{where}: prompt is {len(prompt)} characters; H3 is documented for 7000.")

        jobs.append(
            Job(
                index=index,
                title=segment.title,
                prompt=prompt,
                seconds=frames / FPS,
                frames=frames,
                pictures=selected["Picture"],
                videos=selected["Video"],
                audios=selected["Audio"],
                poses=selected["Pose"],
                seed=seed & 0xFFFFFFFFFFFFFFFF,
                tail_frames=tail,
                warnings=warnings,
            )
        )
    return jobs


def job_key(job: Job, context: dict, previous_key: str = "") -> str:
    """Hash of everything that shapes a segment. Chaining the previous key means a
    change to segment 2 also invalidates 3, 4... whose continuity guides it feeds."""
    payload = {k: v for k, v in asdict(job).items() if k != "warnings"}
    blob = json.dumps(
        {"job": payload, "context": context, "previous": previous_key},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
