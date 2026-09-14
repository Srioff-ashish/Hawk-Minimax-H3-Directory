"""HawkH3References -- gather reference pictures, pose images, videos and audio into one bundle.

Numbering follows connection order and is what scripts use: ``<Picture 1>``,
``<Pose 1>``, ``<Video 1>``, ``<Audio 1>``. The Director renumbers per segment when
a segment only uses some of them.
"""

from __future__ import annotations

import re

import torch
from comfy_api.latest import io, ui

from .common import CATEGORY, H3Refs
from .media import audio_digest, tensor_digest
from .script import DEFAULT_POSE_INSTRUCTION, FPS, LIMITS


class RefBundle:
    """Ordered references. Treated as immutable once built: nodes chain by copying."""

    def __init__(self, pictures=None, videos=None, audios=None, labels=None, poses=None,
                 pose_instruction: str = DEFAULT_POSE_INSTRUCTION):
        self.pictures: list[torch.Tensor] = list(pictures or [])  # each [1, H, W, C]
        self.poses: list[torch.Tensor] = list(poses or [])  # each [1, H, W, C]; sent as pictures, pose only
        self.videos: list[dict] = list(videos or [])  # {"frames": [N, H, W, C], "audio": AUDIO | None}
        self.audios: list[dict] = list(audios or [])  # AUDIO dicts
        self.labels: dict[str, str] = dict(labels or {})  # "Picture 1" -> description
        self.pose_instruction = pose_instruction
        self._digests: dict[tuple[str, int], str] = {}

    def available(self) -> dict[str, int]:
        return {
            "Picture": len(self.pictures),
            "Pose": len(self.poses),
            "Video": len(self.videos),
            "Audio": len(self.audios),
        }

    def video_has_audio(self) -> list[bool]:
        return [video["audio"] is not None for video in self.videos]

    def digest(self, kind: str, number: int) -> str:
        key = (kind, number)
        if key not in self._digests:
            if kind == "Picture":
                value = tensor_digest(self.pictures[number - 1])
            elif kind == "Pose":
                value = tensor_digest(self.poses[number - 1])
            elif kind == "Video":
                video = self.videos[number - 1]
                value = f"{tensor_digest(video['frames'])}+{audio_digest(video['audio'])}"
            else:
                value = audio_digest(self.audios[number - 1])
            self._digests[key] = value
        return self._digests[key]

    def describe(self) -> str:
        lines = []
        for number, picture in enumerate(self.pictures, 1):
            lines.append(f"<Picture {number}> {picture.shape[2]}x{picture.shape[1]}{self._label('Picture', number)}")
        for number, pose in enumerate(self.poses, 1):
            lines.append(f"<Pose {number}> {pose.shape[2]}x{pose.shape[1]}, pose only{self._label('Pose', number)}")
        for number, video in enumerate(self.videos, 1):
            frames = video["frames"]
            sound = ", with soundtrack" if video["audio"] is not None else ""
            lines.append(
                f"<Video {number}> {frames.shape[2]}x{frames.shape[1]}, "
                f"{frames.shape[0] / FPS:.1f}s{sound}{self._label('Video', number)}"
            )
        for number, audio in enumerate(self.audios, 1):
            seconds = audio["waveform"].shape[-1] / int(audio["sample_rate"])
            lines.append(f"<Audio {number}> {seconds:.1f}s{self._label('Audio', number)}")
        return "\n".join(lines) or "(no references)"

    def _label(self, kind: str, number: int) -> str:
        label = self.labels.get(f"{kind} {number}")
        return f" -- {label}" if label else ""


_LABEL = re.compile(
    r"^\s*<?\s*(picture|image|pose|video|audio)\s*_?(\d{1,2})\s*>?\s*[:=\-]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_LABEL_KIND = {"picture": "Picture", "image": "Picture", "pose": "Pose", "video": "Video", "audio": "Audio"}


def parse_labels(text: str) -> dict[str, str]:
    labels = {}
    for line in (text or "").splitlines():
        match = _LABEL.match(line)
        if match:
            labels[f"{_LABEL_KIND[match.group(1).lower()]} {int(match.group(2))}"] = match.group(3)
    return labels


def _retime(frames: torch.Tensor, source_fps: float) -> torch.Tensor:
    """Nearest-frame resample to 24 fps; H3 reads reference video on a 24 fps clock."""
    if abs(source_fps - FPS) < 0.01 or frames.shape[0] < 2:
        return frames
    positions = torch.arange(0, frames.shape[0], source_fps / FPS).round().long()
    return frames[positions.clamp(max=frames.shape[0] - 1)]


def _add_images(target: list, images: dict | None) -> None:
    for image in (images or {}).values():
        if image is None:
            continue
        for index in range(image.shape[0]):
            target.append(image[index : index + 1])


class HawkH3References(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HawkH3References",
            display_name="Hawk H3 References",
            category=CATEGORY,
            description=(
                "Collect up to 9 pictures, 9 pose images, 3 videos (each with an optional soundtrack) "
                "and 3 audio clips. Refer to them in scripts as <Picture 1>, <Pose 1>, <Video 1>, "
                "<Audio 1> in connection order. Chain several of these nodes to add more."
            ),
            search_aliases=["minimax", "h3", "reference", "pose", "hawk"],
            inputs=[
                H3Refs.Input("refs_in", optional=True, tooltip="References from another Hawk H3 References node; new ones are numbered after these."),
                io.Autogrow.Input(
                    "pictures",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("picture", tooltip="Reference picture. A batch adds one picture per frame."),
                        prefix="picture_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "poses",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "pose",
                            tooltip=(
                                "Pose reference: an OpenPose/DWPose skeleton or a photo of someone in the pose. "
                                "Only the body pose and framing are used. A batch adds one pose per frame."
                            ),
                        ),
                        prefix="pose_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("video", tooltip="Reference video frames, 2-15s."),
                        prefix="video_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "video_soundtracks",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("video_soundtrack", tooltip="Soundtrack of the same-numbered video. For a voice you want as its own <Audio N>, use `audios` instead."),
                        prefix="video_soundtrack_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio", tooltip="Standalone reference audio (voice, music, sound)."),
                        prefix="audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.String.Input(
                    "labels",
                    multiline=True,
                    default="",
                    optional=True,
                    placeholder="Picture 1: the heroine's face\nPicture 2: red leather jacket\nPose 1: arms raised\nAudio 1: her voice",
                    tooltip="What each reference is for. Shown to the Story Planner LLM; not sent to H3.",
                ),
                io.String.Input(
                    "pose_instruction",
                    multiline=True,
                    default=DEFAULT_POSE_INSTRUCTION,
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Added to every segment that sends pose references, so H3 copies only the pose. "
                        "{tags} becomes those poses' picture tags. Empty adds nothing."
                    ),
                ),
                io.Float.Input(
                    "video_fps",
                    default=24.0,
                    min=1.0,
                    max=120.0,
                    step=0.001,
                    optional=True,
                    advanced=True,
                    tooltip="Frame rate of the connected videos; they are resampled to H3's 24 fps.",
                ),
            ],
            outputs=[
                H3Refs.Output("refs"),
                io.String.Output("tag_map", tooltip="Which tag means which reference."),
            ],
        )

    @classmethod
    def execute(
        cls,
        refs_in: RefBundle | None = None,
        pictures: dict | None = None,
        poses: dict | None = None,
        videos: dict | None = None,
        video_soundtracks: dict | None = None,
        audios: dict | None = None,
        labels: str = "",
        pose_instruction: str = DEFAULT_POSE_INSTRUCTION,
        video_fps: float = 24.0,
    ) -> io.NodeOutput:
        base = refs_in if refs_in is not None else RefBundle()
        bundle = RefBundle(
            base.pictures, base.videos, base.audios, base.labels, base.poses,
            pose_instruction=pose_instruction,
        )

        # Label numbering is global (after chained refs), so parse onto the merged bundle.
        bundle.labels.update(parse_labels(labels))

        _add_images(bundle.pictures, pictures)
        _add_images(bundle.poses, poses)

        soundtracks = video_soundtracks or {}
        for name, frames in (videos or {}).items():
            if frames is None:
                continue
            if frames.shape[0] < 5:
                raise ValueError(f"{name}: reference videos need at least 5 frames (~0.2s).")
            suffix = name.rsplit("_", 1)[-1]
            bundle.videos.append(
                {"frames": _retime(frames, float(video_fps)), "audio": soundtracks.get(f"video_soundtrack_{suffix}")}
            )

        for audio in (audios or {}).values():
            if audio is not None:
                bundle.audios.append(audio)

        for kind, count in bundle.available().items():
            if count > LIMITS[kind]:
                raise ValueError(
                    f"MiniMax H3 takes at most {LIMITS[kind]} {kind.lower()}s; {count} are connected "
                    f"(including chained references)."
                )

        tag_map = bundle.describe()
        return io.NodeOutput(bundle, tag_map, ui=ui.PreviewText(tag_map))


__all__ = ["HawkH3References", "RefBundle"]
