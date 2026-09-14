"""Tensor helpers: output resolution, frame and audio fitting, seam stitching, digests."""

from __future__ import annotations

import hashlib
import math

import torch

from .script import FPS

CANVAS_MULTIPLE = 32
#: The megapixel scale ComfyUI's Resolution Selector (and the stock template's size table) uses.
MEGAPIXEL = 1024 * 1024


def resolve_resolution(aspect: str, megapixels: float, first_picture: torch.Tensor | None = None) -> tuple[int, int]:
    """Width and height for an aspect ratio at a megapixel budget, snapped to 32.

    Matches the stock template's table: 0.98 MP at 16:9 is H3's native 1344x768.
    """
    if aspect == "match first picture":
        if first_picture is None:
            raise ValueError("aspect_ratio is 'match first picture' but no picture is connected.")
        ratio = first_picture.shape[2] / first_picture.shape[1]
    else:
        left, _, right = aspect.partition(":")
        ratio = float(left) / float(right)

    area = float(megapixels) * MEGAPIXEL
    width = math.sqrt(area * ratio)
    height = width / ratio

    def snap(value: float) -> int:
        return max(CANVAS_MULTIPLE, int(round(value / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)

    return snap(width), snap(height)


def audio_samples(frames: int, sample_rate: int, fps: float = FPS) -> int:
    return int(round(frames * sample_rate / fps))


def fit_frames(frames: torch.Tensor, count: int) -> torch.Tensor:
    """Trim, or pad by holding the last frame, to exactly ``count`` frames."""
    have = frames.shape[0]
    if have == count:
        return frames
    if have > count:
        return frames[:count]
    return torch.cat([frames, frames[-1:].expand(count - have, -1, -1, -1)], dim=0)


def fit_waveform(waveform: torch.Tensor, samples: int) -> torch.Tensor:
    """Trim or zero-pad a ``[B, C, L]`` waveform to ``samples``."""
    length = waveform.shape[-1]
    if length == samples:
        return waveform
    if length > samples:
        return waveform[..., :samples]
    return torch.nn.functional.pad(waveform, (0, samples - length))


def match_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    have = waveform.shape[1]
    if have == channels:
        return waveform
    if have == 1:
        return waveform.expand(-1, channels, -1).contiguous()
    if channels == 1:
        return waveform.mean(dim=1, keepdim=True)
    return waveform[:, :channels]


def stitch_audio(
    accumulated: torch.Tensor | None,
    segment: torch.Tensor,
    trim_samples: int,
    crossfade_samples: int,
) -> torch.Tensor:
    """Append ``segment`` minus its first ``trim_samples`` onto ``accumulated``.

    The trimmed head re-renders the tail of the previous segment (it is the
    continuity guide), so the samples just before the cut line up in time with
    the end of what we already have. They are crossfaded in rather than thrown
    away, which hides the seam. Output length is always
    ``len(accumulated) + len(segment) - trim_samples``, keeping audio locked to video.
    """
    if accumulated is None:
        return segment[..., trim_samples:]

    segment = match_channels(segment, accumulated.shape[1]).to(accumulated)
    fade = max(0, min(int(crossfade_samples), int(trim_samples), accumulated.shape[-1]))
    body = segment[..., trim_samples - fade :]
    if fade == 0:
        return torch.cat([accumulated, body], dim=-1)

    ramp = torch.linspace(0.0, 1.0, fade, dtype=accumulated.dtype, device=accumulated.device)
    blended = accumulated[..., -fade:] * (1.0 - ramp) + body[..., :fade] * ramp
    return torch.cat([accumulated[..., :-fade], blended, body[..., fade:]], dim=-1)


def to_uint8(frames: torch.Tensor) -> torch.Tensor:
    return (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()


def from_uint8(frames: torch.Tensor) -> torch.Tensor:
    return frames.to(torch.float32) / 255.0


def tensor_digest(tensor: torch.Tensor | None) -> str:
    """Cheap content fingerprint: shape plus a stride through the values.

    Enough to notice a swapped reference for the resume cache, without hashing
    several GB of reference video on every run.
    """
    if tensor is None:
        return "none"
    flat = tensor.detach().reshape(-1)
    step = max(1, flat.numel() // 65536)
    hasher = hashlib.sha1(str(tuple(tensor.shape)).encode("utf-8"))
    hasher.update(flat[::step].to(torch.float32).cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def audio_digest(audio: dict | None) -> str:
    if audio is None:
        return "none"
    return f"{tensor_digest(audio['waveform'])}@{int(audio['sample_rate'])}"
