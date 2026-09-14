"""LoRA stack text -> applied LoRAs, with MiniMax H3's per-modality strength split.

The modality split is adapted from Plaguekind's LTX_lora_loader
(github.com/Plaguekind/ComfyUI-Plaguekind-Nodes). H3's transformer is a single
joint-attention stack, so a LoRA delta on ``blocks.*`` cannot be attributed to
one modality. Only these keys are modality-specific:

* video: ``video_patch_proj``, ``final_layer.video_out``
* audio: ``audio_patch_proj``, ``final_layer.audio_out``
* text:  ``condition_proj``, ``token_refiner.*``

Everything else is "joint" and follows the master strength only.

Parsing is pure Python (tests run it without ComfyUI); applying imports ComfyUI lazily.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class LoraEntry:
    name: str
    strength: float = 1.0
    video: float = 1.0
    audio: float = 1.0
    text: float = 1.0


#: Dropdown slots per Hawk H3 LoRA Stack node; chain nodes for more.
LORA_SLOTS = 4
NO_LORA = "None"


def slot_widget_names(slots: int = LORA_SLOTS) -> tuple[list[str], list[str]]:
    """Widget names of the LoRA Stack node: (required, optional) in schema order.
    The node schema and tools/build_workflows.py both use this, so they cannot disagree."""
    required = [name for i in range(1, slots + 1) for name in (f"lora_{i}", f"strength_{i}")]
    optional = [name for i in range(1, slots + 1) for name in (f"video_{i}", f"audio_{i}", f"text_{i}")]
    return required, optional


def entries_from_slots(values: dict, slots: int = LORA_SLOTS) -> list[LoraEntry]:
    """Selected slots, in slot order. 'None' or strength 0 means the slot is off."""
    entries = []
    for i in range(1, slots + 1):
        name = values.get(f"lora_{i}") or NO_LORA
        strength = float(values.get(f"strength_{i}", 1.0))
        if name == NO_LORA or strength == 0.0:
            continue
        entries.append(
            LoraEntry(
                name,
                strength,
                float(values.get(f"video_{i}", 1.0)),
                float(values.get(f"audio_{i}", 1.0)),
                float(values.get(f"text_{i}", 1.0)),
            )
        )
    return entries


_LINE = re.compile(r"^(?P<name>.+?\.(?:safetensors|pt|pth|ckpt|bin))\s*(?::\s*(?P<rest>.*))?$", re.IGNORECASE)


def parse_lora_stack(text: str) -> list[LoraEntry]:
    """Accept one LoRA per line::

        minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors : 1.0
        my_style.safetensors : 0.7 : v=1 a=0.5 t=1
        # disabled.safetensors : 1.0

    or the JSON ``stack_data`` that Plaguekind's loader saves, pasted as-is.
    """
    text = (text or "").strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"lora_stack looks like JSON but does not parse: {exc}") from None
        entries = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("on", True):
                continue
            name = str(row.get("lora") or "").strip()
            if name in ("", "None"):
                continue
            entries.append(
                LoraEntry(
                    name,
                    float(row.get("str", 1.0)),
                    float(row.get("v", 1.0)),
                    float(row.get("a", 1.0)),
                    float(row.get("t", 1.0)),
                )
            )
        return entries

    entries = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            raise ValueError(
                f"lora_stack line {number} ({line!r}) needs a LoRA file name, e.g. "
                f"'my_lora.safetensors : 0.8' or 'my_lora.safetensors : 0.8 : v=1 a=0.5 t=1'."
            )
        entry = LoraEntry(match.group("name").strip())
        for token in re.split(r"[\s:,]+", match.group("rest") or ""):
            if not token:
                continue
            key, _, value = token.partition("=")
            try:
                if not value:
                    entry.strength = float(key)
                elif key.lower() in ("v", "video"):
                    entry.video = float(value)
                elif key.lower() in ("a", "audio"):
                    entry.audio = float(value)
                elif key.lower() in ("t", "text"):
                    entry.text = float(value)
                else:
                    raise ValueError
            except ValueError:
                raise ValueError(f"lora_stack line {number}: cannot read {token!r}.") from None
        entries.append(entry)
    return entries


def modality(key: str) -> str:
    key = key.lower()
    if "video_patch_proj" in key or "final_layer.video_out" in key:
        return "video"
    if "audio_patch_proj" in key or "final_layer.audio_out" in key:
        return "audio"
    if "condition_proj" in key or "token_refiner" in key:
        return "text"
    return "joint"


def apply_lora_stack(model, clip, entries: list[LoraEntry]):
    if not entries:
        return model, clip

    import comfy.utils
    import folder_paths

    try:
        from comfy.sd import load_lora_for_models
    except ImportError:  # pragma: no cover - older layouts
        from comfy.lora import load_lora_for_models

    for entry in entries:
        path = folder_paths.get_full_path("loras", entry.name)
        if not path:
            raise FileNotFoundError(f"LoRA not found in models/loras: {entry.name}")
        weights = comfy.utils.load_torch_file(path, safe_load=True)

        buckets: dict[str, dict] = {"video": {}, "audio": {}, "text": {}, "joint": {}}
        for key, value in weights.items():
            buckets[modality(key)][key] = value
        multipliers = {"video": entry.video, "audio": entry.audio, "text": entry.text, "joint": 1.0}

        for name, bucket in buckets.items():
            strength = entry.strength * multipliers[name]
            if bucket and strength != 0.0:
                model, clip = load_lora_for_models(model, clip, bucket, strength, strength)
    return model, clip
