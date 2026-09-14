"""HawkH3LoraStack -- pick LoRAs from dropdowns; chain the node for as many as you need.

Goes between Hawk H3 Model Loader and Hawk H3 Director. Changing a LoRA here
re-runs only this node, not the model load: ComfyUI applies LoRAs as weight
patches on the already-loaded model.
"""

from __future__ import annotations

import hashlib
import json

import folder_paths
from comfy_api.latest import io, ui

from .common import CATEGORY, H3Pipe
from .lora_stack import LORA_SLOTS, NO_LORA, apply_lora_stack, entries_from_slots


def _slot_inputs() -> list:
    try:
        options = [NO_LORA] + list(folder_paths.get_filename_list("loras"))
    except Exception:
        options = [NO_LORA]

    # Order must match lora_stack.slot_widget_names().
    required, optional = [], []
    for i in range(1, LORA_SLOTS + 1):
        required.append(
            io.Combo.Input(f"lora_{i}", options=options, default=NO_LORA,
                           tooltip=f"LoRA {i} from models/loras. None leaves the slot off.")
        )
        required.append(
            io.Float.Input(f"strength_{i}", default=1.0, min=-10.0, max=10.0, step=0.01,
                           tooltip=f"Strength of LoRA {i}. 0 turns it off.")
        )
    for i in range(1, LORA_SLOTS + 1):
        for key in ("video", "audio", "text"):
            optional.append(
                io.Float.Input(
                    f"{key}_{i}", default=1.0, min=-10.0, max=10.0, step=0.05, optional=True, advanced=True,
                    tooltip=(
                        f"Multiplier for LoRA {i} on H3's {key}-only projection layers. The shared "
                        f"transformer blocks always use the plain strength."
                    ),
                )
            )
    return required + optional


class HawkH3LoraStack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HawkH3LoraStack",
            display_name="Hawk H3 LoRA Stack",
            category=CATEGORY,
            description=(
                f"Adds up to {LORA_SLOTS} LoRAs to a Hawk H3 pipe, picked from dropdowns, with optional "
                f"video / audio / text multipliers. Chain several nodes for more LoRAs."
            ),
            search_aliases=["lora", "minimax", "h3", "hawk"],
            inputs=[H3Pipe.Input("pipe", tooltip="From Hawk H3 Model Loader or another LoRA Stack."), *_slot_inputs()],
            outputs=[
                H3Pipe.Output("pipe", tooltip="Connect to Hawk H3 Director, or to another LoRA Stack."),
                io.Model.Output("model"),
                io.Clip.Output("clip"),
            ],
        )

    @classmethod
    def execute(cls, pipe: dict, **slots) -> io.NodeOutput:
        entries = entries_from_slots(slots)
        if not entries:
            return io.NodeOutput(pipe, pipe["model"], pipe["clip"], ui=ui.PreviewText("no LoRAs selected"))

        model, clip = apply_lora_stack(pipe["model"], pipe["clip"], entries)
        # A new signature, so the Director's resume cache re-renders with the new LoRAs.
        signature = hashlib.sha256(
            json.dumps([pipe["signature"], [vars(e) for e in entries]], sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        lines = []
        for entry in entries:
            split = (entry.video, entry.audio, entry.text)
            extra = f"  (v={entry.video:g} a={entry.audio:g} t={entry.text:g})" if split != (1.0, 1.0, 1.0) else ""
            lines.append(f"{entry.name} @ {entry.strength:g}{extra}")

        return io.NodeOutput(
            {**pipe, "model": model, "clip": clip, "signature": signature},
            model,
            clip,
            ui=ui.PreviewText("\n".join(lines)),
        )


__all__ = ["HawkH3LoraStack"]
