"""HawkH3ModelLoader -- the stock template's whole "Models" group in one node.

UNET -> MiniMax H3 sigma shift -> LoRA stack -> Sol attention -> Sage attention,
in the same order the reference workflow chains them, plus the text encoder and
both VAEs. The Director takes the resulting pipe; the raw MODEL / CLIP / VAE
outputs are there for anything else in the graph.
"""

from __future__ import annotations

import hashlib
import json
import logging

import folder_paths
import nodes
from comfy_api.latest import io, ui
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

from .common import CATEGORY, H3Pipe
from .lora_stack import apply_lora_stack, parse_lora_stack

logger = logging.getLogger("HawkH3")

NONE_FOUND = "(none found)"
ATTENTION_MODES = ["comfy default", "sage", "sol scheduled", "sol scheduled + sage"]
WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


def _choices(folder: str, preferred: str, *hints: str) -> tuple[list[str], str]:
    """Dropdown options plus the best default: the template's file, else a name
    containing every hint, else the first file."""
    try:
        options = list(folder_paths.get_filename_list(folder))
    except Exception:
        options = []
    if not options:
        return [NONE_FOUND], NONE_FOUND
    if preferred in options:
        return options, preferred
    for option in options:
        if all(hint in option.lower() for hint in hints):
            return options, option
    return options, options[0]


def apply_attention(model, mode: str, tau_start: float, tau_end: float):
    """Optional attention backends. Each one degrades to a logged no-op when its
    package is missing, so a shared workflow still runs on a plainer machine."""
    applied: list[str] = []

    if mode.startswith("sol"):
        patch_cls = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3ScheduledSolAttentionPatch")
        if patch_cls is None:
            logger.warning(
                "HawkH3: 'sol scheduled' needs ComfyUI-sol-attn "
                "(github.com/Saganaki22/ComfyUI-sol-attn); continuing without it."
            )
        else:
            try:
                # enabled, tau_start, tau_end, curve, min_tokens, strict, dense_percent,
                # thresh_type, int8_qk, int8_pv, sink_conditioning, dense_blocks
                model = patch_cls().patch(
                    model, True, float(tau_start), float(tau_end), "linear", 4096, False,
                    0.0, "diag", False, False, "exact_kv", "",
                )[0]
                applied.append(f"sol tau {tau_start:g}->{tau_end:g}")
            except Exception as exc:
                logger.warning("HawkH3: Sol attention patch failed (%s); continuing without it.", exc)

    if mode.endswith("sage"):
        from comfy.ldm.modules import attention as comfy_attention

        if not getattr(comfy_attention, "SAGE_ATTENTION_IS_AVAILABLE", False):
            logger.warning("HawkH3: 'sage' needs the sageattention package; continuing without it.")
        else:
            sage = getattr(comfy_attention.attention_sage, "__wrapped__", comfy_attention.attention_sage)

            def attention_override_sage(func, *args, **kwargs):
                return sage(*args, **kwargs)

            model = model.clone()
            options = dict(model.model_options.get("transformer_options", {}))
            options["optimized_attention_override"] = attention_override_sage
            model.model_options["transformer_options"] = options
            applied.append("sage")

    return model, applied


class HawkH3ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        unets, unet_default = _choices(
            "diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "minimax", "ref2va"
        )
        clips, clip_default = _choices(
            "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax_h3"
        )
        vaes, video_default = _choices("vae", "minimax_h3_video_vae_fp16.safetensors", "minimax", "video")
        _, audio_default = _choices("vae", "minimax_h3_audio_vae_fp32.safetensors", "minimax", "audio")

        return io.Schema(
            node_id="HawkH3ModelLoader",
            display_name="Hawk H3 Model Loader",
            category=CATEGORY,
            description=(
                "Loads the MiniMax H3 ref2va model, Qwen3-VL text encoder and both VAEs, "
                "then applies the sigma shift, a LoRA stack and optional Sol / Sage attention."
            ),
            search_aliases=["minimax", "h3", "loader", "hawk"],
            inputs=[
                io.Combo.Input("unet_name", options=unets, default=unet_default),
                io.Combo.Input("clip_name", options=clips, default=clip_default,
                               tooltip="Qwen3-VL 32B MiniMax H3 text encoder (loaded as CLIP type 'minimax')."),
                io.Combo.Input("video_vae", options=vaes, default=video_default),
                io.Combo.Input("audio_vae", options=vaes, default=audio_default),
                io.String.Input(
                    "lora_stack",
                    multiline=True,
                    default="",
                    placeholder=(
                        "one per line:  file.safetensors : strength [: v=1 a=1 t=1]\n"
                        "# lines starting with # are off. Plaguekind stack JSON also works."
                    ),
                    tooltip=(
                        "LoRAs applied in order. v/a/t scale only H3's video, audio and text "
                        "projection layers; the shared transformer blocks follow the main strength. "
                        "Prefer dropdowns? Leave this empty and add Hawk H3 LoRA Stack nodes after the loader."
                    ),
                ),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Combo.Input(
                    "attention",
                    options=ATTENTION_MODES,
                    default="sol scheduled + sage",
                    tooltip=(
                        "Sol needs ComfyUI-sol-attn + Triton; Sage needs sageattention. A "
                        "missing backend is skipped with a log line, never an error."
                    ),
                ),
                io.Combo.Input("weight_dtype", options=WEIGHT_DTYPES, default="default", advanced=True),
                io.Combo.Input("clip_device", options=["default", "cpu"], default="default", advanced=True),
                io.Float.Input("sol_tau_start", default=1.25, min=0.0, max=4.0, step=0.05, advanced=True,
                               tooltip="Sol routing threshold on the first, noisiest step. Higher is faster."),
                io.Float.Input("sol_tau_end", default=0.8, min=0.0, max=4.0, step=0.05, advanced=True,
                               tooltip="Sol routing threshold on the final detail steps. Lower is denser."),
            ],
            outputs=[
                H3Pipe.Output("pipe", tooltip="Connect to Hawk H3 Director."),
                io.Model.Output("model"),
                io.Clip.Output("clip"),
                io.Vae.Output("vae"),
                io.Vae.Output("audio_vae"),
            ],
        )

    @classmethod
    def execute(
        cls,
        unet_name: str,
        clip_name: str,
        video_vae: str,
        audio_vae: str,
        lora_stack: str,
        shift_video: float,
        shift_audio: float,
        attention: str,
        weight_dtype: str = "default",
        clip_device: str = "default",
        sol_tau_start: float = 1.25,
        sol_tau_end: float = 0.8,
    ) -> io.NodeOutput:
        for label, value, folder in (
            ("unet_name", unet_name, "diffusion_models"),
            ("clip_name", clip_name, "text_encoders"),
            ("video_vae", video_vae, "vae"),
            ("audio_vae", audio_vae, "vae"),
        ):
            if value == NONE_FOUND:
                raise FileNotFoundError(
                    f"{label}: no files in models/{folder}. Download the MiniMax H3 weights from "
                    f"https://huggingface.co/Comfy-Org/MiniMax-H3 and restart ComfyUI."
                )

        entries = parse_lora_stack(lora_stack)  # fail on a typo before loading 60 GB

        model = nodes.UNETLoader().load_unet(unet_name, weight_dtype)[0]
        clip = nodes.CLIPLoader().load_clip(clip_name, type="minimax", device=clip_device)[0]
        vae = nodes.VAELoader().load_vae(video_vae)[0]
        avae = nodes.VAELoader().load_vae(audio_vae)[0]

        model = MiniMaxH3SigmaShift.execute(model, shift_video, shift_audio).args[0]
        model, clip = apply_lora_stack(model, clip, entries)
        model, applied = apply_attention(model, attention, sol_tau_start, sol_tau_end)

        # Identifies this exact model setup inside the Director's resume cache.
        signature = hashlib.sha256(
            json.dumps(
                [unet_name, weight_dtype, clip_name, video_vae, audio_vae, shift_video, shift_audio,
                 [vars(e) for e in entries], applied],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]

        pipe = {"model": model, "clip": clip, "vae": vae, "audio_vae": avae, "signature": signature}
        loras = ", ".join(f"{e.name}@{e.strength:g}" for e in entries) or "none"
        summary = (
            f"model: {unet_name}\nclip: {clip_name}\nshift: video {shift_video:g} / audio {shift_audio:g}\n"
            f"loras: {loras}\nattention: {', '.join(applied) or 'comfy default'}"
        )
        return io.NodeOutput(pipe, model, clip, vae, avae, ui=ui.PreviewText(summary))


__all__ = ["HawkH3ModelLoader"]
