"""Hawk MiniMax H3 Director -- the MiniMax H3 reference-to-video workflow as a
handful of nodes, plus multi-segment direction for long videos.

ComfyUI discovers custom nodes by importing this module and calling
``comfy_entrypoint()``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("HawkH3")

__version__ = "0.1.0"


def _avoid_cudnn_attention() -> None:
    """On Blackwell GPUs the cuDNN attention backend has no execution plan for masked attention
    (Krea 2 Identity Edit's reference tokens) and fails with "cuDNN Frontend error: No valid execution
    plans built". ComfyUI's comfy.ops.scaled_dot_product_attention runs every call inside
    sdpa_kernel(SDPA_BACKEND_PRIORITY) with cuDNN in the list, which overrides the global switch, so
    masked calls go through a copy of that list without cuDNN. Unmasked calls (text to image, video)
    keep ComfyUI's order. The global switch still covers code that calls PyTorch directly.
    HAWK_CUDNN_SDP=1 leaves everything as it is."""
    import os

    if os.environ.get("HAWK_CUDNN_SDP", "").strip().lower() in ("1", "true", "on", "yes"):
        return
    try:
        import torch

        if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
            return
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        import comfy.ops
        from torch.nn.attention import SDPBackend, sdpa_kernel

        original = getattr(comfy.ops, "scaled_dot_product_attention", None)
        priority = getattr(comfy.ops, "SDPA_BACKEND_PRIORITY", None)
        if original is None or priority is None or getattr(original, "_hawk_masked", False):
            print("[Hawk H3] cuDNN attention off on this Blackwell GPU", flush=True)
            return
        safe = [backend for backend in priority if backend != SDPBackend.CUDNN_ATTENTION]
        repeat_kv = getattr(comfy.ops, "repeat_kv_for_gqa", None)

        def scaled_dot_product_attention(q, k, v, *args, **kwargs):
            mask = args[0] if args else kwargs.get("attn_mask")
            if mask is None:
                return original(q, k, v, *args, **kwargs)
            if kwargs.get("enable_gqa") and q.shape[-3] != k.shape[-3] and repeat_kv is not None:
                k, v = repeat_kv(k, v, q.shape[-3], -3)
                kwargs["enable_gqa"] = False
            with sdpa_kernel(safe, set_priority=True):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)

        scaled_dot_product_attention._hawk_masked = True
        comfy.ops.scaled_dot_product_attention = scaled_dot_product_attention
        print("[Hawk H3] masked attention skips cuDNN on this Blackwell GPU (HAWK_CUDNN_SDP=1 keeps it)", flush=True)
    except Exception as exc:  # pragma: no cover -- never block loading the nodes
        print(f"[Hawk H3] could not adjust cuDNN attention: {exc!r}", flush=True)


_avoid_cudnn_attention()

try:
    from comfy_api.latest import ComfyExtension
    from comfy_extras import nodes_minimax_h3  # noqa: F401 -- MiniMax H3 support check
except ImportError as exc:  # pragma: no cover
    ComfyExtension = None
    logger.error(
        "Hawk-Minimax-H3-Directory needs a ComfyUI build with MiniMax H3 support "
        "(comfy_extras.nodes_minimax_h3) and the V3 node API. Update ComfyUI and "
        "restart; no Hawk H3 nodes were loaded. (%s)",
        exc,
    )


if ComfyExtension is not None:
    from .hawk_h3.director import HawkH3Director
    from .hawk_h3.loader import HawkH3ModelLoader
    from .hawk_h3.lora_node import HawkH3LoraStack
    from .hawk_h3.planner import HawkH3StoryPlanner
    from .hawk_h3.references import HawkH3References

    NODES = [HawkH3ModelLoader, HawkH3LoraStack, HawkH3References, HawkH3StoryPlanner, HawkH3Director]

    class HawkH3Extension(ComfyExtension):
        async def get_node_list(self) -> list:
            return NODES

    async def comfy_entrypoint() -> HawkH3Extension:
        return HawkH3Extension()

    __all__ = [
        "HawkH3ModelLoader",
        "HawkH3LoraStack",
        "HawkH3References",
        "HawkH3StoryPlanner",
        "HawkH3Director",
        "comfy_entrypoint",
    ]
