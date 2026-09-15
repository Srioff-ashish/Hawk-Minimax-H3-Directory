"""Out-of-GPU-memory recovery for long renders. Pure Python: the ComfyUI hooks are
passed in, so tests/test_memory.py runs without torch.

ComfyUI keeps models cached on the GPU and evicts them only when its own estimate says
it must. When that estimate falls short mid-film (a continuity guide, a decode after a
large sampling pass), one allocation fails and the whole job with it. Unloading the
cached models and trying the step again recovers without reserving VRAM up front.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_oom_retry(
    fn: Callable[[], T],
    *,
    label: str,
    is_oom: Callable[[BaseException], bool],
    recover: Callable[[], None],
    log: Callable[[str], None],
) -> T:
    """Run ``fn``; on an out-of-memory error call ``recover`` and run it once more."""
    try:
        return fn()
    except Exception as exc:
        if not is_oom(exc):
            raise
        log(f"Out of GPU memory during {label}; unloading cached models and retrying once.")
    recover()
    try:
        return fn()
    except Exception as exc:
        if not is_oom(exc):
            raise
        raise RuntimeError(
            f"Out of GPU memory during {label} even after unloading cached models. Lower megapixels "
            f"or segment duration, use fewer LoRAs or ref_image_size=match, or pick a smaller base "
            f"model or text encoder. Finished segments are saved; retry the job to resume."
        ) from exc
