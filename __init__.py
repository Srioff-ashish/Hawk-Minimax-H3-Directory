"""Hawk MiniMax H3 Director -- the MiniMax H3 reference-to-video workflow as a
handful of nodes, plus multi-segment direction for long videos.

ComfyUI discovers custom nodes by importing this module and calling
``comfy_entrypoint()``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("HawkH3")

__version__ = "0.1.0"

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
    from .hawk_h3.planner import HawkH3StoryPlanner
    from .hawk_h3.references import HawkH3References

    NODES = [HawkH3ModelLoader, HawkH3References, HawkH3StoryPlanner, HawkH3Director]

    class HawkH3Extension(ComfyExtension):
        async def get_node_list(self) -> list:
            return NODES

    async def comfy_entrypoint() -> HawkH3Extension:
        return HawkH3Extension()

    __all__ = [
        "HawkH3ModelLoader",
        "HawkH3References",
        "HawkH3StoryPlanner",
        "HawkH3Director",
        "comfy_entrypoint",
    ]
