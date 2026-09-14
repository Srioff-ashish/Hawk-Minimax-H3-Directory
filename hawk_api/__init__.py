"""Hawk H3 API gateway: REST + MCP in front of a private ComfyUI running the Hawk MiniMax H3 pack.

Not a ComfyUI node module -- ComfyUI only imports the pack's root ``__init__.py``,
which never touches this package. Run it with ``deploy/start_pod.sh`` or
``uvicorn hawk_api.app:app``.
"""

__version__ = "0.1.0"
