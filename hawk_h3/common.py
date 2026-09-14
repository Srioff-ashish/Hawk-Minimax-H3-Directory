"""Pieces shared by more than one node."""

from __future__ import annotations

from comfy_api.latest import io

CATEGORY = "Hawk/MiniMax H3"

#: Everything the Director needs from the loader: model, clip, both VAEs, signature.
H3Pipe = io.Custom("HAWK_H3_PIPE")

#: An ordered bundle of reference pictures, videos and audio (references.RefBundle).
H3Refs = io.Custom("HAWK_H3_REFS")

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "match first picture"]
