"""HawkH3StoryPlanner -- an Atlas Cloud LLM turns a brief plus the references into a director script.

With ``segment_count = 1`` it is the stock template's LLM prompt refiner; with more
it plans a multi-segment film whose segments hand off to each other. The reply is
validated against the connected references before it leaves the node, so the
Director never starts a long render on a script that mentions a missing reference.
"""

from __future__ import annotations

import json
import logging
import os

from comfy_api.latest import io, ui

from .atlas import (
    DEFAULT_CHAT_URL,
    MAX_SEED,
    AtlasError,
    chat_completion,
    extract_message_text,
    frame_to_data_uri,
    resolve_api_key,
)
from .common import ASPECT_RATIOS, CATEGORY, H3Refs
from .references import RefBundle
from .script import (
    FPS,
    ScriptError,
    build_jobs,
    drop_unavailable_references,
    parse_script,
    reference_counts_line,
)

logger = logging.getLogger("HawkH3")

_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "planner_system.md")


def default_system_prompt() -> str:
    with open(_PROMPT_FILE, "r", encoding="utf-8") as handle:
        return handle.read()


def _video_sample_indices(frame_count: int, samples: int = 3) -> list[int]:
    if frame_count <= samples:
        return list(range(frame_count))
    return sorted({round(i * (frame_count - 1) / (samples - 1)) for i in range(samples)})


def build_request(
    story: str,
    refs: RefBundle,
    segment_count: int,
    segment_seconds: float,
    aspect_ratio: str,
    image_max_side: int,
) -> tuple[str, list[str]]:
    """The user message text plus the images attached to it, in the order the text describes."""
    images: list[str] = []
    ref_lines: list[str] = []

    for number, picture in enumerate(refs.pictures, 1):
        images.append(frame_to_data_uri(picture, image_max_side))
        label = refs.labels.get(f"Picture {number}", "")
        ref_lines.append(f"<Picture {number}> = attached image {len(images)}{' -- ' + label if label else ''}")

    for number, pose in enumerate(refs.poses, 1):
        images.append(frame_to_data_uri(pose, image_max_side))
        label = refs.labels.get(f"Pose {number}", "")
        ref_lines.append(
            f"<Pose {number}> = attached image {len(images)}, a POSE reference (body pose only)"
            f"{' -- ' + label if label else ''}"
        )

    for number, video in enumerate(refs.videos, 1):
        frames = video["frames"]
        first = len(images) + 1
        stamps = []
        for index in _video_sample_indices(frames.shape[0]):
            images.append(frame_to_data_uri(frames[index], image_max_side))
            stamps.append(f"{index / FPS:.1f}s")
        label = refs.labels.get(f"Video {number}", "")
        sound = ", has its own soundtrack" if video["audio"] is not None else ""
        ref_lines.append(
            f"<Video {number}> = {frames.shape[0] / FPS:.1f}s clip{sound}; attached images "
            f"{first}-{len(images)} are its frames at {', '.join(stamps)}{' -- ' + label if label else ''}"
        )

    for number, audio in enumerate(refs.audios, 1):
        seconds = audio["waveform"].shape[-1] / int(audio["sample_rate"])
        label = refs.labels.get(f"Audio {number}", "")
        ref_lines.append(f"<Audio {number}> = {seconds:.1f}s audio clip (not attached){' -- ' + label if label else ''}")

    count = (
        f"exactly {segment_count} segment(s)"
        if segment_count > 0
        else "as many segments as the story needs (usually 2-8)"
    )
    text = "\n".join(
        [
            "BRIEF:",
            story.strip(),
            "",
            "REFERENCES (global numbering):",
            *(ref_lines or ["(none -- this is a text-only film)"]),
            "",
            "CONSTRAINTS:",
            f"- Write {count}.",
            f"- Target about {segment_seconds:g} seconds per segment (each 5-15s).",
            f"- Frame: {aspect_ratio}.",
            f"- {reference_counts_line(refs.available())}",
            "- Return only the JSON object described in your instructions.",
        ]
    )
    return text, images


class HawkH3StoryPlanner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HawkH3StoryPlanner",
            display_name="Hawk H3 Story Planner (Atlas LLM)",
            category=CATEGORY,
            description=(
                "Plans a MiniMax H3 film: a vision LLM on Atlas Cloud reads the brief and the "
                "references and writes a validated segment script for Hawk H3 Director. "
                "segment_count 1 makes it a single-prompt refiner."
            ),
            search_aliases=["minimax", "h3", "llm", "planner", "prompt", "hawk", "atlas"],
            inputs=[
                io.String.Input(
                    "story",
                    multiline=True,
                    default="",
                    placeholder="What happens, who is in it, the mood, any lines of dialogue...",
                ),
                H3Refs.Input("refs", optional=True),
                io.Int.Input("segment_count", default=3, min=0, max=40,
                             tooltip="How many segments to write. 0 lets the model decide."),
                io.Float.Input("segment_seconds", default=10.0, min=1.0, max=15.0, step=0.5),
                io.Combo.Input("aspect_ratio", options=ASPECT_RATIOS, default="16:9",
                               tooltip="Told to the planner for framing; set the Director's aspect_ratio to match."),
                io.String.Input("model", default="xai/grok-4.3",
                                tooltip="Atlas chat model id. Use a vision-capable model when references are connected."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED, control_after_generate=True,
                             tooltip="Change (or randomize) for a fresh plan; ComfyUI reuses the cached plan otherwise."),
                io.String.Input("system_prompt", multiline=True, default="", optional=True, advanced=True,
                                placeholder="blank = built-in H3 planner instructions (hawk_h3/prompts/planner_system.md)"),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, optional=True, advanced=True),
                io.Int.Input("max_tokens", default=8192, min=256, max=200000, step=256, optional=True, advanced=True),
                io.Boolean.Input("json_mode", default=True, optional=True, advanced=True,
                                 tooltip="Ask Atlas for a JSON object. Turn off for models that reject response_format."),
                io.Int.Input("image_max_side", default=1024, min=256, max=4096, step=64, optional=True, advanced=True),
                io.String.Input("api_url", default=DEFAULT_CHAT_URL, optional=True, advanced=True),
                io.String.Input(
                    "api_key",
                    default="",
                    optional=True,
                    advanced=True,
                    placeholder="blank = use the ATLAS_API_KEY environment variable",
                    tooltip="WARNING: a key typed here is saved into workflow JSON and PNG metadata.",
                ),
                io.Int.Input("timeout", default=240, min=10, max=3600, optional=True, advanced=True),
                io.Int.Input("max_retries", default=3, min=0, max=10, optional=True, advanced=True),
            ],
            outputs=[
                io.String.Output("script", tooltip="Validated JSON script. Connect to Hawk H3 Director's script input."),
                io.String.Output("raw_reply", tooltip="The model's reply as returned, for debugging."),
            ],
        )

    @classmethod
    async def execute(
        cls,
        story: str,
        segment_count: int,
        segment_seconds: float,
        aspect_ratio: str,
        model: str,
        seed: int = 0,
        refs: RefBundle | None = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        json_mode: bool = True,
        image_max_side: int = 1024,
        api_url: str = DEFAULT_CHAT_URL,
        api_key: str = "",
        timeout: int = 240,
        max_retries: int = 3,
    ) -> io.NodeOutput:
        if not story.strip():
            raise ValueError("story is empty. Describe the film you want planned.")
        if not model.strip():
            raise ValueError("model is empty. Enter an Atlas chat model id, e.g. xai/grok-4.3.")
        bundle = refs if refs is not None else RefBundle()
        key = resolve_api_key(api_key)

        text, images = build_request(story, bundle, segment_count, segment_seconds, aspect_ratio, image_max_side)
        content = text if not images else [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": uri}} for uri in images
        ]
        payload: dict = {
            "model": model.strip(),
            "messages": [
                {"role": "system", "content": system_prompt.strip() or default_system_prompt()},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed > 0:
            payload["seed"] = seed % (MAX_SEED + 1)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await chat_completion(api_url, key, payload, timeout=timeout, max_retries=max_retries)
        reply = extract_message_text(response)

        try:
            script, fixes = drop_unavailable_references(parse_script(reply), bundle.available())
            jobs = build_jobs(
                script,
                available=bundle.available(),
                video_has_audio=bundle.video_has_audio(),
                default_seconds=segment_seconds,
                continuity="tail_22",
                pose_instruction=bundle.pose_instruction,
                base_seed=0,
            )
        except ScriptError as exc:
            raise AtlasError(
                f"The planner's reply is not a usable script: {exc}\n\n"
                f"Try again with a new seed, a stronger model, or json_mode toggled.\n\nReply:\n{reply[:2000]}"
            ) from None

        for fix in fixes:
            logger.warning("HawkH3 planner: %s", fix)
        script_json = script.to_json(warnings=fixes)
        total = sum(job.seconds for job in jobs)
        warnings = fixes + [w for job in jobs for w in job.warnings]
        preview = f"{len(jobs)} segments, ~{total:.0f}s\n\n" + "\n\n".join(
            f"[{job.index + 1}] {script.segments[job.index].title or 'untitled'} ({job.seconds:.1f}s)\n"
            f"{script.segments[job.index].prompt}"
            for job in jobs
        )
        if warnings:
            preview = "⚠ " + "\n⚠ ".join(warnings) + "\n\n" + preview
        return io.NodeOutput(script_json, reply, ui=ui.PreviewText(preview))


__all__ = ["HawkH3StoryPlanner"]
