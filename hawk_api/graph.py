"""API-format ComfyUI graphs for the Hawk H3 pack. Pure Python.

The wiring mirrors tools/build_workflows.py. Two ComfyUI rules matter here and are
covered by tests_api/test_graph.py:

* Autogrow slots are flat dotted keys ("pictures.picture_0"); a wrong key is dropped
  silently by ComfyUI, not rejected.
* Links are ``[node_id, output_index]``; seeds are plain ints (control_after_generate
  exists only in UI workflows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hawk_h3.lora_stack import LORA_SLOTS, NO_LORA, slot_widget_names
from hawk_h3.script import DEFAULT_POSE_INSTRUCTION, LIMITS

ATLAS_CHAT_URL = "https://api.atlascloud.ai/v1"

#: role -> (Autogrow group, slot prefix, script tag kind, which loader output to use)
ROLES = {
    "picture": ("pictures", "picture_", "Picture", "image"),
    "pose": ("poses", "pose_", "Pose", "image"),
    "video": ("videos", "video_", "Video", "frames"),
    "audio": ("audios", "audio_", "Audio", "audio"),
    "video_soundtrack": ("video_soundtracks", "video_soundtrack_", None, "audio"),
}
#: role -> asset kinds that can fill it
ROLE_KINDS = {
    "picture": {"image"},
    "pose": {"image"},
    "video": {"video"},
    "audio": {"audio", "video"},
    "video_soundtrack": {"audio", "video"},
}


class GraphError(ValueError):
    """A request that cannot be wired, worded for the API caller."""


@dataclass
class Ref:
    asset_id: str
    kind: str  # image | audio | video
    path: str  # relative to ComfyUI's input folder, e.g. "hawk_api/<id>/face.png"
    role: str
    label: str = ""
    for_video: int | None = None


@dataclass
class RefWiring:
    node: str | None
    available: dict[str, int]
    video_has_audio: list[bool]


@dataclass
class RenderParams:
    run_name: str
    seed: int
    steps: int
    aspect_ratio: str = "16:9"
    megapixels: float = 0.98
    default_seconds: float = 10.0
    sampler_name: str = "res_multistep"
    scheduler: str = "simple"
    continuity: str = "tail_22"
    carry_audio: bool = True
    ref_image_size: str = "match"
    interpolation: str = "off"
    audio_crossfade_ms: int = 60


@dataclass
class BuiltGraph:
    prompt: dict
    nodes: dict = field(default_factory=dict)


class PromptGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add(self, class_type: str, inputs: dict, title: str | None = None) -> str:
        node_id = str(len(self.nodes) + 1)
        node = {"class_type": class_type, "inputs": inputs}
        if title:
            node["_meta"] = {"title": title}
        self.nodes[node_id] = node
        return node_id


def link(node_id: str, index: int = 0) -> list:
    return [node_id, index]


# -------------------------------------------------------------------- pieces


def add_references(g: PromptGraph, refs: list[Ref], pose_instruction: str = DEFAULT_POSE_INSTRUCTION) -> RefWiring:
    empty = {"Picture": 0, "Pose": 0, "Video": 0, "Audio": 0}
    if not refs:
        return RefWiring(None, empty, [])

    loaded: dict[str, dict] = {}

    def output(ref: Ref, want: str) -> list:
        if ref.asset_id not in loaded:
            if ref.kind == "image":
                loaded[ref.asset_id] = {"image": link(g.add("LoadImage", {"image": ref.path}, f"Asset {ref.asset_id}"))}
            elif ref.kind == "audio":
                loaded[ref.asset_id] = {"audio": link(g.add("LoadAudio", {"audio": ref.path}, f"Asset {ref.asset_id}"))}
            else:
                video = g.add("LoadVideo", {"file": ref.path}, f"Asset {ref.asset_id}")
                parts = g.add("GetVideoComponents", {"video": link(video)}, f"Asset {ref.asset_id} parts")
                loaded[ref.asset_id] = {"frames": link(parts, 0), "audio": link(parts, 1), "fps": link(parts, 2)}
        return loaded[ref.asset_id][want]

    inputs: dict = {}
    counts = {role: 0 for role in ROLES}
    labels: list[str] = []
    video_has_audio: list[bool] = []
    soundtracks: list[Ref] = []
    fps = None

    for ref in refs:
        if ref.role not in ROLES:
            raise GraphError(f"Unknown reference role {ref.role!r}; use one of {', '.join(ROLES)}.")
        if ref.kind not in ROLE_KINDS[ref.role]:
            raise GraphError(
                f"Asset {ref.asset_id} is {ref.kind}; role {ref.role!r} needs {' or '.join(sorted(ROLE_KINDS[ref.role]))}."
            )
        if ref.role == "video_soundtrack":
            soundtracks.append(ref)
            continue
        group, prefix, kind, want = ROLES[ref.role]
        index = counts[ref.role]
        if index >= LIMITS[kind]:
            raise GraphError(f"At most {LIMITS[kind]} {ref.role} references are allowed.")
        inputs[f"{group}.{prefix}{index}"] = output(ref, want)
        counts[ref.role] += 1
        if ref.role == "video":
            video_has_audio.append(False)
            fps = fps or output(ref, "fps")
        if ref.label:
            labels.append(f"{kind} {index + 1}: {ref.label}")

    for ref in soundtracks:
        number = ref.for_video
        if not number or not 1 <= number <= counts["video"]:
            raise GraphError(
                f"Soundtrack asset {ref.asset_id} needs for_video between 1 and {counts['video']} "
                f"(the video reference it belongs to)."
            )
        key = f"video_soundtracks.video_soundtrack_{number - 1}"
        if key in inputs:
            raise GraphError(f"Video {number} already has a soundtrack.")
        inputs[key] = output(ref, "audio")
        video_has_audio[number - 1] = True

    inputs["labels"] = "\n".join(labels)
    inputs["pose_instruction"] = pose_instruction
    inputs["video_fps"] = fps if fps is not None else 24.0
    node = g.add("HawkH3References", inputs, "References")
    return RefWiring(
        node,
        {"Picture": counts["picture"], "Pose": counts["pose"], "Video": counts["video"], "Audio": counts["audio"]},
        video_has_audio,
    )


def add_models(g: PromptGraph, models, loras: list) -> tuple[list, list[str]]:
    """Model Loader, then as many 4-slot LoRA Stack nodes as the LoRA list needs.
    Returns the pipe link for the Director and the LoRA Stack node ids."""
    loader = g.add(
        "HawkH3ModelLoader",
        {
            "unet_name": models.unet_name,
            "clip_name": models.clip_name,
            "video_vae": models.video_vae,
            "audio_vae": models.audio_vae,
            "lora_stack": "",
            "shift_video": models.shift_video,
            "shift_audio": models.shift_audio,
            "attention": models.attention,
            "weight_dtype": models.weight_dtype,
            "clip_device": models.clip_device,
            "sol_tau_start": 1.25,
            "sol_tau_end": 0.8,
        },
        "Model Loader",
    )
    pipe = link(loader)
    required, optional = slot_widget_names()
    stacks: list[str] = []
    for start in range(0, len(loras), LORA_SLOTS):
        inputs: dict = {"pipe": pipe}
        for name in required + optional:
            inputs[name] = NO_LORA if name.startswith("lora_") else 1.0
        for slot, lora in enumerate(loras[start : start + LORA_SLOTS], 1):
            inputs[f"lora_{slot}"] = lora.file
            inputs[f"strength_{slot}"] = lora.strength
        node = g.add("HawkH3LoraStack", inputs, f"LoRA Stack {len(stacks) + 1}")
        stacks.append(node)
        pipe = link(node)
    return pipe, stacks


def add_planner(
    g: PromptGraph,
    refs_node: str | None,
    *,
    story: str,
    segment_count: int,
    segment_seconds: float,
    aspect_ratio: str,
    model: str,
    seed: int,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> str:
    inputs = {
        "story": story,
        "segment_count": segment_count,
        "segment_seconds": segment_seconds,
        "aspect_ratio": aspect_ratio,
        "model": model,
        "seed": seed % 2147483648,
        "system_prompt": "",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": True,
        "image_max_side": 1024,
        # The node reads ATLAS_API_KEY from the ComfyUI process; never put the key in a graph.
        "api_url": ATLAS_CHAT_URL,
        "api_key": "",
        "timeout": 240,
        "max_retries": 3,
    }
    if refs_node:
        inputs["refs"] = link(refs_node)
    return g.add("HawkH3StoryPlanner", inputs, "Story Planner")


def add_director(g: PromptGraph, pipe: list, refs_node: str | None, script, params: RenderParams) -> str:
    inputs = {
        "pipe": pipe,
        "script": script,
        "aspect_ratio": params.aspect_ratio,
        "megapixels": params.megapixels,
        "default_seconds": params.default_seconds,
        "steps": params.steps,
        "sampler_name": params.sampler_name,
        "scheduler": params.scheduler,
        "seed": params.seed,
        "continuity": params.continuity,
        "carry_audio": params.carry_audio,
        "ref_image_size": params.ref_image_size,
        "run_name": params.run_name,
        "resume": True,
        "seed_mode": "increment per segment",
        "audio_crossfade_ms": params.audio_crossfade_ms,
        "interpolation": params.interpolation,
        "encode_all_first": True,
        "output_frames": False,
    }
    if refs_node:
        inputs["refs"] = link(refs_node)
    return g.add("HawkH3Director", inputs, "Director")


# ------------------------------------------------------------------ graphs


def plan_graph(refs: list[Ref], planner: dict, pose_instruction: str = DEFAULT_POSE_INSTRUCTION) -> tuple[BuiltGraph, RefWiring]:
    """References -> Story Planner -> Preview Any. No diffusion model is loaded."""
    g = PromptGraph()
    wiring = add_references(g, refs, pose_instruction)
    planner_node = add_planner(g, wiring.node, **planner)
    preview = g.add("PreviewAny", {"source": link(planner_node)}, "Plan")
    return BuiltGraph(g.nodes, {"planner": planner_node, "plan_preview": preview}), wiring


def render_graph(
    refs: list[Ref],
    models,
    loras: list,
    params: RenderParams,
    *,
    script: str | None = None,
    planner: dict | None = None,
    pose_instruction: str = DEFAULT_POSE_INSTRUCTION,
) -> tuple[BuiltGraph, RefWiring]:
    """A render from a finished script, or -- with ``planner`` -- plan and render in one run."""
    if (script is None) == (planner is None):
        raise GraphError("Give exactly one of a script or planner options.")
    g = PromptGraph()
    wiring = add_references(g, refs, pose_instruction)
    pipe, stacks = add_models(g, models, loras)
    nodes: dict = {"lora_stacks": stacks}
    if planner is not None:
        planner_node = add_planner(g, wiring.node, **planner)
        nodes["planner"] = planner_node
        nodes["plan_preview"] = g.add("PreviewAny", {"source": link(planner_node)}, "Plan")
        script_value = link(planner_node)
    else:
        script_value = script
    nodes["director"] = add_director(g, pipe, wiring.node, script_value, params)
    return BuiltGraph(g.nodes, nodes), wiring
