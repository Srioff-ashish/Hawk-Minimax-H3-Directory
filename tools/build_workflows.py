#!/usr/bin/env python3
"""Build the ready-to-load ComfyUI workflows in example_workflows/.

    python tools/build_workflows.py

Widget order for the Hawk nodes is read from their define_schema source (or, for
the LoRA Stack's generated slots, from the helper its schema is built with), so a
workflow can never silently fall out of step with a node: add, remove or rename a
widget and this script refuses to build until the workflow specs below match.
Pure Python -- no ComfyUI needed.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "example_workflows"
AUX_ID = "Srioff-ashish/Hawk-Minimax-H3-Directory"

sys.path.insert(0, str(ROOT))
from hawk_h3.lora_stack import NO_LORA, slot_widget_names  # noqa: E402
from hawk_h3.script import DEFAULT_POSE_INSTRUCTION  # noqa: E402

NODE_FILES = {
    "HawkH3ModelLoader": "loader.py",
    "HawkH3LoraStack": "lora_node.py",
    "HawkH3References": "references.py",
    "HawkH3StoryPlanner": "planner.py",
    "HawkH3Director": "director.py",
}

TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

_INPUT = re.compile(r'\b(H3Pipe|H3Refs|io\.\w+)\.Input\(\s*"(\w+)"')
_WIDGET_KINDS = {"io.String", "io.Int", "io.Float", "io.Boolean", "io.Combo"}
#: Inputs that only exist inside Autogrow templates.
_TEMPLATE_NAMES = {"picture", "video", "video_soundtrack", "audio"}


def widget_order(node_type: str) -> list[str]:
    """The order ComfyUI's frontend stores widgets_values in: required widgets in
    schema order, then optional ones; a seed's control_after_generate value
    follows the seed."""
    if node_type == "HawkH3LoraStack":
        required, optional = slot_widget_names()
        return required + optional

    source = (ROOT / "hawk_h3" / NODE_FILES[node_type]).read_text(encoding="utf-8")
    schema = source[source.index("def define_schema") : source.index("def execute")]
    matches = list(_INPUT.finditer(schema))
    required: list[str] = []
    optional: list[str] = []
    for index, match in enumerate(matches):
        kind, name = match.groups()
        if kind not in _WIDGET_KINDS or name in _TEMPLATE_NAMES:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(schema)
        chunk = schema[match.end() : end]
        names = [name] + (["control_after_generate"] if "control_after_generate" in chunk else [])
        (optional if "optional=True" in chunk else required).extend(names)
    return required + optional


# --------------------------------------------------------------------- builder


def sock(name: str, type_: str, *, optional: bool = False, label: str | None = None, widget: bool = False) -> dict:
    entry = {"localized_name": name, "name": name, "type": type_}
    if label:
        entry["label"] = label
    if optional:
        entry["shape"] = 7
    if widget:
        entry["widget"] = {"name": name}
    return entry


def out(name: str, type_: str) -> dict:
    return {"localized_name": name, "name": name, "type": type_}


class Graph:
    def __init__(self, name: str):
        self.name = name
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self.groups: list[dict] = []

    def add(self, type_, pos, size, *, inputs=(), outputs=(), widgets=None, title=None, core=True, color=None):
        node = {
            "id": len(self.nodes) + 1,
            "type": type_,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": len(self.nodes),
            "mode": 0,
            "inputs": [dict(entry, link=None) for entry in inputs],
            "outputs": [dict(entry, links=[]) for entry in outputs],
            "properties": (
                {"cnr_id": "comfy-core", "Node name for S&R": type_}
                if core
                else {"aux_id": AUX_ID, "Node name for S&R": type_}
            ),
        }
        if title:
            node["title"] = title
        if type_ in NODE_FILES:
            order = widget_order(type_)
            if set(widgets) != set(order):
                raise SystemExit(
                    f"{self.name}: {type_} widgets out of date.\n"
                    f"  missing from spec: {sorted(set(order) - set(widgets))}\n"
                    f"  unknown in spec:   {sorted(set(widgets) - set(order))}"
                )
            node["widgets_values"] = [widgets[key] for key in order]
            node["widgets_values_named"] = {key: widgets[key] for key in order}
        elif widgets is not None:
            node["widgets_values"] = list(widgets)
        if color:
            node["color"], node["bgcolor"] = color
        self.nodes.append(node)
        return node

    def link(self, src: dict, output: str, dst: dict, input_: str) -> None:
        o = next(i for i, e in enumerate(src["outputs"]) if e["name"] == output)
        i = next(i for i, e in enumerate(dst["inputs"]) if e["name"] == input_)
        link_id = len(self.links) + 1
        self.links.append([link_id, src["id"], o, dst["id"], i, src["outputs"][o]["type"]])
        src["outputs"][o]["links"].append(link_id)
        dst["inputs"][i]["link"] = link_id

    def group(self, title, x, y, w, h, color="#3f789e"):
        self.groups.append({"id": len(self.groups) + 1, "title": title, "bounding": [x, y, w, h], "color": color, "flags": {}})

    def to_json(self) -> dict:
        for node in self.nodes:
            for entry in node["outputs"]:
                entry["links"] = entry["links"] or None
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{AUX_ID}/{self.name}")),
            "revision": 0,
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": 0.7, "offset": [620, 80]}},
            "version": 0.4,
        }


# ------------------------------------------------------------------ node kinds


def model_loader(g: Graph, pos) -> dict:
    return g.add(
        "HawkH3ModelLoader", pos, [430, 470], core=False,
        outputs=[out("pipe", "HAWK_H3_PIPE"), out("model", "MODEL"), out("clip", "CLIP"), out("vae", "VAE"), out("audio_vae", "VAE")],
        widgets={
            "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
            "lora_stack": "",
            "shift_video": 12.0,
            "shift_audio": 3.0,
            "attention": "sol scheduled + sage",
            "weight_dtype": "default",
            "clip_device": "default",
            "sol_tau_start": 1.25,
            "sol_tau_end": 0.8,
        },
    )


def lora_stack(g: Graph, pos, loras=()) -> dict:
    required, optional = slot_widget_names()
    widgets = {name: (NO_LORA if name.startswith("lora_") else 1.0) for name in required + optional}
    for slot, (name, strength) in enumerate(loras, 1):
        widgets[f"lora_{slot}"] = name
        widgets[f"strength_{slot}"] = strength
    return g.add(
        "HawkH3LoraStack", pos, [410, 330], core=False,
        inputs=[sock("pipe", "HAWK_H3_PIPE")],
        outputs=[out("pipe", "HAWK_H3_PIPE"), out("model", "MODEL"), out("clip", "CLIP")],
        widgets=widgets,
    )


def models(g: Graph) -> dict:
    """Model Loader -> LoRA Stack (turbo LoRA in slot 1). Returns the node whose pipe feeds the Director."""
    loader = model_loader(g, (0, 0))
    loras = lora_stack(g, (460, 0), [(TURBO_LORA, 1.0)])
    g.link(loader, "pipe", loras, "pipe")
    g.group("1 · Models & LoRAs", -20, -60, 910, 560)
    return loras


def references(g: Graph, pos, *, pictures=0, poses=0, videos=0, soundtracks=0, audios=0, labels="", link_fps=False) -> dict:
    inputs = [sock("refs_in", "HAWK_H3_REFS", optional=True)]
    for group, prefix, count, limit, type_ in (
        ("pictures", "picture_", pictures, 9, "IMAGE"),
        ("poses", "pose_", poses, 9, "IMAGE"),
        ("videos", "video_", videos, 3, "IMAGE"),
        ("video_soundtracks", "video_soundtrack_", soundtracks, 3, "AUDIO"),
        ("audios", "audio_", audios, 3, "AUDIO"),
    ):
        # Connected slots plus the one empty slot the frontend keeps open.
        for index in range(min(count + 1, limit)):
            inputs.append(sock(f"{group}.{prefix}{index}", type_, optional=True, label=f"{prefix}{index}"))
    if link_fps:
        inputs.append(sock("video_fps", "FLOAT", optional=True, widget=True))
    return g.add(
        "HawkH3References", pos, [380, 380], core=False, inputs=inputs,
        outputs=[out("refs", "HAWK_H3_REFS"), out("tag_map", "STRING")],
        widgets={"labels": labels, "pose_instruction": DEFAULT_POSE_INSTRUCTION, "video_fps": 24.0},
    )


def story_planner(g: Graph, pos, *, story, segment_count, segment_seconds, aspect_ratio="16:9") -> dict:
    return g.add(
        "HawkH3StoryPlanner", pos, [440, 560], core=False,
        inputs=[sock("refs", "HAWK_H3_REFS", optional=True)],
        outputs=[out("script", "STRING"), out("raw_reply", "STRING")],
        widgets={
            "story": story,
            "segment_count": segment_count,
            "segment_seconds": segment_seconds,
            "aspect_ratio": aspect_ratio,
            "model": "xai/grok-4.3",
            "seed": 1,
            "control_after_generate": "fixed",
            "system_prompt": "",
            "temperature": 0.7,
            "max_tokens": 16384,
            "json_mode": True,
            "image_max_side": 1024,
            "api_url": "https://api.atlascloud.ai/v1",
            "api_key": "",
            "timeout": 240,
            "max_retries": 3,
        },
    )


def director(g: Graph, pos, *, script, run_name, link_script=False, megapixels=0.98, aspect_ratio="16:9",
             default_seconds=10.0, continuity="tail_22") -> dict:
    inputs = [sock("pipe", "HAWK_H3_PIPE"), sock("refs", "HAWK_H3_REFS", optional=True)]
    if link_script:
        inputs.append(sock("script", "STRING", widget=True))
    return g.add(
        "HawkH3Director", pos, [520, 900], core=False, inputs=inputs,
        outputs=[out("video", "VIDEO"), out("frames", "IMAGE"), out("audio", "AUDIO"), out("prompts", "STRING"), out("info", "STRING")],
        widgets={
            "script": "" if link_script else script,
            "aspect_ratio": aspect_ratio,
            "megapixels": megapixels,
            "default_seconds": default_seconds,
            "steps": 8,
            "sampler_name": "res_multistep",
            "scheduler": "simple",
            "seed": 0,
            "control_after_generate": "fixed",
            "continuity": continuity,
            "carry_audio": True,
            "ref_image_size": "match",
            "run_name": run_name,
            "resume": True,
            "seed_mode": "increment per segment",
            "audio_crossfade_ms": 60,
            "interpolation": "off",
            "encode_all_first": True,
            "output_frames": False,
            "music_volume_db": -3.0,
            "scene_volume_db": 0.0,
            "music_fade_seconds": 2.0,
            "mute_generated_music": True,
        },
    )


def load_image(g: Graph, pos, filename, title) -> dict:
    return g.add("LoadImage", pos, [300, 330], title=title, outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
                 widgets=[filename, "image"], color=("#223", "#335"))


def load_audio(g: Graph, pos, filename, title) -> dict:
    return g.add("LoadAudio", pos, [300, 140], title=title, outputs=[out("AUDIO", "AUDIO")],
                 widgets=[filename, None, None], color=("#223", "#335"))


def load_video(g: Graph, pos, filename, title) -> dict:
    return g.add("LoadVideo", pos, [300, 330], title=title, outputs=[out("VIDEO", "VIDEO")],
                 widgets=[filename, "image"], color=("#223", "#335"))


def video_components(g: Graph, pos) -> dict:
    return g.add(
        "GetVideoComponents", pos, [240, 150], inputs=[sock("video", "VIDEO")],
        outputs=[out("images", "IMAGE"), out("audio", "AUDIO"), out("fps", "FLOAT"), out("bit_depth", "COMBO"), out("color_space", "COMBO")],
    )


def save_video(g: Graph, pos, prefix) -> dict:
    return g.add("SaveVideo", pos, [620, 520], inputs=[sock("video", "VIDEO")], outputs=[out("video", "VIDEO")],
                 widgets=[prefix, "auto", "auto"])


def preview(g: Graph, pos, title) -> dict:
    return g.add("PreviewAny", pos, [620, 360], title=title, inputs=[sock("source", "*")],
                 outputs=[out("STRING", "STRING")], widgets=[None, None, None])


def note(g: Graph, pos, size, text, title="Read me") -> dict:
    return g.add("MarkdownNote", pos, size, title=title, widgets=[text], color=("#222", "#000"))


# ------------------------------------------------------------------- workflows

MODELS_NOTE = """
**Models** ([Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3))
- `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `vae/minimax_h3_video_vae_fp16.safetensors`
- `vae/minimax_h3_audio_vae_fp32.safetensors`
- `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`

**LoRAs:** pick up to 4 in **Hawk H3 LoRA Stack**. Need more? Add another LoRA Stack node between it and the Director (pipe → pipe). Without the turbo LoRA, raise the Director's `steps` to ~30.

No Sol / Sage installed? They are skipped automatically, or set `attention` to `comfy default`.

Docs: https://github.com/Srioff-ashish/Hawk-Minimax-H3-Directory/tree/main/docs
"""


def workflow_single_clip() -> Graph:
    g = Graph("01_single_clip")
    note(g, (-520, 0), [470, 780], f"""## 01 · Single clip

The simplest setup: one reference picture, one prompt, one H3 clip with native audio.

**To run**
1. Load your own picture in **Reference picture** (the bundled `example.png` is only a placeholder).
2. Edit the script in **Hawk H3 Director**. Keep `<Picture 1>` where you describe the person.
3. Queue. The video appears on the Director and in **Save Video**; files also go to `output/hawk_h3/example_single_clip/`.

Change `duration:` in the script (1–15 s) and `aspect_ratio` / `megapixels` on the Director.
{MODELS_NOTE}""")
    loras = models(g)
    image = load_image(g, (0, 540), "example.png", "Reference picture")
    refs = references(g, (340, 540), pictures=1, labels="Picture 1: the main character")
    direct = director(
        g, (940, 0), run_name="example_single_clip",
        script=(
            "duration: 6\n"
            "<Picture 1> defines the person's face and hair. They stand by a tall window in soft morning light, "
            "turn from the view toward camera and give a small, warm smile. Slow push-in from a medium shot to a close-up.\n"
            "Sound: quiet room tone, distant birdsong, fabric rustle as they turn. Music N/A."
        ),
    )
    save = save_video(g, (1520, 0), "video/hawk_h3_single_clip")
    info = preview(g, (1520, 560), "Render report")
    g.link(loras, "pipe", direct, "pipe")
    g.link(image, "IMAGE", refs, "pictures.picture_0")
    g.link(refs, "refs", direct, "refs")
    g.link(direct, "video", save, "video")
    g.link(direct, "info", info, "source")
    g.group("2 · References", -20, 520, 760, 400)
    g.group("3 · Direct & render", 920, -60, 1240, 1000)
    return g


FILM_SCRIPT = """style: Cinematic live-action, soft overcast light, 35mm lens feel, light film grain. Native audio. No subtitles, no on-screen text.
---
title: Arrival
duration: 8
pictures: 1, 2, 3
<Picture 1> defines her face and hair. <Picture 2> defines her coat: take its colour, fabric and cut, not the person wearing it. <Picture 3> defines the station platform.
She steps off a tram onto the rain-wet platform, adjusts her coat collar and looks up at the station clock. Slow push-in from a medium-wide shot.
Sound: tram brakes hiss, distant announcements, light rain on the canopy. Music N/A.
---
title: The call
duration: 10
pictures: 1, 2
audios: 1
<Picture 1> defines her face. <Picture 2> defines her coat. <Audio 1> is her voice: its timbre and calm pace.
She turns toward camera, takes out her phone and answers: "I'm here. Where are you?" The camera tracks backwards at walking pace in a medium close-up.
Sound: footsteps on wet tiles, her voice close and clear, rain continues.
---
title: The wave
duration: 8
pictures: 1
She lowers the phone, spots someone off-screen left and waves, breaking into a wide smile, and ends in the pose from <Pose 1>. The camera pans left to follow her gaze, then settles.
Sound: rain, a distant voice calling her name.
---
title: Cafe (hard cut)
duration: 10
continuity: off
pictures: 1, 2
audios: 1
Hard cut. <Picture 1> defines her face and hair. <Picture 2> defines her coat, now hanging on the chair behind her. <Audio 1> is her voice.
A warm cafe interior at night, rain streaking the window. She sits across from camera, wraps both hands around a cup and laughs: "You haven't changed at all."
Static medium shot, then a slow push-in. Sound: cafe murmur, cups clinking, rain on the glass."""

FILM_LABELS = (
    "Picture 1: her face and hair\nPicture 2: her coat\nPicture 3: the station platform\n"
    "Pose 1: one arm raised high in a wave\nAudio 1: her voice"
)


def film_references(g: Graph, x: int, y: int) -> dict:
    face = load_image(g, (x, y), "face.png", "Picture 1 · face")
    coat = load_image(g, (x, y + 370), "coat.png", "Picture 2 · coat")
    place = load_image(g, (x, y + 740), "platform.png", "Picture 3 · location")
    wave = load_image(g, (x, y + 1110), "pose_wave.png", "Pose 1 · waving pose")
    voice = load_audio(g, (x, y + 1480), "voice.mp3", "Audio 1 · voice sample")
    refs = references(g, (x + 340, y), pictures=3, poses=1, audios=1, labels=FILM_LABELS)
    g.link(face, "IMAGE", refs, "pictures.picture_0")
    g.link(coat, "IMAGE", refs, "pictures.picture_1")
    g.link(place, "IMAGE", refs, "pictures.picture_2")
    g.link(wave, "IMAGE", refs, "poses.pose_0")
    g.link(voice, "AUDIO", refs, "audios.audio_0")
    return refs


def workflow_multi_segment_film() -> Graph:
    g = Graph("02_multi_segment_film")
    note(g, (-520, 0), [470, 980], f"""## 02 · Multi-segment film (~34 s)

Four H3 segments joined into one video. Segments 1→2→3 flow continuously: each starts from the last ~1 s of frames and audio of the one before. Segment 4 is a hard cut (`continuity: off`).

**To run**
1. Load your references: **face**, **coat**, **location** pictures, a **waving pose** (an OpenPose skeleton or any photo of someone in that pose) and a clean **voice** sample (5–15 s, one speaker).
2. Queue. Segments are saved to `output/hawk_h3/example_film/` as they finish.
3. Edit a segment and queue again: earlier segments are reused; only the edited one and those after it re-render. Keep the Director's **seed on fixed**.

**Script tips**
- Numbers in `<Picture N>` follow the order on **Hawk H3 References** (see its tag map).
- `pictures: 1, 2` sends only those references in a segment: faster, less drift.
- Continuity segments should *continue* the action, not re-introduce the scene.
- `<Pose 1>` in segment 3 makes her end in the waving pose. A pose is only sent to segments that mention it, with an automatic "pose only, not identity or clothes" instruction.

**Try first at low cost:** set `megapixels` to 0.4 and `run_name` to `example_film_preview`.
{MODELS_NOTE}""")
    loras = models(g)
    refs = film_references(g, 0, 580)
    direct = director(g, (940, 0), run_name="example_film", script=FILM_SCRIPT)
    save = save_video(g, (1520, 0), "video/hawk_h3_film")
    prompts = preview(g, (1520, 560), "Encoded prompts (after tag renumbering)")
    info = preview(g, (1520, 960), "Render report")
    g.link(loras, "pipe", direct, "pipe")
    g.link(refs, "refs", direct, "refs")
    g.link(direct, "video", save, "video")
    g.link(direct, "prompts", prompts, "source")
    g.link(direct, "info", info, "source")
    g.group("2 · References", -20, 520, 780, 1700)
    g.group("3 · Direct & render", 920, -60, 1240, 1420)
    return g


def workflow_llm_story_planner() -> Graph:
    g = Graph("03_llm_story_planner")
    note(g, (-520, 0), [470, 1080], f"""## 03 · LLM story planner

An Atlas Cloud vision LLM reads your brief and references and writes the segment script; the Director renders it.

**Before starting ComfyUI** set your key: `export ATLAS_API_KEY=...` (leave the node's `api_key` blank, otherwise the key is saved into this workflow).

**To run**
1. Load your references (face, coat, location, waving pose, voice) and edit the **labels** on Hawk H3 References: the planner reads them and decides where each pose happens.
2. Write your brief in **Hawk H3 Story Planner → story**. Set `segment_count` and `segment_seconds`.
3. **Review first:** select the Director and press **Ctrl+B** (bypass), queue. Read the plan in **Plan preview**. Change the planner `seed` for a different plan.
4. Un-bypass the Director (Ctrl+B again) and queue. The plan is cached, so rendering starts straight away.

The Director is set to a **cheap preview** (`megapixels 0.4`, run `example_planned_film_preview`). When the film works, set `megapixels 0.98` and a new `run_name` for the final render.

**Hand-edit the plan:** copy the JSON from Plan preview into the Director's `script` box, disconnect the planner's `script` link, edit, queue.
{MODELS_NOTE}""")
    loras = models(g)
    refs = film_references(g, 0, 580)
    plan = story_planner(
        g, (820, 580),
        story=(
            "A woman arrives by tram in a rainy city to meet an old friend she has not seen in ten years. "
            "She calls from the platform, spots the friend across the station and waves (use the waving pose), "
            "and they end up laughing together "
            "in a warm cafe. Understated, warm, a little bittersweet. Her lines: \"I'm here. Where are you?\" "
            "and later \"You haven't changed at all.\""
        ),
        segment_count=4,
        segment_seconds=10.0,
    )
    plan_view = preview(g, (820, 1200), "Plan preview (script JSON)")
    direct = director(g, (1340, 0), run_name="example_planned_film_preview", script="", link_script=True, megapixels=0.4)
    save = save_video(g, (1920, 0), "video/hawk_h3_planned_film")
    info = preview(g, (1920, 560), "Render report")
    g.link(loras, "pipe", direct, "pipe")
    g.link(refs, "refs", plan, "refs")
    g.link(refs, "refs", direct, "refs")
    g.link(plan, "script", direct, "script")
    g.link(plan, "script", plan_view, "source")
    g.link(direct, "video", save, "video")
    g.link(direct, "info", info, "source")
    g.group("2 · References", -20, 520, 780, 1700)
    g.group("3 · Plan (Atlas LLM)", 800, 520, 660, 1100, color="#8a5a2b")
    g.group("4 · Direct & render", 1320, -60, 1240, 1000)
    return g


def workflow_video_reference() -> Graph:
    g = Graph("04_video_motion_and_voice")
    note(g, (-520, 0), [470, 940], f"""## 04 · Motion and voice from a video

Takes the **movement and camera** from a reference video and the **voice** from its soundtrack, and puts them on the person from a reference picture.

**To run**
1. **Reference video**: a 2–15 s clip with the movement you want and one clear speaker. Trim longer clips first.
2. **Reference picture**: the person who should appear.
3. Queue.

**How it is wired**
- Video frames → `videos` = `<Video 1>` (motion, timing, camera).
- Video audio → `audios` (not `video_soundtracks`) = `<Audio 1>`, a voice you can name in prompts.
- Video fps → `video_fps`, so any frame rate is resampled to H3's 24 fps.

Segment 1 uses only the motion (`audios: none`), segment 2 only the voice (`videos: none`). Always say what a video must **not** contribute (person, clothes, location).
{MODELS_NOTE}""")
    loras = models(g)
    clip = load_video(g, (0, 580), "reference.mp4", "Reference video")
    parts = video_components(g, (0, 950))
    face = load_image(g, (0, 1140), "face.png", "Reference picture")
    refs = references(
        g, (360, 580), pictures=1, videos=1, audios=1, link_fps=True,
        labels="Picture 1: the person to show\nVideo 1: the movement and camera move to copy\nAudio 1: the voice",
    )
    direct = director(
        g, (940, 0), run_name="example_motion_voice", default_seconds=8.0,
        script=(
            "style: Natural handheld documentary look, daylight. Native audio. No subtitles.\n"
            "---\n"
            "title: Motion\n"
            "duration: 8\n"
            "pictures: 1\n"
            "videos: 1\n"
            "audios: none\n"
            "<Picture 1> defines the person's face and hair. <Video 1> supplies ONLY the body movement, its timing and "
            "the camera move; do not take the person, clothes or location from it.\n"
            "They perform that movement on a sunny rooftop terrace with plants and a city skyline behind them, wearing a plain white t-shirt.\n"
            "Sound: light wind, footsteps on wooden decking, distant traffic. Music N/A.\n"
            "---\n"
            "title: Talk\n"
            "duration: 8\n"
            "pictures: 1\n"
            "videos: none\n"
            "audios: 1\n"
            "<Picture 1> defines their face. <Audio 1> is their voice: take its timbre, accent and pace.\n"
            "They finish the movement, catch their breath, look into the lens and say: \"Okay. Your turn.\"\n"
            "Medium close-up, handheld. Sound: breathing, wind, their voice close and clear."
        ),
    )
    save = save_video(g, (1520, 0), "video/hawk_h3_motion_voice")
    info = preview(g, (1520, 560), "Render report")
    g.link(loras, "pipe", direct, "pipe")
    g.link(clip, "VIDEO", parts, "video")
    g.link(parts, "images", refs, "videos.video_0")
    g.link(parts, "audio", refs, "audios.audio_0")
    g.link(parts, "fps", refs, "video_fps")
    g.link(face, "IMAGE", refs, "pictures.picture_0")
    g.link(refs, "refs", direct, "refs")
    g.link(direct, "video", save, "video")
    g.link(direct, "info", info, "source")
    g.group("2 · References", -20, 520, 800, 1000)
    g.group("3 · Direct & render", 920, -60, 1240, 1000)
    return g


POSE_SCRIPT = """style: Contemporary dance film, empty white studio, soft top light, slow elegant camera. Native audio. Music: sparse solo piano. No subtitles.
---
title: Rise
duration: 6
<Picture 1> defines the dancer's face, hair and black leotard.
She stands still in the centre of the studio, then slowly lifts both arms and ends in the pose from <Pose 1>. Slow push-in from a full shot.
Sound: soft footfalls, fabric movement, piano.
---
title: Lunge
duration: 6
<Picture 1> defines the dancer's face and leotard.
She flows down out of the stretch into the pose from <Pose 2> and holds it for a beat. The camera arcs slowly to her side.
Sound: a slow breath, bare feet sliding on the floor, piano.
---
title: Bow
duration: 6
<Picture 1> defines the dancer's face and leotard.
She rises out of the lunge, steps back and finishes in the bow from <Pose 3>. The camera settles into a static wide shot.
Sound: the piano holds a final chord, then silence."""


def workflow_pose_sequence() -> Graph:
    g = Graph("05_pose_guided_sequence")
    note(g, (-520, 0), [470, 1080], f"""## 05 · Pose-guided sequence

A dancer moves through three key poses across three continuous segments. Each segment ends in the pose it names, and the next segment starts from it.

**To run**
1. **Picture 1**: the dancer (face and outfit).
2. **Pose 1–3**: one image per key pose. Use OpenPose / DWPose skeleton renders, or photos of anyone in the pose: only the pose is used.
3. Queue.

**How poses work**
- Connect pose images to the **poses** input of Hawk H3 References; they are numbered `<Pose 1>`, `<Pose 2>`… separately from pictures.
- A segment sends only the poses its prompt mentions. Add `poses: 1, 2` to a segment to choose explicitly.
- Write *when* the pose happens: "…and ends in the pose from `<Pose 2>`".
- Each segment gets an automatic instruction to copy only the pose, not the face, clothes or style of the pose image. Edit it in `pose_instruction` (advanced) on the References node.
- Pictures + poses count toward H3's 9 images per segment.

**Tip:** land poses at the **end** of a segment. With continuity on, the next segment then starts exactly in that pose.
{MODELS_NOTE}""")
    loras = models(g)
    dancer = load_image(g, (0, 580), "dancer.png", "Picture 1 · dancer")
    poses = [load_image(g, (0, 950 + 370 * i), f"pose_{i + 1}.png", f"Pose {i + 1}") for i in range(3)]
    refs = references(
        g, (340, 580), pictures=1, poses=3,
        labels=(
            "Picture 1: the dancer's face, hair and black leotard\n"
            "Pose 1: standing, both arms stretched overhead\n"
            "Pose 2: deep lunge, one arm reaching forward\n"
            "Pose 3: a low bow, one hand on the chest"
        ),
    )
    direct = director(g, (940, 0), run_name="example_pose_sequence", default_seconds=6.0, script=POSE_SCRIPT)
    save = save_video(g, (1520, 0), "video/hawk_h3_pose_sequence")
    prompts = preview(g, (1520, 560), "Encoded prompts (poses become pictures)")
    info = preview(g, (1520, 960), "Render report")
    g.link(loras, "pipe", direct, "pipe")
    g.link(dancer, "IMAGE", refs, "pictures.picture_0")
    for index, pose in enumerate(poses):
        g.link(pose, "IMAGE", refs, f"poses.pose_{index}")
    g.link(refs, "refs", direct, "refs")
    g.link(direct, "video", save, "video")
    g.link(direct, "prompts", prompts, "source")
    g.link(direct, "info", info, "source")
    g.group("2 · References & poses", -20, 520, 780, 1560)
    g.group("3 · Direct & render", 920, -60, 1240, 1420)
    return g


WORKFLOWS = [
    workflow_single_clip,
    workflow_multi_segment_film,
    workflow_llm_story_planner,
    workflow_video_reference,
    workflow_pose_sequence,
]


def validate(workflow: dict, name: str) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    for link_id, src, src_slot, dst, dst_slot, type_ in workflow["links"]:
        output = nodes[src]["outputs"][src_slot]
        target = nodes[dst]["inputs"][dst_slot]
        assert link_id in (output["links"] or []), f"{name}: link {link_id} missing on output"
        assert target["link"] == link_id, f"{name}: link {link_id} missing on input"
        assert output["type"] == type_, f"{name}: link {link_id} type {type_} != {output['type']}"
        assert target["type"] in (type_, "*"), f"{name}: link {link_id} {type_} into {target['type']} ({target['name']})"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for build in WORKFLOWS:
        graph = build()
        data = graph.to_json()
        validate(data, graph.name)
        path = OUT / f"{graph.name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}: {len(data['nodes'])} nodes, {len(data['links'])} links")


if __name__ == "__main__":
    main()
