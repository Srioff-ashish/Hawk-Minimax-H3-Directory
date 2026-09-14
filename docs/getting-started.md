# Getting started

This page takes you from nothing to a rendered clip, then to a three-segment film.

## 1. Requirements

- **ComfyUI with MiniMax H3 support** ([ComfyUI#15224](https://github.com/Comfy-Org/ComfyUI/pull/15224) or any later release). If your ComfyUI is older, update it first. The pack logs `needs a ComfyUI build with MiniMax H3 support` and loads no nodes otherwise.
- **An NVIDIA GPU with enough VRAM for H3.** The ref2va model is int8 and the text encoder is a 32B Qwen3-VL. ComfyUI offloads between them, but plan for a high-VRAM card and plenty of system RAM.
- **Python packages:** nothing extra. Everything the pack imports ships with ComfyUI.

## 2. Install the pack

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Srioff-ashish/Hawk-Minimax-H3-Directory.git
```

Restart ComfyUI. In the node search, type **hawk h3**. You should see four nodes under **Hawk / MiniMax H3**:

- Hawk H3 Model Loader
- Hawk H3 References
- Hawk H3 Story Planner (Atlas LLM)
- Hawk H3 Director

To update later: `cd ComfyUI/custom_nodes/Hawk-Minimax-H3-Directory && git pull`, then restart.

## 3. Download the models

From [huggingface.co/Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| File | Put it in |
|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` *(or the `int8_convrot` variant)* | `ComfyUI/models/text_encoders/` |
| `minimax_h3_video_vae_fp16.safetensors` | `ComfyUI/models/vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae/` |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` *(recommended)* | `ComfyUI/models/loras/` |

The turbo LoRA is what makes the default **8 steps** look good. Without it, use about 30 steps.

Restart ComfyUI (or press **R** to refresh) after adding files so the dropdowns see them.

## 4. Optional extras

None of these are required. When one is missing, the matching setting is skipped with a line in the console, not an error. The exception is interpolation, which stops the run up front.

| Install | Unlocks | Why you'd want it |
|---|---|---|
| `pip install sageattention` (in ComfyUI's Python) | Model Loader → `attention: sage` | Faster sampling |
| [ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) + Triton | Model Loader → `attention: sol scheduled` | Faster still on long / high-res segments |
| [ComfyUI-VFI](https://github.com/GACLove/ComfyUI-VFI) | Director → `interpolation: 48 / 60 fps (RIFE)` | Smoother playback |
| [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) | **Load Video UI** node | Trim a reference video visually and get its audio |
| [comfyui-deno-custom-nodes](https://github.com/Deno2026/comfyui-deno-custom-nodes) | **Deno Multi Image Loader** | Load many reference pictures in one node |
| [HawkNodes](https://github.com/Srioff-ashish/HawkNodes) | **Hawk Atlas I2I** | Prepare references, e.g. put a character in a specific outfit |

## 5. Set your Atlas key (only for the Story Planner)

The Story Planner calls an LLM on [Atlas Cloud](https://atlascloud.ai). Set the key as an environment variable **before** starting ComfyUI:

```bash
# macOS / Linux
export ATLAS_API_KEY="your-key"
python main.py

# Windows (PowerShell)
$env:ATLAS_API_KEY = "your-key"
python main.py
```

Leave the node's `api_key` field blank. ComfyUI saves every widget value into workflow files and PNG metadata, so a key typed into the node gets shared along with your workflows.

You don't need a key if you write scripts yourself.

---

## Example workflows

The quickest start is a ready-made graph. Drag a file from [`example_workflows/`](../example_workflows) onto the ComfyUI canvas, or open **Workflow → Browse Templates** and pick it under *Hawk-Minimax-H3-Directory*.

| File | Use it to |
|---|---|
| `01_single_clip.json` | Render one clip from one picture (same as step 6 below) |
| `02_multi_segment_film.json` | Render a four-segment film with continuous joins and a hard cut (same as step 7) |
| `03_llm_story_planner.json` | Have the Atlas LLM write the script, review it, then render (same as step 8) |
| `04_video_motion_and_voice.json` | Copy movement from a reference video and use its voice |

After loading:
1. Pick your own files in every **Load Image / Load Audio / Load Video** node. The names in them (`face.png`, `voice.mp3`, `reference.mp4`) are placeholders and show as missing until you replace them.
2. Check the four model dropdowns on **Hawk H3 Model Loader**.
3. Read the *Read me* note on the canvas, then queue.

The steps below build the same graphs by hand, which is the best way to understand them.

## 6. Your first clip (one segment, no LLM)

1. **Add `Hawk H3 Model Loader`.** Check that the four model dropdowns show the files from step 3. Put this in `lora_stack`:
   ```
   minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors : 1.0
   ```
   If you haven't installed Sol or Sage, set `attention` to `comfy default` (or leave it; it will just skip them).

2. **Add `Load Image`** and pick a clear photo of a person's face.

3. **Add `Hawk H3 References`.** Connect the image to `pictures → picture_0`. A new empty slot appears each time you connect one.

4. **Add `Hawk H3 Director`.**
   - `pipe` ← Model Loader `pipe`
   - `refs` ← References `refs`
   - Replace the `script` text with:
     ```
     duration: 6
     <Picture 1> defines the woman's face and hair. She sits at a café window in soft
     morning light, lifts a coffee cup, looks out at the rain and smiles. Slow push-in
     from a medium shot to a close-up. Sound: rain on the glass, quiet café murmur,
     cup set down on a saucer. Music N/A.
     ```

5. **Add `Save Video`** and connect Director `video` → `video`.

6. **Queue.** The Director shows a video preview when it's done. Files also land in `ComfyUI/output/hawk_h3/hawk_h3/`.

## 7. Your first film (three segments)

Keep the same graph and change only the Director's `script`:

```
style: Cinematic live-action, soft overcast light, 35mm lens feel, light film grain. Native audio. Music N/A. No subtitles.
---
title: Arrival
duration: 8
<Picture 1> defines the woman's face and hair. She steps off a tram onto a rain-wet platform and looks up at the station clock. Slow push-in. Sound: tram brakes hiss, distant announcements, light rain on the canopy.
---
title: The call
duration: 8
She turns toward camera, takes out her phone and answers: "I'm here. Where are you?" The camera tracks backwards at walking pace. Sound: footsteps on wet tiles, her voice clear and close.
---
title: The wave
duration: 6
She lowers the phone, spots someone off-screen left and waves, breaking into a smile. Camera pans left to follow her gaze. Sound: rain, a distant voice calling her name.
```

Queue it. Here's what happens:

1. All three prompts are encoded first.
2. Segment 1 renders and is saved.
3. Segment 2 renders **starting from the last ~1 second of segment 1**, so she's in the same place, mid-motion. Then segment 3 does the same from segment 2.
4. The three segments are joined into `hawk_h3_final.mp4` with the audio crossfaded at the seams.

Now change only segment 3's prompt and queue again. Segments 1 and 2 are reused from disk and only segment 3 renders. That's resume; see [Long videos](long-videos.md).

## 8. Let the LLM write the script

1. Add **`Hawk H3 Story Planner`**. Connect References `refs` → Planner `refs`.
2. Write a brief in `story`, for example: *"A woman arrives in a rainy city to meet an old friend. Warm reunion. 3 scenes."*
3. Set `segment_count` to 3 and `segment_seconds` to 8.
4. Connect Planner `script` → Director `script`. The script widget turns into an input.
5. Queue. The Planner shows its plan on the node; the Director renders it.

The Planner caches its result. Change its `seed` to get a different plan. To read a plan before spending GPU time on it, see the [review-then-render tip](nodes.md#hawk-h3-story-planner-atlas-llm).

**Next:** [Node reference](nodes.md) · [Writing scripts](scripts.md) · [Recipes](recipes.md)
