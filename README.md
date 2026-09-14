# Hawk MiniMax H3 Director

ComfyUI nodes that run the **MiniMax H3 reference-to-video** workflow in five nodes instead of ~40, and **direct long videos** made of many related H3 segments that flow into each other.

```
Hawk H3 Model Loader ─► Hawk H3 LoRA Stack ──pipe────────┐
Hawk H3 References ────refs──┬───────────────────────────┤
                             └─► Hawk H3 Story Planner ──script──► Hawk H3 Director ──video──► Save Video
```

## Example workflows

Drag a file from [`example_workflows/`](example_workflows) onto ComfyUI, or open it from **Workflow → Browse Templates** under this pack's name. Each one has a *Read me* note on the canvas.

| Workflow | What it shows |
|---|---|
| [`01_single_clip.json`](example_workflows/01_single_clip.json) | One picture, one prompt, one clip: the minimal setup |
| [`02_multi_segment_film.json`](example_workflows/02_multi_segment_film.json) | A four-segment ~34 s film: face, outfit, location and voice references, continuous joins and a hard cut |
| [`03_llm_story_planner.json`](example_workflows/03_llm_story_planner.json) | An Atlas LLM writes the script from a brief; review the plan, then render a cheap preview |
| [`04_video_motion_and_voice.json`](example_workflows/04_video_motion_and_voice.json) | Movement and camera from a reference video, the voice from its audio, on the person from a picture |
| [`05_pose_guided_sequence.json`](example_workflows/05_pose_guided_sequence.json) | A dancer moves through three key poses from pose images across three continuous segments |

Replace the placeholder file names in the Load Image / Load Audio / Load Video nodes with your own files.

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, models, your first clip and your first film |
| [Node reference](docs/nodes.md) | Every input and output, file layout, size and duration tables |
| [Writing scripts](docs/scripts.md) | Script format, reference tags, writing prompts H3 follows |
| [Long videos](docs/long-videos.md) | Continuity modes, resume and re-rendering, previews, consistency, performance |
| [Recipes](docs/recipes.md) | Ready setups: outfits, voices, motion copy, LLM stories, vertical shorts, ads, migrating the old workflow |
| [Troubleshooting](docs/troubleshooting.md) | Every error message and what to do |

## Nodes

| Node | Replaces in the stock template | What it does |
|---|---|---|
| **Hawk H3 Model Loader** | UNETLoader, CLIPLoader, 2× VAELoader, ModelSamplingMiniMaxH3, LoRA stack loader, Sol attention patch, Sage attention patch | Loads the ref2va model, Qwen3-VL text encoder and both VAEs. Applies the sigma shift, a LoRA stack with per-modality (video/audio/text) strengths, and optional Sol + Sage attention. Missing optional backends are skipped with a log line. |
| **Hawk H3 LoRA Stack** | The LoRA loader's dropdown stack | Up to 4 LoRAs per node, picked from dropdowns with strength and optional video/audio/text multipliers. Chain nodes for more. Changing a LoRA doesn't reload the model. |
| **Hawk H3 References** | The loose image / video / audio inputs of MiniMaxH3ReferenceToVideo | Up to 9 pictures, 9 pose images (body pose only, `<Pose N>`), 3 videos (with an optional paired soundtrack) and 3 audio clips in one bundle. Picture batches expand to one picture per frame. Chainable. Optional labels tell the planner what each reference is for. |
| **Hawk H3 Story Planner** | HawkAtlasLLM + prompt switch + string concat | A vision LLM on Atlas Cloud reads your brief and the references and writes a segment script. The reply is validated against the connected references before it leaves the node. `segment_count = 1` makes it a single-prompt refiner. |
| **Hawk H3 Director** | MiniMaxH3ReferenceToVideo, RandomNoise, KSamplerSelect, BasicScheduler, BasicGuider, SamplerCustomAdvanced, VAEDecode, VAEDecodeAudio, CreateVideo, duration math, resolution selector | Renders the script: one segment or forty. Outputs the whole film as one VIDEO with synced audio, plus frames, audio, the exact prompts and a JSON report. |

## Long videos: how segments connect

For every segment after the first, the Director takes the last frames of the previous segment and their audio. It anchors them at frame 0 of the new segment, using the same mechanism as ComfyUI's stock *Add Guide for MiniMax H3* node. H3 then continues the motion, identity, lighting and room tone instead of starting cold. The re-rendered head is trimmed off and its audio is crossfaded into the seam, so nothing plays twice.

| `continuity` | Carried over | Use for |
|---|---|---|
| `tail_22` (default) | last 22 frames (~0.9s) + audio | continuous action across segments |
| `tail_39` | last 39 frames (~1.6s) + audio | fast motion; costs more tokens |
| `tail_5` | last 5 frames + audio | lighter, still keeps direction of motion |
| `last_frame` | one still, no audio | a soft scene change that should match the last composition |
| `off` | nothing | hard cuts, new locations, time jumps |

Set it per node, and override it per segment in the script.

### Efficiency

- **One text-encoder pass.** All segments' text and references are encoded before any sampling. The 32B text encoder and the DiT therefore swap in VRAM once, not once per segment.
- **Resume.** Each finished segment is saved to `output/hawk_h3/<run_name>/` with a hash of everything that produced it. After a crash, an interrupt or a prompt edit, re-queue: only changed segments render again, along with every segment after them, since each one feeds the next segment's continuity guide. **Keep `seed` on *fixed*** for this; a randomized seed changes every segment.
- **Fewer reference tokens.** Each segment can list only the references it needs (`pictures: 1,3`). Tags are renumbered for that segment automatically, and every reference left out makes each sampling step cheaper.
- **Bounded RAM.** Finished segments live on disk. Only the continuity tail and the audio stay in memory. The final MP4 is assembled by stream copy, and full-film frames are only materialised if you enable `output_frames`.
- **Encode cache.** Re-running with a different seed, steps or sampler reuses the text conditioning already in memory.

## Script format

Plain text. Segments are separated by a line of `---`. Each block may begin with headers; everything after them is the prompt. A `style:` block is prepended to every segment.

```
style: Cinematic live-action, soft overcast light, 35mm lens feel, light grain. Native audio, music N/A.
---
title: Arrival
duration: 8
pictures: 1, 2
<Picture 1> defines her face and copper hair. <Picture 2> defines the green bomber jacket.
She steps off a tram onto a rain-wet platform and looks up at the station clock. Slow push-in.
Sound: tram brakes hiss, distant announcements, light rain.
---
title: The call
duration: 10
pictures: 1
audios: 1
<Audio 1> is her voice. She walks toward camera, answers her phone: "I'm here. Where are you?"
Tracking shot backwards at walking pace. Sound: footsteps on wet tiles, her voice close.
---
title: Rooftop
continuity: off
seed: 1234
Hard cut. Night, the city skyline...
```

| Header | Meaning |
|---|---|
| `title:` | label for logs and the report |
| `duration:` | seconds (1–15), snapped to H3's 17k+5 frame grid; default from the node |
| `pictures:` `videos:` `audios:` | which references this segment sends: `1,3`, `all` (default) or `none` |
| `continuity:` | `off`, `last_frame`, `tail_5`, `tail_22`, `tail_39` |
| `seed:` | fixed seed for this segment |

JSON works too, which is what the Story Planner emits: `{"style": "...", "segments": [{"title", "duration", "pictures", "videos", "audios", "continuity", "seed", "prompt"}]}`.

### Reference tags

- Write `<Picture N>`, `<Video N>`, `<Audio N>` using the **global** numbering, meaning the order references were connected to Hawk H3 References. `Image 1`, `@image1` and `<image_1>` are normalised for you.
- If a segment only sends some references, the Director renumbers the tags to what H3 actually receives.
- A video's paired soundtrack takes an audio label ahead of the standalone clips inside H3; this is accounted for automatically. If you want a clip's voice as its own `<Audio N>`, like the stock template does, connect the audio to `audios`, not `video_soundtracks`.
- A tag that points at a missing or unselected reference stops the run **before** anything renders.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Srioff-ashish/Hawk-Minimax-H3-Directory.git
```

Restart ComfyUI. You need a ComfyUI build with MiniMax H3 support ([ComfyUI#15224](https://github.com/Comfy-Org/ComfyUI/pull/15224)).

### Models ([Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3))

```
models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors   (or the int8_convrot variant)
models/vae/minimax_h3_video_vae_fp16.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors
models/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors   (optional, for 8-step sampling)
```

### Optional companions

| Package | Enables |
|---|---|
| `sageattention` (pip) | `attention: sage` |
| [ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) + Triton | `attention: sol scheduled` |
| [ComfyUI-VFI](https://github.com/GACLove/ComfyUI-VFI) | `interpolation: 48 / 60 fps (RIFE)` |
| [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) | `Load Video UI` for trimming reference videos |
| [comfyui-deno-custom-nodes](https://github.com/Deno2026/comfyui-deno-custom-nodes) | `Deno Multi Image Loader` for picture batches |
| [HawkNodes](https://github.com/Srioff-ashish/HawkNodes) | `Hawk Atlas I2I` for preparing reference images (e.g. a character in a given outfit) |

### Atlas Cloud key (Story Planner)

Export `ATLAS_API_KEY` before starting ComfyUI and leave the node's `api_key` blank. ComfyUI saves widget values into workflow JSON and PNG metadata, so a key typed into the node travels with every workflow you share.

## Recommended settings

The defaults match the stock template:

- sampler `res_multistep`, scheduler `simple` (try `beta`/`normal` for reference-heavy prompts), shift 12 video / 3 audio
- 8 steps with the turbo LoRA at 1.0; ~30 without it
- `megapixels 0.98` at 16:9 gives H3's native 1344×768
- segments of 8–12 s with `tail_22` for flowing action; `ref_image_size: max` when identity matters more than speed

## Development

```bash
python -m unittest discover -s tests
```

The script, tag and LoRA-stack logic is pure Python and is tested without ComfyUI or torch.

## Credits

Built on ComfyUI's MiniMax H3 nodes. LoRA modality split adapted from [ComfyUI-Plaguekind-Nodes](https://github.com/Plaguekind/ComfyUI-Plaguekind-Nodes). Sol attention via [ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn). Atlas client from [HawkNodes](https://github.com/Srioff-ashish/HawkNodes). The planner's prompting guide is adapted from the MiniMax H3 Prompt Engineer system prompt used in the original workflow.

MIT License.
