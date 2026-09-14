# Node reference

Every input and output of the four nodes. Inputs marked *advanced* are hidden until you expand the node's advanced section.

- [Hawk H3 Model Loader](#hawk-h3-model-loader)
- [Hawk H3 References](#hawk-h3-references)
- [Hawk H3 Story Planner](#hawk-h3-story-planner-atlas-llm)
- [Hawk H3 Director](#hawk-h3-director)

---

## Hawk H3 Model Loader

Loads everything the Director needs, then patches the model in the same order the stock MiniMax H3 template does:

**diffusion model → sigma shift → LoRAs → Sol attention → Sage attention**

### Inputs

| Input | Default | What it does |
|---|---|---|
| `unet_name` | `minimax_h3_ref2va_pruned_int8_convrot` if present | The H3 **ref2va** diffusion model from `models/diffusion_models`. The `fl2va` model used by the stock text-to-video and image-to-video templates is a different model and won't work here. |
| `clip_name` | `qwen3vl_32b_minimax_h3_nvfp4_awq` if present | The Qwen3-VL text encoder from `models/text_encoders`, loaded as CLIP type `minimax`. |
| `video_vae` | `minimax_h3_video_vae_fp16` if present | Video VAE from `models/vae`. |
| `audio_vae` | `minimax_h3_audio_vae_fp32` if present | Audio VAE from `models/vae`. |
| `lora_stack` | empty | LoRAs to apply, one per line. See [LoRA stack format](#lora-stack-format). |
| `shift_video` | 12.0 | Flow shift for the video stream. Higher pushes more steps toward coarse structure. Leave at 12 unless you know why. |
| `shift_audio` | 3.0 | Flow shift for the audio stream. |
| `attention` | `sol scheduled + sage` | `comfy default`, `sage`, `sol scheduled`, or `sol scheduled + sage`. A backend that isn't installed is skipped with a console warning. |
| `weight_dtype` *(advanced)* | `default` | Load the model as fp8 to save VRAM (`fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2`). |
| `clip_device` *(advanced)* | `default` | `cpu` keeps the text encoder off the GPU: slower to encode, more VRAM for sampling. |
| `sol_tau_start` *(advanced)* | 1.25 | Sol sparsity on the first, noisiest step. Higher is faster and looser. |
| `sol_tau_end` *(advanced)* | 0.8 | Sol sparsity on the last detail steps. Lower is denser and more accurate. |

### Outputs

| Output | Use |
|---|---|
| `pipe` | Connect to **Hawk H3 Director**. Carries the model, text encoder, both VAEs, and a fingerprint of the setup used by the resume cache. |
| `model`, `clip`, `vae`, `audio_vae` | The same objects, for other ComfyUI nodes. |

The node shows a short summary of what it loaded and which attention backends actually applied.

### LoRA stack format

One LoRA per line: file name, then strength, then optional per-modality multipliers.

```
minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors : 1.0
my_style.safetensors : 0.7 : v=1 a=0.5 t=1
subfolder/character.safetensors : 0.8
# switched_off.safetensors : 1.0
```

- The strength defaults to 1.0 if omitted.
- `v=` / `a=` / `t=` multiply the strength for H3's **video**, **audio** and **text** projection layers only. H3's main transformer blocks are shared by all three modalities, so they always use the plain strength. `a=0` does not fully mute a LoRA's effect on sound; it only skips the audio-specific layers.
- A line starting with `#` is ignored.
- You can also paste the JSON `stack_data` from Plaguekind's *LoRA Loader Stack* node; rows with `"on": false` are skipped.

A typo in a line stops the node **before** any model loads, so you don't wait a minute to find out.

---

## Hawk H3 References

Collects the reference files and fixes their numbering. The numbers are how scripts refer to them.

### Inputs

| Input | What to connect |
|---|---|
| `refs_in` *(optional)* | Another References node's `refs`. Its references come first; this node's are numbered after them. Use it to split a big set across nodes. |
| `pictures` | Up to 9 images. Each connection adds a slot. **A batch of images adds one picture per image**, so a 3-image batch becomes `<Picture 1>`–`<Picture 3>`. |
| `videos` | Up to 3 videos as IMAGE frames (from Load Video UI, VHS Load Video, etc.). 2–15 s each; at least 5 frames. |
| `video_soundtracks` | The audio of the **same-numbered** video: `video_soundtrack_0` belongs to `video_0`. Use this when the sound belongs to the clip, e.g. its ambience or the performance's timing. |
| `audios` | Up to 3 standalone audio clips: a voice sample, a music bed, a sound effect. |
| `labels` *(optional)* | One line per reference saying what it is for. Shown to the Story Planner and in `tag_map`; **not** sent to H3. |
| `video_fps` *(advanced)* | Frame rate of the connected videos. They're resampled to H3's 24 fps. Leave at 24 if your loader already outputs 24 fps (Load Video UI does by default). |

`labels` format:

```
Picture 1: her face and hair
Picture 2: green bomber jacket
Video 1: walking pace and camera move
Audio 1: her voice
```

### Soundtrack or standalone audio?

The same audio file can go into either input, and H3 treats the two differently:

| Connected to | H3 sees it as | Script tag |
|---|---|---|
| `video_soundtracks` | part of that video | none of its own; refer to the video (`the voice in <Video 1>`) |
| `audios` | its own reference | `<Audio N>` |

If you want a video's **voice** as a distinct voice reference you can name in prompts, connect the audio to `audios`. That's also how the original stock workflow was wired.

### Outputs

| Output | Use |
|---|---|
| `refs` | Connect to the Director and/or the Story Planner. |
| `tag_map` | Text such as `<Picture 1> 1024x1536 -- her face`. Also shown on the node. Use it to check numbering before writing a script. |

### Limits

At most **9 pictures, 3 videos and 3 audio clips** in total, counting chained `refs_in`. The node errors if you exceed them.

---

## Hawk H3 Story Planner (Atlas LLM)

Sends your brief and the references to a vision LLM on Atlas Cloud and gets back a script. The reply is **checked before it leaves the node**: if it mentions a reference you didn't connect or returns malformed JSON, the Planner errors and shows the reply, instead of letting the Director start a long render.

What the LLM receives:
- every picture as an image
- three frames (start, middle, end) of each reference video, labelled with timestamps
- a text line for each audio clip with its length and label (the audio itself is not sent)
- your brief, the segment count, the target length per segment and the aspect ratio

### Inputs

| Input | Default | What it does |
|---|---|---|
| `story` | empty | Your brief. Plot, characters, mood, locations, exact dialogue lines you want. More specific briefs give more usable scripts. |
| `refs` *(optional)* | — | From Hawk H3 References. Without it the LLM writes a text-only film. |
| `segment_count` | 3 | Number of segments to write. **0** lets the LLM decide (usually 2–8). **1** turns the node into a single-prompt refiner. |
| `segment_seconds` | 10 | Target length per segment (1–15). The LLM may vary it per segment. |
| `aspect_ratio` | 16:9 | Told to the LLM so it frames shots correctly. **Set the Director to the same value.** |
| `model` | `xai/grok-4.3` | Any Atlas chat model id. Use a vision-capable model when references are connected. |
| `seed` | 0 | ComfyUI reuses the last plan while inputs are unchanged. Change the seed (or set it to randomize) for a new plan. Also sent to Atlas when above 0. |
| `system_prompt` *(advanced)* | blank | Blank uses the built-in H3 planning guide in `hawk_h3/prompts/planner_system.md`. Paste your own to override it, but keep the JSON output format. |
| `temperature` *(advanced)* | 0.7 | Lower is more literal; higher is more inventive. |
| `max_tokens` *(advanced)* | 8192 | Raise it for many long segments if replies get cut off. |
| `json_mode` *(advanced)* | on | Asks Atlas for strict JSON. Turn it off if a model rejects `response_format`. |
| `image_max_side` *(advanced)* | 1024 | Pictures are downscaled to this before upload. |
| `api_url` *(advanced)* | `https://api.atlascloud.ai/v1` | Change only for a proxy. |
| `api_key` *(advanced)* | blank | Blank reads `ATLAS_API_KEY`. A typed key is saved into workflow files. |
| `timeout` *(advanced)* | 240 | Seconds to wait for the reply. |
| `max_retries` *(advanced)* | 3 | Retries on rate limits and server errors. Auth errors are never retried. |

### Outputs

| Output | Use |
|---|---|
| `script` | Clean, validated JSON. Connect to the Director's `script`. |
| `raw_reply` | Exactly what the model returned, for debugging. |

The node displays the plan (titles, durations, prompts, warnings) so you can read it before rendering.

**Tip: review, then render.** Connect the Planner's `script` to a **Preview Any** node as well as to the Director. Bypass the Director (select it, Ctrl+B) and queue: only the Planner runs, since Preview Any is an output node, and you can read the plan. Then un-bypass the Director and queue again. The Planner result is cached, so the second queue goes straight to rendering.

**To hand-edit the plan**, copy the JSON text from that Preview Any node into the Director's `script` widget, disconnect the Planner from `script`, and edit the JSON. The preview shown *on the Planner node itself* is a readable summary, not a script, so don't paste that.

---

## Hawk H3 Director

Renders the script. For each segment it runs exactly what the stock template runs (reference conditioning → guider → sampler → video and audio decode), chains the segments, and writes one finished video.

### Inputs

| Input | Default | What it does |
|---|---|---|
| `pipe` | — | From Hawk H3 Model Loader. |
| `script` | an example | The segments to render. Plain text or JSON; see [Writing scripts](scripts.md). Can be typed or connected from the Planner. |
| `refs` *(optional)* | — | From Hawk H3 References. |
| `aspect_ratio` | 16:9 | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9`, `9:21`, or `match first picture` (uses `<Picture 1>`'s shape). |
| `megapixels` | 0.98 | Output size. 0.98 at 16:9 is H3's native 1344×768. See the [size table](#output-sizes). |
| `default_seconds` | 10 | Length of segments that don't set `duration:`. |
| `steps` | 8 | Sampling steps. 8 with the turbo LoRA; about 30 without it. |
| `sampler_name` | `res_multistep` | Any ComfyUI sampler. |
| `scheduler` | `simple` | `beta` or `normal` often do better on reference-heavy prompts. |
| `seed` | 0, *fixed* | Base seed. **Keep it fixed**: resume depends on the seed staying the same. |
| `continuity` | `tail_22` | How segments hand off: `off`, `last_frame`, `tail_5`, `tail_22`, `tail_39`. See [Long videos](long-videos.md#continuity-modes). Scripts can override it per segment. |
| `carry_audio` | on | Also hand the previous segment's tail **audio** to the next, so voices and room tone continue across the join. |
| `ref_image_size` | `match` | `match` shrinks reference pictures to the output's pixel area (fast). `max` keeps up to a 2048 px short edge for stronger identity, but reference tokens ride along every step, so it can be several times slower. |
| `run_name` | `hawk_h3` | Folder name under `output/hawk_h3/`. **Use a new name per project** so runs don't overwrite each other. |
| `resume` | on | Reuse saved segments whose inputs haven't changed. Off renders everything again and overwrites the folder's segments. |
| `seed_mode` *(advanced)* | `increment per segment` | Segment *n* gets `seed + n - 1`. `same for all` gives every segment the same seed. A script's `seed:` always wins. |
| `audio_crossfade_ms` *(advanced)* | 60 | Crossfade length at each continuity join. Hard cuts (`continuity: off`) are never faded. |
| `interpolation` *(advanced)* | `off` | `48 fps (RIFE)` or `60 fps (RIFE)`. Applied per segment. Needs ComfyUI-VFI. |
| `encode_all_first` *(advanced)* | on | Encode every segment's text and references before sampling, so the text encoder and the video model each load once. Turn it off only if RAM is too tight to hold all conditionings at once. |
| `output_frames` *(advanced)* | off | On: `frames` returns every frame of the whole film. Off: only the last rendered segment's frames. A few minutes of film at full size can need tens of GB of RAM, so leave it off unless the next node really needs all frames. |

### Outputs

| Output | Use |
|---|---|
| `video` | The whole film with audio. Connect to **Save Video**. (The Director already saves `<run_name>_final.mp4`; Save Video makes another copy with your own filename.) |
| `frames` | IMAGE frames (see `output_frames`). Useful for upscalers or a preview. |
| `audio` | The full stitched soundtrack. |
| `prompts` | The exact text each segment was encoded with, after `style:` was added and tags were renumbered. Check this when a result ignores a reference. |
| `info` | JSON report: resolution, each segment's length, seed, continuity frames, whether it came from cache, render time, file paths, and all warnings. |

### Files written

```
ComfyUI/output/hawk_h3/<run_name>/
├── segment_001.mp4      the segment as it appears in the film (joined part trimmed)
├── segment_001.pt       continuity tail + audio, used to resume
├── segment_001.json     settings fingerprint, seed, prompt
├── segment_002.mp4 …
└── <run_name>_final.mp4 all segments joined, audio crossfaded
```

The individual `segment_###.mp4` files are handy for editing in another program. Their audio isn't crossfaded; the final file's is.

### Output sizes

| megapixels | 16:9 | 9:16 | 1:1 | 4:3 | 21:9 |
|---|---|---|---|---|---|
| 0.4 | 864×480 | 480×864 | 640×640 | 736×576 | 992×416 |
| 0.6 | 1056×608 | 608×1056 | 800×800 | 928×672 | 1216×512 |
| 0.8 | 1216×672 | 672×1216 | 928×928 | 1056×800 | 1408×608 |
| **0.98** | **1344×768** | **768×1344** | **1024×1024** | **1184×864** | **1536×672** |
| 1.5 | 1664×928 | 928×1664 | 1248×1248 | 1440×1088 | 1920×832 |

### Durations

H3 renders frame counts on a fixed grid (17k + 5 frames at 24 fps), so durations round **up** slightly:

| You ask for | Frames | Actual length |
|---|---|---|
| 3 s | 73 | 3.04 s |
| 5 s | 124 | 5.17 s |
| 6 s | 158 | 6.58 s |
| 8 s | 192 | 8.00 s |
| 10 s | 243 | 10.12 s |
| 12 s | 294 | 12.25 s |
| 15 s | 362 | 15.08 s (the maximum) |

Anything above 15 s is clamped to 15 s with a warning.
