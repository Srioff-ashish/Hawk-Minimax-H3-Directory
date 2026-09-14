# Troubleshooting

Find your message or symptom below. Error text is shown as the node's error popup; warnings appear in the ComfyUI console, prefixed `HawkH3:`, and in the Director's `info` output.

- [Installation](#installation)
- [Model Loader](#model-loader)
- [References](#references)
- [Scripts](#scripts)
- [Story Planner](#story-planner)
- [Director](#director)
- [Quality problems](#quality-problems)

---

## Installation

**The Hawk H3 nodes don't appear.**
Look for this in the console at startup:
`Hawk-Minimax-H3-Directory needs a ComfyUI build with MiniMax H3 support`
Your ComfyUI is too old. Update it (`git pull` in the ComfyUI folder, or the Manager's *Update ComfyUI*) and restart.
If that line isn't there, check that the folder is `ComfyUI/custom_nodes/Hawk-Minimax-H3-Directory` and contains `__init__.py` directly (not a nested folder of the same name).

**A dropdown shows `(none found)`.**
No files in that model folder. See [Getting started → models](getting-started.md#3-download-the-models), then restart ComfyUI or press **R**.

---

## Model Loader

**`unet_name: no files in models/diffusion_models…`** (or `clip_name`, `video_vae`, `audio_vae`)
Same as above: the model folder is empty or the file is somewhere else.

**A LoRA doesn't appear in the LoRA Stack dropdown.**
Put the file in `ComfyUI/models/loras/` (subfolders are fine), then press **R** in ComfyUI or restart.

**I need more than 4 LoRAs.**
Chain another **Hawk H3 LoRA Stack** node: previous LoRA Stack `pipe` → new LoRA Stack `pipe` → Director `pipe`. Or list them all as lines in the Model Loader's `lora_stack`.

**`LoRA not found in models/loras: <name>`**
The name in `lora_stack` must match the file's path relative to `models/loras`, including subfolders and extension (`subfolder/my_lora.safetensors`).

**`lora_stack line N (…) needs a LoRA file name…`** / **`cannot read …`**
Use `file.safetensors : 0.8` or `file.safetensors : 0.8 : v=1 a=0.5 t=1`. Only `v`, `a` and `t` are accepted as keys.

**Console: `'sol scheduled' needs ComfyUI-sol-attn…` / `Sol attention patch failed…` / `'sage' needs the sageattention package…`**
Not errors: the backend was skipped and sampling uses ComfyUI's default attention. Install it (see [optional extras](getting-started.md#4-optional-extras)) or set `attention` to what you have. The node's summary shows which backends actually applied.

**Wrong-looking results or shape errors right after loading.**
Make sure `unet_name` is the **ref2va** model, not the `fl2va` model, and `clip_name` is the MiniMax H3 Qwen3-VL encoder.

---

## References

**`MiniMax H3 takes at most 9 pictures; 11 are connected (including chained references).`**
A batch input counts every image in it. Remove some, or send them from a separate References node to a different Director.

**`video_0: reference videos need at least 5 frames`**
The video input is empty or trimmed too short. Reference videos should be 2–15 s.

**Numbering isn't what I expected.**
Read `tag_map` on the node. Chained `refs_in` references come first. Batches expand in order. Empty slots are skipped.

---

## Scripts

All of these appear **before anything renders**.

| Message | Fix |
|---|---|
| `The script is empty…` | Type a prompt, or connect the Planner's `script`. |
| `The script has no segment with prompt text.` | Every block contains only headers. Add prompt lines below them. |
| `Segment N mentions <Picture 4> but only 3 picture(s) are connected.` | Connect the reference or fix the tag. Check `tag_map`. |
| `Segment N mentions <Picture 2> but its pictures list leaves it out.` | Add `2` to that segment's `pictures:` or remove the mention. |
| `Segment N asks for picture [5] but only 3 picture(s) are connected.` | The `pictures:` list names a reference that isn't connected. |
| `…picture numbers run 1-9; got [0]` | Numbers start at 1. Use `none` for no pictures. |
| `Segment N: duration 'soon' is not a number of seconds.` | Use `8`, `8s` or `7.5`. |
| `Segment N: continuity 'x' is not one of off, last_frame, tail_5, tail_22, tail_39.` | Use one of those. |
| `The script looks like JSON but does not parse: …` | The text starts with `{` or `[` but is broken JSON. Fix it, or remove the leading bracket to use plain text. |
| `A JSON script needs a "segments" list…` | Wrap segments as `{"segments": [ … ]}`. |

**A prompt word turned into a tag.**
Phrases like "video 2" or "image 1" are converted to tags. Rephrase ("the second video", "a two-minute video").

**Warnings (render continues):**
- `…is past H3's ~15s trained range; clamped to 15s.` Split it into two segments.
- `…a 39-frame continuity guide does not fit a 56-frame segment; using 22.` The segment is too short for its tail.
- `…N reference files; H3 is documented for at most 12.` Send fewer references in that segment.
- `…prompt is N characters; H3 is documented for 7000.` Shorten the prompt or the `style:` block, which is added to every segment.

---

## Story Planner

**`No Atlas API key…`**
Set `ATLAS_API_KEY` before starting ComfyUI ([how](getting-started.md#5-set-your-atlas-key-only-for-the-story-planner)), or fill `api_key` (it will be saved in workflows).

**`Atlas API error 401` / `403`**
Invalid or unauthorised key. Check it on Atlas Cloud.

**`Atlas API error 400 … not found`**
The `model` id isn't available on your account. Ids are case-sensitive and vendor-prefixed. List yours:
```bash
curl -s https://api.atlascloud.ai/v1/models -H "Authorization: Bearer $ATLAS_API_KEY"
```

**`Atlas API error 400` (other)**
Usually `json_mode` on a model that doesn't support it, or images sent to a text-only model. Turn `json_mode` off, or use a vision model.

**`Atlas returned an empty message (finish_reason=length)`**
Raise `max_tokens`.

**`The planner's reply is not a usable script: …`**
The LLM mentioned a reference that isn't connected, or broke the format. The message includes the reply. Try another `seed`, a stronger model, or `json_mode` on. If it keeps inventing references, add `labels` and state in `story` which references exist.

**`Atlas request timed out…`**
Raise `timeout`; large reference sets with big images take longer. Lowering `image_max_side` also helps.

**The plan doesn't change when I queue again.**
ComfyUI caches unchanged nodes. Change the Planner `seed`.

---

## Director

**Everything re-renders every time.**
- The `seed` widget is set to *randomize*. Set it to **fixed**.
- The Planner produced a new script (its seed changed), so every prompt changed.
- `resume` is off.
- Something in the fingerprint changed: see [what's included](long-videos.md#what-a-segments-fingerprint-includes). Changing a segment's **title** counts.

**Nothing happens when I queue again.**
ComfyUI saw no input changes and reused the Director's last output. That's expected; change something to render.

**`interpolation needs ComfyUI-VFI…`**
Install [ComfyUI-VFI](https://github.com/GACLove/ComfyUI-VFI) or set `interpolation: off`.

**`aspect_ratio is 'match first picture' but no picture is connected.`**
Connect a picture or pick a fixed ratio.

**`Segment N decoded at X Hz but earlier segments at Y Hz…`**
Old segments in the run folder were made with a different audio VAE. Use a new `run_name`, delete the folder, or queue once with `resume` off.

**Out of memory (CUDA OOM).**
See [Performance and memory](long-videos.md#performance-and-memory). Finished segments are safe on disk: after changing settings that are *not* in the fingerprint (e.g. `encode_all_first`), queue again to continue where it stopped.

**Where are my files?**
`ComfyUI/output/hawk_h3/<run_name>/`. The `info` output lists every path.

**A visible jump or a repeated moment at a join.**
- Jump in position or lighting: use a longer tail (`tail_22` → `tail_39`) and make sure the next prompt continues the action instead of re-describing the scene.
- A repeated motion: the next prompt restarts an action that already finished. Start that prompt from the end state.
- Audio click: raise `audio_crossfade_ms` to 100–150. This only reassembles the final file; no segments re-render.

---

## Quality problems

**The face drifts from the reference.**
Send the face picture in every segment with that character; name distinctive traits; use `ref_image_size: max`; remove references that compete (another person's photo, a busy video); consider a scheduler of `beta` or `normal`.

**H3 copies the wrong thing from a reference** (the outfit photo's model, the motion video's location).
Give each reference a narrow job and say what it must **not** provide: "`<Video 1>` supplies only the camera move; do not take the person or place from it."

**Dialogue is cut off or rushed.**
The line is too long for the shot. Budget about 2.5 words per second or lengthen the segment.

**Soft or smeary output.**
Without the turbo LoRA, raise `steps` to ~30. With it, keep 8. Raise `megapixels`. Lower `sol_tau_start` if using Sol.

**Unwanted subtitles or random text.**
Add `No subtitles, no on-screen text.` to `style:`.

**Music changes at every join.**
Each segment generates its own score. Use `carry_audio: on` with a tail mode, describe the same music in `style:`, or write `Music N/A` and add music in an editor.
