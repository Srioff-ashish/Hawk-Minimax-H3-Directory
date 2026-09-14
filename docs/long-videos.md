# Long videos

One H3 generation tops out around 15 seconds. The Director makes longer films by rendering a chain of segments and joining them so they read as one continuous video. This page explains how the join works, how to control it, and how to iterate without re-rendering everything.

- [How segments are joined](#how-segments-are-joined)
- [Continuity modes](#continuity-modes)
- [Mixing continuous scenes and hard cuts](#mixing-continuous-scenes-and-hard-cuts)
- [Resume and re-rendering](#resume-and-re-rendering)
- [Cheap previews first](#cheap-previews-first)
- [Keeping a character consistent over many segments](#keeping-a-character-consistent-over-many-segments)
- [Performance and memory](#performance-and-memory)

---

## How segments are joined

With continuity on, segment 2 isn't rendered from scratch:

```
segment 1  ██████████████████████████████▓▓▓▓▓▓        (last 22 frames = the "tail")
                                         │
                                         ▼ anchored at frame 0
segment 2                                ▓▓▓▓▓▓██████████████████████████████▓▓▓▓▓▓
                                         └trim┘
final film ██████████████████████████████▓▓▓▓▓▓██████████████████████████████▓▓▓▓▓▓ …
```

1. The last frames of segment 1, and the audio under them, are given to segment 2 as a **guide pinned at its first frame**. This is the same mechanism as ComfyUI's stock *Add Guide for MiniMax H3* node.
2. H3 renders segment 2 as a continuation of that clip: same people, place, light, motion direction and background sound.
3. The first frames of segment 2 repeat the tail, so they're **trimmed** from the film.
4. The audio across the join is **crossfaded** (`audio_crossfade_ms`, 60 ms by default) to hide any click or level jump.

So a film of segments at 8 s each with `tail_22` is:
`8.00 + (8.00 − 0.92) + (8.00 − 0.92) …` seconds. The `info` output reports the exact total.

The prompt of segment 2 should **continue** the action, not restart the scene. See [Writing scripts](scripts.md#writing-prompts-h3-follows).

---

## Continuity modes

Set the default on the Director (`continuity`) and override per segment with `continuity:` in the script.

| Mode | Carries into the next segment | Trimmed | Best for |
|---|---|---|---|
| `tail_22` *(default)* | last 22 frames (0.92 s) + audio | 22 frames | Continuous action: walking, talking, a camera move that keeps going |
| `tail_39` | last 39 frames (1.63 s) + audio | 39 frames | Fast or complex motion (dancing, fights, vehicles) where direction matters. Costs more tokens per step. |
| `tail_5` | last 5 frames (0.21 s) + audio | 5 frames | Lighter hand-off; keeps position and motion direction with less influence |
| `last_frame` | one still frame, **no audio** | 1 frame | A soft change of beat that should match the last composition but not its motion |
| `off` | nothing | nothing | Hard cuts: new location, time jump, montage |

Notes:

- `carry_audio` off keeps the visual hand-off but lets each segment's sound start fresh.
- If a segment is too short for its tail (e.g. `tail_39` on a 1-second segment), the Director uses the largest tail that fits and warns you.
- A longer tail is not always better. Anchoring a lot of footage makes H3 cling to it; if a segment should *change* what's happening, `tail_5` or `last_frame` gives it more freedom.

---

## Mixing continuous scenes and hard cuts

Think in scenes. Inside a scene, segments continue; between scenes, cut.

```
style: Neo-noir, rain, sodium streetlights, anamorphic lens flare. Native audio. No subtitles.
---
title: Scene 1 - Alley (1/2)
duration: 10
pictures: 1, 2
<Picture 1> defines the detective's face and grey stubble. <Picture 2> defines the long tan trench coat.
He walks down a wet alley, stops under a flickering lamp and lights a cigarette. Tracking from behind, then he turns into profile.
Sound: rain on metal, footsteps in puddles, lighter click. Music: low synth drone.
---
title: Scene 1 - Alley (2/2)
duration: 8
He exhales smoke and looks up at a lit window. Slow push-in to a close-up. Sound: rain, distant siren.
---
title: Scene 2 - Office
continuity: off
duration: 10
pictures: 1, 2, 3
Hard cut. <Picture 3> defines the cramped office. The detective sits at a desk under a green lamp, spreading photographs...
---
title: Scene 2 - Office (2/2)
duration: 8
He picks up one photograph and turns it toward camera...
```

The first segment of a scene should re-establish who, where and the look, since it starts fresh. The following segments just continue.

---

## Resume and re-rendering

Every finished segment is saved with a fingerprint of everything that produced it. When you queue again with the same `run_name` and `resume` on, a segment is **reused** if its fingerprint still matches.

### What a segment's fingerprint includes

- its prompt (after `style:` is added and tags are renumbered), title, duration, seed, continuity frames, and which references it uses
- the content of those references (swapping a picture invalidates the segments that use it)
- the model setup: model files, LoRAs and strengths, shifts, attention backend
- output size, steps, sampler, scheduler, `ref_image_size`, `carry_audio`, interpolation fps
- **the previous segment's fingerprint**

Because each fingerprint includes the previous one, changing a segment re-renders that segment **and every segment after it**. Later segments begin from its tail, so they have to be redone too. Earlier segments are always reused.

### Not in the fingerprint (safe to change)

- `audio_crossfade_ms`: applied when the final file is joined
- `encode_all_first`, `output_frames`, `resume` itself
- anything on the Story Planner, *unless* it changes the script text

### Common situations

| You want to… | Do this |
|---|---|
| Continue after a crash, out-of-memory error or Cancel | Queue again, unchanged. Finished segments are skipped. |
| Retry only the **last** segment with a different take | Add `seed: 12345` to that segment (or change its prompt) and queue. |
| Fix segment 3 of 10 | Edit segment 3 and queue. Segments 1–2 are reused; 3–10 re-render. |
| Change a middle segment **without** re-rendering what follows | You can't keep them consistent with the old tail. If the change is small, accept the re-render; otherwise set `continuity: off` on the next segment to make that a cut. |
| Get different takes of the whole film | Change `run_name` (keeps the old run) or the base `seed`. |
| Start completely over in the same folder | Turn `resume` off for one queue, or delete `output/hawk_h3/<run_name>/`. |

⚠ **Seed must stay fixed.** If the Director's seed widget is set to *randomize*, every queue changes every seed and nothing is ever reused.

⚠ **Retitling a segment re-renders it.** Titles are part of the fingerprint.

---

## Cheap previews first

Test the story, pacing and hand-offs cheaply, then render the real thing:

1. `run_name: mystory_preview`, `megapixels: 0.4` (864×480 at 16:9), optionally shorter `default_seconds`.
2. Queue. Fix prompts until the preview flows.
3. `run_name: mystory_final`, `megapixels: 0.98`, `ref_image_size: max` if identity matters.
4. Queue.

Size is part of the fingerprint, so the final run renders fresh, and a separate `run_name` keeps the preview folder intact.

A preview at the same seed is **not** a frame-accurate miniature of the final: a different size means a different sample. It is a reliable test of the script, timing and reference jobs.

---

## Keeping a character consistent over many segments

Continuity carries identity from one segment to the next, but small drift adds up over a long chain. Keep re-anchoring:

- **Send the face reference in every segment the character appears in** (`pictures: 1` or more), and name the identifying traits in the prompt each time.
- **Use a clean reference set:** a sharp front or three-quarter face, a full-body wardrobe shot, and a close detail (logo, tattoo) if it matters. Compatible references beat many references.
- **`ref_image_size: max`** for the final render when the face must hold.
- **Put the look in `style:`** so lighting and colour don't wander.
- **Voices:** connect a clean voice sample to `audios`, send it in segments where the character speaks (`audios: 1`), and write `<Audio 1> is her voice` in those prompts. `carry_audio` also helps keep the tone across joins.
- **Land key poses at segment ends.** A `<Pose N>` reached at the end of a segment gives the next segment a precise, known starting pose. See [pose references](scripts.md#pose-references).
- **Cut occasionally.** A `continuity: off` scene change that re-establishes the character from references resets accumulated drift.

---

## Performance and memory

What the Director already does for you:

| Optimisation | Effect |
|---|---|
| Encode all segments first | The 32B text encoder and the video model each load once per queue, instead of swapping every segment. |
| Encode cache | Re-queuing with a different seed, steps or sampler reuses the text conditioning still in memory. |
| Segments to disk as they finish | RAM holds one segment plus a short tail, not the whole film. |
| Stream-copy join | The final MP4 is assembled without decoding and re-encoding every frame. |
| Per-segment reference lists | Fewer reference tokens on every sampling step. |

What you can tune:

| Problem | Try |
|---|---|
| Out of VRAM while sampling | Lower `megapixels`; `weight_dtype: fp8_e4m3fn`; `clip_device: cpu`; use `ref_image_size: match`; send fewer references per segment |
| Out of RAM during encoding | `encode_all_first: off` (each segment is encoded right before it renders) |
| Out of RAM at the end | Keep `output_frames` off |
| Too slow | Turbo LoRA + 8 steps; `attention: sol scheduled + sage`; `tail_5` instead of `tail_39`; fewer references per segment; `megapixels 0.8` |
| Choppy motion | `interpolation: 48 fps (RIFE)` (needs ComfyUI-VFI) |
