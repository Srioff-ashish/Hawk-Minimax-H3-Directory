# Recipes

Working setups for common jobs. Each lists the graph, the settings that matter, and a script to start from. Unless a recipe says otherwise, the Model Loader uses the turbo LoRA and the Director keeps its defaults.

- [1. Single clip from a face photo](#1-single-clip-from-a-face-photo)
- [2. Character in a specific outfit](#2-character-in-a-specific-outfit)
- [3. Talking character with a voice reference](#3-talking-character-with-a-voice-reference)
- [4. Copy a motion or camera move from a video](#4-copy-a-motion-or-camera-move-from-a-video)
- [5. A one-minute story written by the LLM](#5-a-one-minute-story-written-by-the-llm)
- [6. Vertical short for social media](#6-vertical-short-for-social-media)
- [7. Montage of hard cuts](#7-montage-of-hard-cuts)
- [8. Product ad with readable text](#8-product-ad-with-readable-text)
- [9. Rebuilding the original MiniMax H3 workflow](#9-rebuilding-the-original-minimax-h3-workflow)
- [10. Hit specific body poses](#10-hit-specific-body-poses)

---

## 1. Single clip from a face photo

**Graph:** Load Image → References `picture_0` · Model Loader → Director · Director `video` → Save Video

```
duration: 6
<Picture 1> defines the man's face, round glasses and short grey hair.
He sits in a sunlit library, closes a heavy book and looks up toward camera with a quiet smile.
Static medium shot, then a slow push-in to a close-up.
Sound: book thud, pages settling, distant clock ticking. Music N/A.
```

---

## 2. Character in a specific outfit

**Graph:** face photo → `picture_0`, outfit photo → `picture_1`, optional location photo → `picture_2`.

Optional preparation: use **Hawk Atlas I2I** (HawkNodes) to make a single clean image of the character already wearing the outfit. Feed that as `picture_0` for the strongest result.

References `labels`:
```
Picture 1: her face and hair
Picture 2: the dress
Picture 3: the ballroom
```

```
duration: 10
<Picture 1> defines her face, dark wavy hair and green eyes. <Picture 2> defines the dress: take its colour, fabric and cut exactly, not the model wearing it. <Picture 3> defines the ballroom and chandelier light.
She walks down the marble staircase, one hand on the rail, the dress trailing behind. Camera cranes down slowly with her, ending on a full-length shot at the bottom.
Sound: heels on marble, soft orchestra from the hall, murmuring guests.
```

The phrase "not the model wearing it" stops H3 from borrowing the outfit photo's face.

---

## 3. Talking character with a voice reference

**Graph:** face photo → `picture_0`; clean voice sample (5–15 s, one speaker, no music) via **Load Audio** → `audios` `audio_0`.

If the voice comes from a video (Load Video UI), connect the video's `audio` output to `audios`, not `video_soundtracks`, so it becomes its own `<Audio 1>`.

```
style: Natural documentary look, soft window light, handheld micro-movement. Native audio, music N/A.
---
title: Intro
duration: 8
pictures: 1
audios: 1
<Picture 1> defines her face and hair. <Audio 1> is her voice: take its timbre, accent and relaxed pace.
She sits on a sofa, looks into the lens and says: "Hi, I'm Maya. Let me show you where I work."
Medium close-up, static. Sound: quiet room tone, her voice close and clear.
---
title: Walk
duration: 10
pictures: 1
audios: 1
<Picture 1> defines her face. <Audio 1> is her voice.
She stands up and walks toward the doorway, talking over her shoulder: "It's small, but the light is perfect."
Handheld follow from behind, she turns into profile at the door.
```

Keep lines short enough for the duration (about 2.5 words per second).

---

## 4. Copy a motion or camera move from a video

**Graph:** **Load Video UI** (trim to 2–10 s of the move) `images` → References `video_0`; face photo → `picture_0`.

```
duration: 8
pictures: 1
videos: 1
<Picture 1> defines the dancer's face and hair. <Video 1> supplies ONLY the dance moves, their timing and the circling camera; do not take the person, clothes or location from it.
She dances in an empty concrete parking garage at night under fluorescent tubes, wearing a black tracksuit.
Sound: sneakers squeaking on concrete, echoing space, a steady electronic beat.
```

Say explicitly what the video must **not** contribute; otherwise H3 may copy its person or place.

---

## 5. A one-minute story written by the LLM

**Graph:**
```
Load Image (hero) ─┐
Load Image (city) ─┼► References ─refs─┬─► Story Planner ─script─► Director ─video─► Save Video
Load Audio (voice)─┘                   └──────────────────────────► Director
Model Loader ─pipe─────────────────────────────────────────────────► Director
```

References `labels`:
```
Picture 1: the courier, main character
Picture 2: the city at dusk
Audio 1: the courier's voice
```

Story Planner:
- `story`:
  > A bicycle courier races across a city at dusk to deliver an envelope before a shop closes. Near-misses in traffic, a shortcut through a market, arriving just as the shutter comes down, then the shopkeeper opens it again with a smile. Upbeat, warm, light humour. She says one line at the end: "Special delivery. Just in time?"
- `segment_count`: 6 · `segment_seconds`: 10 · `aspect_ratio`: 16:9

Director: `run_name: courier_preview`, `megapixels: 0.4`, `continuity: tail_22`.

Also connect the Planner's `script` to a **Preview Any** node.

Workflow:
1. Bypass the Director (Ctrl+B), queue, and read the plan in Preview Any.
2. Not happy? Change the Planner `seed` or refine `story`, queue again.
3. Un-bypass the Director, queue the preview.
4. Switch to `run_name: courier_final`, `megapixels: 0.98`, queue.

To tweak one segment by hand: copy the JSON from Preview Any into the Director's `script` widget, disconnect the Planner from `script`, edit that segment's `prompt`, and queue. Segments before the edit are reused. The text shown on the Planner node itself is a summary, not a script.

---

## 6. Vertical short for social media

Director: `aspect_ratio: 9:16`, `megapixels: 0.98` (768×1344). Story Planner `aspect_ratio: 9:16` too.

```
style: Vertical 9:16 phone-shot look, bright daylight, vivid colour, quick energy. Native audio.
---
title: Hook
duration: 4
pictures: 1
<Picture 1> defines his face. Extreme close-up, he raises an eyebrow and says: "You're making coffee wrong."
---
title: Show
duration: 10
pictures: 1, 2
<Picture 2> defines the copper pour-over kettle. He pulls back to a medium shot and pours in a slow spiral, steam rising. Top-down cut at 00:05 of the bloom.
Sound: water pouring, gentle bubbling.
---
title: Payoff
duration: 5
He sips, closes his eyes and grins at camera. Sound: satisfied exhale.
```

Short hooks work: the Director accepts durations under 5 s (they round up to the frame grid, e.g. 4 s → 4.46 s).

---

## 7. Montage of hard cuts

Director `continuity: off` (or `continuity: off` on each segment). Every segment is independent.

```
style: Travel montage, golden hour, handheld, 35mm grain. Native audio. Music: one continuous acoustic guitar motif.
---
duration: 4
pictures: 1
<Picture 1> defines her face. She laughs on a ferry deck, hair blown by wind.
---
duration: 4
pictures: 1
She bites into street food at a night market, neon behind her.
---
duration: 4
pictures: 1
She reaches a mountain summit at sunrise and throws her arms up.
```

Music can't truly flow across hard cuts, because each segment generates its own. For one continuous song, lay the music over the final video in an editor, or render the montage with `continuity: tail_5` and `carry_audio: on` so the sound bed carries through.

---

## 8. Product ad with readable text

H3 can render legible text when you give the exact words and how they appear.

```
style: Premium product commercial, black studio, rim light, slow elegant camera. Native audio. No other text anywhere.
---
title: Reveal
duration: 8
pictures: 1
<Picture 1> defines the matte-black water bottle, its silver cap and the white "AERO" wordmark. The logo appears only on the bottle.
The bottle rotates slowly on a turntable while a light sweep crosses it. Macro push-in on the cap.
Sound: soft whoosh, deep sub hit as the light passes.
---
title: Title card
duration: 5
The bottle settles centre frame. The words "STAY COLD. 24 HOURS." rack into focus above it in thin white sans-serif capitals and hold, no animation.
Sound: single low piano note, then silence.
```

Spell out the text, typeface feel, colour and animation, and forbid other text.

---

## 9. Rebuilding the original MiniMax H3 workflow

How the old ~40-node graph maps onto this pack:

| Old nodes | New |
|---|---|
| UNETLoader, CLIPLoader, 2× VAELoader, ModelSamplingMiniMaxH3, MiniMaxH3ScheduledSolAttentionPatch, PathchSageAttentionKJ | **Hawk H3 Model Loader** |
| LTX_lora_loader | **Hawk H3 LoRA Stack** (pick each LoRA in a slot, chain nodes for more), or paste the old loader's stack JSON into the Model Loader's `lora_stack` |
| LoadImage ×N, DenoMultiImageLoader, Any Switch, the ref inputs of MiniMaxH3ReferenceToVideo | **Hawk H3 References** (keep the loaders; connect their images into `pictures`) |
| LoadVideoUI `images` / `audio` | References `videos` + `audios` (as before, audio as a standalone clip) |
| LoadAudio | References `audios` |
| HawkAtlasLLM, StringConcatenate, PrimitiveStringMultiline, LLM Prompt Refiner Switch, PreviewAny | **Hawk H3 Story Planner** with `segment_count: 1`, or type the prompt in the Director's `script` |
| MiniMaxH3ReferenceToVideo, RandomNoise, KSamplerSelect, BasicScheduler, BasicGuider, SamplerCustomAdvanced, VAEDecode, VAEDecodeAudio, CreateVideo, Float (Duration), Math Expression, Resolution Selector | **Hawk H3 Director** (`default_seconds`, `aspect_ratio`, `megapixels`, `steps`, `sampler_name`, `scheduler`) |
| HawkAtlasI2I | Unchanged: keep it before the References node to prepare pictures |
| SaveVideo | Unchanged |

Old settings, new names:

| Old | New |
|---|---|
| Float (Duration) = 15 | `default_seconds: 15` |
| Resolution Selector 1:1, 0.8 MP | `aspect_ratio: 1:1`, `megapixels: 0.8` (928×928) |
| BasicScheduler simple, 8 steps | `scheduler: simple`, `steps: 8` |
| KSamplerSelect res_multistep | `sampler_name: res_multistep` |
| Sigma shift 12 / 3 | `shift_video: 12`, `shift_audio: 3` |
| Sol tau 1.25 → 0.8 | `attention: sol scheduled + sage`, `sol_tau_start: 1.25`, `sol_tau_end: 0.8` |
| LLM system prompt | Built in (a multi-segment version); or paste yours into the Planner's `system_prompt` and keep its JSON output section |

---

## 10. Hit specific body poses

Ready-made: [`example_workflows/05_pose_guided_sequence.json`](../example_workflows/05_pose_guided_sequence.json).

**Graph:** character picture → References `picture_0`; one image per key pose → References `poses` (`pose_0`, `pose_1`, …).

To make skeleton pose images from photos, run the photos through an OpenPose or DWPose preprocessor (e.g. from comfyui_controlnet_aux) and connect its output to `poses`. Plain photos of someone in the pose also work.

References `labels`:
```
Picture 1: the fighter's face and red gi
Pose 1: guard stance, fists up
Pose 2: high front kick
```

```
style: Martial-arts film, dojo at dusk, warm side light, slow-motion accents. Native audio.
---
title: Guard
duration: 5
<Picture 1> defines the fighter's face, topknot and red gi.
He bows, steps back and settles into the stance from <Pose 1>. Static medium-wide shot.
Sound: wooden floor creak, a sharp exhale.
---
title: Kick
duration: 6
<Picture 1> defines the fighter's face and red gi.
He explodes forward and freezes at the top of the kick from <Pose 2>, then lands softly. The camera pushes in during the kick.
Sound: gi snap, whoosh, foot landing on wood.
```

Notes:
- Mentioning `<Pose 2>` is what sends pose 2 to that segment. Segments without a pose mention send none.
- Check the Director's `prompts` output: poses appear as `<Picture 2>` there, followed by the pose-only instruction.
- If the pose image's clothes or face leak into the result, switch to a skeleton render, or strengthen `pose_instruction` on the References node.
