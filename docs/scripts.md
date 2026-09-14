# Writing scripts

A script tells the Director what to render: a list of segments, each with a prompt and optional settings. You can type it into the Director's `script` box or have the [Story Planner](nodes.md#hawk-h3-story-planner-atlas-llm) write it.

- [Plain-text format](#plain-text-format)
- [JSON format](#json-format)
- [Reference tags](#reference-tags)
- [Choosing references per segment](#choosing-references-per-segment)
- [Pose references](#pose-references)
- [Writing prompts H3 follows](#writing-prompts-h3-follows)
- [Checklist before a long render](#checklist-before-a-long-render)

---

## Plain-text format

- Separate segments with a line containing only `---`.
- A segment can start with **header lines** (`name: value`). The first line that isn't a header starts the prompt; everything after it is the prompt, including line breaks.
- A block that starts with `style:` is the **style block**. Its text is placed in front of every segment's prompt.

```
style: Cinematic live-action, warm sunset light, 50mm lens, gentle film grain. Native audio. No subtitles.
---
title: Rooftop
duration: 8
pictures: 1, 2
<Picture 1> defines his face, short black hair and beard. <Picture 2> defines the rooftop garden.
He waters the plants, then turns toward the skyline. Slow arc left around him.
Sound: soft wind, water trickling, distant traffic. Music N/A.
---
title: Visitor
duration: 10
He hears the door, looks back and smiles: "You made it." Camera holds a medium shot.
Sound: metal door creak, his voice warm and close.
```

A script with no `---` and no headers is a single segment. Plain prompt text is a valid script.

### Headers

| Header | Values | Default | Meaning |
|---|---|---|---|
| `title:` | any text | none | Name shown in logs, the plan and the `info` report. |
| `duration:` | seconds: `8`, `8s`, `7.5` | Director `default_seconds` | Segment length. Rounds up to H3's frame grid; 15 s max. |
| `pictures:` | `1, 3` · `all` · `none` | `all` | Which reference pictures this segment sends. |
| `videos:` | same | `all` | Which reference videos. |
| `audios:` | same | `all` | Which standalone audio clips. |
| `poses:` | same | poses mentioned in the prompt | Which pose references to send. Normally leave it out: mentioning `<Pose 2>` sends pose 2. |
| `continuity:` | `off` · `last_frame` · `tail_5` · `tail_22` · `tail_39` | Director `continuity` | How this segment starts from the previous one. Ignored on the first segment. Aliases: `cut`, `hard cut`, `none` → `off`; `tail` → `tail_22`. |
| `seed:` | whole number | base seed (+ segment number) | Pin this segment's seed. |

`images:` works as an alias for `pictures:`, and `seconds:` for `duration:`. Header names ignore case.

A header only counts at the **top** of a block. A line like `Title: The Return` written in the middle of a prompt stays part of the prompt.

---

## JSON format

The Story Planner outputs this, and you can write it by hand:

```json
{
  "style": "Cinematic live-action, warm sunset light, 50mm lens. Native audio. No subtitles.",
  "segments": [
    {
      "title": "Rooftop",
      "duration": 8,
      "pictures": [1, 2],
      "videos": [],
      "audios": [],
      "continuity": "inherit",
      "seed": null,
      "prompt": "<Picture 1> defines his face... "
    },
    {
      "title": "Visitor",
      "duration": 10,
      "prompt": "He hears the door..."
    }
  ]
}
```

- Only `prompt` is required per segment. Missing fields use the same defaults as the plain-text headers.
- `null` for `pictures`/`videos`/`audios` means **all**; `[]` means **none**.
- `"continuity": "inherit"` (or omitting it) uses the Director setting.
- A bare string works as a segment: `{"segments": ["prompt one", "prompt two"]}`.
- Code fences (```` ```json ````) and text around the JSON are ignored, so you can paste an LLM reply directly.

---

## Reference tags

Tags point a prompt at a reference. Numbering follows the order references were connected to **Hawk H3 References** (see its `tag_map` output):

| Tag | Refers to |
|---|---|
| `<Picture 1>` … `<Picture 9>` | pictures |
| `<Video 1>` … `<Video 3>` | videos |
| `<Audio 1>` … `<Audio 3>` | standalone audio clips (the `audios` input) |
| `<Pose 1>` … `<Pose 9>` | pose references (the `poses` input) |

**Always use this global numbering**, even in segments that only use some references. The Director converts tags into what H3 needs for each segment (next section).

### Forms that are recognised

These are all converted to the proper tag:

| You write | Becomes |
|---|---|
| `<Picture 2>`, `<picture2>`, `<image_2>` | `<Picture 2>` |
| `@image2`, `@picture 2` | `<Picture 2>` |
| `Image 2`, `picture 2` | `<Picture 2>` |
| `Video 1`, `@video1` | `<Video 1>` |
| `Audio 1`, `@audio1` | `<Audio 1>` |
| `Pose 1`, `@pose1`, `<pose_1>` | `<Pose 1>` |

The plain-word form catches ordinary phrases too: "a video 2 minutes long" becomes `<Video 2>`. Avoid putting a number straight after the words *picture, image, pose, video* or *audio* in normal prose.

### Mistakes are caught before rendering

The whole script is checked before the first segment renders. These stop the run immediately with a clear message:

- a tag for a reference that isn't connected (`<Picture 4>` with 3 pictures)
- a tag for a reference the segment left out (`pictures: 1` but the prompt mentions `<Picture 2>`)
- a `pictures:` / `videos:` / `audios:` number that isn't connected
- a bad duration, seed or continuity value

---

## Choosing references per segment

By default every segment sends every reference. Listing only what a segment needs is usually better:

- **Faster:** every reference picture adds tokens that are processed on *every* sampling step.
- **Less drift:** H3 isn't distracted by a location photo in a close-up that doesn't need it.

```
---
title: Close-up
pictures: 1
<Picture 1> defines her face. Extreme close-up as she reads the letter...
---
title: Wide
pictures: 1, 3
<Picture 1> defines her face. <Picture 3> defines the lighthouse and the cliff...
```

### How renumbering works

H3 only sees the references a segment sends, numbered from 1. With `pictures: 1, 3`, H3 receives two pictures, so the Director rewrites:

```
you write:      <Picture 3> stands beside <Picture 1>
H3 receives:    <Picture 2> stands beside <Picture 1>
```

You never do this yourself; the Director's `prompts` output shows the rewritten text.

### Audio numbering with video soundtracks

Inside H3, a video's soundtrack (connected to `video_soundtracks`) also takes an audio number, ahead of the standalone clips. The Director accounts for this. With `<Video 1>` carrying a soundtrack, your `<Audio 1>` is sent to H3 as `<Audio 2>`. Keep writing the global numbers.

---

## Pose references

Pose images (connected to the References node's `poses` input) tell H3 what body pose to hit, without borrowing the face or clothes in them. Mention a pose at the moment it should happen:

```
style: Contemporary dance film, white studio, soft top light. Native audio.
---
title: Rise
duration: 6
<Picture 1> defines the dancer's face and black leotard.
She slowly lifts both arms and ends in the pose from <Pose 1>. Slow push-in.
---
title: Lunge
duration: 6
<Picture 1> defines the dancer's face and leotard.
She flows down into the pose from <Pose 2> and holds it for a beat.
```

What happens per segment:

- **Only mentioned poses are sent.** Segment "Rise" sends pose 1, segment "Lunge" sends pose 2. `poses: 1, 2` overrides this; `poses: none` sends none. An explicit list without a mention still sends the pose, but H3 won't know when to use it.
- **Poses become pictures.** H3 only knows pictures, so poses are sent after the segment's pictures and renumbered. With `<Picture 1>` in the segment, `<Pose 1>` reaches H3 as `<Picture 2>`.
- **A pose-only instruction is appended** (the References node's `pose_instruction`), so H3 doesn't copy the pose image's identity, clothing or style.
- **Images are limited to 9 per segment**, pictures and poses combined.

The Director's `prompts` output shows the final text, which is the quickest way to check what H3 was told.

Tips:
- Write the pose as the **end state of a movement** ("ends in", "lands in", "freezes in") rather than a static description.
- Put a pose at the **end of a segment** when the next segment should start from it.
- Use one pose per beat. Several poses in a short segment make H3 rush or skip some.

---

## Writing prompts H3 follows

H3 follows a prompt written like a director's brief much better than a list of keywords. For each segment, in this order:

1. **Reference jobs.** Give every reference a narrow role.
   `<Picture 1> defines her face, copper hair and the mole under her left eye. <Picture 2> defines the green bomber jacket. <Audio 1> is her voice: timbre and warm delivery.`
2. **Opening.** What's on screen at the first frame.
3. **Action, in playback order, as visible behaviour.**
   ✗ *She feels confident.* ✓ *She straightens a cuff, looks at the sunrise and walks past camera with a small smile.*
4. **Camera.** One main move per shot: static, push in, pull out, pan, tilt, truck, pedestal, arc, tracking, handheld, over-the-shoulder, overhead. Say what we see after the move.
5. **Dialogue and sound.** Exact words in quotes and who says them; sound effects next to the action that causes them; ambience; `Music N/A` if you want no score.
6. **Ending.** The final pose. In multi-segment films, this is where the next segment picks up.

### Rules of thumb

- **Fit the action to the length.** 5 s ≈ one beat; 8 s ≈ one or two shots; 15 s ≈ a short shot list. Speech runs about 2.5 words per second, so a 6-second shot holds roughly one sentence.
- **Timed shots inside one segment** are fine: `[Shot 1] … [Shot 2] At 00:05, cut to a close-up of the ticket …`. Keep timestamps inside the segment's duration.
- **Name the traits that make someone recognisable** (face shape, hairline, scar, jacket buttons) instead of "keep the same person".
- **Resolve conflicts in words:** "hair from `<Picture 1>`, coat from `<Picture 2>`".
- **Put the shared look in `style:`** once, not in every segment.
- **In continuity segments, don't re-establish.** The segment already starts where the last one ended. Write "She keeps walking…", not "A woman walks into a station…".
- **Voice references:** say which character gets the voice, and write *new* lines for them.
- **Avoid:** keyword soup, unassigned references, paragraphs of dialogue in short segments, conflicting camera directions, and subtitles or on-screen text unless you specify the exact words.
- **Prompts over ~7000 characters** (style included) trigger a warning; H3 is documented up to that length.

---

## Checklist before a long render

- [ ] `tag_map` on the References node matches the numbers in the script.
- [ ] Director `aspect_ratio` matches what the Planner was told.
- [ ] `seed` is **fixed** and `run_name` is new for this project.
- [ ] Render a cheap preview first: `megapixels 0.4` and a different `run_name` (see [Long videos → previews](long-videos.md#cheap-previews-first)).
- [ ] Read the Director's `prompts` output after the first segment if something looks off.
