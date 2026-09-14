# Hawk MiniMax H3 Director — Documentation

Hawk MiniMax H3 Director is a ComfyUI node pack for **MiniMax H3 reference-to-video**. You give it reference pictures, video and audio plus a written script. It renders one clip, or a long film made of many clips that flow into each other, with native audio.

## Start here

| If you want to… | Read |
|---|---|
| Install the pack and render your first clip | [Getting started](getting-started.md) |
| Load a ready-made graph and just press Queue | [Example workflows](../example_workflows) (see [Getting started → example workflows](getting-started.md#example-workflows)) |
| Know what every input and output does | [Node reference](nodes.md) |
| Write scripts, use reference tags, write prompts H3 follows | [Writing scripts](scripts.md) |
| Make long videos: continuity, resume, re-rendering one segment | [Long videos](long-videos.md) |
| Copy a working setup for a common job | [Recipes](recipes.md) |
| Fix an error message | [Troubleshooting](troubleshooting.md) |
| Run it as an API and use it from Claude or Grok chats | [API (REST + MCP)](api.md) |
| Run that API on Google Colab | [API on Google Colab](colab.md) |

## The pack in one picture

```
                         ┌──────────────────────────┐
  pictures ──┐           │                          │
  videos  ───┼─► Hawk H3 References ─refs─┬─────────►│                          │
  audio   ───┘                            │         │                          │
                                          ▼         │     Hawk H3 Director     ├─video──► Save Video
  your brief ──────────► Hawk H3 Story Planner ─script─►│                      │
                         (optional, Atlas LLM)          │                      │
                                                        │                      │
  Hawk H3 Model Loader ─► Hawk H3 LoRA Stack ──pipe────►│                      │
                                                        └──────────────────────┘
```

- **Model Loader** loads the model once.
- **LoRA Stack** adds LoRAs from dropdowns. Chain more of them for more LoRAs. It feeds the Director.
- **References** bundles what the video should look and sound like.
- **Story Planner** (optional) writes the script for you using an LLM.
- **Director** renders the script and outputs one finished video with audio.

## Words used in these docs

| Term | Meaning |
|---|---|
| **Segment** | One H3 generation, 1–15 seconds. A film is a list of segments played back to back. |
| **Script** | The text (or JSON) that lists the segments: prompt, duration, which references each one uses. |
| **Reference** | A picture, video or audio clip that guides the result: a face, an outfit, a location, a motion, a voice. |
| **Tag** | How a prompt points at a reference: `<Picture 1>`, `<Video 1>`, `<Audio 1>`. |
| **Continuity** | Carrying the end of one segment into the start of the next so they join smoothly. |
| **Run** | One folder of saved segments under `output/hawk_h3/<run_name>/`. Re-running the same run resumes it. |
