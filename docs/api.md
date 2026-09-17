# API: plan and render from your chats or scripts

The `hawk_api` gateway turns the long-video + LLM pipeline into a web service on your GPU pod. You can drive it:

- **From Claude or Grok chats**, through MCP: "plan a 3-segment film from my references, then render a preview".
- **From scripts**, through REST: upload files, start a plan or render job, poll it, download the MP4.

```
Claude / Grok chat ──MCP──┐
curl / scripts ──REST─────┼─► hawk_api (port 8000, token auth, job database)
Upload page (browser) ────┘            │ localhost only
                                       ▼
                         ComfyUI 127.0.0.1:8188 + Hawk H3 nodes + models
```

There is also **Hawk H3 Studio**, a web page at `<base URL>/t/<token>/studio`. Upload images, pick LoRAs, set duration and resolution, write a prompt (or let the AI planner write segments), then watch progress and play the finished videos. It uses the same REST API. Open `/studio` without the token in the path and it asks for the token once, keeping it in that browser.

The gateway builds the same graph as the example workflows (Model Loader → LoRA Stack → References → Story Planner → Director). All rendering happens in your ComfyUI; the gateway only wires, queues and tracks jobs.

- [1. Set up the pod](#1-set-up-the-pod)
- [2. Choose LoRAs (loras.json)](#2-choose-loras-lorasjson)
- [3. Connect Claude](#3-connect-claude)
- [4. Connect Grok](#4-connect-grok)
- [5. Using it from a chat](#5-using-it-from-a-chat)
- [6. REST reference](#6-rest-reference)
- [Agent: an autonomous video director](#agent-an-autonomous-video-director)
- [7. MCP tools](#7-mcp-tools)
- [8. Jobs, progress and resume](#8-jobs-progress-and-resume)
- [9. Security](#9-security)
- [10. Troubleshooting](#10-troubleshooting)

---

## 1. Set up the pod

> **Using Google Colab?** Skip this section and use the notebook instead: [Running the API on Google Colab](colab.md). The rest of this page (LoRAs, connecting chats, REST and MCP) applies to both.

Works on any GPU pod that keeps running (RunPod, Vast, Lambda…). The examples use RunPod paths.

1. **ComfyUI with the pack and models.** Install ComfyUI in `/workspace/ComfyUI`, clone this repo into `custom_nodes/`, and download the H3 models and the turbo LoRA ([Getting started](getting-started.md)). Check that a workflow renders in the ComfyUI UI once before adding the API.
2. **Expose one HTTP port: 8000.** In the RunPod pod settings, add `8000` to *Expose HTTP Ports*. Your public URL becomes `https://<pod-id>-8000.proxy.runpod.net`. **Do not expose 8188**: ComfyUI has no authentication.
3. **Set environment variables** (pod template env, or export in a terminal):

   | Variable | Required | Meaning |
   |---|---|---|
   | `HAWK_API_TOKEN` | yes | Your secret. At least 16 characters: `openssl rand -hex 24` |
   | `ATLAS_API_KEY` | for planning | Atlas Cloud key, read by the Story Planner node inside ComfyUI |
   | `PUBLIC_BASE_URL` | yes | `https://<pod-id>-8000.proxy.runpod.net`, used in download and upload links |
   | `DATA_DIR` | no | Job database and `loras.json`. Default `/workspace/hawk_api_data` (keep it on the persistent volume) |
   | `COMFY_DIR` | no | Default `/workspace/ComfyUI` |
   | `MAX_UPLOAD_MB` | no | Largest reference file, default 2048 |
   | `HAWK_UNET`, `HAWK_CLIP`, `HAWK_VIDEO_VAE`, `HAWK_AUDIO_VAE` | no | Model file names, if yours differ from the defaults |
   | `HAWK_ATTENTION` | no | `sol scheduled + sage` (default), `sage`, `sol scheduled`, `comfy default` |
   | `HAWK_PLANNER_MODEL` | no | Atlas chat model for planning, default `xai/grok-4.3` |

4. **Start everything:**
   ```bash
   bash /workspace/ComfyUI/custom_nodes/Hawk-Minimax-H3-Directory/deploy/start_pod.sh
   ```
   The script installs `requirements-api.txt`, starts ComfyUI on `127.0.0.1:8188` with a 2 GB upload limit, waits until it's up, then starts the API on port 8000. ComfyUI's log is `$DATA_DIR/comfyui.log`.

5. **Check it:**
   ```bash
   curl https://<pod-id>-8000.proxy.runpod.net/healthz
   # {"ok": true, "comfy": "ok", "loras": "ok"}
   ```
   `"loras": "degraded"` means a required default LoRA is missing; see the next section.

The API schema is browsable at `https://<pod-id>-8000.proxy.runpod.net/docs` (no token needed to read it; every call still needs the token, so use curl or a client with the header).

---

## 2. Choose LoRAs (loras.json)

Every render gets the LoRAs from `$DATA_DIR/loras.json`, created on first start from [`deploy/loras.example.json`](../deploy/loras.example.json). Edit it any time; changes apply to the next request.

```json
{
  "defaults": [
    {"name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength": 1.0, "required": true, "turbo": true}
  ],
  "presets": {
    "realism": [{"name": "h3-realism-people", "strength": 0.8}]
  }
}
```

- **`defaults`** are added to every render. **`required: true`** means renders are refused while that file is missing from `ComfyUI/models/loras`, instead of quietly rendering without it.
- **`presets`** are named extra sets a request picks with `lora_preset`.
- **`turbo: true`** (or `turbo` in the file name) makes renders default to **8 steps**; without a turbo LoRA they default to **30**. A request's `steps` always wins.
- **`name`** can be the exact path in `models/loras`, just the file name, or a unique part of it (`"turbo"`).

On every render the gateway asks ComfyUI which files are actually in `models/loras` (`GET /models/loras`, so new files count without a restart):

| Situation | Result |
|---|---|
| A requested name matches one file | Used; the job lists the exact file under `loras` |
| No match, or several matches | **422** before anything is queued, with the closest file names |
| A required default is missing | **422** "Required default LoRA … is missing on the pod"; `/healthz` says `degraded` |
| A non-required default is missing | Skipped, with a warning on the job |

Per request, in `settings`: `"loras": [{"name": "singularity", "strength": 0.7}]` adds LoRAs, `"lora_preset": "realism"` adds a preset, `"use_default_loras": false` skips the defaults, and a strength of `0` removes one (e.g. `{"name": "turbo", "strength": 0}`).

After the render, the job's `loras_applied` lists what the LoRA Stack nodes actually reported applying. If that differs from the request, the job gets a warning.

To see what's on the pod: `GET /v1/options` (REST) or `list_options` (MCP) returns `available_loras`, `default_loras` (with `present: true/false`) and `lora_presets`.

---

## 3. Connect Claude

### claude.ai (web, desktop and mobile apps)

1. **Settings → Connectors → Add custom connector.**
2. Name: `Hawk H3`. URL:
   ```
   https://<pod-id>-8000.proxy.runpod.net/t/<HAWK_API_TOKEN>/mcp
   ```
   The token inside the URL is the authentication, because custom connectors can't send your own headers. **Treat this URL like a password.**
3. Leave OAuth fields empty and save. In a chat, enable the connector from the tools menu.

### Claude Code / Claude Desktop config

These can send a header, so the token stays out of the URL:
```bash
claude mcp add --transport http hawk-h3 https://<pod-id>-8000.proxy.runpod.net/mcp \
  --header "Authorization: Bearer $HAWK_API_TOKEN"
```

---

## 4. Connect Grok

**Grok app (grok.com):** open **Connectors → New Connector → Custom** and enter the MCP URL. If the form offers a header or API-key field, use `https://<pod-id>-8000.proxy.runpod.net/mcp` with `Authorization: Bearer <token>`. Otherwise use the token URL form, `…/t/<token>/mcp`. See [xAI connectors](https://docs.x.ai/grok/connectors).

**xAI API (your own scripts):** add the server to the request's tools as a remote MCP tool. Its server URL is `…/t/<token>/mcp`, or `…/mcp` together with xAI's authorization field set to `Bearer <token>`. See [xAI remote MCP tools](https://docs.x.ai/developers/tools/remote-mcp).

---

## 5. Using it from a chat

The connector tells the assistant how the pipeline works. Plain requests are enough:

> **You:** I want to make a short film. Give me the upload link.
>
> **Assistant:** *(calls `upload_page_link`)* Open this link and upload your files…
>
> **You:** Uploaded: a1b2c3 is her face, d4e5f6 is the red coat, 778899 is a waving pose, 101112 is her voice. Plan a 3-segment rainy-city reunion, 10 s each, 16:9.
>
> **Assistant:** *(calls `plan_film`, then `get_job` until done)* Here's the plan… want changes?
>
> **You:** Make segment 2 shorter and render a cheap preview.
>
> **Assistant:** *(edits the script, calls `render_film` with `megapixels: 0.4`, polls `get_job`)* Segment 2/3 done… Finished: *(download link)*

Tips:
- **Ask for a preview first** (`megapixels 0.4`), then "render the final at 0.98 with the same script".
- **Long renders take minutes to hours.** The assistant should poll `get_job` rather than wait; if a chat times out, ask it later to "check job <id>".
- **Upload once, reuse the ids.** Assets stay on the pod, and `list_references` shows them.

---

## 6. REST reference

All endpoints need `Authorization: Bearer <token>` (or the `/t/<token>/` prefix), except `/healthz` and signed links.

```bash
API=https://<pod-id>-8000.proxy.runpod.net
AUTH="Authorization: Bearer $HAWK_API_TOKEN"
```

### Upload references

```bash
curl -H "$AUTH" -F files=@face.png -F files=@voice.mp3 "$API/v1/assets"
# {"assets": [{"id": "a1b2c3d4e5f6", "kind": "image", "filename": "face.png", ...}, ...]}

curl -H "$AUTH" -H 'content-type: application/json' \
     -d '{"url": "https://example.com/pose.png"}' "$API/v1/assets/from-url"
```

`GET /v1/assets` lists uploads. The browser upload page is `GET /upload` (via a signed link from `upload_page_link`, or `/t/<token>/upload`).

### References in requests

```json
"references": [
  {"asset_id": "a1b2c3d4e5f6", "role": "picture", "label": "her face"},
  {"asset_id": "b2c3d4e5f6a1", "role": "pose",    "label": "waving"},
  {"asset_id": "c3d4e5f6a1b2", "role": "video",   "label": "walking motion"},
  {"asset_id": "d4e5f6a1b2c3", "role": "audio",   "label": "her voice"},
  {"asset_id": "e5f6a1b2c3d4", "role": "video_soundtrack", "for_video": 1}
]
```

| role | asset kinds | script tag |
|---|---|---|
| `picture` | image | `<Picture N>`: identity, outfit, product, place |
| `pose` | image | `<Pose N>`: body pose only |
| `video` | video | `<Video N>`: motion, timing, camera |
| `audio` | audio, or video (uses its sound) | `<Audio N>`: voice, music, sound |
| `video_soundtrack` | audio or video | none; belongs to video `for_video` |

N counts each role in list order. Limits are 9 pictures, 9 poses, 3 videos and 3 audio; a segment sends at most 9 images.

### Plan (LLM writes the script)

```bash
curl -H "$AUTH" -H 'content-type: application/json' "$API/v1/plans" -d '{
  "story": "A woman arrives in a rainy city to meet an old friend. Warm, bittersweet.",
  "references": [{"asset_id": "a1b2c3d4e5f6", "role": "picture", "label": "her face"}],
  "segment_count": 3, "segment_seconds": 10, "aspect_ratio": "16:9"
}'
# 202 {"id": "…", "kind": "plan", "status": "planning", ...}
```

Poll `GET /v1/jobs/<id>` until `status` is `done`; `script` holds the plan (JSON text).

### Render

Give exactly **one** script source:

```bash
# A. Render a finished plan as-is (its references are reused when you omit them)
curl -H "$AUTH" -H 'content-type: application/json' "$API/v1/videos" \
  -d '{"plan_job_id": "<plan id>", "settings": {"megapixels": 0.4}}'

# B. Render an edited or hand-written script
curl -H "$AUTH" -H 'content-type: application/json' "$API/v1/videos" -d '{
  "references": [{"asset_id": "a1b2c3d4e5f6", "role": "picture"}],
  "script": "duration: 6\n<Picture 1> smiles at the camera in soft window light. Slow push-in.",
  "settings": {"aspect_ratio": "9:16", "loras": [{"name": "realism", "strength": 0.7}]}
}'

# C. Plan and render in one job
curl -H "$AUTH" -H 'content-type: application/json' "$API/v1/videos" -d '{
  "references": [{"asset_id": "a1b2c3d4e5f6", "role": "picture"}],
  "story": {"story": "A dancer rehearses alone at night.", "segment_count": 2, "segment_seconds": 8}
}'
```

`settings` (all optional):

| Field | Default | Notes |
|---|---|---|
| `aspect_ratio` | `16:9` | `16:9 9:16 1:1 4:3 3:4 21:9 9:21 "match first picture"` |
| `megapixels` | 0.98 | 0.4 for previews |
| `default_seconds` | 10 | Segments without `duration:` |
| `steps` | 8 with turbo LoRA, else 30 | |
| `sampler_name`, `scheduler` | `res_multistep`, `simple` | |
| `seed` | random, then fixed for the job | Same seed + same script = resumable |
| `continuity` | `tail_22` | `off last_frame tail_5 tail_22 tail_39` |
| `carry_audio` | true | |
| `ref_image_size` | `match` | `max` for stronger identity (slower) |
| `interpolation` | `off` | `48 fps (RIFE)`, `60 fps (RIFE)`; needs ComfyUI-VFI |
| `audio_crossfade_ms` | 60 | |
| `music_asset_id` | none | **Music bed**: an uploaded audio asset mixed under the whole film (looped or trimmed, faded out). One continuous track instead of H3 composing new music per segment. Don't also list it in `references` |
| `music_volume_db`, `scene_volume_db` | -3, 0 | Levels of the music bed and of the rendered sound |
| `music_fade_seconds` | 2 | Fade-out at the end |
| `mute_generated_music` | true | Sets every segment's music to N/A so only the music bed plays |
| `loras`, `lora_preset`, `use_default_loras` | | See [LoRAs](#2-choose-loras-lorasjson) |
| `attention` | server setting | |
| `unet_name` | server's model (`HAWK_UNET`) | A ref2va file from `/v1/options` → `diffusion_models`, by name or a unique part (`"bf16"`). bf16 = best quality, slowest; int8/fp8 faster. fl2va models are refused |
| `clip_name` | server's encoder (`HAWK_CLIP`) | A Qwen3-VL file from `text_encoders`, e.g. `"int8"` or `"bf16"` |

Script format and tag rules: [Writing scripts](scripts.md).

### Jobs and downloads

| Method | Path | |
|---|---|---|
| GET | `/v1/jobs/<id>` | Status, progress, script, LoRAs, links |
| GET | `/v1/jobs` | Recent jobs |
| POST | `/v1/jobs/<id>/cancel` | Stop a queued or running job |
| POST | `/v1/jobs/<id>/retry` | Resume a failed or cancelled job |
| GET | `/v1/jobs/<id>/video` | Final MP4 |
| GET | `/v1/jobs/<id>/segments/<n>` | One segment's MP4 |
| GET | `/v1/options` | Base models, text encoders, LoRA files, defaults, presets, samplers… |
| GET | `/healthz` | No auth: ComfyUI and LoRA status |

A finished render's job:

```json
{
  "id": "5c0e…", "kind": "render", "status": "done",
  "progress": {"segments_done": 3, "segments_total": 3},
  "script": "…",
  "steps": 8, "steps_reason": "8 steps because turbo LoRA … is applied",
  "loras": [{"requested": "minimax_h3_ref2v_turbo…", "file": "minimax_h3_ref2v_turbo….safetensors", "strength": 1.0, "turbo": true, "source": "default"}],
  "loras_applied": [{"file": "minimax_h3_ref2v_turbo….safetensors", "strength": 1.0}],
  "seed": 81234567, "run_name": "api_5c0e1b2a3d4f5061",
  "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
  "queue_position": null,
  "final_prompts": "### Segment 1 …  the exact text MiniMax H3 encoded: <Pose N> rewritten to <Picture k>, style and pose notes inserted",
  "video_url": "https://…/v1/jobs/5c0e…/video?exp=…&sig=…",
  "segment_urls": ["https://…/segments/1?exp=…&sig=…", "…"],
  "warnings": [], "error": null, "resumable": false
}
```

`video_url` and `segment_urls` are **signed links**: they open in any browser without the token and expire after 7 days (`LINK_TTL_SECONDS`). Fetch the job again for fresh ones.

Errors are JSON `{"error": "…", "details": {…}}` with status 401 (token), 404 (unknown job), 409 (wrong state, e.g. cancelling a finished job), 422 (bad request: script, references, LoRAs) or 503 (ComfyUI unreachable).

---

## Agent: an autonomous video director

Studio's **Agent** page (and the `/v1/agent` API) runs a chat model that does whole video tasks by itself: it reads the options, plans, renders, waits for the render, retries failures and hands you the video link.

- **Models:** any chat model on your Atlas account. The picker lists Grok 4.6 (default, `HAWK_AGENT_MODEL`), Grok 4.5, Grok 4.3 first. `GET /v1/agent/models` returns the list with vision support and prices.
- **Persona:** free text per chat ("a Bollywood ad-film director who loves warm colours"). You can also tell the agent "be a …" in the chat. Safety rules are fixed and not changed by the persona: no sexual content involving anyone who appears under 18, and no sexual or nude content of real, identifiable people.
- **Tools:** exactly the MCP tools of this server, listed and called **in-process**, so the agent never goes through the tunnel and gets new tools automatically. It also has `wait_for_job` (waits on the server without spending tokens), `set_persona` and `rename_chat`.
- **History:** every chat is stored in `DATA_DIR/jobs.sqlite3`. The last 30 messages are sent to the model verbatim; very long chats are summarised automatically, and the full history stays in the database.
- **Runs on the server:** a message starts a background run that continues if you close the browser. Limits per message: 40 model steps and 3 hours. **Stop** ends it after the current step. A server restart marks a running chat as interrupted.
- **Protocol:** Atlas does not advertise tool calling for Grok, so the model answers every turn with JSON: `{"say": "…", "actions": [{"tool": "…", "args": {…}}], "done": false}`. Replies in any other shape get one repair round.
- **Cost:** each chat shows tokens used and the estimated cost from Atlas prices.

| Method | Path | |
|---|---|---|
| GET | `/v1/agent/models` | Chat models on the Atlas account |
| POST | `/v1/agent/sessions` | `{title?, persona?, model?}` → new chat |
| GET | `/v1/agent/sessions` | Chats, most recent first |
| GET | `/v1/agent/sessions/<id>?after=<message id>` | The chat and its messages after that id |
| PATCH | `/v1/agent/sessions/<id>` | Change `title`, `persona` or `model` |
| POST | `/v1/agent/sessions/<id>/messages` | `{text, attachments: [asset_id]}` → 202, the agent starts working; 409 while it is still working |
| POST | `/v1/agent/sessions/<id>/stop` | Stop after the current step |
| DELETE | `/v1/agent/sessions/<id>` | Delete the chat |

### Images

`POST /v1/images` (and the MCP / agent tool `generate_image`) creates images with Atlas Cloud's `generateImage` API: `{prompt, reference_asset_ids?, model?, size?, n (1-4), seed?}`. Without references it uses `bytedance/seedream-v5.0-pro/text-to-image`; with references it switches to `bytedance/seedream-v5.0-pro/edit` and sends those images, so you can keep a face and change the outfit, scene or style. Results are stored as image assets and can be used right away as `picture` references in plans and renders.

Every asset in API responses carries a signed `file_url`, and images a `thumb_url` (a 320 px JPEG, via `GET /v1/assets/<id>/file?w=320`). Signed links open without the token, so Studio and the agent chat show thumbnails of uploaded and generated images.

The API process needs `ATLAS_API_KEY` (the Colab launcher passes it already). The **planner model** is also choosable: `/v1/options` returns `planner_models`, and `/v1/plans`, `story` and MCP `plan_film` take `model`.

---

## 7. MCP tools

| Tool | What it does |
|---|---|
| `upload_page_link` | Signed link to the browser upload page |
| `add_reference_from_url` | Fetch a public file URL onto the pod |
| `list_references` | Uploaded assets |
| `generate_image` | Generate or edit images with Atlas (Seedream v5.0 Pro by default); results become image assets |
| `list_options` | Base models, text encoders and LoRAs on the pod, defaults, presets, samplers, aspect ratios |
| `plan_film` | Start a plan job |
| `render_film` | Start a render: `script`, `plan_job_id` or `story`, plus `references`, `settings`, `loras`, `lora_preset` |
| `get_job` | Status, progress, script, video link |
| `list_jobs` | Recent jobs |
| `cancel_job` / `retry_job` | Stop / resume |

The server's instructions teach the assistant the flow, the reference roles and the script rules. Tools return immediately; rendering happens in the background.

---

## 8. Jobs, progress and resume

**Statuses:** `queued` → `planning` (LLM step, if any) → `rendering` → `done`, or `failed` / `cancelled`.

**Progress** has two levels:
- `segments_done / segments_total` and `current_segment` (its title) update when a segment finishes. A 3-segment film shows 0/3, then 1/3…
- `steps_done / steps_total` is the sampler inside the current segment (e.g. 5/8), so a long segment still shows movement. It restarts for each segment.

**Resume:** each render keeps one `run_name` and one `seed`. The Director saves every finished segment under `ComfyUI/output/hawk_h3/<run_name>/`. `retry` resubmits the same graph, and the Director reuses finished segments. For one-call jobs whose plan had already arrived, it re-renders that exact plan instead of asking the LLM again.

**Restarts:** ComfyUI keeps its queue in memory. If ComfyUI or the pod restarts mid-render, the gateway notices within about a minute and marks the job `failed` with `resumable: true` and a message saying so. Call `retry` once ComfyUI is back. The job database survives in `DATA_DIR`.

One render runs at a time. Further jobs wait in ComfyUI's queue with status `queued` and `queue_position` (1 = next after the current render), and switch to `planning` or `rendering` when ComfyUI starts them.

**Out of GPU memory:** the Director retries a step once after unloading every cached model, which covers ComfyUI underestimating what a segment needs. If it runs out again, the job fails with a message naming the step. Lower `megapixels` or the segment length, use fewer LoRAs, or pick a smaller `unet_name`/`clip_name`, then `retry`: finished segments are reused.

**Planner slips:** if the LLM lists a reference number that doesn't exist (for example `poses: [4]` with no poses) and never mentions it in the prompt, the number is dropped and a warning is added to the job instead of failing the plan.

---

## 9. Security

- **Only port 8000 is public.** ComfyUI listens on 127.0.0.1.
- **The token** is required for every request except `/healthz` and signed links. Rotate it by changing `HAWK_API_TOKEN` and restarting. Old signed links and connector URLs stop working, so re-add the connector.
- **Token URLs** (`/t/<token>/mcp`) put the secret in the URL. Use them only in connector settings, never in shared chats or screenshots. Prefer the header form wherever a client supports it.
- **Signed links** grant access to one file (or the upload page) until they expire. Anyone with the link can download or upload.
- **The Atlas key** stays in ComfyUI's environment and is never written into graphs, jobs or responses.
- **Uploads** are limited to image, audio and video types and `MAX_UPLOAD_MB`, and stored under `ComfyUI/input/hawk_api/<asset id>/`.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `401 Missing or invalid token` | Check the `Authorization: Bearer` header or the `/t/<token>/` URL. Signed links expire; fetch the job again. |
| `/healthz` says `"comfy": "unreachable"` | ComfyUI crashed or is still loading. Check `$DATA_DIR/comfyui.log`. |
| `"loras": "degraded"` / render refused for a required LoRA | Put the file in `ComfyUI/models/loras/`, or fix its `name` in `loras.json`. |
| `422 LoRA 'x' is not in ComfyUI's models/loras. Closest files: …` | Use one of the suggested names; `list_options` shows all files. |
| `422 Script problem: Segment 2 mentions <Picture 3> but only 2 picture(s) are connected` | Tags count per role in the order of `references`. |
| `422 ComfyUI rejected the graph: …` | A node or model is missing on the pod: update the pack in `custom_nodes`, check model file names and `HAWK_*` variables. |
| Job `failed`, `resumable: true`, "ComfyUI no longer knows this job" | ComfyUI restarted. Call retry. |
| Job `failed` with a node error (e.g. out of memory) | Lower `megapixels`, fewer references per segment, then retry. |
| Planning fails with an Atlas error | `ATLAS_API_KEY` must be set in the environment that starts ComfyUI. |
| Upload fails for a large video | Raise `MAX_UPLOAD_MB` (it also sets ComfyUI's `--max-upload-size`) and restart. |
| Connector form asks for OAuth credentials | Cancel. This API uses no OAuth: add the connector with the `…/t/<token>/mcp` URL (or `…/mcp` plus a Bearer header where the form has a header field) |
| Chat assistant doesn't see the tools | Enable the connector in that chat; for claude.ai check the URL includes `/t/<token>/mcp`. |

---

## Development

```bash
pip install -r requirements-api.txt aiohttp
python -m unittest discover -s tests_api      # graph, LoRA, auth, and an end-to-end run against a fake ComfyUI
```

The end-to-end test starts the real gateway and a fake ComfyUI that speaks ComfyUI's HTTP and websocket formats. It exercises uploads, plan → render, progress, LoRA checks, signed downloads, a ComfyUI restart with retry, cancel, and the MCP tools through the MCP client. It doesn't replace a real render on the pod.
