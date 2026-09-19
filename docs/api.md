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
| GET | `/v1/jobs?view=summary&since=<server_time>` | Recent jobs. `view=summary` drops scripts, prompts and segment links; `since` returns only jobs changed after the `server_time` of your previous call |
| POST | `/v1/jobs/<id>/cancel` | Stop a queued or running job |
| POST | `/v1/jobs/<id>/retry` | Resume a failed or cancelled job |
| GET | `/v1/jobs/<id>/video` | Final MP4, streamed inline with byte ranges (seekable). `?download=1` saves it as a file |
| GET | `/v1/jobs/<id>/segments/<n>` | One segment's MP4 (same options) |
| GET | `/v1/jobs/<id>/thumb?w=640` | JPEG poster frame of a finished render |
| POST | `/v1/jobs/<id>/drive` | Copy a finished render to Google Drive (again) |
| GET / PUT | `/v1/drive/export` | Drive copy settings: `{"enabled": true, "folder": "Hawk H3/Videos", "segments": false}` plus `mounted` |
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
  "title": "Chai ad on a Mumbai rooftop",
  "video_url": "https://…/v1/jobs/5c0e…/video?exp=…&sig=…",
  "download_url": "https://…/v1/jobs/5c0e…/video?exp=…&sig=…&download=1",
  "thumb_url": "https://…/v1/jobs/5c0e…/thumb?exp=…&sig=…&w=640",
  "drive": {"status": "ready", "path": "Hawk H3/Videos/2026-09-17/Chai_ad_on_a_Mumbai_rooftop_5c0e1b2a.mp4", "file_id": "1AbC…",
            "preview_url": "https://drive.google.com/file/d/1AbC…/preview", "view_url": "…/view",
            "download_url": "https://drive.google.com/uc?id=1AbC…&export=download", "thumb_url": "…"},
  "segment_urls": ["https://…/segments/1?exp=…&sig=…", "…"],
  "warnings": [], "error": null, "resumable": false
}
```

`video_url` and `segment_urls` are **signed links**: they open in any browser without the token and expire after 7 days (`LINK_TTL_SECONDS`). Fetch the job again for fresh ones. Links lasting a day or more expire at a whole UTC day, so a file keeps the same URL all day and browsers cache it (`Cache-Control: private, max-age=86400`).

**Fast delivery.** When `COMFY_OUTPUT_DIR` points at ComfyUI's output folder on the same machine, videos are served straight from disk; otherwise they're proxied from ComfyUI with the `Range` header forwarded. JSON responses over 1 KB are gzipped; media never is.

**Google Drive copy.** With Google Drive mounted (`HAWK_DRIVE_ROOT`, default `/content/drive/MyDrive`) and export enabled (the default), every finished render is copied to `<folder>/<date>/<title>_<id>.mp4`. `drive.status` goes `copying` → `syncing` → `ready` once Drive reports the file id (read from the mount's `user.drive.id` attribute), or `copied` if the mount doesn't expose the id, or `failed` with `error`. Drive's `preview_url` and `download_url` stream from Google's servers, not the tunnel, but only for a Google account that can see the file: the Drive owner, or anyone if you share the folder.

Errors are JSON `{"error": "…", "details": {…}}` with status 401 (token), 404 (unknown job), 409 (wrong state, e.g. cancelling a finished job), 422 (bad request: script, references, LoRAs) or 503 (ComfyUI unreachable).

---

## Agent: an autonomous video director

Studio's **Agent** page (and the `/v1/agent` API) runs a chat model that does whole video tasks by itself: it reads the options, plans, renders, waits for the render, retries failures and hands you the video link.

- **Models:** any chat model on your Atlas account. The picker lists Grok 4.6 (default, `HAWK_AGENT_MODEL`), Grok 4.5, Grok 4.3 first. `GET /v1/agent/models` returns the list with vision support and prices.
- **Persona:** free text per chat ("a Bollywood ad-film director who loves warm colours"). You can also tell the agent "be a …" in the chat. Safety rules are fixed and not changed by the persona: no sexual content involving anyone who appears under 18, and no sexual or nude content of real, identifiable people.
- **Tools:** exactly the MCP tools of this server, listed and called **in-process**, so the agent never goes through the tunnel and gets new tools automatically. It also has `wait_for_job` (waits on the server without spending tokens), `set_persona`, `rename_chat` and `describe_tool`. Tools with large argument schemas (`render_film`, `plan_film`) are listed by description only until the chat uses them or calls `describe_tool`. That saves about 1,500 tokens per call in chats that don't make videos.
- **History:** every chat is stored in `DATA_DIR/jobs.sqlite3` and stays there in full. What is sent to the model is kept small in three ways:
  - **Trimming (free):** tool results from before your latest message are cut to what later turns refer to: ids, links, engine, status, scores and errors. Long action arguments such as image prompts are cut to 300 characters. The current turn is sent in full.
  - **Automatic summary:** once the history sent with each call passes about 20,000 tokens (`HAWK_AGENT_COMPACT_TOKENS`), everything but the last 10 messages (`HAWK_AGENT_KEEP_MESSAGES`) is folded into the chat's summary. One call to `HAWK_AGENT_SUMMARY_MODEL` does it (default DeepSeek V4.1 Flash, whatever the chat's model; the chat's model is the fallback).
  - **Compact button** (`POST /v1/agent/sessions/<id>/compact`): summarises all but the last 4 messages now and reports the estimated tokens per call before and after. It returns 409 while the agent is working.
- **Runs on the server:** a message starts a background run that continues if you close the browser. Limits per message: 40 model steps and 3 hours. **Stop** ends it after the current step. A server restart marks a running chat as interrupted.
- **Protocol:** Atlas does not advertise tool calling for Grok, so the model answers every turn with JSON: `{"say": "…", "actions": [{"tool": "…", "args": {…}}], "done": false}`. Replies in any other shape get one repair round.
- **Cost:** each chat shows tokens used and the estimated cost from Atlas prices. Each reply shows its own call's tokens in and out; hover for the cost. Stored assistant messages carry `usage: {model, in, out, cached, cost_usd}`. `cached` is filled when the provider reports prompt-cache hits.

| Method | Path | |
|---|---|---|
| GET | `/v1/agent/models` | Chat models on the Atlas account |
| POST | `/v1/agent/sessions` | `{title?, persona?, model?, name?, avatar_asset_id?}` → new chat |
| GET | `/v1/agent/sessions` | Chats, most recent first |
| GET | `/v1/agent/sessions/<id>?after=<message id>` | The chat and its messages after that id |
| PATCH | `/v1/agent/sessions/<id>` | Change `title`, `persona`, `model`, `name` or `avatar_asset_id` (`""` removes the avatar) |
| POST | `/v1/agent/sessions/<id>/messages` | `{text, attachments: [asset_id]}` → 202, the agent starts working; 409 while it is still working |
| POST | `/v1/agent/sessions/<id>/talk` | `{rounds: 1–10}` → group chats: the characters talk to each other; stops when they pause or on Stop; a message from you joins in |
| POST | `/v1/agent/sessions/<id>/stop` | Stop after the current step |
| POST | `/v1/agent/sessions/<id>/compact` | Summarise all but the last 4 messages now → `{compacted, before, after, session}` (estimated tokens per call) |
| DELETE | `/v1/agent/sessions/<id>` | Delete the chat |

**Group chats.** A chat can hold a `cast` of up to 4 characters (`[{name, persona, avatar_asset_id}]`, the first is the lead; with several, each needs a name). One model call per turn voices all of them: replies carry `lines: [{speaker, say}]` (at most 8 per reply) and characters may talk to each other. `@Name` in your message gets only that character; otherwise one or two who fit answer, and "everyone" / "sab" gets all of them. `set_persona` / `set_avatar` / `remove_character` take `speaker`, so "Riya, show me a picture of you" sets Riya's avatar, and a new speaker in `set_persona` adds a character. `/talk` runs up to 10 rounds of them talking among themselves (one model call per round).

**Adaptive personas.** Tick **🌱 Adaptive** in a chat's header (or PATCH `{adaptive: true}`) and its characters grow: they pick up your preferences, nicknames, in-jokes, shared memories and how the relationship is going, and in group chats what they learn about and feel for each other, including during "Let them talk". When something meaningful changes, the reply carries `grow: [{speaker, note}]` (at most one note per character); each note is stored on the character as `growth`, shown in the chat as "🌱 Maya: …", and fed back into every later prompt. Notes record who a character is becoming (a feeling, an attitude towards you or another character, a habit, a preference), not events, and only when something shifts. At 8 notes the chat's model merges them into at most 4 denser ones, keeping the strongest emotional shifts even when they are old; the chat itself is never touched, and the long-chat summary also keeps how the characters feel and why. Core identity never changes through growth: name, age (always an adult), background and the platform rules. The Persona panel lists what each character has picked up; forget one note with ×, or **Reset all**. Through the API, `cast[].growth` omitted keeps it and `[]` clears it. Turning Adaptive off stops new growth but keeps what was learned.

**Persona name and avatar.** Each chat has its own. `persona_name` is `name` when set, otherwise the name the persona gives itself ("You are Maya, …" → Maya). Ask the agent for a picture of itself: it runs `generate_image`, then `set_avatar` with the new image, and Studio shows that face and name on every reply, in the chat header and in the chat list. The agent is told its avatar's asset id and uses it as the picture reference for later images or videos of itself. Sessions include `avatar_url` (a signed thumbnail) and `avatar_file_url`.

### Media library and Google Drive import

Every uploaded, imported or generated file is an **asset** in a **collection** (default `Uploads`) with optional **tags**. Identical files are stored once: uploading the same content again returns the existing asset with `duplicate: true`.

- **Upload in bulk:** `POST /v1/assets` accepts many `files` plus `collection` and comma-separated `tags` form fields. Studio's **Media** page uploads files or whole folders three at a time (drag and drop works too); files over 100 MB can't pass the Cloudflare tunnel, so import those from Drive.
- **Google Drive (Colab):** mount Drive in the notebook (`from google.colab import drive; drive.mount('/content/drive')`). `GET /v1/drive?path=` browses My Drive; `POST /v1/drive/import` `{paths, recursive, collection?, tags}` imports files or folders in the background, copying them on the server straight into ComfyUI's input folder (no tunnel, no size limit). `GET /v1/imports/<id>` reports progress. The Colab launcher sets `COMFY_INPUT_DIR` and `HAWK_DRIVE_ROOT`.
- **Organise:** `GET /v1/library?kind=&collection=&tag=&q=` (with collection and tag counts), `PATCH /v1/assets/<id>`, `POST /v1/assets/bulk` `{ids, action: move|tag|untag|delete}`, `DELETE /v1/assets/<id>`. Video thumbnails need ffmpeg on the server.
- **Use:** in Studio select media and choose *Use in Create*, *Attach to agent* or *Set as music bed*, or open the media picker from Create (references, music bed) and the agent chat. MCP tools for chats and the agent: `list_references` (filters), `list_collections`, `organize_assets`, `browse_drive`, `import_from_drive`, `get_import`.

### Editable prompts

Studio's **Prompts** page (or `/v1/prompts`) lets you rewrite the **agent** prompt and the **story planner** prompt completely. The agent prompt uses `{{PERSONA}}`, `{{PIPELINE}}` and `{{TOOLS}}` placeholders (the tool list is appended if you remove `{{TOOLS}}`); keep its JSON reply format. The planner prompt must keep the JSON output with a `segments` list. Every save keeps the previous version (last 30), and reset returns to the built-in text. Prompts are stored in `DATA_DIR/prompts.json`.

The server always appends a short, non-editable **platform rules** block (no sexual content involving anyone who appears under 18; no sexual or nude content of real, identifiable people) to the agent prompt and to an edited planner prompt. These are instructions to the model, not a content filter.

| Method | Path | |
|---|---|---|
| GET | `/v1/prompts` | Both prompts: text, default, history, placeholders, warnings, platform rules |
| GET | `/v1/prompts/<agent\|planner>` | One prompt |
| PUT | `/v1/prompts/<agent\|planner>` | `{text}`: save a new version |
| POST | `/v1/prompts/<agent\|planner>/reset` | Back to the built-in prompt |

### Images

`POST /v1/images` (and the MCP / agent tool `generate_image`) creates images: `{prompt, engine?, loras?, steps?, reference_asset_ids?, model?, size?, n (1-4), seed?}`. `GET /v1/images/options` (MCP `image_options`) reports whether local Krea 2 is installed and idle, its LoRA catalogue, and the Atlas models.

| `engine` | What runs | Notes |
|---|---|---|
| `auto` (default) | Krea 2 → z-image/turbo → Seedream | Local Krea 2 when it's installed and ComfyUI is idle; otherwise z-image/turbo; Seedream if that fails too. The result's `tried` lists what was skipped and why. Change the default with `HAWK_IMAGE_ENGINE`. |
| `local` | Krea 2 Turbo on the pod's GPU | Free and private, with Krea 2 LoRAs. Waits behind a running render instead of falling back. Model `krea` / `local` means the same. |
| `turbo` | Atlas `z-image/turbo` | Fast, about $0.01 an image; sizes 512–2048 a side (default 1024x1536); `n` runs as parallel requests. `HAWK_IMAGE_MODEL` sets the Atlas text-to-image default. |
| `seedream` | Atlas `bytedance/seedream-v5.0-pro/text-to-image` | Best quality. Sizes snap to Atlas's nearest preset. Up to 2.36 MP is the 1.5K tier, about $0.036 an image (default 1328x1776; 1024x1024 becomes 1536x1536 at the same price). Larger sizes such as 2048x2048 bill the 2K tier, about $0.072. One image per request, so `n` runs as parallel requests. |
| `seedream-lite` | Atlas `bytedance/seedream-v5.0-lite` | 2K and above only (default 1664x2496), about $0.032 an image, a little below Pro in quality. The cheapest way to get a 2K+ image. Model `seedream-lite` / `lite` means the same. |
| `auto` / `local`, with `reference_asset_ids` | Krea 2 Identity Edit on the pod | 1 image, or 2 (the scene first, then the person to place in it). Free, keeps a face while changing outfit, pose, scene, light or style; `ref_boost` is the likeness dial (4 default, 1 looser, above about 8 fights removals). Falls back to Seedream edit when not installed, busy, or given more than 2 images. |
| `seedream` / `turbo`, with `reference_asset_ids` | `bytedance/seedream-v5.0-pro/edit` | Up to 10 images. z-image can't edit, so `turbo` switches here and says so in `note`. |
| `seedream-lite`, with `reference_asset_ids` | `bytedance/seedream-v5.0-lite/edit` | Same as Lite above. |

Atlas results carry `cost_usd`, an estimate from Atlas's discounted list prices (September 2026). Seedream adds about $0.003 for each reference image after the first. `GET /v1/images/options` lists the prices under `atlas.prices_usd`.

**Krea 2 LoRAs.** `loras: [{name, strength?}]` picks LoRAs by file name, a unique part of it, or the catalogue label (`"realism"`, `"darkbrush"`). Without `strength`, the catalogue's recommended value is used. A LoRA's trigger words are added to the prompt, and its recommended steps, sampler and scheduler replace the Turbo defaults (8 steps, cfg 1, euler, simple). The catalogue lives in `$DATA_DIR/image_loras.json` (copied from [deploy/image_loras.example.json](../deploy/image_loras.example.json) on first use; edit the copy). Each entry has `kind` (`realism`, `detail`, `style`, `adult`), `strength`, `range`, optional `trigger`, `steps`, `sampler` and `scheduler`, and `notes` that the agent reads. Any other LoRA file with "krea" in its name shows up as kind `other`.

**Safety.** Local generation has no provider moderation, so the API checks it: prompts that mention minors (child, teen, schoolgirl, "16 year old" …) are refused with 422. One image may stack up to three `adult` LoRAs (`max_adult_loras`, default 3, lower it to be stricter); the result's `note` warns when their combined strength goes above 2.0, where they tend to over-cook. The catalogue's notes steer the agent: SNOFS + Mystic XXX is the GO-TO adult pair (NSFW Master is added as a third only when needed), and Enhancer is marked AVOID. The catalogue has a `version`: when a newer one ships with the code, the pod's copy is replaced and the old one kept as `image_loras.json.bak`. A LoRA error (unknown name, too many adult LoRAs) returns 422 instead of silently falling back to another engine. Adult LoRAs are for fictional adults only; the agent uses them only when you explicitly ask for adult content, and never for real, identifiable people.

**Krea 2 edit.** Needs the [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) node pack (`Krea2EditModelPatch`, `Krea2EditGroundedEncode`) and the [krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit) LoRA (`krea2_identity_edit_v1_2.safetensors`, or its `_r128` / `_r64` low-VRAM variants) in `models/loras`; `local.edit` in `/v1/images/options` says what is missing. The graph follows the pack's workflow: the identity LoRA at 1.0 first (your other LoRAs after it), `Krea2EditModelPatch` with the source image as context tokens (`fit` geometry, `ref_boost`), `Krea2EditGroundedEncode` for the instruction and an empty negative, both grounded on the source at 768 px, `EmptySD3LatentImage` at the source's aspect ratio and about 1 MP (or `size`), 10 steps, cfg 1, euler, simple. `n` edits run as separate prompts with consecutive seeds. Edits of uploaded photos may show real people, so they are refused when the instruction is sexual (nude, naked, topless, undress, sexual, explicit, porn and similar words) or an adult LoRA is used. "Uploaded" means anything not made here (device uploads, Drive and URL imports) and any image made from one, followed back through every edit. Images made here from a prompt are fictional characters and follow the normal rules. The instruction check is a word list, so the adult-LoRA block is the stronger half; Seedream edits are moderated by Atlas.

**Graph.** Krea 2 runs as `UNETLoader (krea2_turbo_fp8_scaled)` → `LoraLoaderModelOnly` × N → `KSampler`, with `CLIPLoader (qwen3vl_4b_fp8_scaled, type krea2)` → `CLIPTextEncode` as the positive, `ConditioningZeroOut` as the negative, and `VAELoader (qwen_image_vae)` → `VAEDecode` → `SaveImage`. If a configured file is missing, the API uses another precision of the same model it finds (bf16 first, then fp16, then fp8), so swapping `qwen3vl_4b_fp8_scaled` for `qwen3vl_4b_bf16` needs no setting; `local.files` in `/v1/images/options` shows the files in use. `HAWK_KREA_UNET`, `HAWK_KREA_CLIP` and `HAWK_KREA_VAE` pin exact names.

Results are stored as image assets in the **Generated** collection, tagged with the engine (`krea2`, `z-image`, `seedream`), and can be used right away as `picture` references in plans and renders. In Studio, **Media → ✨ Generate** does the same: prompt, engine, size, count, seed, reference images and a Krea 2 LoRA picker with strength sliders. Select images and press **✨ Edit / remix** to start from them.

The agent follows the same order for base images: Krea 2 when the GPU is free, else z-image/turbo. It checks results with **`inspect_image`**: a vision model scores each image against the brief and flags faces, hands, anatomy, wrong outfit or setting and garbled text. It asks the chat's own model first when that model can see images, then Grok 4.6, then Grok 4.3, until one returns a usable verdict. Some models refuse to review certain images, adult ones in particular. When a batch fails (every image "retry", or the best scores below 6), the next `generate_image` with engine `auto` in that task moves up a rung: local Krea 2, then z-image/turbo, then Seedream. Edits go from Krea 2 straight to Seedream. The result's `engine_note` says so. Takes with LoRAs stay on Krea 2. The ladder resets with the user's next message. It picks Krea LoRAs from `image_options` (a realism or detail LoRA for photo portraits, a style LoRA by its trigger). For a picture of itself or a character it writes a detailed 60–120 word prompt from the persona, makes 2 options, inspects them and sets the best as the avatar.

Every asset in API responses carries a signed `file_url`, and images a `thumb_url` (a 320 px JPEG, via `GET /v1/assets/<id>/file?w=320`). Signed links open without the token, so Studio and the agent chat show thumbnails of uploaded and generated images.

The API process needs `ATLAS_API_KEY` (the Colab launcher passes it already). The **planner model** is also choosable: `/v1/options` returns `planner_models`, and `/v1/plans`, `story` and MCP `plan_film` take `model`.

---

## 7. MCP tools

| Tool | What it does |
|---|---|
| `upload_page_link` | Signed link to the browser upload page |
| `add_reference_from_url` | Fetch a public file URL onto the pod |
| `list_references` | Uploaded assets |
| `generate_image` | Generate or edit images: local Krea 2 with LoRAs, z-image/turbo or Seedream (`engine`, `loras`); results become image assets |
| `image_options` | Whether local Krea 2 is installed and idle, its LoRAs with recommended strengths, and the Atlas image models |
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
