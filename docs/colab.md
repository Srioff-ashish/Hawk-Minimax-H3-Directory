# Running the API on Google Colab (G4)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Srioff-ashish/Hawk-Minimax-H3-Directory/blob/main/deploy/colab/Hawk_H3_API_Colab.ipynb)

The notebook [`deploy/colab/Hawk_H3_API_Colab.ipynb`](../deploy/colab/Hawk_H3_API_Colab.ipynb) runs the whole stack on one Colab GPU, so Claude or Grok can plan and render from a chat:

```
Claude / Grok ──► https://<random>.trycloudflare.com ──► Hawk H3 API (127.0.0.1:8000) ──► ComfyUI (127.0.0.1:8188)
                  Cloudflare quick tunnel                  inside the Colab runtime
```

Each session starts from nothing. The notebook installs everything, downloads the models, starts ComfyUI and the API, and opens a tunnel with a new URL. When the runtime stops, everything is gone, including models, job history and rendered videos.

- [Already have ComfyUI and the models? One cell](#already-have-comfyui-and-the-models-one-cell)
- [What you need](#what-you-need)
- [Run a session](#run-a-session)
- [Connect Claude or Grok each session](#connect-claude-or-grok-each-session)
- [Limits of this setup](#limits-of-this-setup)
- [Settings](#settings)
- [Troubleshooting](#troubleshooting)

---

## Already have ComfyUI and the models? One cell

If your own notebook already installs ComfyUI and downloads the MiniMax H3 models and LoRAs, add this cell after your download cells and **don't start ComfyUI yourself**. The cell starts it, so ComfyUI gets the Hawk nodes and your Atlas key. Then add the watch cell below it.

```python
#@title Hawk H3 · node pack + API + tunnel (models already downloaded)
import importlib, os, subprocess, sys

COMFY_DIR = "/content/ComfyUI"  #@param {type:"string"}
ATTENTION = "sol scheduled"  #@param ["sol scheduled", "comfy default"]
#@markdown Leave blank to auto-detect in COMFY_DIR/models (ref2va model, qwen3vl text encoder, minimax_h3 VAEs, ref2v turbo LoRA).
UNET_NAME = ""  #@param {type:"string"}
CLIP_NAME = ""  #@param {type:"string"}
TURBO_LORA = ""  #@param {type:"string"}

PACK_DIR = f"{COMFY_DIR}/custom_nodes/Hawk-Minimax-H3-Directory"
if not os.path.isdir(PACK_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Srioff-ashish/Hawk-Minimax-H3-Directory", PACK_DIR], check=True)
else:
    subprocess.run(["git", "-C", PACK_DIR, "pull", "--ff-only"], check=True)

sys.path.insert(0, f"{PACK_DIR}/deploy/colab")
import hawk_colab
importlib.reload(hawk_colab)

hawk_colab.install(COMFY_DIR, PACK_DIR, sol=ATTENTION.startswith("sol"), comfy_requirements=False)
session = hawk_colab.start(
    COMFY_DIR,
    PACK_DIR,
    attention=ATTENTION,
    unet_name=UNET_NAME or None,
    clip_name=CLIP_NAME or None,
    turbo_lora=TURBO_LORA or None,
    token=hawk_colab.colab_secret("HAWK_API_TOKEN"),
    atlas_api_key=hawk_colab.colab_secret("ATLAS_API_KEY"),
)
```

```python
#@title Hawk H3 · watch jobs (keep running)
hawk_colab.watch(session, interval=60)
```

What the first cell does:
- **Installs only what's missing:** this node pack into `custom_nodes`, the API's Python packages, Sol attention and `cloudflared`. It doesn't reinstall ComfyUI's requirements (`comfy_requirements=False`) and doesn't download any models.
- **Finds your files** under `ComfyUI/models`. It needs the **ref2va** diffusion model (fl2va doesn't work with this pack), the `qwen3vl…minimax` text encoder and both `minimax_h3` VAEs. If several match, it prefers pruned int8, then the NVFP4 text encoder, then the `ref2v…turbo…4step_v0.1` LoRA. It prints what it picked; type a name in the form to override it. Subfolder paths such as `h3/minimax_h3_ref2va_bf16.safetensors` work.
- **Sets up LoRAs:** the turbo LoRA it finds becomes the required default LoRA (8 steps). Your other LoRAs in `models/loras` are available by name in requests; ask the chat for `list_options`.
- **Starts** ComfyUI on 127.0.0.1:8188, the tunnel and the API, then prints the connector links.

If your notebook already started ComfyUI:
- **Started after this pack was installed** (for example, on a second run of the cell): it's reused. For planning it must have `ATLAS_API_KEY` in its environment.
- **Started before this pack was installed:** the cell stops with a message. Stop that ComfyUI (interrupt its cell or run `!pkill -f 'ComfyUI/main.py'`), then run the cell again.

Add the same Secrets as below (`ATLAS_API_KEY`, and ideally `HAWK_API_TOKEN`).

---

## What you need

- **A paid Colab plan with G4 GPUs.** G4 is an NVIDIA RTX PRO 6000 Blackwell with 96 GB of VRAM. H3 fits comfortably, and the notebook's NVFP4 text encoder runs natively on it.
- **A positive compute-unit balance** for the whole session. [Colab's rules](https://research.google.com/colaboratory/faq.html) forbid web services and web UIs on runtimes that run free of charge; a paid plan with a positive balance lifts that restriction. G4 uses compute units quickly, so stop the runtime when you're done.
- **An Atlas Cloud API key** if you want LLM planning. Rendering your own scripts works without it.

## Run a session

1. **Open the notebook**, using the badge above or File → Open notebook → GitHub → `Srioff-ashish/Hawk-Minimax-H3-Directory`.
2. **Runtime → Change runtime type → G4.**
3. **Add secrets** (🔑 on the left sidebar), each with *Notebook access* on:

   | Secret | Needed? | Purpose |
   |---|---|---|
   | `ATLAS_API_KEY` | for planning | The Story Planner calls Atlas Cloud |
   | `HAWK_API_TOKEN` | recommended | Your API secret (`openssl rand -hex 24`). Without it a new random token is made every session |
   | `HF_TOKEN` | optional | Faster or authenticated Hugging Face downloads |

   Secrets stay in your Colab account and are never written into the notebook or its outputs, except the token, which is printed inside the connector URL for you to copy.

4. **Run the cells in order:**

   | Cell | Time | What happens |
   |---|---|---|
   | 1 · Settings | instant | Choose models, extra LoRAs, attention |
   | 2 · Install | ~5 min | Clones ComfyUI, this pack and Sol attention; installs requirements. Installs a Blackwell-capable torch only if Colab's lacks it |
   | 3 · Download | ~5–15 min | About 45 GB from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), plus your extra LoRAs |
   | 4 · Start | ~2–5 min | Starts ComfyUI on 127.0.0.1, opens the tunnel, starts the API with the tunnel URL, checks it's reachable, prints your links |
   | 5 · Watch | keep running | Prints job status every minute; reopens the tunnel and restarts the API if either dies |

5. **Copy the printed links** (next section).

The **first render of a session** is slower. Sol attention's Triton kernels compile for each new resolution and length, and with nothing persisted every session pays that again. Later renders at the same size are fast.

## Use the Studio web page

The start cell also prints a **Studio** link: `https://….trycloudflare.com/t/<token>/studio`. Open it in any browser, including a phone:

1. **References:** drop images (or audio/video). For each one choose *Picture* (who, what, where) or *Pose* (body pose only), and click its tag, e.g. `<Picture 1>`, to put it in the prompt.
2. **Prompt:** *Direct prompt* renders what you write. *AI planner* writes a multi-segment plan from your idea: *Write plan* lets you edit it first, *Plan & render* does both.
3. **Video:** duration (per segment in planner mode), segment count, aspect ratio and resolution. Preview 0.4 MP is fastest; Native 0.98 MP is H3's normal size. **Base model** and **Text encoder** list the ref2va and Qwen3-VL files in your ComfyUI: int8 for quick previews, bf16 for the best quality (slowest, most VRAM).
4. **LoRAs:** tick any LoRA from the server's `models/loras` and set its strength. The turbo LoRA is on by default.
5. **Generate.** The right side shows progress by segment and sampling step, then the video player and download links.

Like the connector, the link changes every session, and uploads are limited to 100 MB (use the URL box for bigger files).

### Media page and Google Drive

**Media** in Studio's sidebar holds every image, video and audio file: upload many files or whole folders, drop them on the page, or import from Google Drive. For Drive, run this once in a new notebook cell and allow access:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The same mount turns on **fast video delivery** (next section). Then *Import → From Google Drive* browses My Drive; pick files or folders, a collection and tags. Drive imports are copied on the Colab machine, so they're fast and have no 100 MB limit. Select media to use it in Create, attach it to an agent chat, or set a music bed; Create and the agent chat also have a *From media* picker.

### Videos play from Google Drive

Downloading through the tunnel gets slow, especially while a render is using the CPU. With Drive mounted, the API copies every finished video to **My Drive → Hawk H3 → Videos → <date>**. Once Drive has synced it (usually under a minute), Studio plays and downloads it **from Google's servers**, without the tunnel:

- Video cards show a poster frame. Nothing downloads until you press play.
- The player and **Download** use Drive when the video is marked *In Drive*. Until then, or if Drive can't play it yet, they use the server, which streams with seeking. In a video's details, *Play from server* forces the server, and *Save to Drive* copies it again.
- Drive playback needs a browser signed in to the Google account whose Drive is mounted. To watch on other accounts or devices, share the *Hawk H3* folder in Drive with *Anyone with the link*.
- **Connect & settings → Video delivery** sets the folder, turns the copy on or off, can also copy segment files, and switches playback between Drive and the server.

### Agent page

**Agent** in Studio's sidebar is an autonomous director. Start a chat, pick the model (Grok 4.6 by default, or Grok 4.3 and any other Atlas model), optionally set a **Persona**, attach files and describe the video. It plans, renders, waits and fixes problems by itself, shows every tool it uses, and plays the result in the chat. It runs on the Colab server, so you can close the browser; press **Stop** to end a run. It uses the Studio tools in-process, never through the tunnel. Details: [API → Agent](api.md#agent-an-autonomous-video-director).

Tick **🌱 Adaptive** on a chat to let its characters grow from what you talk about and from their conversations with each other. What they pick up shows in the chat and in the Persona panel, where you can remove it. Their name, age and the safety rules never change.

### Local images with Krea 2

Studio and the agent can make images on the Colab GPU with **Krea 2 Turbo** and its LoRAs: free, private, and without Atlas moderation. Install it in a notebook cell after ComfyUI is set up. It needs:

| Folder in `/content/ComfyUI/models` | File (Hugging Face `Comfy-Org/Krea-2`) |
|---|---|
| `diffusion_models` | `krea2_turbo_fp8_scaled.safetensors` |
| `text_encoders` | `qwen3vl_4b_fp8_scaled.safetensors` |
| `vae` | `qwen_image_vae.safetensors` |
| `loras` | Your Krea 2 LoRAs, e.g. `krea2_realism_v1`, `krea2_realistic_snapshot`, `krea2_enhancer`, `snofs_photodetail_slider`, `krea2_darkbrush`, `krea2_sunsetblur` |

Keep Civitai downloads authenticated with a Colab secret (🔑 in the sidebar, e.g. `CIVITAI_TOKEN`, read with `userdata.get`). Never paste the token into a cell. Save LoRAs with the file names in [deploy/image_loras.example.json](../deploy/image_loras.example.json) so their recommended strengths, trigger words and step counts apply; other files with "krea" in the name still show up. No ComfyUI restart is needed: the API re-reads the model folders.

Use it from **Media → ✨ Generate** (engine *Auto* or *Krea 2 (local)*, tick LoRAs, set strengths), or just ask the agent for an image. *Auto* uses Krea 2 when the GPU is idle, **Z-Image Turbo** on Atlas while a video is rendering, and **Seedream** if both fail; the agent also moves to Seedream when it isn't happy with a result. Editing reference images always uses Seedream.

Krea 2 shares the GPU with the video models. ComfyUI may unload the H3 models to fit it, so the next video render spends a minute or so reloading them. A video render that starts while Krea 2 runs waits its turn.

In **AI planner** mode on the Create page, **Planner model** picks which Atlas model writes the plan; a warning appears when the model can't see your reference images.

## Connect Claude or Grok each session

The quick-tunnel URL is new every session, so the connector has to be updated every time.

**Claude (claude.ai, desktop, mobile):**
1. Settings → Connectors. Remove the previous session's Hawk H3 connector.
2. *Add custom connector*, and paste the printed `https://….trycloudflare.com/t/<token>/mcp` URL.
3. Enable it in your chat's tools menu.

**Grok:** Connectors → New Connector → Custom. Use the `…/t/<token>/mcp` URL, or the `…/mcp` URL with the `Authorization: Bearer <token>` header if the form has a header field.

**Claude Code:**
```bash
claude mcp remove hawk-h3 2>/dev/null
claude mcp add --transport http hawk-h3 https://<random>.trycloudflare.com/mcp \
  --header "Authorization: Bearer <token>"
```

Then work in the chat as described in [API → Using it from a chat](api.md#5-using-it-from-a-chat). For example: *"Give me the upload link"* → upload → *"Plan a 3-segment film"* → *"Render a preview at 0.4 megapixels"*.

## Limits of this setup

| Limit | Cause | What to do |
|---|---|---|
| **URL changes every session** | Cloudflare quick tunnels have random, temporary names | Update the connector each session. A stable URL needs ngrok's free static domain or a Cloudflare named tunnel on your own domain |
| **Uploads through the tunnel ≤ 100 MB** | Cloudflare's free proxy limit | Big reference videos: put them online (Drive share link, Hugging Face…) and use `add_reference_from_url`; that download happens inside Colab, not through the tunnel |
| **Nothing survives the session** | Chosen setup: no Drive | Download finished videos before stopping (the `video_url` links die with the runtime). Unfinished renders can't be resumed in a new session |
| **Session time limits** | Colab idle and maximum runtime limits | Keep cell 5 running and the tab open during long renders; plan long films as several shorter jobs |
| **One render at a time** | One GPU | Jobs queue: they show `queued` with their place in line (`#1` = next) until ComfyUI starts them |
| **Quick tunnels are for testing** | Cloudflare's terms: no uptime guarantee, 200 concurrent requests | Fine for personal use from your chats; not for sharing publicly |

## Settings

| Setting | Options | Notes |
|---|---|---|
| `diffusion_model` | pruned int8 **(default, 21 GB)**, pruned fp8, pruned bf16 (40 GB), full int8 (34 GB), full bf16 (66 GB) | All are ref2va models. Larger = slower download, possibly slightly better quality |
| `text_encoder` | nvfp4 **(default, 16 GB)**, int8 (27 GB), bf16 (52 GB) | NVFP4 is native on Blackwell |
| `extra_loras` | comma-separated `owner/repo/path/file.safetensors` or direct URLs | Downloaded into `models/loras`; use them by name in `settings.loras` or `lora_preset` |
| `attention` | `sol scheduled` **(default)**, `comfy default` | Sol falls back to normal attention by itself if its kernel can't run |
| `install_rife_interpolation` | off / on | Needed only for `interpolation: 48/60 fps` |
| `pack_branch` | `main` | Which branch of this repo to run |
| `HAWK_IMAGE_ENGINE` | `auto` **(default)**, `local`, `turbo`, `seedream` | Default image engine for Studio and the agent. Set `os.environ["HAWK_IMAGE_ENGINE"]` in a cell before starting the API |

The turbo LoRA is always downloaded and is required by default ([loras.json](api.md#2-choose-loras-lorasjson)). To use extra LoRAs in every render, edit `/content/hawk_api_data/loras.json` during the session. It resets next session; to keep a change, add it to `deploy/loras.example.json` in your fork.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No CUDA GPU found` | Runtime → Change runtime type → G4, then run from cell 2 |
| `Not enough disk` | Pick smaller models in cell 1 |
| Generate says *Krea 2 isn't installed* | The message lists the missing files; check they're in `/content/ComfyUI/models/{diffusion_models,text_encoders,vae}` with those exact names |
| Images come from Z-Image Turbo instead of Krea 2 | A video was rendering, so *Auto* didn't wait. Pick *Krea 2 (local)* to queue behind it |
| Download very slow or 401/403 | Add an `HF_TOKEN` secret |
| `ComfyUI exited during startup` | Run the *Show logs* cell. A missing package usually means cell 2 didn't finish; rerun it |
| `Cloudflare quick tunnel did not start` | Rerun cell 4. Cloudflare occasionally refuses new quick tunnels for a few minutes |
| `API runs locally but the tunnel URL is not reachable yet` | Wait a minute and open the Health link; new trycloudflare names take a moment to resolve |
| A render has **fewer segments than scenes** (4 sent, 3 made) and scene 1 shows up inside every segment | A `style:` line written straight above scene 1 used to swallow the whole first block. Fixed: `style:` now covers only its own paragraph. The parser also runs inside ComfyUI, so update and restart both: `!git -C /content/ComfyUI/custom_nodes/Hawk-Minimax-H3-Directory pull`, then `hawk_colab.restart_api(session)` and `hawk_colab.restart_comfyui(session)` (models load again on the next render) |
| Connector form asks for **OAuth credentials** (Client ID, Authorization Endpoint…) | Cancel it; this API uses no OAuth. Use the URL that contains `/t/<token>/mcp`. If you already did, update the pack and reload the API: `!git -C /content/ComfyUI/custom_nodes/Hawk-Minimax-H3-Directory pull` then `hawk_colab.restart_api(session)`, and add the connector again |
| Connector or API says **401 invalid token** although the token matches the printed one | An API from an earlier run of the start cell is still holding port 8000 with its old token. Update the pack and restart the API: the launcher now stops leftovers first (see the update cell in [Already have ComfyUI…](#already-have-comfyui-and-the-models-one-cell)), or run `!pkill -f hawk_api.app:create_app` then `hawk_colab.restart_api(session)` |
| Claude says the connector can't connect | The URL is from an old session. Copy the current one from cell 4 (or from cell 5's output after a tunnel restart) |
| Chat reports `422 Required default LoRA … missing` | Cell 3 didn't finish; rerun it, then retry |
| Planning fails with an Atlas error | Add the `ATLAS_API_KEY` secret with notebook access, then rerun cell 4 |
| `HawkH3Director: Allocation on device 0 would exceed allowed memory` | Update the pack and restart ComfyUI: the Director now unloads cached models and retries the step once. If the error says `even after unloading cached models`, lower megapixels or duration, or choose an int8 model/text encoder. Retry the job: finished segments are reused |
| Colab disconnected mid-render | The session and all files are gone; start a new session and render again (use a lower `megapixels` preview first) |
