"""Google Colab launcher: ComfyUI + the Hawk H3 API behind a Cloudflare quick tunnel.

Used by deploy/colab/Hawk_H3_API_Colab.ipynb, or on its own from a notebook that
already has ComfyUI and the models (see docs/colab.md). A session:

1. install this pack's API requirements, Sol attention and cloudflared
   (and ComfyUI + its requirements when missing)
2. download the MiniMax H3 models and LoRAs from Hugging Face (optional)
3. start ComfyUI on 127.0.0.1:8188, or reuse one already running with the Hawk nodes
4. open a Cloudflare quick tunnel to the API port and read its URL
5. start the API with that URL, then print the connector links

Only the standard library is imported at module level; the pure helpers are
unit-tested in tests_api/test_colab.py.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

MODEL_REPO = "Comfy-Org/MiniMax-H3"

DIFFUSION_MODELS = {
    "ref2va pruned int8 (21 GB, recommended)": ("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", 20.97),
    "ref2va pruned fp8 (21 GB)": ("diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors", 20.96),
    "ref2va pruned bf16 (40 GB)": ("diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors", 40.23),
    "ref2va full int8 (34 GB)": ("diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors", 34.04),
    "ref2va full bf16 (66 GB)": ("diffusion_models/minimax_h3_ref2va_bf16.safetensors", 66.28),
}
TEXT_ENCODERS = {
    "nvfp4 (16 GB, recommended on G4)": ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15.69),
    "int8 (27 GB)": ("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", 27.14),
    "bf16 (52 GB)": ("text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", 51.51),
}
VAES = [
    ("vae/minimax_h3_video_vae_fp16.safetensors", 5.21),
    ("vae/minimax_h3_audio_vae_fp32.safetensors", 0.61),
]
TURBO_LORA = ("loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1.96)

#: Auto-detection: which file wins when several match.
UNET_PREFERENCE = ["pruned_int8", "pruned_fp8", "pruned_bf16", "int8", "fp8", "bf16"]
CLIP_PREFERENCE = ["nvfp4", "int8", "bf16"]
TURBO_PREFERENCE = ["4step_v0.1_comfyui_bf16", "4step", "8step"]

COMFY_PORT = 8188
API_PORT = 8000
#: `pkill -f` pattern for this launcher's own tunnel only. A notebook may run a second
#: quick tunnel for the ComfyUI UI (e.g. `cloudflared tunnel --url http://localhost:8188`);
#: restarting the API must not kill it.
API_TUNNEL_PATTERN = rf"cloudflared tunnel --no-autoupdate --url http://127\.0\.0\.1:{API_PORT}"
#: Cloudflare's free proxy rejects request bodies over 100 MB.
TUNNEL_UPLOAD_MB = 100
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
EXTRA_NODES = {
    "sol": "https://github.com/Saganaki22/ComfyUI-sol-attn",
    "vfi": "https://github.com/GACLove/ComfyUI-VFI",
}
_MODEL_EXT = (".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf")


# ------------------------------------------------------------------ pure helpers


@dataclass(frozen=True)
class Download:
    folder: str  # under ComfyUI/models
    filename: str
    repo: str | None = None
    path: str | None = None
    revision: str = "main"
    url: str | None = None
    size_gb: float = 0.0


_HF_URL = re.compile(r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/(?:resolve|blob)/(?P<rev>[^/]+)/(?P<path>[^?#]+)")


def parse_lora_source(spec: str) -> Download:
    """``owner/repo/path/file.safetensors``, a huggingface.co resolve/blob URL, or any direct URL."""
    spec = spec.strip()
    match = _HF_URL.match(spec)
    if match:
        path = urllib.parse.unquote(match["path"])
        return Download("loras", os.path.basename(path), repo=match["repo"], path=path, revision=match["rev"])
    if spec.startswith(("http://", "https://")):
        name = os.path.basename(urllib.parse.unquote(urllib.parse.urlsplit(spec).path))
        if not name.lower().endswith(_MODEL_EXT):
            raise ValueError(f"LoRA URL {spec!r} must end in a model file name such as .safetensors.")
        return Download("loras", name, url=spec)
    parts = [part for part in spec.split("/") if part]
    if len(parts) >= 3 and spec.lower().endswith(_MODEL_EXT):
        return Download("loras", parts[-1], repo="/".join(parts[:2]), path="/".join(parts[2:]))
    raise ValueError(
        f"Can't read LoRA source {spec!r}. Use 'owner/repo/path/file.safetensors' or a direct download URL."
    )


def _repo_file(path: str, size_gb: float) -> Download:
    folder, _, filename = path.partition("/")
    return Download(folder, filename, repo=MODEL_REPO, path=path, size_gb=size_gb)


def manifest(diffusion_model: str, text_encoder: str, extra_loras: str | list[str] = "", turbo_lora: bool = True) -> list[Download]:
    if diffusion_model not in DIFFUSION_MODELS:
        raise ValueError(f"Unknown diffusion model {diffusion_model!r}. Choices: {', '.join(DIFFUSION_MODELS)}")
    if text_encoder not in TEXT_ENCODERS:
        raise ValueError(f"Unknown text encoder {text_encoder!r}. Choices: {', '.join(TEXT_ENCODERS)}")
    items = [_repo_file(*DIFFUSION_MODELS[diffusion_model]), _repo_file(*TEXT_ENCODERS[text_encoder])]
    items += [_repo_file(*vae) for vae in VAES]
    if turbo_lora:
        items.append(_repo_file(*TURBO_LORA))
    specs = re.split(r"[\n,]+", extra_loras) if isinstance(extra_loras, str) else list(extra_loras)
    items += [parse_lora_source(spec) for spec in specs if spec.strip()]
    return items


def estimated_gb(downloads: list[Download]) -> float:
    return round(sum(item.size_gb for item in downloads), 2)


def find_tunnel_url(log_text: str) -> str | None:
    matches = re.findall(r"https://[-a-z0-9]+\.trycloudflare\.com", log_text)
    return matches[-1] if matches else None


def needs_blackwell_torch(info: dict) -> bool:
    """True when the GPU is Blackwell (compute capability 12.x) but torch was built without its kernels."""
    capability = info.get("cap")
    if not capability or capability[0] < 12:
        return False
    return not any(arch in ("sm_120", "sm_121") for arch in info.get("arch") or [])


def pick_model(files: list[str], required: tuple[str, ...], prefer: list[str]) -> str | None:
    """First file whose path contains every `required` word, favouring `prefer` words in its file name."""
    candidates = [name for name in files if all(word in name.lower() for word in required)]
    for word in prefer:
        for name in candidates:
            if word in os.path.basename(name).lower():
                return name
    return candidates[0] if candidates else None


def pick_models(files: dict[str, list[str]]) -> dict[str, str | None]:
    """Choose the H3 files from ``{"diffusion_models": [...], "text_encoders": [...], "vae": [...], "loras": [...]}``."""
    return {
        "unet_name": pick_model(files.get("diffusion_models", []), ("ref2va",), UNET_PREFERENCE),
        "clip_name": pick_model(files.get("text_encoders", []), ("qwen3vl", "minimax"), CLIP_PREFERENCE),
        "video_vae": pick_model(files.get("vae", []), ("minimax_h3_video",), []),
        "audio_vae": pick_model(files.get("vae", []), ("minimax_h3_audio",), []),
        "turbo_lora": pick_model(files.get("loras", []), ("ref2v", "turbo"), TURBO_PREFERENCE),
    }


def lora_config(example: dict, turbo_lora: str | None) -> dict:
    """The API's loras.json for this session: the example's presets, with the turbo LoRA
    that is actually on disk as the required default (none when there is no turbo LoRA)."""
    config = {key: value for key, value in example.items() if key != "defaults"}
    config["defaults"] = (
        [{"name": turbo_lora, "strength": 1.0, "required": True, "turbo": True}] if turbo_lora else []
    )
    return config


# ------------------------------------------------------------- process helpers


def _run(cmd: list[str], cwd: str | None = None, check: bool = True, quiet: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=quiet)
    if check and result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or ""))[-3000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{output}")
    return result


def _pip(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-m", "pip", "install", "-q", *args], check=check)


def _pip_requirements(path: str) -> None:
    if _pip("-r", path, check=False).returncode == 0:
        return
    print(f"Some packages in {path} failed; installing one by one.")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            requirement = line.split("#", 1)[0].strip()
            if requirement and _pip(requirement, check=False).returncode != 0:
                print(f"  skipped {requirement}")


def _clone(url: str, dest: str, branch: str | None = None) -> None:
    if os.path.isdir(dest):
        return
    cmd = ["git", "clone", "--depth", "1"] + (["-b", branch] if branch else []) + [url, dest]
    _run(cmd)


def torch_info() -> dict:
    code = (
        "import json, torch\n"
        "ok = torch.cuda.is_available()\n"
        "print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda,"
        " 'arch': torch.cuda.get_arch_list() if ok else [],"
        " 'cap': list(torch.cuda.get_device_capability()) if ok else None,"
        " 'name': torch.cuda.get_device_name() if ok else None}))"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout.strip().splitlines()[-1])


def colab_secret(name: str) -> str | None:
    """A value from Colab's Secrets panel, or None outside Colab / when not set or not shared."""
    try:
        from google.colab import userdata
    except ImportError:
        return os.environ.get(name) or None
    try:
        return userdata.get(name) or None
    except Exception as exc:  # SecretNotFoundError, NotebookAccessError
        if "access" in type(exc).__name__.lower():
            print(f"Secret {name} exists but this notebook has no access: toggle 'Notebook access' in the Secrets panel.")
        return None


def log_tail(path: str, lines: int = 40) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return "(no log yet)"


def _http_status(url: str, headers: dict | None = None, timeout: float = 10.0) -> int | None:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return None


def _http_json(url: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except Exception:
        return None


def _wait_http(url: str, proc: subprocess.Popen, log_path: str, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{label} exited during startup. Last log lines:\n{log_tail(log_path)}")
        if _http_status(url, timeout=5) == 200:
            return
        time.sleep(3)
    raise RuntimeError(f"{label} did not answer at {url} within {timeout:.0f}s. Last log lines:\n{log_tail(log_path)}")


def list_model_files(comfy_dir: str, *folders: str) -> list[str]:
    """Model files under ComfyUI/models/<folder>, as ComfyUI names them (relative, '/'-separated)."""
    names = set()
    for folder in folders:
        root = os.path.join(comfy_dir, "models", folder)
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for filename in files:
                if filename.lower().endswith(_MODEL_EXT):
                    names.add(os.path.relpath(os.path.join(dirpath, filename), root).replace(os.sep, "/"))
    return sorted(names)


def resolve_models(
    comfy_dir: str,
    *,
    diffusion_model: str | None = None,
    text_encoder: str | None = None,
    unet_name: str | None = None,
    clip_name: str | None = None,
    video_vae: str | None = None,
    audio_vae: str | None = None,
    turbo_lora: str | None = None,
) -> dict[str, str | None]:
    """Explicit names win, then notebook labels, then whatever is on disk."""
    files = {
        "diffusion_models": list_model_files(comfy_dir, "diffusion_models", "unet"),
        "text_encoders": list_model_files(comfy_dir, "text_encoders", "clip"),
        "vae": list_model_files(comfy_dir, "vae"),
        "loras": list_model_files(comfy_dir, "loras"),
    }
    chosen = pick_models(files)
    if diffusion_model:
        chosen["unet_name"] = os.path.basename(DIFFUSION_MODELS[diffusion_model][0])
    if text_encoder:
        chosen["clip_name"] = os.path.basename(TEXT_ENCODERS[text_encoder][0])
    for key, value in (("unet_name", unet_name), ("clip_name", clip_name), ("video_vae", video_vae),
                       ("audio_vae", audio_vae), ("turbo_lora", turbo_lora)):
        if value:
            chosen[key] = value

    folder_for = {"unet_name": "diffusion_models", "clip_name": "text_encoders", "video_vae": "vae", "audio_vae": "vae"}
    missing = [key for key in folder_for if not chosen[key]]
    if missing:
        found = "\n".join(f"  models/{folder}: {', '.join(names) or '(empty)'}" for folder, names in files.items())
        raise RuntimeError(
            f"Could not find {', '.join(missing)} under {comfy_dir}/models. Set the name(s) explicitly.\n"
            f"The ref2va diffusion model (not fl2va), a qwen3vl minimax text encoder and both minimax_h3 VAEs are needed.\n"
            f"Found:\n{found}"
        )
    return chosen


# ------------------------------------------------------------------ install


def install(comfy_dir: str, pack_dir: str, sol: bool = True, vfi: bool = False, comfy_requirements: bool = True) -> None:
    """Everything the API needs. ``comfy_requirements=False`` skips ComfyUI's own
    requirements for a ComfyUI that is already installed and working."""
    info = torch_info()
    print(f"GPU: {info.get('name')} (capability {info.get('cap')}), torch {info.get('version')} CUDA {info.get('cuda')}")
    if not info.get("cap"):
        raise RuntimeError("No CUDA GPU found. Runtime -> Change runtime type -> G4 GPU.")
    if needs_blackwell_torch(info):
        print("This torch build has no Blackwell kernels; installing a CUDA 12.8 build.")
        _pip("--upgrade", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128")

    fresh = not os.path.isdir(comfy_dir)
    _clone("https://github.com/comfyanonymous/ComfyUI", comfy_dir)
    if fresh or comfy_requirements:
        _pip_requirements(os.path.join(comfy_dir, "requirements.txt"))
    _pip_requirements(os.path.join(pack_dir, "requirements-api.txt"))

    custom_nodes = os.path.join(comfy_dir, "custom_nodes")
    if sol:
        _clone(EXTRA_NODES["sol"], os.path.join(custom_nodes, "ComfyUI-sol-attn"))
    if vfi:
        vfi_dir = os.path.join(custom_nodes, "ComfyUI-VFI")
        _clone(EXTRA_NODES["vfi"], vfi_dir)
        if os.path.exists(os.path.join(vfi_dir, "requirements.txt")):
            _pip_requirements(os.path.join(vfi_dir, "requirements.txt"))

    cloudflared = cloudflared_path()
    if not os.path.exists(cloudflared):
        print("Downloading cloudflared")
        urllib.request.urlretrieve(CLOUDFLARED_URL, cloudflared)
        os.chmod(cloudflared, 0o755)
    print("Install finished.")


def cloudflared_path() -> str:
    return "/content/cloudflared" if os.path.isdir("/content") else os.path.abspath("cloudflared")


# ----------------------------------------------------------------- download


def download_models(comfy_dir: str, downloads: list[Download], hf_token: str | None = None) -> None:
    models = os.path.join(comfy_dir, "models")
    needed = estimated_gb(downloads)
    free = shutil.disk_usage(comfy_dir).free / 1e9
    print(f"Downloading up to {needed:.1f} GB ({free:.0f} GB free on disk).")
    if free < needed + 5:
        raise RuntimeError(f"Not enough disk: need about {needed + 5:.0f} GB, {free:.0f} GB free. Pick smaller models.")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _pip("huggingface_hub")
        from huggingface_hub import hf_hub_download

    staging = os.path.join(comfy_dir, ".hf_downloads")
    for item in downloads:
        dest_dir = os.path.join(models, item.folder)
        dest = os.path.join(dest_dir, item.filename)
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  have {item.folder}/{item.filename}")
            continue
        started = time.time()
        print(f"  fetching {item.folder}/{item.filename}" + (f" ({item.size_gb:.1f} GB)" if item.size_gb else ""), flush=True)
        if item.repo:
            path = hf_hub_download(item.repo, item.path, revision=item.revision, local_dir=staging, token=hf_token)
            shutil.move(path, dest)
        else:
            partial = dest + ".part"
            request = urllib.request.Request(item.url, headers={"User-Agent": "hawk-colab"})
            with urllib.request.urlopen(request) as response, open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 24)
            os.replace(partial, dest)
        print(f"    done in {time.time() - started:.0f}s", flush=True)
    shutil.rmtree(staging, ignore_errors=True)
    print("All models ready.")


# ------------------------------------------------------------------- session


@dataclass
class Session:
    comfy_dir: str
    pack_dir: str
    token: str
    env: dict
    log_dir: str
    public_url: str = ""
    #: name -> Popen; "comfyui" is None when an already-running ComfyUI is reused.
    procs: dict = field(default_factory=dict)

    def log(self, name: str) -> str:
        return os.path.join(self.log_dir, f"{name}.log")

    def signed(self, path: str, ttl: int = 7 * 24 * 3600) -> str:
        if self.pack_dir not in sys.path:
            sys.path.insert(0, self.pack_dir)
        from hawk_api.auth import sign_path

        return self.public_url + sign_path(self.token, path, ttl)

    def summary(self) -> str:
        url = self.public_url
        return "\n".join([
            "",
            "Hawk H3 API is running",
            "=" * 60,
            "Studio (web UI: upload, LoRAs, prompt, watch videos):",
            f"  {url}/t/{self.token}/studio",
            "",
            f"API base URL        {url}",
            f"Health              {url}/healthz",
            f"API schema          {url}/docs",
            "",
            "Claude custom connector URL (Settings -> Connectors -> Add custom connector):",
            f"  {url}/t/{self.token}/mcp",
            "",
            "Clients that send headers (Claude Code, Grok connector with auth, xAI API):",
            f"  {url}/mcp   with   Authorization: Bearer {self.token}",
            "",
            f"Upload page (valid 7 days)  {self.signed('/upload')}",
            "",
            "Remember:",
            "  - This URL changes every Colab session: update the connector each time.",
            f"  - Uploads through the tunnel are limited to {TUNNEL_UPLOAD_MB} MB; use add_reference_from_url for bigger files.",
            "  - Download finished videos before the runtime stops; nothing is kept.",
            "  - Treat the connector URL and token like a password.",
            "=" * 60,
        ])


def _popen(cmd: list[str], cwd: str, env: dict, log_path: str) -> subprocess.Popen:
    handle = open(log_path, "a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def _start_comfyui(session: Session) -> None:
    base = f"http://127.0.0.1:{COMFY_PORT}"
    if _http_status(f"{base}/queue", timeout=3) == 200:
        if _http_json(f"{base}/object_info/HawkH3Director"):
            session.procs["comfyui"] = None
            print(
                f"Reusing the ComfyUI already running on port {COMFY_PORT}. For LLM planning it must have been "
                "started with ATLAS_API_KEY in its environment."
            )
            return
        raise RuntimeError(
            f"A ComfyUI is already running on port {COMFY_PORT} without the Hawk H3 nodes (it was started before "
            "they were installed). Stop it -- interrupt your ComfyUI cell, or run `!pkill -f 'ComfyUI/main.py'` -- "
            "then run this cell again so ComfyUI restarts with the nodes and your Atlas key."
        )
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT), "--max-upload-size", "2048"]
    session.procs["comfyui"] = _popen(cmd, session.comfy_dir, session.env, session.log("comfyui"))
    print("Starting ComfyUI (first start loads nodes; a few minutes)...", flush=True)
    _wait_http(f"{base}/queue", session.procs["comfyui"], session.log("comfyui"), 900, "ComfyUI")
    if not _http_json(f"{base}/object_info/HawkH3Director"):
        raise RuntimeError(f"ComfyUI started but the Hawk H3 nodes did not load. Log:\n{log_tail(session.log('comfyui'), 80)}")
    print("ComfyUI is up with the Hawk H3 nodes.")


def _start_tunnel(session: Session) -> None:
    log_path = session.log("tunnel")
    open(log_path, "w").close()
    cmd = [cloudflared_path(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{API_PORT}"]
    session.procs["tunnel"] = _popen(cmd, session.comfy_dir, session.env, log_path)
    deadline = time.time() + 90
    while time.time() < deadline:
        url = find_tunnel_url(log_tail(log_path, 200))
        if url:
            session.public_url = url
            print(f"Tunnel: {url}")
            return
        if session.procs["tunnel"].poll() is not None:
            break
        time.sleep(2)
    raise RuntimeError(f"Cloudflare quick tunnel did not start. Log:\n{log_tail(log_path)}")


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) != 0


def _free_api_port(timeout: float = 20.0) -> None:
    """Stop a Hawk H3 API left over from an earlier run of the start cell.

    A leftover server keeps port 8000 with its old token and old code; a new one then
    fails to bind and exits, while health checks still succeed against the old one."""
    if port_free(API_PORT):
        return
    print(f"Stopping an older Hawk H3 API still running on port {API_PORT}.")
    subprocess.run(["pkill", "-f", "hawk_api.app:create_app"], capture_output=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_free(API_PORT):
            return
        time.sleep(1)
    raise RuntimeError(
        f"Port {API_PORT} is still in use by another program. Find it with `!fuser -v {API_PORT}/tcp` "
        f"(or `!ps aux | grep uvicorn`) and stop it, then run this cell again."
    )


def _start_api(session: Session) -> None:
    _free_api_port()
    env = dict(session.env, PUBLIC_BASE_URL=session.public_url)
    cmd = [sys.executable, "-m", "uvicorn", "--factory", "hawk_api.app:create_app",
           "--host", "127.0.0.1", "--port", str(API_PORT), "--proxy-headers"]
    session.procs["api"] = _popen(cmd, session.pack_dir, env, session.log("api"))
    # /openapi.json is served by the API alone; /healthz stays 503 while ComfyUI is down or restarting.
    _wait_http(f"http://127.0.0.1:{API_PORT}/openapi.json", session.procs["api"], session.log("api"), 180, "Hawk H3 API")
    time.sleep(2)
    accepted = _http_status(
        f"http://127.0.0.1:{API_PORT}/v1/jobs?limit=1", headers={"Authorization": f"Bearer {session.token}"}
    )
    if session.procs["api"].poll() is not None or accepted != 200:
        raise RuntimeError(
            f"The API answering on port {API_PORT} is not the one just started (session token -> {accepted}). "
            f"Last log lines:\n{log_tail(session.log('api'))}"
        )
    deadline = time.time() + 120
    while time.time() < deadline:  # the new trycloudflare hostname needs a moment to resolve
        if _http_status(f"{session.public_url}/openapi.json", timeout=10) == 200:
            print("API reachable through the tunnel.")
            return
        time.sleep(5)
    print("Warning: the API runs locally but the tunnel URL is not reachable yet; try the health link in a minute.")


def _write_lora_config(pack_dir: str, data_dir: str, turbo_lora: str | None) -> None:
    path = os.path.join(data_dir, "loras.json")
    if os.path.exists(path):
        print(f"Keeping existing {path}")
        return
    with open(os.path.join(pack_dir, "deploy", "loras.example.json"), "r", encoding="utf-8") as handle:
        example = json.load(handle)
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(lora_config(example, turbo_lora), handle, indent=2)
    if not turbo_lora:
        print("Note: no ref2v turbo LoRA found in models/loras; renders default to 30 steps.")


def start(
    comfy_dir: str,
    pack_dir: str,
    *,
    diffusion_model: str | None = None,
    text_encoder: str | None = None,
    unet_name: str | None = None,
    clip_name: str | None = None,
    video_vae: str | None = None,
    audio_vae: str | None = None,
    turbo_lora: str | None = None,
    attention: str = "sol scheduled",
    token: str | None = None,
    atlas_api_key: str | None = None,
    log_dir: str = "/content/hawk_logs",
) -> Session:
    """Start ComfyUI (or reuse it), the tunnel and the API. Model names left out are
    detected from ComfyUI/models."""
    os.makedirs(log_dir, exist_ok=True)
    models = resolve_models(
        comfy_dir, diffusion_model=diffusion_model, text_encoder=text_encoder, unet_name=unet_name,
        clip_name=clip_name, video_vae=video_vae, audio_vae=audio_vae, turbo_lora=turbo_lora,
    )
    print("Models:")
    for key, value in models.items():
        print(f"  {key:11} {value or '(none)'}")
    if not atlas_api_key:
        print("Note: no ATLAS_API_KEY secret, so planning (plan_film / story) will fail. Scripts still render.")
    if not token:
        token = secrets.token_hex(24)
        print("No HAWK_API_TOKEN secret; generated a token for this session (shown below).")

    data_dir = "/content/hawk_api_data" if os.path.isdir("/content") else os.path.abspath("hawk_api_data")
    _write_lora_config(pack_dir, data_dir, models["turbo_lora"])

    env = dict(os.environ)
    env.update({
        "HAWK_API_TOKEN": token,
        "COMFY_URL": f"http://127.0.0.1:{COMFY_PORT}",
        "DATA_DIR": data_dir,
        "MAX_UPLOAD_MB": str(TUNNEL_UPLOAD_MB),
        "COMFY_INPUT_DIR": os.path.join(comfy_dir, "input"),
        "COMFY_OUTPUT_DIR": os.path.join(comfy_dir, "output"),
        "HAWK_DRIVE_ROOT": "/content/drive/MyDrive",
        "HAWK_UNET": models["unet_name"],
        "HAWK_CLIP": models["clip_name"],
        "HAWK_VIDEO_VAE": models["video_vae"],
        "HAWK_AUDIO_VAE": models["audio_vae"],
        "HAWK_ATTENTION": attention,
    })
    if atlas_api_key:
        env["ATLAS_API_KEY"] = atlas_api_key

    session = Session(comfy_dir, pack_dir, token, env, log_dir)
    _start_comfyui(session)
    # API tunnels from an earlier run of this cell would keep serving old URLs.
    # Only ours: a ComfyUI UI tunnel started by the notebook keeps running.
    subprocess.run(["pkill", "-f", API_TUNNEL_PATTERN], capture_output=True)
    _start_tunnel(session)
    _start_api(session)
    print(session.summary())
    return session


def status_line(session: Session) -> str:
    health = _http_status(f"http://127.0.0.1:{API_PORT}/healthz")
    jobs = []
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{API_PORT}/v1/jobs?limit=5", headers={"Authorization": f"Bearer {session.token}"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            jobs = json.load(response).get("jobs", [])
    except Exception:
        pass
    parts = [f"{time.strftime('%H:%M:%S')} api={'ok' if health == 200 else health}"]
    for job in jobs:
        progress = job.get("progress") or {}
        done, total = progress.get("segments_done"), progress.get("segments_total")
        detail = f" {done}/{total}" if total else ""
        if job["status"] == "queued" and job.get("queue_position"):
            detail = f" #{job['queue_position']}"
        parts.append(f"{job['kind']} {job['id'][:8]} {job['status']}{detail}")
    return " | ".join(parts)


def watch(session: Session, interval: int = 60) -> None:
    """Print status every `interval` seconds and restart the tunnel or API if they die.
    Stop the cell to stop watching; the services keep running."""
    try:
        while True:
            comfy = session.procs.get("comfyui")
            comfy_dead = comfy.poll() is not None if comfy is not None else \
                _http_status(f"http://127.0.0.1:{COMFY_PORT}/queue", timeout=5) != 200
            if comfy_dead:
                print("ComfyUI stopped." + (f" Last log lines:\n{log_tail(session.log('comfyui'))}" if comfy else ""))
                print("Run the start cell again.")
                return
            if session.procs["tunnel"].poll() is not None:
                print("Tunnel stopped; opening a new one (the URL changes).")
                session.procs["api"].terminate()
                _start_tunnel(session)
                _start_api(session)
                print(session.summary())
            elif session.procs["api"].poll() is not None:
                print(f"API stopped; restarting. Last log lines:\n{log_tail(session.log('api'))}")
                _start_api(session)
            print(status_line(session), flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped watching. Services are still running.")


def restart_api(session: Session) -> None:
    """Reload the API code (e.g. after `git pull`) keeping ComfyUI and the tunnel URL."""
    api = session.procs.get("api")
    if api is not None and api.poll() is None:
        api.terminate()
        try:
            api.wait(timeout=20)
        except subprocess.TimeoutExpired:
            api.kill()
    _start_api(session)
    print(session.summary())


def restart_comfyui(session: Session, extra_args: list[str] | None = None, force: bool = False) -> None:
    """Reload the Hawk H3 node code (e.g. after `git pull`): stop ComfyUI and start it again.
    Models load again on the next render. Refuses while a prompt is running unless force=True."""
    base = f"http://127.0.0.1:{COMFY_PORT}"
    queue = _http_json(f"{base}/queue", timeout=5) or {}
    if queue.get("queue_running") and not force:
        raise RuntimeError("ComfyUI is rendering. Wait for it to finish (or cancel it), or pass force=True.")
    proc = session.procs.get("comfyui")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not port_free(COMFY_PORT):  # started by another cell
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], capture_output=True)
        subprocess.run(["pkill", "-f", f"main.py.*--port {COMFY_PORT}"], capture_output=True)
        subprocess.run(["fuser", "-k", f"{COMFY_PORT}/tcp"], capture_output=True)
    deadline = time.time() + 60
    while not port_free(COMFY_PORT) and time.time() < deadline:
        time.sleep(1)
    if not port_free(COMFY_PORT):
        raise RuntimeError(f"Port {COMFY_PORT} is still in use. Stop ComfyUI yourself (`!fuser -k {COMFY_PORT}/tcp`) and retry.")
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT), "--max-upload-size", "2048", *(extra_args or [])]
    session.procs["comfyui"] = _popen(cmd, session.comfy_dir, session.env, session.log("comfyui"))
    print("Restarting ComfyUI...", flush=True)
    _wait_http(f"{base}/queue", session.procs["comfyui"], session.log("comfyui"), 900, "ComfyUI")
    if not _http_json(f"{base}/object_info/HawkH3Director"):
        raise RuntimeError(f"ComfyUI restarted but the Hawk H3 nodes did not load. Log:\n{log_tail(session.log('comfyui'), 80)}")
    print("ComfyUI is up with the updated Hawk H3 nodes. The first render reloads the models.")


def show_logs(session: Session, lines: int = 60) -> None:
    for name in ("comfyui", "tunnel", "api"):
        print(f"----- {name} -----\n{log_tail(session.log(name), lines)}")


def stop(session: Session) -> None:
    for name, proc in session.procs.items():
        if proc is not None and proc.poll() is None:
            proc.terminate()
            print(f"stopped {name}")
