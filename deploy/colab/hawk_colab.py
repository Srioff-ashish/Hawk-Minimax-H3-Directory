"""Google Colab launcher: ComfyUI + the Hawk H3 API behind a Cloudflare quick tunnel.

Used by deploy/colab/Hawk_H3_API_Colab.ipynb. Every session starts from scratch:

1. install ComfyUI, this pack, Sol attention and the API requirements
2. download the MiniMax H3 models and LoRAs from Hugging Face
3. start ComfyUI on 127.0.0.1:8188 (never exposed)
4. open a Cloudflare quick tunnel to the API port and read its URL
5. start the API with that URL, then print the connector links

Nothing survives the runtime. Only the standard library is imported at module level;
the pure helpers (model manifest, LoRA sources, tunnel URL, torch check) are
unit-tested in tests_api/test_colab.py.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
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

COMFY_PORT = 8188
API_PORT = 8000
#: Cloudflare's free proxy rejects request bodies over 100 MB.
TUNNEL_UPLOAD_MB = 100
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
EXTRA_NODES = {
    "sol": "https://github.com/Saganaki22/ComfyUI-sol-attn",
    "vfi": "https://github.com/GACLove/ComfyUI-VFI",
}


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
_MODEL_EXT = (".safetensors", ".pt", ".pth", ".ckpt", ".bin")


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


def _wait_http(url: str, proc: subprocess.Popen, log_path: str, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{label} exited during startup. Last log lines:\n{log_tail(log_path)}")
        if _http_status(url, timeout=5) == 200:
            return
        time.sleep(3)
    raise RuntimeError(f"{label} did not answer at {url} within {timeout:.0f}s. Last log lines:\n{log_tail(log_path)}")


# ------------------------------------------------------------------ install


def install(comfy_dir: str, pack_dir: str, sol: bool = True, vfi: bool = False) -> None:
    info = torch_info()
    print(f"GPU: {info.get('name')} (capability {info.get('cap')}), torch {info.get('version')} CUDA {info.get('cuda')}")
    if not info.get("cap"):
        raise RuntimeError("No CUDA GPU found. Runtime -> Change runtime type -> G4 GPU.")
    if needs_blackwell_torch(info):
        print("This torch build has no Blackwell kernels; installing a CUDA 12.8 build.")
        _pip("--upgrade", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128")

    _clone("https://github.com/comfyanonymous/ComfyUI", comfy_dir)
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
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT), "--max-upload-size", "2048"]
    session.procs["comfyui"] = _popen(cmd, session.comfy_dir, session.env, session.log("comfyui"))
    print("Starting ComfyUI (first start loads nodes; a few minutes)...", flush=True)
    _wait_http(f"http://127.0.0.1:{COMFY_PORT}/queue", session.procs["comfyui"], session.log("comfyui"), 900, "ComfyUI")
    print("ComfyUI is up.")


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


def _start_api(session: Session) -> None:
    env = dict(session.env, PUBLIC_BASE_URL=session.public_url)
    cmd = [sys.executable, "-m", "uvicorn", "--factory", "hawk_api.app:create_app",
           "--host", "127.0.0.1", "--port", str(API_PORT), "--proxy-headers"]
    session.procs["api"] = _popen(cmd, session.pack_dir, env, session.log("api"))
    _wait_http(f"http://127.0.0.1:{API_PORT}/healthz", session.procs["api"], session.log("api"), 180, "Hawk H3 API")
    deadline = time.time() + 120
    while time.time() < deadline:  # the new trycloudflare hostname needs a moment to resolve
        if _http_status(f"{session.public_url}/healthz", timeout=10) == 200:
            print("API reachable through the tunnel.")
            return
        time.sleep(5)
    print("Warning: the API runs locally but the tunnel URL is not reachable yet; try the health link in a minute.")


def start(
    comfy_dir: str,
    pack_dir: str,
    *,
    diffusion_model: str,
    text_encoder: str,
    attention: str = "sol scheduled",
    token: str | None = None,
    atlas_api_key: str | None = None,
    log_dir: str = "/content/hawk_logs",
) -> Session:
    os.makedirs(log_dir, exist_ok=True)
    if not atlas_api_key:
        print("Note: no ATLAS_API_KEY secret, so planning (plan_film / story) will fail. Scripts still render.")
    if not token:
        token = secrets.token_hex(24)
        print("No HAWK_API_TOKEN secret; generated a token for this session (shown below).")
    env = dict(os.environ)
    env.update({
        "HAWK_API_TOKEN": token,
        "COMFY_URL": f"http://127.0.0.1:{COMFY_PORT}",
        "DATA_DIR": "/content/hawk_api_data" if os.path.isdir("/content") else os.path.abspath("hawk_api_data"),
        "MAX_UPLOAD_MB": str(TUNNEL_UPLOAD_MB),
        "HAWK_UNET": os.path.basename(DIFFUSION_MODELS[diffusion_model][0]),
        "HAWK_CLIP": os.path.basename(TEXT_ENCODERS[text_encoder][0]),
        "HAWK_ATTENTION": attention,
    })
    if atlas_api_key:
        env["ATLAS_API_KEY"] = atlas_api_key

    session = Session(comfy_dir, pack_dir, token, env, log_dir)
    _start_comfyui(session)
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
        parts.append(f"{job['kind']} {job['id'][:8]} {job['status']}" + (f" {done}/{total}" if total else ""))
    return " | ".join(parts)


def watch(session: Session, interval: int = 60) -> None:
    """Print status every `interval` seconds and restart the tunnel or API if they die.
    Stop the cell to stop watching; the services keep running."""
    try:
        while True:
            if session.procs["comfyui"].poll() is not None:
                print(f"ComfyUI stopped. Last log lines:\n{log_tail(session.log('comfyui'))}")
                print("Run the Start cell again.")
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


def show_logs(session: Session, lines: int = 60) -> None:
    for name in ("comfyui", "tunnel", "api"):
        print(f"----- {name} -----\n{log_tail(session.log(name), lines)}")


def stop(session: Session) -> None:
    for name, proc in session.procs.items():
        if proc.poll() is None:
            proc.terminate()
            print(f"stopped {name}")
