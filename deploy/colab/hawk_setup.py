"""One-call setup for the Hawk H3 Studio Colab notebook: secrets, Sage, custom nodes, model files and LoRAs.

The notebook keeps only data (which models, which LoRAs, which nodes) and calls ``setup()``; the logic lives here,
so a fix reaches the next run with a ``git pull`` instead of a hand-edited cell. Behaviour is the notebook's own,
moved, with three speedups:

* model files download 4 at a time (the link is the limit, not the disk);
* a custom node whose commit did not change skips its ``pip install -r``;
* the Sage wheel is reused from Drive when one was built for this exact torch and Python.

Nothing here imports torch or ComfyUI at module level, so it is importable (and testable) anywhere.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DRIVE = "/content/drive/MyDrive"
WHEEL_CACHE = f"{DRIVE}/hawk_wheels"
NODE_STAMP = ".hawk_installed_commit"
LORA_RECEIPT = ".hawk_lora_removals.json"
CHECK_NODES = ("PH3ImageConditioning", "MiniMaxH3SigmaShift", "LoraLoaderBypassModelOnly")


def _sh(cmd: str, env: dict | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        print("$", cmd, flush=True)
    result = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.stdout and not quiet:
        print(result.stdout, flush=True)
    result.check_returncode()
    return result


# --------------------------------------------------------------------------- secrets

def load_secrets(required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict[str, bool]:
    """Copy Colab secrets into the environment. A missing optional secret is skipped, not a crash.

    ``os.environ[key] = userdata.get(key)`` raised TypeError for any secret that was not set, which is why every
    optional key used to be mandatory in practice."""
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:  # not on Colab: whatever is already in the environment counts
        userdata = None

    def read(key: str) -> str | None:
        if userdata is not None:
            try:
                value = userdata.get(key)
            except Exception:  # SecretNotFoundError / NotebookAccessError
                value = None
            if value:
                return str(value)
        return os.environ.get(key) or None

    found, missing = {}, []
    for key in (*required, *optional):
        value = read(key)
        if value:
            os.environ[key] = value
        found[key] = bool(value)
        if not value and key in required:
            missing.append(key)
    print("secrets:", "  ".join(f"{k} {'ok' if v else '-'}" for k, v in found.items()))
    if missing:
        raise SystemExit(f"Missing required Colab secrets: {', '.join(missing)}")
    return found


# --------------------------------------------------------------------------- sage

def build_sage(cache_root: str = WHEEL_CACHE, arch: str = "12.0", cuda: str = "13.0") -> str:
    """SageAttention built against this runtime's torch, cached on Drive per torch + Python version.

    The notebook's own recipe: nvcc for the torch's CUDA, source switched to C++20 (torch 2.14 headers need it),
    built for one GPU arch, checked with one real call on the GPU before the wheel is cached."""
    from importlib.metadata import version

    torch_version = version("torch")
    python = f"py{sys.version_info.major}{sys.version_info.minor}"
    cache = f"{cache_root}/sage-2.2.0-torch{torch_version}-{python}"
    print("torch", torch_version, "| cache", cache, flush=True)
    _sh(f"{sys.executable} -m pip uninstall -y -q sageattention || true", quiet=True)
    cached = glob.glob(f"{cache}/sageattention-*.whl")
    wheel = cached[0] if cached else ""
    if wheel:
        _sh(f'{sys.executable} -m pip install -q --no-deps "{wheel}"')
    else:
        cuda_home = f"/usr/local/cuda-{cuda}"
        if not os.path.exists(f"{cuda_home}/bin/nvcc"):
            _sh("cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
                "cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && apt-get -qq update"
                f" && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y cuda-toolkit-{cuda.replace('.', '-')} >/dev/null")
        _sh("rm -rf /content/SageAttention /content/sage_wheel && "
            "git clone -q https://github.com/thu-ml/SageAttention /content/SageAttention")
        setup_py = "/content/SageAttention/setup.py"
        with open(setup_py, encoding="utf-8") as handle:
            source = handle.read()
        flags = source.count("-std=c++17")
        if flags != 2:
            raise RuntimeError(f"expected 2 -std=c++17 flags in SageAttention's setup.py, found {flags}")
        with open(setup_py, "w", encoding="utf-8") as handle:
            handle.write(source.replace("-std=c++17", "-std=c++20"))
        env = {**os.environ, "CUDA_HOME": cuda_home, "PATH": f"{cuda_home}/bin:" + os.environ["PATH"],
               "TORCH_CUDA_ARCH_LIST": arch, "MAX_JOBS": "16", "EXT_PARALLEL": "4", "NVCC_APPEND_FLAGS": "--threads 8"}
        print("building SageAttention (this takes a while)...", flush=True)
        result = subprocess.run(f"{sys.executable} -m pip wheel . -v --no-build-isolation --no-deps -w /content/sage_wheel",
                                shell=True, cwd="/content/SageAttention", env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open("/content/sage_build.log", "w", encoding="utf-8") as handle:
            handle.write(result.stdout)
        if result.returncode:
            lines = result.stdout.splitlines()
            hits = [l[:400] for l in lines if re.search(r"\berror\b|fatal|Traceback|No such file|Killed", l, re.I)]
            print("=== FIRST ERROR LINES ===\n" + "\n".join(hits[:40]))
            print("\n=== LAST 20 LINES ===\n" + "\n".join(l[:400] for l in lines[-20:]))
            raise RuntimeError("Sage build failed; full log in /content/sage_build.log")
        wheel = glob.glob("/content/sage_wheel/sageattention-*.whl")[0]
        _sh(f'{sys.executable} -m pip install -q --no-deps "{wheel}"')
    _sh(f'''cd /tmp && CUDA_LAUNCH_BLOCKING=1 {sys.executable} -c "
import torch; from sageattention import sageattn
q=k=v=torch.randn(1,24,32768,128,device='cuda',dtype=torch.bfloat16)
sageattn(q,k,v,tensor_layout='HND'); torch.cuda.synchronize(); print('SAGE 2 OK on', torch.cuda.get_device_name())"''')
    if not cached and os.path.isdir(DRIVE):  # only a wheel that passed the GPU check is ever reused
        os.makedirs(cache, exist_ok=True)
        shutil.copy(wheel, cache)
    return wheel


# --------------------------------------------------------------------------- ComfyUI and nodes

def install_comfyui(workspace: str, requirements: bool = True) -> str:
    """ComfyUI on origin/master (P-H3-Image needs master); a reset, because a pull fails silently on a tag."""
    if not os.path.isdir(f"{workspace}/.git"):
        _sh(f"git clone -q https://github.com/comfyanonymous/ComfyUI.git {workspace}", quiet=True)
    before = _git(workspace, "rev-parse", "HEAD")
    _sh(f"git -C {workspace} fetch -q origin master && git -C {workspace} checkout -q master && "
        f"git -C {workspace} reset -q --hard origin/master", quiet=True)
    after = _git(workspace, "rev-parse", "HEAD")
    if requirements and (before != after or not os.path.exists(f"{workspace}/{NODE_STAMP}")):
        _sh(f"{sys.executable} -m pip install -q -r {workspace}/requirements.txt", quiet=True)
        with open(f"{workspace}/{NODE_STAMP}", "w", encoding="utf-8") as handle:
            handle.write(after)
    print("ComfyUI", _git(workspace, "log", "-1", "--format=%h %cs"))
    return after


def _git(path: str, *args: str) -> str:
    return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True).stdout.strip()


def install_nodes(workspace: str, nodes: dict[str, str]) -> dict[str, str]:
    """Clone or pull each custom node; install its requirements only when its commit changed."""
    os.makedirs(f"{workspace}/custom_nodes", exist_ok=True)
    report = {}
    for name, url in nodes.items():
        folder = f"{workspace}/custom_nodes/{name}"
        if os.path.isdir(folder):
            subprocess.run(["git", "-C", folder, "pull", "-q"], check=False)
        else:
            subprocess.run(["git", "clone", "-q", "--depth", "1", url, folder], check=False)
        commit = _git(folder, "rev-parse", "HEAD")
        stamp = f"{folder}/{NODE_STAMP}"
        done = os.path.exists(stamp) and open(stamp, encoding="utf-8").read().strip() == commit
        requirements = f"{folder}/requirements.txt"
        if os.path.exists(requirements) and not done:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", requirements], check=False)
        if commit:
            with open(stamp, "w", encoding="utf-8") as handle:
                handle.write(commit)
        report[name] = "unchanged" if done else ("installed" if commit else "FAILED")
        print(f"  {report[name]:<9} {name}")
    return report


def check_core_nodes(workspace: str) -> dict[str, bool]:
    """The core ComfyUI nodes the P-H3-Image graphs need, and the ph3 engine in this pack."""
    found = {}
    sources = [f"{workspace}/nodes.py", *glob.glob(f"{workspace}/comfy_extras/*.py")]
    text = ""
    for path in sources:
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                text += handle.read()
        except OSError:
            continue
    for node in CHECK_NODES:
        found[node] = node in text
    engine = f"{workspace}/custom_nodes/Hawk-Minimax-H3-Directory/hawk_api/local_images.py"
    found["ph3 engine"] = os.path.exists(engine) and "def ph3_graph" in open(engine, encoding="utf-8").read()
    for node, ok in found.items():
        print(f"  {'ok     ' if ok else 'MISSING'} {node}")
    return found


# --------------------------------------------------------------------------- model files

def model_path(models_dir: str, entry: tuple) -> str:
    """(repo, path in repo, models folder, saved name or None, GB or None) -> where it lands."""
    _repo, path, folder, name, _gb = entry
    return os.path.join(models_dir, folder, name or path.rsplit("/", 1)[-1])


def _have(out: str, gb: float | None) -> bool:
    if not os.path.exists(out) or os.path.exists(out + ".aria2"):
        return False
    size = os.path.getsize(out) / 1e9
    return gb is None or abs(size - gb) <= gb * 0.01


def _aria2(url: str, out: str, headers: tuple[str, ...] = (), connections: int = 16) -> bool:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    args = []
    for header in headers:
        args += ["--header", header]
    result = subprocess.run(
        ["aria2c", f"-x{connections}", f"-s{connections}", "-k1M", "--continue=true", "--max-tries=5",
         "--retry-wait=5", "--auto-file-renaming=false", "--allow-overwrite=true", "--summary-interval=0",
         "--console-log-level=warn", "--download-result=hide", *args, "-d", os.path.dirname(out),
         "-o", os.path.basename(out), url])
    return result.returncode == 0 and os.path.exists(out)


def download_models(models_dir: str, files: list[tuple], hf_token: str | None = None, parallel: int = 4) -> list[str]:
    """Hugging Face files, several at once. Returns the paths that failed."""
    if not shutil.which("aria2c"):
        _sh("apt-get -qq install -y aria2 > /dev/null", quiet=True)
    headers = (f"Authorization: Bearer {hf_token}",) if hf_token else ()
    todo, failed = [], []
    for entry in files:
        out = model_path(models_dir, entry)
        if _have(out, entry[4]):
            print(f"have  {os.path.relpath(out, models_dir)}")
        else:
            todo.append(entry)
    total = sum(e[4] or 0 for e in todo)
    print(f"{len(todo)} to download (~{total:.0f} GB listed), {shutil.disk_usage(models_dir if os.path.isdir(models_dir) else '/').free / 1e9:.0f} GB free", flush=True)

    def one(entry: tuple) -> tuple[tuple, bool, float]:
        repo, path = entry[0], entry[1]
        started = time.time()
        ok = _aria2(f"https://huggingface.co/{repo}/resolve/main/{path}", model_path(models_dir, entry), headers,
                    connections=max(4, 16 // max(1, parallel // 2)))
        return entry, ok, time.time() - started

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for future in as_completed([pool.submit(one, entry) for entry in todo]):
            entry, ok, seconds = future.result()
            label = os.path.relpath(model_path(models_dir, entry), models_dir)
            print(f"{'done' if ok else 'FAILED'}  {label}  ({seconds:.0f}s)", flush=True)
            if not ok:
                failed.append(entry[1])
    print("failed:", failed or "none")
    return failed


# --------------------------------------------------------------------------- LoRAs

def is_safetensors(path: str) -> bool:
    """A real .safetensors starts with a uint64 header length and then JSON. Civitai answers a bad token with a 200
    and an HTML page, which a size check alone lets through."""
    try:
        if os.path.getsize(path) < 1_000_000:
            return False
        with open(path, "rb") as handle:
            head = handle.read(9)
        return len(head) == 9 and 0 < int.from_bytes(head[:8], "little") < 100_000_000 and head[8:9] == b"{"
    except OSError:
        return False


def _civitai(out: str, version: int, token: str) -> bool:
    url = f"https://civitai.com/api/download/models/{version}?token={token}"
    for _attempt in range(2):
        subprocess.run(["wget", "-q", "--content-disposition=off", "-O", out, url], check=False)
        if is_safetensors(out):
            return True
        if os.path.exists(out):
            os.remove(out)
    return False


def sync_loras(lora_dir: str, loras: dict[str, dict[str, int]], token: str, remove: list[str] = (),
               dry_run: bool = True, parallel: int = 3) -> dict:
    """One folder per family; loose files moved in; missing ones fetched from Civitai; the removal list applied.

    Removals are recorded in a receipt so a file you later put back by hand is never taken away again, and nothing
    on the install list is ever deleted, whatever the removal list says."""
    if not token:
        raise SystemExit("Set CIVITAI_TOKEN first. Civitai returns an HTML login page without it.")
    os.makedirs(lora_dir, exist_ok=True)
    moved = 0
    for family, files in loras.items():
        os.makedirs(f"{lora_dir}/{family}", exist_ok=True)
        for name in files:
            flat, nested = f"{lora_dir}/{name}", f"{lora_dir}/{family}/{name}"
            if os.path.exists(flat) and not os.path.exists(nested):
                os.replace(flat, nested)
                print(f"moved {name} -> {family}/")
                moved += 1

    todo = []
    skipped = 0
    for family, files in loras.items():
        for name, version in files.items():
            out = f"{lora_dir}/{family}/{name}"
            if is_safetensors(out):
                skipped += 1
                continue
            if os.path.exists(out):
                os.remove(out)  # a half-download or an HTML error page
            todo.append((family, name, version, out))
    failed, fetched = [], 0
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {pool.submit(_civitai, out, version, token): (family, name, version) for family, name, version, out in todo}
        for future in as_completed(futures):
            family, name, version = futures[future]
            if future.result():
                fetched += 1
                print(f"   ok  {family}/{name}", flush=True)
            else:
                failed.append(f"{family}/{name} (v{version})")
    print(f"\n{moved} moved, {fetched} downloaded, {skipped} already present, {len(failed)} failed")
    for item in failed:
        print("  failed -", item, "(check CIVITAI_TOKEN and 'Show mature content' on the account)")

    receipt_path = f"{lora_dir}/{LORA_RECEIPT}"
    try:
        with open(receipt_path, encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError):
        receipt = {}
    keep = {os.path.abspath(f"{lora_dir}/{fam}/{name}") for fam, files in loras.items() for name in files}
    freed, newly = 0.0, {}
    for name in remove:
        if name in receipt:
            continue
        here = next((p for p in (f"{lora_dir}/klein/{name}", f"{lora_dir}/{name}") if os.path.isfile(p)), None)
        if here is None:
            newly[name] = {"at": time.strftime("%Y-%m-%d %H:%M"), "gb": 0.0, "note": "already absent"}
            continue
        if os.path.abspath(here) in keep:
            print(f"  REFUSED       {name} -- still on the install list")
            continue
        size = os.path.getsize(here) / 2**30
        print(f"  {'would remove' if dry_run else 'removing'}  {os.path.relpath(here, lora_dir)}  ({size:.2f} GB)")
        freed += size
        if not dry_run:
            os.remove(here)
            newly[name] = {"at": time.strftime("%Y-%m-%d %H:%M"), "gb": round(size, 2)}
    if newly and not dry_run:  # a dry run leaves no trace, or the next run would think the work was done
        receipt.update(newly)
        tmp = receipt_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
        os.replace(tmp, receipt_path)
    if dry_run and freed:
        print(f"{freed:.1f} GB would be freed -- set DRY_RUN_LORA_DELETE = False and re-run to delete.")
    for family in loras:
        folder = f"{lora_dir}/{family}"
        names = [f for f in os.listdir(folder) if f.endswith(".safetensors")]
        size = sum(os.path.getsize(f"{folder}/{f}") for f in names)
        print(f"  {family:6} {len(names):2}/{len(loras[family]):<2} files  {size / 2**30:5.1f} GB")
    loose = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
    if loose:
        print(f"\nstill loose in loras/ (classified by file name): {loose}")
    return {"moved": moved, "fetched": fetched, "skipped": skipped, "failed": failed, "freed_gb": round(freed, 2)}


# --------------------------------------------------------------------------- everything

def setup(*, workspace: str, nodes: dict[str, str], model_files: list[tuple], loras: dict[str, dict[str, int]],
          remove_loras: list[str] = (), dry_run_lora_delete: bool = True, sage: bool = True,
          parallel_downloads: int = 4) -> dict:
    """ComfyUI, nodes, Sage, model files and LoRAs, in the order the notebook ran them."""
    started = time.time()
    install_comfyui(workspace)
    print("\ncustom nodes:")
    install_nodes(workspace, nodes)
    check_core_nodes(workspace)
    if sage:
        print("\nSageAttention:")
        build_sage()
    print("\nmodel files:")
    failed = download_models(f"{workspace}/models", model_files, os.environ.get("HF_TOKEN"), parallel_downloads)
    print("\nLoRAs:")
    lora_report = sync_loras(f"{workspace}/models/loras", loras, os.environ.get("CIVITAI_TOKEN", "").strip(),
                             list(remove_loras), dry_run_lora_delete)
    print(f"\nsetup finished in {(time.time() - started) / 60:.1f} min")
    return {"failed_models": failed, "loras": lora_report}
