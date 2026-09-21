"""Job store (SQLite) and HawkService -- the one place REST and MCP both call."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
import random
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid

import httpx

from hawk_h3.script import ScriptError, build_jobs, parse_script

from . import graph as graphs
from . import image_engines
from .atlas import AtlasClient, AtlasError
from .auth import sign_path
from .prompts import PLATFORM_RULES, PromptStore
from .comfy_client import ComfyClient, ComfyError, ComfyNotFound, ComfyValidationError
from .config import ModelSettings, Settings
from .local_images import LocalImageEngine, LocalImageError
from .loras import (
    LoraError,
    LoraSpec,
    ResolvedLora,
    choose_steps,
    compare_applied,
    default_status,
    load_config,
    parse_applied,
    resolve_name,
    resolve_request,
)
from .schemas import PlannerOptions, PlanRequest, ReferenceIn, RenderSettings, VideoRequest

log = logging.getLogger("hawk_api")

ACTIVE = ("queued", "planning", "rendering")
FINISHED = ("done", "failed", "cancelled")
#: A job whose prompt ComfyUI no longer knows is only declared lost after this grace period.
LOST_AFTER_SECONDS = 20.0
DEFAULT_COLLECTION = "Uploads"
# The Atlas model ids live in image_engines, where the engine descriptors name them.
IMAGE_MODEL = image_engines.IMAGE_MODEL
IMAGE_EDIT_MODEL = image_engines.IMAGE_EDIT_MODEL
IMAGE_LITE_MODEL = image_engines.IMAGE_LITE_MODEL
IMAGE_LITE_EDIT_MODEL = image_engines.IMAGE_LITE_EDIT_MODEL
IMAGE_FAST_MODEL = image_engines.IMAGE_FAST_MODEL
#: Aliases for the ``model`` field (an Atlas model id), which is not the same vocabulary as ``engine``:
#: model "z-image" still means the Atlas one, while engine "z-image" now means the local engine.
IMAGE_ALIASES = {
    "turbo": IMAGE_FAST_MODEL, "z-image": IMAGE_FAST_MODEL, "zimage": IMAGE_FAST_MODEL, "z-image-turbo": IMAGE_FAST_MODEL,
    "fast": IMAGE_FAST_MODEL, "cheap": IMAGE_FAST_MODEL,
    "seedream": IMAGE_MODEL, "quality": IMAGE_MODEL, "best": IMAGE_MODEL,
    "seedream-lite": IMAGE_LITE_MODEL, "lite": IMAGE_LITE_MODEL,
}
# Seedream 5 sizes (Atlas presets). Pro bills by output pixels: up to 2.36 MP is the 1.5K tier, above it the 2K tier at
# twice the price, so Pro stays at 1.5K unless a bigger size is asked for. Lite starts at 2K.
SEEDREAM_15K_PIXELS = 2_359_296
SEEDREAM_PRO_15K = ((1536, 1536), (1776, 1328), (1328, 1776), (2048, 1152), (1152, 2048), (1024, 1024))
SEEDREAM_PRO_2K = ((2048, 2048), (2304, 1728), (1728, 2304), (2720, 1530), (1530, 2720), (2496, 1664), (1664, 2496))
SEEDREAM_LITE = ((2048, 2048), (2304, 1728), (1728, 2304), (2848, 1600), (1600, 2848), (2496, 1664), (1664, 2496))
# Estimated USD per image on Atlas (discounted list prices, Sept 2026); reported as cost_usd so the agent can budget.
IMAGE_PRICES = {"z-image": 0.01, "pro-1.5k": 0.036, "pro-2k": 0.072, "lite": 0.032}
SEEDREAM_EXTRA_REFERENCE = 0.003  # each reference image after the first


def _text_only_image_model(model: str) -> bool:
    return model.startswith("z-image/")


def _parse_size(size: str | None) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*[x*×]\s*(\d+)\s*", size or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def _star_size_loose(size: str) -> str:
    parsed = _parse_size(size)
    return f"{parsed[0]}*{parsed[1]}" if parsed else size


def seedream_size(model: str, size: str | None) -> tuple[str, str]:
    """(Atlas "W*H", price tier) for a Seedream 5 request: the preset closest in aspect ratio (larger on a tie, same
    price). Pro uses the 1.5K tier unless the request itself is above 2.36 MP. No size means portrait 2:3."""
    parsed = _parse_size(size)
    if size and parsed is None:
        raise RequestError(f"size {size!r} should look like 1024x1536.")
    width, height = parsed or (1024, 1536)
    if "lite" in model:
        presets, tier = SEEDREAM_LITE, "lite"
    elif width * height > SEEDREAM_15K_PIXELS:
        presets, tier = SEEDREAM_PRO_2K, "pro-2k"
    else:
        presets, tier = SEEDREAM_PRO_15K, "pro-1.5k"
    want = math.log(width / height)
    best = min(presets, key=lambda p: (round(abs(math.log(p[0] / p[1]) - want), 3), -p[0] * p[1]))
    return f"{best[0]}*{best[1]}", tier


def _star_size(size: str) -> str:
    """z-image takes "W*H", each side 512-2048."""
    match = re.fullmatch(r"\s*(\d+)\s*[x*×]\s*(\d+)\s*", size or "")
    if not match:
        raise RequestError(f"size {size!r} should look like 1024x1536.")
    width, height = int(match.group(1)), int(match.group(2))
    if not (512 <= width <= 2048 and 512 <= height <= 2048):
        raise RequestError("z-image/turbo sizes run 512-2048 on each side, e.g. 1024x1536 or 1536x1536.")
    return f"{width}*{height}"
THUMB_WIDTHS = (160, 320, 640, 1280, 2048)  # 1280 / 2048: the full-screen viewer
#: models folder -> (file-name family the Director can use, label for errors)
MODEL_FAMILIES = {"diffusion_models": ("ref2va", "Base model"), "text_encoders": ("qwen3vl", "Text encoder")}


def model_family(folder: str, files: list[str]) -> list[str]:
    family = MODEL_FAMILIES[folder][0]
    return [name for name in files if family in os.path.basename(name).lower()]


def normalize_tags(tags) -> list[str]:
    if isinstance(tags, str):
        tags = tags.split(",")
    seen: list[str] = []
    for tag in tags or []:
        tag = re.sub(r"\s+", " ", str(tag)).strip().lower()[:40]
        if tag and tag not in seen:
            seen.append(tag)
    return seen[:30]


def _hash_file(fileobj) -> str | None:
    """sha256 of a seekable file object, rewound afterwards; None when it cannot seek."""
    try:
        start = fileobj.tell()
        digest = hashlib.sha256()
        for chunk in iter(lambda: fileobj.read(1 << 20), b""):
            digest.update(chunk)
        fileobj.seek(start)
        return digest.hexdigest()
    except (AttributeError, OSError, ValueError):
        return None


def job_title(job: dict) -> str:
    """A short human title: the plan's title, the first segment title(s), or the first prompt line."""
    script = job.get("script") or ""
    try:
        data = json.loads(script)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        segments = data.get("segments") or []
        if data.get("title"):
            return str(data["title"])[:80]
        if segments and isinstance(segments[0], dict) and segments[0].get("title"):
            first = str(segments[0]["title"])
            return f"{first} +{len(segments) - 1}" if len(segments) > 1 else first
    titles = re.findall(r"^\s*title\s*:\s*(.+)$", script, re.IGNORECASE | re.MULTILINE)
    if titles:
        return f"{titles[0].strip()} +{len(titles) - 1}" if len(titles) > 1 else titles[0].strip()
    for line in script.splitlines():
        line = line.strip()
        if line and not re.match(r"^(-{3,}|(title|duration|seconds|pictures|images|videos|audios|poses|seed|continuity|style)\s*:)", line, re.I):
            return line[:77] + "..." if len(line) > 80 else line
    return "Plan" if job.get("kind") == "plan" else "Video"


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def drive_links(drive: dict | None) -> dict | None:
    if not drive:
        return None
    links = dict(drive)
    file_id = drive.get("file_id")
    if file_id and not str(file_id).lower().startswith("local"):  # "local-<n>": upload not finished
        links.update(
            preview_url=f"https://drive.google.com/file/d/{file_id}/preview",
            view_url=f"https://drive.google.com/file/d/{file_id}/view",
            download_url=f"https://drive.google.com/uc?id={file_id}&export=download",
            thumb_url=f"https://drive.google.com/thumbnail?id={file_id}&sz=w640",
        )
    return links


def _image_type(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x89PNG"):
        return "png", "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg", "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return "png", "image/png"


def script_warnings(text: str) -> list[str]:
    """The planner's notes about what it fixed, carried in its JSON script."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    warnings = data.get("warnings") if isinstance(data, dict) else None
    return [str(item) for item in warnings] if isinstance(warnings, list) else []


class RequestError(ValueError):
    """Bad input -> HTTP 422."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class NotFound(LookupError):
    """-> HTTP 404."""


class Conflict(RuntimeError):
    """-> HTTP 409."""


class Unavailable(RuntimeError):
    """ComfyUI unreachable -> HTTP 503."""


def asset_kind(filename: str, content_type: str | None) -> str:
    guess = (content_type or "").split(";")[0].strip().lower()
    if not guess or guess == "application/octet-stream":
        guess = mimetypes.guess_type(filename)[0] or ""
    for kind in ("image", "audio", "video"):
        if guess.startswith(kind + "/"):
            return kind
    raise RequestError(f"{filename!r} is not an image, audio or video file (type {guess or 'unknown'}).")


# ------------------------------------------------------------------- store


class Store:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, prompt_id TEXT,
        data TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS jobs_prompt ON jobs(prompt_id);
    CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(self._SCHEMA)
            self._db.commit()

    def add_asset(self, asset: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO assets VALUES (?, ?, ?)", (asset["id"], json.dumps(asset), asset["created_at"])
            )
            self._db.commit()

    def get_asset(self, asset_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM assets WHERE id = ?", (asset_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_assets(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM assets ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def all_assets(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM assets ORDER BY created_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete_asset(self, asset_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
            self._db.commit()

    def save_job(self, job: dict) -> None:
        job["updated_at"] = time.time()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job["id"], job["kind"], job["status"], job.get("prompt_id"), json.dumps(job),
                 job["created_at"], job["updated_at"]),
            )
            self._db.commit()

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def find_by_prompt(self, prompt_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM jobs WHERE prompt_id = ?", (prompt_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_jobs(self, limit: int = 50, statuses: tuple[str, ...] | None = None) -> list[dict]:
        query, args = "SELECT data FROM jobs", []
        if statuses:
            query += f" WHERE status IN ({','.join('?' * len(statuses))})"
            args.extend(statuses)
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(query, args).fetchall()
        return [json.loads(row[0]) for row in rows]


# ----------------------------------------------------------------- service


def _combo_options(info: dict, name: str) -> list:
    for section in ("required", "optional"):
        spec = (info.get("input") or {}).get(section, {}).get(name)
        if not spec:
            continue
        if isinstance(spec[0], list):
            return spec[0]
        if len(spec) > 1 and isinstance(spec[1], dict):
            return list(spec[1].get("options") or [])
    return []


class HawkService:
    #: Set by create_app when Google Drive is mounted: generated images are copied there in the background.
    drive_exporter = None

    def __init__(self, settings: Settings, store: Store | None = None, comfy: ComfyClient | None = None):
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self.comfy = comfy or ComfyClient(settings.comfy_url)
        self.atlas = AtlasClient(settings.atlas_url, settings.atlas_api_key)
        self.prompts = PromptStore(settings.data_dir)
        self.image_engines = image_engines.ImageEngineStore(settings.data_dir)
        #: Called with the job id when a render finishes (e.g. the Google Drive exporter).
        self.render_done_hooks: list = []
        self.local_images = LocalImageEngine(self)
        self._model_cache: dict[str, tuple[float, list[str]]] = {}
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        load_config(self.settings.loras_path)  # create loras.json on first start
        self._tasks = [
            asyncio.create_task(self.comfy.listen(self.handle_event)),
            asyncio.create_task(self._reconcile_loop()),
        ]
        health = await self.health()
        if health.get("loras") == "degraded":
            log.warning("hawk_api: required default LoRAs missing on the pod: %s", health.get("missing_required_loras"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.comfy.close()

    # ----------------------------------------------------------- LoRAs

    async def available_models(self, folder: str, refresh: bool = False) -> list[str]:
        stamp, files = self._model_cache.get(folder, (0.0, []))
        if refresh or time.monotonic() - stamp > self.settings.lora_cache_seconds:
            try:
                files = await self.comfy.list_models(folder)
            except ComfyError as exc:
                raise Unavailable(str(exc)) from exc
            self._model_cache[folder] = (time.monotonic(), files)
        return files

    async def available_loras(self, refresh: bool = False, images: bool = False) -> list[str]:
        """LoRA files for video renders. ComfyUI keeps one models/loras folder, so the Krea 2 image LoRAs sit next to
        the MiniMax H3 ones; they are left out here (and video LoRAs never reach image generation, which matches
        against its own catalogue). images=True returns the folder as it is."""
        files = await self.available_models("loras", refresh)
        if images:
            return files
        image_files = await self.local_images.lora_basenames()
        return [f for f in files if os.path.basename(f).lower() not in image_files]

    async def choose_model(self, requested: str | None, folder: str, default: str) -> str:
        """A per-render base model or text encoder, matched like LoRA names."""
        if not requested or not requested.strip():
            return default
        _, label = MODEL_FAMILIES[folder]
        for refresh in (False, True):  # a file may have been added since the cached listing
            files = await self.available_models(folder, refresh=refresh)
            choices = model_family(folder, files)
            try:
                return resolve_name(requested, choices, label=label, folder=folder)
            except LoraError as exc:
                error = exc
        if folder == "diffusion_models":
            try:
                other = resolve_name(requested, files)
            except LoraError:
                other = None
            if other and "fl2va" in os.path.basename(other).lower():
                raise RequestError(
                    f"{other} is an fl2va model (first/last-frame video); the Director needs a ref2va model. "
                    f"Choices: {', '.join(choices) or 'none found'}.",
                    {"requested": requested, "choices": choices},
                )
        raise RequestError(f"{error} Choices: {', '.join(choices) or 'none found'}.", {**error.details, "choices": choices})

    async def resolve_loras(self, settings: RenderSettings) -> tuple[list[ResolvedLora], list[str]]:
        config = load_config(self.settings.loras_path)
        specs = [LoraSpec(item.name, item.strength) for item in settings.loras]
        kwargs = dict(loras=specs, preset=settings.lora_preset, use_defaults=settings.use_default_loras)
        try:
            return resolve_request(config, await self.available_loras(), **kwargs)
        except LoraError:
            # A file may have been added since the cached listing.
            try:
                return resolve_request(config, await self.available_loras(refresh=True), **kwargs)
            except LoraError as exc:
                await self._reject_image_lora(specs, exc)
                raise RequestError(str(exc), exc.details) from None

    async def _reject_image_lora(self, specs, error) -> None:
        """Say so plainly when a render asks for one of the Krea 2 image LoRAs instead of just "not found"."""
        wanted = str(error.details.get("requested") or "").strip().lower()
        if not wanted:
            return
        for name in await self.local_images.lora_basenames():
            if wanted in (name, name.rsplit(".", 1)[0]) or (len(wanted) > 3 and wanted in name):
                raise RequestError(
                    f"{error.details['requested']!r} is a Krea 2 image LoRA; video renders use the MiniMax H3 LoRAs "
                    "in models/loras. Use list_loras to see them, or generate_image for a picture.",
                    {**error.details, "image_lora": name},
                ) from None

    # ---------------------------------------------------------- info

    async def health(self) -> dict:
        comfy_ok = await self.comfy.ping()
        result: dict = {"ok": comfy_ok, "comfy": "ok" if comfy_ok else "unreachable"}
        if comfy_ok:
            try:
                rows = default_status(load_config(self.settings.loras_path), await self.available_loras(refresh=True))
                missing = [row["name"] for row in rows if row["required"] and not row["present"]]
                result["loras"] = "degraded" if missing else "ok"
                if missing:
                    result["missing_required_loras"] = missing
            except Exception as exc:
                result["loras"] = f"error: {exc}"
        return result

    async def options(self) -> dict:
        available = await self.available_loras(refresh=True)
        config = load_config(self.settings.loras_path)
        director = await self.comfy.object_info("HawkH3Director")
        try:
            planner_models = await self.atlas.list_models()
        except AtlasError as exc:
            log.warning("hawk_api: Atlas model list unavailable: %s", exc)
            planner_models = []
        return {
            "planner_models": planner_models,
            "default_planner_model": self.settings.planner_model,
            "default_agent_model": self.settings.agent_model,
            "diffusion_models": model_family("diffusion_models", await self.available_models("diffusion_models", refresh=True)),
            "text_encoders": model_family("text_encoders", await self.available_models("text_encoders", refresh=True)),
            "default_unet": self.settings.models.unet_name,
            "default_clip": self.settings.models.clip_name,
            "available_loras": available,
            "default_loras": default_status(config, available),
            "lora_presets": {name: [dataclasses.asdict(spec) for spec in specs] for name, specs in config.presets.items()},
            "samplers": _combo_options(director, "sampler_name"),
            "schedulers": _combo_options(director, "scheduler"),
            "aspect_ratios": _combo_options(director, "aspect_ratio"),
            "continuity_modes": _combo_options(director, "continuity"),
            "reference_roles": list(graphs.ROLES),
            "defaults": {"models": dataclasses.asdict(self.settings.models), "planner_model": self.settings.planner_model},
        }

    # ---------------------------------------------------------- assets

    async def add_asset(
        self,
        filename: str,
        fileobj,
        content_type: str | None = None,
        size: int | None = None,
        *,
        collection: str | None = None,
        tags=None,
        source: dict | None = None,
        local_path: str | None = None,
        dedupe: bool = True,
    ) -> dict:
        """Store one file as an asset. Identical content already in the library is not stored
        twice: the existing asset comes back with ``duplicate: true``."""
        kind = asset_kind(filename, content_type)
        digest = await asyncio.to_thread(_hash_file, fileobj)
        if dedupe and digest:
            existing = next((a for a in self.store.all_assets() if a.get("sha256") == digest), None)
            if existing is not None:
                return dict(existing, duplicate=True)
        asset_id = uuid.uuid4().hex[:12]
        base = os.path.basename(filename.replace("\\", "/"))
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._") or f"file{os.path.splitext(base)[1]}"
        mime = content_type or mimetypes.guess_type(safe)[0] or "application/octet-stream"
        input_dir = self.settings.comfy_input_dir
        if local_path and input_dir and os.path.isdir(input_dir):
            # Same machine as ComfyUI: copy straight into its input folder (no size limit, no HTTP).
            target_dir = os.path.join(input_dir, "hawk_api", asset_id)
            os.makedirs(target_dir, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, local_path, os.path.join(target_dir, safe))
            path = f"hawk_api/{asset_id}/{safe}"
        else:
            try:
                uploaded = await self.comfy.upload(fileobj, safe, f"hawk_api/{asset_id}", mime)
            except ComfyError as exc:
                raise Unavailable(str(exc)) from exc
            subfolder = uploaded.get("subfolder") or ""
            path = f"{subfolder}/{uploaded['name']}" if subfolder else uploaded["name"]
        asset = {
            "id": asset_id,
            "kind": kind,
            "filename": base,
            "path": path,
            "size": size,
            "created_at": time.time(),
            "collection": (collection or "").strip() or DEFAULT_COLLECTION,
            "tags": normalize_tags(tags),
            "sha256": digest,
            "source": source or {"type": "upload"},
        }
        self.store.add_asset(asset)
        return asset

    # ------------------------------------------------------------ library

    def search_assets(self, *, kind=None, collection=None, tag=None, query=None, limit: int = 100, offset: int = 0) -> tuple[list[dict], int]:
        items = self.store.all_assets()
        needle = (query or "").strip().lower()
        tag = (tag or "").strip().lower()
        matches = [
            a for a in items
            if (not kind or a.get("kind") == kind)
            and (not collection or a.get("collection", DEFAULT_COLLECTION) == collection)
            and (not tag or tag in (a.get("tags") or []))
            and (not needle or needle in f"{a['filename']} {a['id']} {' '.join(a.get('tags') or [])} {a.get('collection', '')}".lower())
        ]
        return matches[offset : offset + limit], len(matches)

    def collections(self) -> list[dict]:
        groups: dict[str, dict] = {}
        for asset in self.store.all_assets():
            name = asset.get("collection") or DEFAULT_COLLECTION
            group = groups.setdefault(name, {"name": name, "count": 0, "kinds": {}, "updated_at": 0})
            group["count"] += 1
            group["kinds"][asset["kind"]] = group["kinds"].get(asset["kind"], 0) + 1
            group["updated_at"] = max(group["updated_at"], asset["created_at"])
        return sorted(groups.values(), key=lambda g: g["updated_at"], reverse=True)

    def tags(self) -> list[dict]:
        counts: dict[str, int] = {}
        for asset in self.store.all_assets():
            for tag in asset.get("tags") or []:
                counts[tag] = counts.get(tag, 0) + 1
        return [{"name": name, "count": count} for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]

    def update_asset(self, asset_id: str, *, collection=None, tags=None, add_tags=None, remove_tags=None, filename=None,
                     owner=None) -> dict:
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise NotFound(f"No asset {asset_id!r}.")
        if owner:  # the character who made it, in a group chat: {"name", "member", "session"}
            asset["by"] = {k: str(owner.get(k) or "")[:80] for k in ("name", "member", "session")}
            add_tags = list(add_tags or []) + [f"by {owner.get('name')}"] if owner.get("name") else add_tags
        if collection is not None and collection.strip():
            asset["collection"] = collection.strip()
        if tags is not None:
            asset["tags"] = normalize_tags(tags)
        if add_tags:
            asset["tags"] = normalize_tags((asset.get("tags") or []) + normalize_tags(add_tags))
        if remove_tags:
            drop = set(normalize_tags(remove_tags))
            asset["tags"] = [t for t in asset.get("tags") or [] if t not in drop]
        if filename is not None and filename.strip():
            asset["filename"] = os.path.basename(filename.strip())
        self.store.add_asset(asset)
        return asset

    def set_job_owner(self, job_id: str, owner: dict) -> None:
        """Whose video this is, in a group chat. Missing jobs are ignored: ownership is never worth an error."""
        job = self.store.get_job(job_id)
        if job is None or not owner:
            return
        job["by"] = {k: str(owner.get(k) or "")[:80] for k in ("name", "member", "session")}
        self.store.save_job(job)

    def delete_asset(self, asset_id: str) -> None:
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise NotFound(f"No asset {asset_id!r}.")
        self.store.delete_asset(asset_id)
        input_dir = self.settings.comfy_input_dir
        if input_dir and asset["path"].startswith(f"hawk_api/{asset_id}/"):
            shutil.rmtree(os.path.join(input_dir, "hawk_api", asset_id), ignore_errors=True)
        thumbs = os.path.join(self.settings.data_dir, "thumbs")
        if os.path.isdir(thumbs):
            for name in os.listdir(thumbs):
                if name.startswith(f"{asset_id}_"):
                    os.remove(os.path.join(thumbs, name))

    async def add_asset_from_url(self, url: str, filename: str | None = None) -> dict:
        if not re.match(r"^https?://", url):
            raise RequestError("Only http(s) URLs can be fetched.")
        limit = self.settings.max_upload_mb * 1024 * 1024
        with tempfile.TemporaryFile() as buffer:
            size = 0
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0)) as client:
                    async with client.stream("GET", url) as response:
                        if response.status_code != 200:
                            raise RequestError(f"Could not download {url} ({response.status_code}).")
                        content_type = response.headers.get("content-type")
                        async for chunk in response.aiter_bytes(1 << 20):
                            size += len(chunk)
                            if size > limit:
                                raise RequestError(f"{url} is larger than {self.settings.max_upload_mb} MB.")
                            buffer.write(chunk)
            except httpx.HTTPError as exc:
                raise RequestError(f"Could not download {url}: {exc}") from None
            name = filename or os.path.basename(httpx.URL(url).path) or "download"
            if not os.path.splitext(name)[1] and content_type:
                name += mimetypes.guess_extension(content_type.split(";")[0]) or ""
            buffer.seek(0)
            return await self.add_asset(name, buffer, content_type, size, collection="From URLs", source={"type": "url", "url": url})

    def list_assets(self, limit: int = 100) -> list[dict]:
        return self.store.list_assets(limit)

    def asset_view(self, asset: dict) -> dict:
        """The stored asset plus signed links: the file, and a thumbnail for images."""
        base, token, ttl = self.settings.public_base_url, self.settings.token, self.settings.link_ttl_seconds
        link = base + sign_path(token, f"/v1/assets/{asset['id']}/file", ttl)
        view = dict(asset, file_url=link)
        if asset.get("kind") in ("image", "video"):
            view["thumb_url"] = f"{link}&w=320"
        view.setdefault("collection", DEFAULT_COLLECTION)
        view.setdefault("tags", [])
        return view

    def local_asset_path(self, asset: dict) -> str | None:
        input_dir = self.settings.comfy_input_dir
        path = os.path.join(input_dir, asset["path"]) if input_dir else ""
        return path if path and os.path.isfile(path) else None

    async def image_dimensions(self, asset: dict) -> tuple[int, int] | None:
        """Width and height of an image asset, or None when Pillow can't read it."""
        try:
            from PIL import Image

            data = await self.asset_bytes(asset)
            with Image.open(io.BytesIO(data)) as image:
                return image.size
        except Exception:
            return None

    async def asset_bytes(self, asset: dict) -> bytes:
        local = self.local_asset_path(asset)
        if local:
            return await asyncio.to_thread(lambda: open(local, "rb").read())
        subfolder, _, filename = asset["path"].rpartition("/")
        try:
            _, _, body = await self.comfy.view(filename, subfolder, "input")
        except ComfyNotFound as exc:
            raise NotFound(f"The file for asset {asset['id']} is gone from ComfyUI's input folder.") from exc
        except ComfyError as exc:
            raise Unavailable(str(exc)) from None
        return b"".join([chunk async for chunk in body])

    async def asset_file(self, asset_id: str, width: int = 0) -> tuple[bytes, str]:
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise NotFound(f"No asset {asset_id!r}.")
        mime = mimetypes.guess_type(asset["filename"])[0] or "application/octet-stream"
        if asset["kind"] == "video" and width > 0:
            return await self._video_thumb(asset, width), "image/jpeg"
        if asset["kind"] != "image" or width <= 0:
            return await self.asset_bytes(asset), mime
        width = min((w for w in THUMB_WIDTHS if w >= width), default=THUMB_WIDTHS[-1])
        cache = os.path.join(self.settings.data_dir, "thumbs", f"{asset_id}_{width}.jpg")
        if os.path.isfile(cache):
            with open(cache, "rb") as handle:
                return handle.read(), "image/jpeg"
        data = await self.asset_bytes(asset)
        try:
            from PIL import Image
        except ImportError:  # no Pillow: send the original
            return data, mime

        def make() -> bytes | None:
            try:
                image = Image.open(io.BytesIO(data))
                image.thumbnail((width, width * 4))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=82 if width <= 640 else 90)
            except Exception:
                return None
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "wb") as handle:
                handle.write(buffer.getvalue())
            return buffer.getvalue()

        thumb = await asyncio.to_thread(make)
        return (thumb, "image/jpeg") if thumb else (data, mime)

    async def _video_thumb(self, asset: dict, width: int) -> bytes:
        width = min((w for w in THUMB_WIDTHS if w >= width), default=THUMB_WIDTHS[-1])
        cache = os.path.join(self.settings.data_dir, "thumbs", f"{asset['id']}_{width}.jpg")
        if os.path.isfile(cache):
            with open(cache, "rb") as handle:
                return handle.read()
        if not shutil.which("ffmpeg"):
            raise NotFound("No video thumbnail: ffmpeg is not installed on the server.")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        source = self.local_asset_path(asset)
        temp = None
        if source is None:
            temp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(asset["filename"])[1] or ".mp4", delete=False)
            temp.write(await self.asset_bytes(asset))
            temp.close()
            source = temp.name
        try:
            for seek in ("1", "0"):
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", seek, "-i", source, "-frames:v", "1",
                    "-vf", f"scale={width}:-2", cache, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await process.wait()
                if process.returncode == 0 and os.path.isfile(cache):
                    with open(cache, "rb") as handle:
                        return handle.read()
        finally:
            if temp is not None:
                os.remove(temp.name)
        raise NotFound("Could not make a thumbnail for this video.")

    async def generate_images(
        self,
        prompt: str,
        *,
        model: str | None = None,
        reference_asset_ids: list[str] | tuple = (),
        size: str | None = None,
        n: int = 1,
        seed: int | None = None,
        engine: str | None = None,
        loras: list[dict] | None = None,
        steps: int | None = None,
        max_adult_loras: int = 3,
        ref_boost: float | None = None,
    ) -> dict:
        """Make images and store them as assets, ready to use as picture references.

        engine "auto" walks the ladder for this request -- generate or edit -- trying each engine until one
        answers, and naming the ones it skipped in "tried". Any engine id from image_engines pins a single
        engine instead, in which case a failure is reported rather than worked around. A refusal from a local
        engine's content check stops the walk outright: it is never retried somewhere else."""
        if not (prompt or "").strip():
            raise RequestError("Describe the image to generate.")
        sources = []
        for asset_id in reference_asset_ids or []:
            asset = self.store.get_asset(asset_id)
            if asset is None:
                raise RequestError(f"Unknown asset_id {asset_id!r}.")
            if asset["kind"] != "image":
                raise RequestError(f"{asset['filename']} is {asset['kind']}; image edits need image assets.")
            sources.append(asset)

        model = (model or "").strip()
        raw = (engine or "").strip().lower()
        action = "edit" if sources else "generate"
        notes, tried = [], []

        # "model" names an Atlas model (or an alias for one); naming a local one there means the engine.
        if model.lower() in ("krea", "krea2", "krea-2", "local"):
            raw, model = "krea2", ""
        elif model:
            raw = raw or "atlas"
        raw = raw or self.settings.image_engine

        atlas_override = ""
        if raw == "auto":
            ladder = self.image_ladder(action)
        elif raw == "atlas":  # a model id was given: use exactly that, on the Atlas path
            ladder = ["seedream"]
            atlas_override = IMAGE_ALIASES.get(model.lower(), model) or (
                IMAGE_EDIT_MODEL if sources else self.settings.image_model)
        else:
            pinned = image_engines.resolve(raw)
            if not pinned:
                raise RequestError(f"engine {raw!r} should be auto or one of: {', '.join(image_engines.ENGINES)}.")
            if raw in image_engines.MOVED:
                notes.append(image_engines.MOVED[raw])
            ladder = [pinned]

        # An Atlas engine that only makes images from text still has somewhere to go when given references.
        if sources and len(ladder) == 1:
            only = image_engines.get(ladder[0])
            if only and not only.edit and not only.local:
                notes.append(f"{only.label} can't use reference images, so Seedream edit made this one.")
                ladder, atlas_override = ["seedream"], IMAGE_EDIT_MODEL

        references: list[str] = []

        async def atlas_references() -> list[str]:
            """The sources as data URIs, built once and only if an Atlas engine actually runs."""
            if sources and not references:
                for asset in sources:
                    mime = mimetypes.guess_type(asset["filename"])[0] or "image/png"
                    references.append(f"data:{mime};base64," + base64.b64encode(await self.asset_bytes(asset)).decode("ascii"))
            return references

        last_error, said_loras = None, False
        for index, engine_id in enumerate(ladder):
            spec = image_engines.get(engine_id)
            more = index + 1 < len(ladder)
            if not spec.supports(action):
                tried.append({"engine": engine_id, "skipped": f"{spec.label} can't {action} images."})
                continue
            if sources and len(sources) > spec.max_refs:
                why = f"{spec.label} takes at most {spec.max_refs} reference image(s); {len(sources)} were given."
                if not more:
                    raise RequestError(why)
                tried.append({"engine": engine_id, "skipped": why})
                continue

            if spec.local:
                try:
                    local, used = await self._local_image(
                        spec, action, prompt, sources, size=size, n=n, seed=seed, loras=loras, steps=steps,
                        ref_boost=ref_boost, max_adult_loras=max_adult_loras)
                except LocalImageError as exc:
                    # A content refusal is final. Walking on would hand the same prompt to the next engine,
                    # and eventually to a paid one whose moderation is not ours -- so stop the whole ladder.
                    if exc.fatal or not more:
                        # carry any note with it: an engine name that changed meaning explains itself here
                        raise RequestError(" ".join([str(exc), *notes]).strip()) from None
                    tried.append({"engine": spec.tag_for(action), "skipped": str(exc)})
                    continue
                notes.extend(local.warnings)
                return await self._image_result(prompt, local.images, used, spec.tag_for(action), notes, tried,
                                                reference_asset_ids, engine_id=engine_id,
                                                extra={"loras": local.loras, "seconds": local.seconds})

            if loras and not said_loras:
                notes.append("Image LoRAs only apply to the local engines; ignored here.")
                said_loras = True
            name = atlas_override or (spec.atlas_edit_model if sources else spec.atlas_model)
            if engine_id == "turbo" and not atlas_override:
                name = self.settings.image_model or spec.atlas_model  # HAWK_IMAGE_MODEL replaces this rung's model
            try:
                used, images, cost = await self._atlas_images(prompt, name, await atlas_references(), size, n, seed, notes)
            except RequestError as exc:
                last_error = exc
                if not more:
                    raise
                tried.append({"engine": engine_id, "skipped": str(exc)[:300]})
                continue
            tag = "z-image" if used.startswith("z-image/") else "seedream" if "seedream" in used else "atlas"
            return await self._image_result(prompt, images, used, tag, notes, tried, reference_asset_ids,
                                            engine_id=image_engines.id_for_tag(tag), extra={"cost_usd": round(cost, 4)})
        raise last_error or RequestError("No image engine could make this image.")

    def image_ladder(self, action: str = "generate") -> list[str]:
        """Engine ids to try for this action, best first, as Studio has them ordered."""
        return self.image_engines.order(action)

    async def _local_image(self, spec, action: str, prompt: str, sources: list, **kwargs):
        """Run one local engine. Returns its result and the model name to report."""
        if spec.id != "krea2":
            # Klein and Z-Image get their graphs in a later commit; until then they are never in the ladder,
            # and naming one explicitly says so instead of pretending.
            raise LocalImageError(f"{spec.label} is not wired up on this pod yet; use engine auto.")
        if action == "edit":
            local = await self.local_images.edit(
                prompt, sources, size=kwargs["size"], n=kwargs["n"], seed=kwargs["seed"], loras=kwargs["loras"],
                steps=kwargs["steps"], ref_boost=kwargs["ref_boost"], max_adult_loras=kwargs["max_adult_loras"])
            return local, "krea2/identity-edit"
        local = await self.local_images.generate(
            prompt, size=kwargs["size"], n=kwargs["n"], seed=kwargs["seed"], loras=kwargs["loras"],
            steps=kwargs["steps"], max_adult_loras=kwargs["max_adult_loras"])
        return local, "krea2/turbo"

    def _ladder_view(self, action: str, settings: dict) -> list[dict]:
        rows = []
        for row in settings[action]:
            spec = image_engines.get(row["engine"])
            rows.append({"engine": spec.id, "label": spec.label, "where": spec.where, "enabled": row["enabled"],
                         "cost_usd": IMAGE_PRICES.get(spec.price_key, 0.0), "max_refs": spec.max_refs})
        return rows

    async def image_options(self) -> dict:
        local = await self.local_images.status()
        engines = self.image_engines.view()
        ladders = {action: self._ladder_view(action, engines) for action in ("generate", "edit")}
        return {
            "default_engine": self.settings.image_engine,
            "generate_ladder": ladders["generate"],
            "edit_ladder": ladders["edit"],
            "would_use": {action: (self.image_ladder(action) or [None])[0] for action in ("generate", "edit")},
            "busy": engines["busy"],
            "engine_warnings": engines["warnings"],
            "local": local,
            "atlas": {"configured": self.atlas.configured, "text_to_image": self.settings.image_model,
                      "quality": IMAGE_MODEL, "edit": IMAGE_EDIT_MODEL, "lite": IMAGE_LITE_MODEL,
                      "prices_usd": {"z-image/turbo": IMAGE_PRICES["z-image"], "seedream 1.5K (up to 2.36 MP)": IMAGE_PRICES["pro-1.5k"],
                                     "seedream 2K": IMAGE_PRICES["pro-2k"], "seedream-lite (2K+)": IMAGE_PRICES["lite"]}},
            "sizes": ["1024x1024", "1024x1536", "1536x1024", "896x1600", "1600x896"],
        }

    async def _atlas_images(self, prompt: str, model: str, references: list[str], size, n: int, seed,
                            notes: list) -> tuple[str, list[bytes], float]:
        """(model used, images, estimated USD)."""
        if references and _text_only_image_model(model):
            notes.append(f"{model} can't use reference images, so Seedream edit made this one.")
            model = IMAGE_EDIT_MODEL
        if references and model.endswith("/text-to-image"):
            model = model[: -len("/text-to-image")] + "/edit"
        if references and model == IMAGE_LITE_MODEL:
            model = IMAGE_LITE_EDIT_MODEL
        if not references and model == IMAGE_LITE_EDIT_MODEL:
            model = IMAGE_LITE_MODEL
        if not references and model.endswith("/edit"):
            raise RequestError(f"{model} edits images: pass reference_asset_ids, or use a text-to-image model.")
        payload: dict = {"model": model, "prompt": prompt.strip()}
        count = max(1, n or 1)
        # Both engines make one image per request, so n runs as parallel requests.
        if _text_only_image_model(model):  # z-image: "W*H", each side 512-2048
            payload["size"] = _star_size(size) if size else "1024*1536"
            payload["prompt_extend"] = False
            payloads = [{**payload, "seed": (seed + index) if seed is not None else -1} for index in range(count)]
            each = IMAGE_PRICES["z-image"]
        else:
            payload["size"], tier = seedream_size(model, size)
            if references:
                payload["images"] = references
            payloads = [{**payload, **({"seed": seed + index} if seed is not None else {})} for index in range(count)]
            each = IMAGE_PRICES.get(tier, 0.0) + SEEDREAM_EXTRA_REFERENCE * max(0, len(references) - 1)
            if size and payload["size"] != _star_size_loose(size):
                notes.append(f"Seedream made {payload['size'].replace('*', 'x')} (its nearest preset to {size}).")
        try:
            batches = await asyncio.gather(*(self.atlas.generate_image(body) for body in payloads))
        except AtlasError as exc:
            raise RequestError(str(exc)) from None
        images = [image for batch in batches for image in batch]
        return model, images, each * len(images)

    async def _image_result(self, prompt: str, images: list[bytes], model: str, engine_tag: str, notes: list, tried: list,
                            reference_asset_ids, extra: dict | None = None, engine_id: str = "") -> dict:
        stem = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40] or "image"
        # "engine" is the registry id; the agent reads it to know which rung made an image, instead of
        # guessing from the model name. Assets written before this field fall back to image_engines.id_for_tag.
        source = {"type": "generated", "engine": engine_id or image_engines.id_for_tag(engine_tag), "generator": model,
                  "prompt": prompt.strip()[:500], "references": list(reference_asset_ids or [])}
        if extra and extra.get("loras"):
            source["loras"] = extra["loras"]
        assets = []
        for number, data in enumerate(images, 1):
            extension, mime = _image_type(data)
            asset = await self.add_asset(
                f"gen_{stem}_{number}.{extension}", io.BytesIO(data), mime, len(data), collection="Generated",
                tags=["generated", engine_tag], source=source,
            )
            assets.append(self.asset_view(asset))
        # create_app sets drive_exporter: new images are copied into Drive in the background, like finished renders
        to_drive = bool(self.drive_exporter and self.drive_exporter.schedule_assets(assets))
        result = {"model": model, "engine": engine_tag, "assets": assets, "saving_to_drive": to_drive}
        if notes:
            result["note"] = " ".join(notes)
        if tried:
            result["tried"] = tried
        if extra:
            result.update(extra)
        return result

    def _refs(self, references: list[ReferenceIn]) -> list[graphs.Ref]:
        refs = []
        for reference in references:
            asset = self.store.get_asset(reference.asset_id)
            if asset is None:
                raise RequestError(f"Unknown asset_id {reference.asset_id!r}. Upload the file first.")
            refs.append(graphs.Ref(asset["id"], asset["kind"], asset["path"], reference.role, reference.label, reference.for_video))
        return refs

    # ------------------------------------------------------------ jobs

    def _planner_args(self, options: PlannerOptions, seed: int) -> dict:
        return {
            "story": options.story,
            "segment_count": options.segment_count,
            "segment_seconds": options.segment_seconds,
            "aspect_ratio": options.aspect_ratio,
            "model": options.model or self.settings.planner_model,
            "seed": seed,
            "temperature": options.temperature,
            # Blank keeps the node's built-in guide; an edited planner prompt gets the platform rules appended.
            "system_prompt": (f"{custom.rstrip()}\n\n{PLATFORM_RULES}" if (custom := self.prompts.custom("planner")) else ""),
        }

    @staticmethod
    def _validate_script(text: str, available: dict, video_has_audio: list, settings: RenderSettings) -> int:
        try:
            jobs = build_jobs(
                parse_script(text),
                available=available,
                video_has_audio=video_has_audio,
                default_seconds=settings.default_seconds,
                continuity=settings.continuity,
                base_seed=0,
            )
        except ScriptError as exc:
            raise RequestError(f"Script problem: {exc}") from None
        return len(jobs)

    @staticmethod
    def _new_job(kind: str, job_id: str | None = None, **fields) -> dict:
        now = time.time()
        job = {
            "id": job_id or str(uuid.uuid4()),
            "kind": kind,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "prompt_id": None,
            "attempts": 0,
            "script": None,
            "segments_total": None,
            "segments_done": 0,
            "current_segment": None,
            "steps_done": 0,
            "steps_total": None,
            "outputs": {},
            "loras": [],
            "loras_applied_by_node": {},
            "loras_applied": [],
            "warnings": [],
            "error": None,
            "resumable": False,
        }
        job.update(fields)
        return job

    async def create_plan(self, request: PlanRequest) -> dict:
        refs = self._refs(request.references)
        seed = request.seed if request.seed is not None else random.randrange(1, 2**31)
        try:
            built, wiring = graphs.plan_graph(refs, self._planner_args(request, seed))
        except graphs.GraphError as exc:
            raise RequestError(str(exc)) from None
        job = self._new_job(
            "plan",
            request=request.model_dump(),
            refs=[dataclasses.asdict(ref) for ref in refs],
            graph=built.prompt,
            nodes=built.nodes,
            available=wiring.available,
            video_has_audio=wiring.video_has_audio,
            seed=seed,
        )
        return await self._submit(job)

    async def create_video(self, request: VideoRequest) -> dict:
        settings = request.settings
        references = request.references
        script_text: str | None = None
        planner: dict | None = None

        if request.plan_job_id:
            plan = self.store.get_job(request.plan_job_id)
            if plan is None or plan["kind"] != "plan":
                raise RequestError(f"No plan job {request.plan_job_id!r}.")
            if plan["status"] != "done" or not plan.get("script"):
                raise RequestError(f"Plan {request.plan_job_id} is {plan['status']}; wait until it is done.")
            script_text = plan["script"]
            if not references:
                references = [ReferenceIn(**item) for item in plan["request"].get("references", [])]
        elif request.script is not None:
            script_text = request.script if isinstance(request.script, str) else json.dumps(request.script)

        refs = self._refs(references)
        loras, warnings = await self.resolve_loras(settings)
        steps, steps_reason = choose_steps(loras, settings.steps)
        seed = settings.seed if settings.seed is not None else random.randrange(1, 2**48)
        job_id = str(uuid.uuid4())
        music_path = None
        if settings.music_asset_id:
            music = self.store.get_asset(settings.music_asset_id)
            if music is None:
                raise RequestError(f"Unknown music_asset_id {settings.music_asset_id!r}. Upload the track first.")
            if music["kind"] != "audio":
                raise RequestError(f"music_asset_id must be an audio file (mp3, wav, m4a...); {music['filename']} is {music['kind']}.")
            music_path = music["path"]
        params = graphs.RenderParams(
            run_name=f"api_{job_id.replace('-', '')[:16]}",
            seed=seed,
            steps=steps,
            aspect_ratio=settings.aspect_ratio,
            megapixels=settings.megapixels,
            default_seconds=settings.default_seconds,
            sampler_name=settings.sampler_name,
            scheduler=settings.scheduler,
            continuity=settings.continuity,
            carry_audio=settings.carry_audio,
            ref_image_size=settings.ref_image_size,
            interpolation=settings.interpolation,
            audio_crossfade_ms=settings.audio_crossfade_ms,
            music_path=music_path,
            music_volume_db=settings.music_volume_db,
            scene_volume_db=settings.scene_volume_db,
            music_fade_seconds=settings.music_fade_seconds,
            mute_generated_music=settings.mute_generated_music,
        )
        defaults = self.settings.models
        models = dataclasses.replace(
            defaults,
            attention=settings.attention or defaults.attention,
            unet_name=await self.choose_model(settings.unet_name, "diffusion_models", defaults.unet_name),
            clip_name=await self.choose_model(settings.clip_name, "text_encoders", defaults.clip_name),
        )
        if request.story is not None:
            planner = self._planner_args(request.story, seed % 2**31)

        try:
            built, wiring = graphs.render_graph(refs, models, loras, params, script=script_text, planner=planner)
        except graphs.GraphError as exc:
            raise RequestError(str(exc)) from None

        segments_total = None
        if script_text is not None:
            segments_total = self._validate_script(script_text, wiring.available, wiring.video_has_audio, settings)

        job = self._new_job(
            "render",
            job_id=job_id,
            request=request.model_dump(),
            refs=[dataclasses.asdict(ref) for ref in refs],
            graph=built.prompt,
            nodes=built.nodes,
            available=wiring.available,
            video_has_audio=wiring.video_has_audio,
            seed=seed,
            run_name=params.run_name,
            params=dataclasses.asdict(params),
            models=dataclasses.asdict(models),
            script=script_text,
            segments_total=segments_total,
            loras=[dataclasses.asdict(entry) for entry in loras],
            steps=steps,
            steps_reason=steps_reason,
            warnings=warnings,
        )
        return await self._submit(job)

    async def _submit(self, job: dict) -> dict:
        """Queue the graph. The job stays `queued` until ComfyUI starts it (execution_start)."""
        # The first attempt reuses the job id; retries need a fresh id because
        # ComfyUI keeps finished prompt ids in its history.
        job["prompt_id"] = job["id"] if job["attempts"] == 0 else str(uuid.uuid4())
        job["status"] = "queued"
        self.store.save_job(job)  # saved first so events arriving mid-submit find it
        try:
            await self.comfy.submit(job["graph"], job["prompt_id"])
        except ComfyValidationError as exc:
            job.update(status="failed", error=f"ComfyUI rejected the graph: {exc}")
            self.store.save_job(job)
            raise RequestError(job["error"], exc.details) from None
        except ComfyError as exc:
            job.update(status="failed", error=str(exc), resumable=True)
            self.store.save_job(job)
            raise Unavailable(str(exc)) from None
        current = self.store.get_job(job["id"]) or job  # events may already have moved it on
        current["attempts"] = job["attempts"] + 1
        self.store.save_job(current)
        await self._refresh_queue()
        return self.store.get_job(job["id"]) or current

    @staticmethod
    def _started_status(job: dict) -> str:
        if job["kind"] == "plan" or (job["nodes"].get("planner") and not job.get("script")):
            return "planning"
        return "rendering"

    async def _refresh_queue(self) -> None:
        """Promote queued jobs ComfyUI has started and number the ones still waiting."""
        if not self.store.list_jobs(limit=1, statuses=("queued",)):
            return
        try:
            running, pending = await self.comfy.queue_state()
        except ComfyError:
            return
        positions = {prompt_id: number for number, prompt_id in enumerate(pending, 1)}
        for job in self.store.list_jobs(limit=500, statuses=("queued",)):
            prompt_id = job.get("prompt_id")
            if prompt_id in running:
                job.update(status=self._started_status(job), queue_position=None)
            elif prompt_id in positions and job.get("queue_position") != positions[prompt_id]:
                job["queue_position"] = positions[prompt_id]
            else:
                continue
            self.store.save_job(job)

    def get_job(self, job_id: str) -> dict:
        job = self.store.get_job(job_id)
        if job is None:
            raise NotFound(f"No job {job_id!r}.")
        return job

    def list_jobs(self, limit: int = 50) -> list[dict]:
        return self.store.list_jobs(limit)

    async def cancel(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job["status"] not in ACTIVE:
            raise Conflict(f"Job {job_id} is already {job['status']}.")
        try:
            await self.comfy.cancel(job["prompt_id"])
        except ComfyError as exc:
            log.warning("hawk_api: cancel reached no ComfyUI (%s)", exc)
        job.update(status="cancelled", resumable=job["kind"] == "render")
        self.store.save_job(job)
        return job

    async def retry(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job["status"] in ACTIVE:
            raise Conflict(f"Job {job_id} is still {job['status']}.")
        if job["status"] == "done":
            raise Conflict(f"Job {job_id} already finished.")
        if job["kind"] == "render" and job.get("script") and job["nodes"].get("planner"):
            # One-call job whose plan already arrived: render that exact script so the
            # Director's resume cache matches instead of asking the LLM for a new plan.
            try:
                built, _ = graphs.render_graph(
                    [graphs.Ref(**ref) for ref in job["refs"]],
                    ModelSettings(**job["models"]),
                    [ResolvedLora(**entry) for entry in job["loras"]],
                    graphs.RenderParams(**job["params"]),
                    script=job["script"],
                )
            except graphs.GraphError as exc:
                raise RequestError(str(exc)) from None
            job.update(graph=built.prompt, nodes=built.nodes)
        job.update(error=None, resumable=False, outputs={}, loras_applied_by_node={}, loras_applied=[], segments_done=0)
        return await self._submit(job)

    # ---------------------------------------------------------- events

    async def handle_event(self, event: dict) -> None:
        data = event.get("data") or {}
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            return
        job = self.store.find_by_prompt(prompt_id)
        if job is None or job["status"] in FINISHED:
            return
        kind = event.get("type")
        if kind == "execution_start":
            if job["status"] == "queued":
                job.update(status=self._started_status(job), queue_position=None)
                self.store.save_job(job)
            await self._refresh_queue()
        elif kind == "hawk_h3.segment":
            job["segments_done"] = int(data.get("done") or 0)
            job["segments_total"] = int(data.get("total") or job.get("segments_total") or 0)
            job["current_segment"] = data.get("title") or None
            job["steps_done"] = 0
            job["status"] = "rendering"
            self.store.save_job(job)
        elif kind == "progress" and str(data.get("node")) == job["nodes"].get("director"):
            # The sampler's per-step bar reports under the Director node. The Director's own
            # segment bar arrives right after its hawk_h3.segment event and is skipped here.
            value, maximum = int(data.get("value") or 0), int(data.get("max") or 0)
            if maximum == job.get("segments_total") and value == job.get("segments_done"):
                return
            job["steps_done"], job["steps_total"] = value, maximum
            job["status"] = "rendering"
            self.store.save_job(job)
        elif kind == "executed":
            self._apply_output(job, str(data.get("node")), data.get("output") or {})
            self.store.save_job(job)
        elif kind == "execution_success":
            await self._finish(job)
            await self._refresh_queue()
        elif kind == "execution_error":
            message = f"{data.get('node_type', 'node')}: {str(data.get('exception_message', '')).strip()}"
            self._fail(job, message)
            await self._refresh_queue()
        elif kind == "execution_interrupted":
            job.update(status="cancelled", resumable=job["kind"] == "render")
            self.store.save_job(job)
            await self._refresh_queue()

    def _apply_output(self, job: dict, node: str, output: dict) -> None:
        texts = output.get("text") or []
        if node == job["nodes"].get("plan_preview") and texts:
            job["script"] = texts[0]
            for warning in script_warnings(texts[0]):
                if warning not in job["warnings"]:
                    job["warnings"].append(warning)
            if job["kind"] == "render":
                settings = RenderSettings(**job["request"]["settings"])
                try:
                    job["segments_total"] = len(
                        build_jobs(
                            parse_script(texts[0]),
                            available=job["available"],
                            video_has_audio=job["video_has_audio"],
                            default_seconds=settings.default_seconds,
                            continuity=settings.continuity,
                            base_seed=0,
                        )
                    )
                except ScriptError as exc:
                    job["warnings"].append(f"Could not read the plan: {exc}")
                job["status"] = "rendering"
        elif node == job["nodes"].get("prompt_preview") and texts:
            job["final_prompts"] = texts[0]
        elif node in job["nodes"].get("lora_stacks", []) and texts:
            job["loras_applied_by_node"][node] = parse_applied(texts[0])
        elif node == job["nodes"].get("director"):
            images = output.get("images") or []
            if images:
                first = images[0]
                job["outputs"]["video"] = {
                    "filename": first["filename"],
                    "subfolder": first.get("subfolder", ""),
                    "type": first.get("type", "output"),
                }

    @staticmethod
    def _stack_applied(job: dict, node: str) -> list[tuple[str, float]]:
        """LoRAs a LoRA Stack applied. A stack ComfyUI served from its cache (same LoRAs as an
        earlier render) sends no report; its output came from a run with these exact inputs,
        whose LoRA names ComfyUI validated, so read them from the submitted graph."""
        if node in job["loras_applied_by_node"]:
            return job["loras_applied_by_node"][node]
        inputs = ((job.get("graph") or {}).get(node) or {}).get("inputs") or {}
        applied = []
        for key in sorted((k for k in inputs if re.fullmatch(r"lora_\d+", k)), key=lambda k: int(k[5:])):
            name, strength = inputs[key], inputs.get(f"strength_{key[5:]}", 1.0)
            if isinstance(name, str) and name != "None" and isinstance(strength, (int, float)) and float(strength) != 0.0:
                applied.append((name, float(strength)))
        return applied

    def _fail(self, job: dict, message: str, resumable: bool | None = None) -> None:
        job.update(status="failed", error=message, resumable=job["kind"] == "render" if resumable is None else resumable)
        self.store.save_job(job)

    async def _finish(self, job: dict) -> None:
        job = self.store.get_job(job["id"]) or job
        if job["status"] in FINISHED:
            return
        try:
            history = await self.comfy.history(job["prompt_id"])
        except ComfyError:
            history = None
        if history:
            for node, output in (history.get("outputs") or {}).items():
                self._apply_output(job, str(node), output)
            status = history.get("status") or {}
            if status.get("status_str") == "error":
                message = next(
                    (
                        f"{data.get('node_type', 'node')}: {str(data.get('exception_message', '')).strip()}"
                        for name, data in status.get("messages", [])
                        if name == "execution_error"
                    ),
                    "ComfyUI reported an error.",
                )
                return self._fail(job, message)

        if job["kind"] == "plan":
            if not job.get("script"):
                return self._fail(job, "The planner finished without a script.", resumable=False)
        else:
            if not job["outputs"].get("video"):
                return self._fail(job, "The render finished but the Director reported no video.")
            applied = [pair for node in job["nodes"].get("lora_stacks", []) for pair in self._stack_applied(job, node)]
            job["loras_applied"] = [{"file": name, "strength": strength} for name, strength in applied]
            if job["nodes"].get("lora_stacks") or job["loras"]:
                job["warnings"].extend(compare_applied(job["loras"], applied))
            if job.get("segments_total"):
                job["segments_done"] = job["segments_total"]
        job["status"] = "done"
        self.store.save_job(job)
        if job["kind"] == "render":
            for hook in self.render_done_hooks:
                try:
                    hook(job["id"])
                except Exception:
                    log.exception("hawk_api: render-done hook failed")

    async def reconcile(self) -> None:
        """Catch up on anything the websocket missed, and flag jobs ComfyUI forgot
        (it keeps history in memory, so a restart loses running prompts)."""
        active = self.store.list_jobs(limit=500, statuses=ACTIVE)
        if not active:
            return
        await self._refresh_queue()
        try:
            queued = await self.comfy.queue_ids()
        except ComfyError:
            return
        for job in active:
            if not job.get("prompt_id"):
                continue
            try:
                history = await self.comfy.history(job["prompt_id"])
            except ComfyError:
                return
            if history:
                await self._finish(job)
            elif job["prompt_id"] not in queued and time.time() - job["updated_at"] > LOST_AFTER_SECONDS:
                self._fail(
                    job,
                    "ComfyUI no longer knows this job (it probably restarted). Retry it: finished segments are reused.",
                    resumable=True,
                )

    async def _reconcile_loop(self) -> None:
        while True:
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("hawk_api: reconcile failed")
            await asyncio.sleep(self.settings.reconcile_seconds)

    # ---------------------------------------------------------- views

    def job_summary(self, job: dict) -> dict:
        """The job list's light view: no scripts, prompts or segment links (open the job for those)."""
        view = self.job_view(job)
        for key in ("script", "final_prompts", "segment_urls", "steps_reason"):
            view.pop(key, None)
        if view.get("error") and len(view["error"]) > 400:
            view["error"] = view["error"][:400] + "..."
        view["title"] = job_title(job)
        view["segments_available"] = job.get("segments_done") or 0
        return view

    def job_view(self, job: dict) -> dict:
        base, token, ttl = self.settings.public_base_url, self.settings.token, self.settings.link_ttl_seconds
        view = {
            "id": job["id"],
            "kind": job["kind"],
            "status": job["status"],
            "queue_position": job.get("queue_position") if job["status"] == "queued" else None,
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "progress": {
                "segments_done": job.get("segments_done", 0),
                "segments_total": job.get("segments_total"),
                "current_segment": job.get("current_segment"),
                "steps_done": job.get("steps_done", 0),
                "steps_total": job.get("steps_total"),
            },
            "script": job.get("script"),
            "error": job.get("error"),
            "resumable": job.get("resumable", False),
            "warnings": job.get("warnings", []),
            "title": job_title(job),
            "by": job.get("by"),  # the character whose video this is, in a group chat
        }
        if job["kind"] == "render":
            view.update(
                seed=job.get("seed"),
                steps=job.get("steps"),
                steps_reason=job.get("steps_reason"),
                loras=job.get("loras", []),
                loras_applied=job.get("loras_applied", []),
                run_name=job.get("run_name"),
                unet_name=(job.get("models") or {}).get("unet_name"),
                clip_name=(job.get("models") or {}).get("clip_name"),
                music_asset_id=((job.get("request") or {}).get("settings") or {}).get("music_asset_id"),
                final_prompts=job.get("final_prompts"),
                video_url=None,
                segment_urls=[],
            )
            if job["status"] == "done" and job["outputs"].get("video"):
                view["video_url"] = base + sign_path(token, f"/v1/jobs/{job['id']}/video", ttl)
                view["download_url"] = view["video_url"] + "&download=1"
                view["thumb_url"] = base + sign_path(token, f"/v1/jobs/{job['id']}/thumb", ttl) + "&w=640"
            view["drive"] = drive_links(job.get("drive"))
            done = job.get("segments_done") or 0
            view["segment_urls"] = [
                base + sign_path(token, f"/v1/jobs/{job['id']}/segments/{number}", ttl) for number in range(1, done + 1)
            ]
        return view

    def video_location(self, job_id: str) -> tuple[str, str, str]:
        """(filename, subfolder, type) of a finished render's video."""
        job = self.get_job(job_id)
        video = job["outputs"].get("video")
        if job["kind"] != "render" or job["status"] != "done" or not video:
            raise NotFound(f"Job {job_id} has no finished video yet.")
        return video["filename"], video["subfolder"], video.get("type", "output")

    def segment_location(self, job_id: str, number: int) -> tuple[str, str, str]:
        job = self.get_job(job_id)
        if job["kind"] != "render" or number < 1:
            raise NotFound("No such segment.")
        return f"segment_{number:03d}.mp4", f"hawk_h3/{job['run_name']}", "output"

    def local_output(self, filename: str, subfolder: str, type_: str = "output") -> str | None:
        """The file on this machine when ComfyUI's output folder is local (served with byte ranges)."""
        root = self.settings.comfy_output_dir if type_ == "output" else self.settings.comfy_input_dir
        if not root:
            return None
        path = os.path.realpath(os.path.join(root, subfolder, filename))
        return path if path.startswith(os.path.realpath(root) + os.sep) and os.path.isfile(path) else None

    async def open_remote(self, filename: str, subfolder: str, type_: str, range_header: str | None) -> dict:
        try:
            return await self.comfy.view_range(filename, subfolder, type_, range_header)
        except ComfyNotFound as exc:
            raise NotFound(str(exc)) from None
        except ComfyError as exc:
            raise Unavailable(str(exc)) from None

    async def open_video(self, job_id: str):
        return await self._view(*self.video_location(job_id))

    async def open_segment(self, job_id: str, number: int):
        return await self._view(*self.segment_location(job_id, number))

    async def job_thumb(self, job_id: str, width: int = 640) -> bytes:
        """A JPEG frame from a finished render (1 s in), cached."""
        width = min((w for w in THUMB_WIDTHS if w >= width), default=THUMB_WIDTHS[-1])
        cache = os.path.join(self.settings.data_dir, "thumbs", f"job_{job_id}_{width}.jpg")
        if os.path.isfile(cache):
            return await asyncio.to_thread(_read_bytes, cache)
        if not shutil.which("ffmpeg"):
            raise NotFound("No thumbnail: ffmpeg is not installed on the server.")
        filename, subfolder, type_ = self.video_location(job_id)
        source, temp = self.local_output(filename, subfolder, type_), None
        if source is None:
            _, _, body = await self._view(filename, subfolder, type_)
            temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            async for chunk in body:
                temp.write(chunk)
            temp.close()
            source = temp.name
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        try:
            for seek in ("1", "0"):
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", seek, "-i", source, "-frames:v", "1",
                    "-vf", f"scale={width}:-2", cache, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await process.wait()
                if process.returncode == 0 and os.path.isfile(cache):
                    return await asyncio.to_thread(_read_bytes, cache)
        finally:
            if temp is not None:
                os.remove(temp.name)
        raise NotFound("Could not make a thumbnail for this video.")

    async def _view(self, filename: str, subfolder: str, type_: str = "output"):
        try:
            return await self.comfy.view(filename, subfolder, type_)
        except ComfyNotFound as exc:
            raise NotFound(str(exc)) from None
        except ComfyError as exc:
            raise Unavailable(str(exc)) from None

    def upload_page_link(self) -> str:
        return self.settings.public_base_url + sign_path(self.settings.token, "/upload", self.settings.link_ttl_seconds)
