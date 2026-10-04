"""Job store (SQLite) and HawkService -- the one place REST and MCP both call."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import glob
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

from hawk_h3.script import ScriptError, build_jobs, max_segment_seconds, parse_script, reference_counts_line

from . import content_credentials
from . import graph as graphs
from . import google_images
from . import image_engines
from . import models_llm
from . import pose_guide
from .atlas import AtlasClient, AtlasError, routing_block as atlas_routing
from .auth import sign_path
from .prompts import PLATFORM_RULES, PromptStore
from .comfy_client import ComfyClient, ComfyError, ComfyNotFound, ComfyValidationError
from .config import LLMSettings, LLMSettingsStore, ModelSettings, ModelStore, Settings
from . import local_images
from .local_images import LocalImageEngine, LocalImageError
from .loras import (
    LoraError,
    LoraSpec,
    ResolvedLora,
    choose_steps,
    compare_applied,
    drop_baked_turbo,
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
IMAGE_PRICES = {"z-image": 0.01, "pro-1.5k": 0.036, "pro-2k": 0.072, "lite": 0.032,
                # Google's 1K prices; bigger outputs bill higher, see google_images.PRICES
                "nano-banana": google_images.price("nano-banana"),
                "nano-banana-pro": google_images.price("nano-banana-pro"),
                "nano-banana-lite": google_images.price("nano-banana-lite")}
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
#: models folder -> (every word a file name must contain to be one the Director can load, label for errors).
#:
#: The text encoder needs both words. "qwen3vl" alone was enough while H3 owned that name, but the image
#: engines moved in beside it -- Krea 2 loads qwen3vl_4b and Qwen Image 2.1 loads qwen3vl_8b, in the same
#: folder -- and those are narrower models: 4096 wide against the 32B's 5120. Offering one to a render does
#: not fail in the loader, it fails deep in the first matmul with "mat1 and mat2 shapes cannot be
#: multiplied", which names no file. This is the same pair hawk_colab's own detection has always required.
#: Longest side of an image handed to the planner. The node uses 1024; a bigger picture costs tokens without
#: telling the planner anything new about a shot it only has to describe.
PLANNER_IMAGE_SIDE = 1024
#: Seconds into a video clip to sample, so the planner sees how a reference clip starts, sits and ends.
PLANNER_VIDEO_STAMPS = (0.0, 1.5, 3.0)
# deepseek-v4-pro reasons before it writes, and its reasoning counts against this: at 8192 a full plan
# (5-7k tokens of script) came back empty with finish_reason=length. Same cap as the director's.
PLANNER_MAX_TOKENS = 15_000

#:
#: Each folder takes any one of several word sets. An H3 "hybrid" is fl2va with ref2va's reference pathway (the
#: later blocks' adaln_proj) grafted back in, so it reads references like ref2va does -- 10Eros-Max beta5 is
#: one, and is named "..._h3_TURBO-hybrid_beta5..." with no "ref2va" in it at all.
MODEL_FAMILIES = {"diffusion_models": ((("ref2va",), ("h3", "hybrid")), "Base model"),
                  "text_encoders": ((("qwen3vl", "minimax"),), "Text encoder")}


def model_family(folder: str, files: list[str]) -> list[str]:
    alternatives, _ = MODEL_FAMILIES[folder]
    return [name for name in files
            if any(all(word in os.path.basename(name).lower() for word in words) for words in alternatives)]


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


#: How much of an image's prompt is kept on the asset. It was 500, which fitted the keyword-style prompts
#: the earlier engines wanted but cuts a Qwen Image 2.1 prompt off in its second sentence -- that model is
#: asked for four to five hundred *words*, so the record kept about a seventh of it, ending mid-word. The
#: prompt is what makes a picture reproducible by hand, so it is stored whole at any length anyone writes.
PROMPT_RECORD_LIMIT = 4000


class Store:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS asset_history (id TEXT PRIMARY KEY, data TEXT NOT NULL, deleted_at REAL NOT NULL);
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

    def list_assets(self, limit: int = 100, by: str = "", session: str = "") -> list[dict]:
        """Newest first. by matches a character's name or their member id, so a camera roll is a query.

        Filtered in SQL rather than after the LIMIT, which would return a page of everything and then throw
        most of it away.
        """
        where, args = [], []
        if by:
            where.append("(lower(json_extract(data, '$.by.name')) = ? OR json_extract(data, '$.by.member') = ?)")
            args += [by.strip().lower(), by.strip()]
        if session:
            where.append("json_extract(data, '$.by.session') = ?")
            args.append(session.strip())
        sql = "SELECT data FROM assets" + (" WHERE " + " AND ".join(where) if where else "")
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY created_at DESC LIMIT ?", (*args, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def all_assets(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM assets ORDER BY created_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    #: What a deleted asset leaves behind: enough to say where it came from, nothing to show.
    HISTORY_FIELDS = ("id", "filename", "kind", "sha256", "created_at", "source")

    def delete_asset(self, asset_id: str) -> None:
        """Delete an asset, keeping its history. Images made from it name it as a reference, and a reference
        with no record counts as an upload -- so deleting one generated image used to put every image made from
        it under the upload rules for good."""
        with self._lock:
            row = self._db.execute("SELECT data FROM assets WHERE id = ?", (asset_id,)).fetchone()
            if row:
                data = json.loads(row[0])
                kept = {key: data.get(key) for key in self.HISTORY_FIELDS if key in data}
                self._db.execute("INSERT OR REPLACE INTO asset_history VALUES (?, ?, ?)",
                                 (asset_id, json.dumps(kept), time.time()))
            self._db.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
            self._db.commit()

    def get_history(self, asset_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT data, deleted_at FROM asset_history WHERE id = ?", (asset_id,)).fetchone()
        return dict(json.loads(row[0]), deleted_at=row[1]) if row else None

    def put_history(self, record: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO asset_history VALUES (?, ?, ?)",
                             (record["id"], json.dumps(record), time.time()))
            self._db.commit()

    def drop_history(self, asset_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM asset_history WHERE id = ?", (asset_id,))
            self._db.commit()

    def lineage(self, asset_id: str) -> dict | None:
        """An asset for the purpose of tracing where an image came from: live, or deleted with its history."""
        return self.get_asset(asset_id) or self.get_history(asset_id)

    def referenced_by(self, asset_id: str) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM assets WHERE EXISTS (SELECT 1 FROM json_each(data, '$.source.references') "
                "WHERE json_each.value = ?)", (asset_id,)).fetchall()
        return [row[0] for row in rows]

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
        self.llm_settings = LLMSettingsStore(settings.data_dir)
        self.prompts = PromptStore(settings.data_dir)
        self.image_engines = image_engines.ImageEngineStore(settings.data_dir)
        self.render_models = ModelStore(settings.data_dir)
        #: Called with the job id when a render finishes (e.g. the Google Drive exporter).
        self.render_done_hooks: list = []
        self.local_images = LocalImageEngine(self)
        self._model_cache: dict[str, tuple[float, list[str]]] = {}
        self._llm_clients: dict[tuple[str, str, str], AtlasClient] = {}  # per url+key+routing; see the atlas property
        self._tasks: list[asyncio.Task] = []
        #: Plans in flight. A plan is written by the gateway rather than queued in ComfyUI, so it runs as a
        #: task; held here so stop() can cancel one instead of leaving a job stuck at "running".
        self._plan_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------ properties

    def llm(self) -> LLMSettings:
        """The provider and models in force right now: the stored choice over the pod's own."""
        return self.llm_settings.resolve(self.settings.llm_overrides)

    def planner_model(self) -> str:
        """The planner's configured chain, as written. Comma-separated, best first; see models_llm.chain."""
        return self.llm().planner_model_override or self.settings.planner_model

    async def model_catalogue(self) -> list[dict]:
        """The chat models the provider in force serves, or [] when it cannot be asked.

        [] means "do not second-guess the setting" to models_llm, so a /models call that fails leaves every
        role sending exactly the id it was configured with.
        """
        try:
            return await self.atlas.list_models()
        except AtlasError as exc:
            log.warning("hawk_api: model list unavailable, using configured ids as-is: %s", exc)
            return []

    async def model_for(self, role: str, catalogue: list[dict] | None = None, *, vision: bool = False) -> str:
        """The id to send for one role on whichever provider is switched on. See hawk_api/models_llm.py.

        ``vision`` is for a role whose need for image support depends on the call rather than the role.
        """
        llm = self.llm()
        configured = {
            "planner": llm.planner_model_override or self.settings.planner_model,
            "director": llm.agent_model_override or self.settings.agent_model,
            "prose": llm.agent_prose_model_override or self.settings.agent_prose_model,
            "summary": llm.agent_summary_model_override or self.settings.agent_summary_model,
            "vision": llm.agent_vision_model_override or self.settings.agent_vision_model,
        }.get(role, "")
        if catalogue is None:
            catalogue = await self.model_catalogue()
        return models_llm.resolve(role, configured, catalogue, vision=vision)

    async def model_report(self, catalogue: list[dict] | None = None) -> dict:
        """Per role: what is configured, what it resolves to here, and whether that is what was asked for.

        Returned when the LLM settings are saved, because switching provider silently re-points every role and
        the alternative is finding out from a render that fails much later.
        """
        llm = self.llm()
        if catalogue is None:
            catalogue = await self.model_catalogue()
        return models_llm.report({
            "planner": llm.planner_model_override or self.settings.planner_model,
            "director": llm.agent_model_override or self.settings.agent_model,
            "prose": llm.agent_prose_model_override or self.settings.agent_prose_model,
            "summary": llm.agent_summary_model_override or self.settings.agent_summary_model,
            "vision": llm.agent_vision_model_override or self.settings.agent_vision_model,
        }, catalogue)

    def forget_llm_clients(self) -> None:
        """Drop the cached clients so the next call is built from the settings just saved."""
        self._llm_clients.clear()

    @property
    def atlas(self) -> AtlasClient:
        """The chat service for planning and agent turns, built from the settings as they stand.

        Resolved on every access rather than once at startup, so a change in Studio reaches the next request
        without a restart -- the same arrangement as ModelStore and ImageEngineStore.

        One client is kept per (url, key) pair rather than one per provider with its fields rewritten: a
        client carries a cached model list, so mutating the key on a live one went on serving the previous
        account's models until that cache expired.
        """
        llm = self.llm()
        openrouter_key = llm.openrouter_api_key or self.settings.openrouter_api_key
        atlas_key = llm.atlas_api_key_override or self.settings.atlas_api_key
        # OpenRouter when it is chosen, and also when it is the only key the pod has: a runtime given an
        # OPENROUTER_API_KEY and no ATLAS_API_KEY would otherwise call Atlas unauthenticated and report
        # itself as having no key at all.
        if llm.llm_provider == "openrouter" or (openrouter_key and not atlas_key):
            # Atlas serves its own models, so only OpenRouter gets a routing block: there, one model id is
            # served by many services at different prices and different weights, and left unfiltered the
            # same call can come back at a different quality each time.
            url, key, routing = llm.openrouter_url, openrouter_key, atlas_routing(llm.openrouter_routing)
        else:
            url, key, routing = self.settings.atlas_url, atlas_key, None
        return self._llm_client(url, key, routing, llm.openrouter_routing if routing else "")

    @property
    def image_atlas(self) -> AtlasClient:
        """The Atlas client the paid image engines use, whatever service is answering chat.

        Seedream and z-image/turbo are Atlas models reached over an Atlas-only endpoint
        (/api/v1/model/generateImage), so ``where="atlas"`` describes the engine, not a preference.
        Choosing OpenRouter for chat used to re-point these calls there too, where that path does not
        exist: every paid image came back as a 404 body for the agent to explain in prose.
        """
        llm = self.llm()
        return self._llm_client(self.settings.atlas_url, llm.atlas_api_key_override or self.settings.atlas_api_key,
                                None, "")

    @property
    def google(self) -> google_images.GoogleImageClient:
        """The Gemini client for the Nano Banana engines and their SFW check, from the key in force.

        Rebuilt on every access: it holds nothing but the key, so a key saved in Studio applies at once."""
        return google_images.GoogleImageClient(self.llm().google_api_key or self.settings.google_api_key)

    def _llm_client(self, url: str, key: str, routing: dict | None, routing_key: str) -> AtlasClient:
        """One client per (url, key, routing), kept because a client carries a cached model list."""
        cache_key = (url, key, routing_key)
        found = self._llm_clients.get(cache_key)
        if found is None:
            found = self._llm_clients[cache_key] = AtlasClient(url, key, routing=routing)
        return found

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        load_config(self.settings.loras_path)  # create loras.json on first start
        # A plan runs as a task in this process, so a restart loses it with nothing in ComfyUI to reconcile
        # against. Left alone the job would poll as "running" for ever; retry re-runs it.
        for job in self.store.list_jobs(limit=500, statuses=ACTIVE):
            if job["kind"] == "plan" and not job.get("prompt_id"):
                job.update(status="failed", resumable=True,
                           error="The plan was interrupted by a server restart. Retry it.")
                self.store.save_job(job)
        self._tasks = [
            asyncio.create_task(self.comfy.listen(self.handle_event)),
            asyncio.create_task(self._reconcile_loop()),
        ]
        health = await self.health()
        if health.get("loras") == "degraded":
            log.warning("hawk_api: required default LoRAs missing on the pod: %s", health.get("missing_required_loras"))

    async def stop(self) -> None:
        for task in self._plan_tasks:
            task.cancel()
        if self._plan_tasks:
            await asyncio.gather(*self._plan_tasks, return_exceptions=True)
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

    async def available_loras(self, refresh: bool = False, family: str = "h3") -> list[str]:
        """LoRA files of one family. ComfyUI keeps a single models/loras folder, so MiniMax H3, Krea 2, Qwen Image 2.1
        and Z-Image files all sit in it together and a file from the wrong family produces garbage rather than an
        error. family="" returns the folder as it is."""
        files = await self.available_models("loras", refresh)
        if not family:
            return files
        if family != "h3":
            return [f for f in files if image_engines.family_of(f) == family]
        image_files = await self.local_images.lora_basenames()
        return [f for f in files if os.path.basename(f).lower() not in image_files]

    async def listed_name(self, folder: str, name: str) -> str:
        """`name` as ComfyUI lists it. A configured bare file name becomes its subfolder path -- an encoder kept
        in text_encoders/h3/ is "h3/qwen3vl_..." to the loader, which refuses the bare name as "not in list"."""
        if not name or "/" in name:
            return name
        for refresh in (False, True):  # a file may have been moved since the cached listing
            files = await self.available_models(folder, refresh=refresh)
            if name in files:
                return name
            match = next((f for f in files if f.rsplit("/", 1)[-1] == name), None)
            if match:
                return match
        return name

    async def choose_model(self, requested: str | None, folder: str, default: str) -> str:
        """A per-render base model or text encoder, matched like LoRA names."""
        if not requested or not requested.strip():
            return await self.listed_name(folder, default)
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
                    f"{other} is an fl2va model (first/last-frame video); the Director needs a ref2va model or "
                    f"an H3 fl2va/ref2va hybrid. "
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
        """Say which model a LoRA belongs to when a render asks for an image one, instead of just "not found"."""
        wanted = str(error.details.get("requested") or "").strip().lower()
        if not wanted:
            return
        for name in await self.local_images.lora_basenames():
            if wanted in (name, name.rsplit(".", 1)[0]) or (len(wanted) > 3 and wanted in name):
                family = image_engines.family_of(name)
                raise RequestError(
                    f"{error.details['requested']!r} is a {image_engines.family_label(family)} image LoRA; video "
                    "renders use the MiniMax H3 LoRAs in models/loras. Use list_loras to see them, or "
                    "generate_image for a picture.",
                    {**error.details, "image_lora": name, "family": family},
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
            try:
                gone = await self.missing_render_models()
                result["models"] = "degraded" if gone else "ok"
                if gone:
                    result["ok"] = False
                    result["missing_models"] = gone
            except Exception as exc:
                result["models"] = f"error: {exc}"
        return result

    async def missing_render_models(self) -> list[str]:
        """Configured H3 files that are not in ComfyUI's folders. A render would fail at the loader."""
        models = self.render_models.resolve(self.settings.models)
        gone = []
        for folder, name in (("diffusion_models", models.unet_name), ("text_encoders", models.clip_name),
                             ("vae", models.video_vae), ("vae", models.audio_vae)):
            if not name:
                continue
            files = await self.available_models(folder, refresh=True)
            if not any(f == name or f.rsplit("/", 1)[-1] == name for f in files):
                gone.append(f"{folder}/{name}")
        return gone

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
            # The resolved id, not the configured chain: a dropdown has to preselect something it lists.
            "default_planner_model": await self.model_for("planner", planner_models),
            "default_agent_model": await self.model_for("director", planner_models),
            "model_chains": {"planner": self.planner_model(),
                             "agent": self.llm().agent_model_override or self.settings.agent_model},
            "diffusion_models": model_family("diffusion_models", await self.available_models("diffusion_models", refresh=True)),
            "text_encoders": model_family("text_encoders", await self.available_models("text_encoders", refresh=True)),
            "default_unet": self.render_models.resolve(self.settings.models).unet_name,
            "default_clip": self.render_models.resolve(self.settings.models).clip_name,
            "available_loras": available,
            "default_loras": default_status(config, available),
            "lora_presets": {name: [dataclasses.asdict(spec) for spec in specs] for name, specs in config.presets.items()},
            "samplers": _combo_options(director, "sampler_name"),
            "schedulers": _combo_options(director, "scheduler"),
            "aspect_ratios": _combo_options(director, "aspect_ratio"),
            "continuity_modes": _combo_options(director, "continuity"),
            "reference_roles": list(graphs.ROLES),
            "defaults": {"models": dataclasses.asdict(self.render_models.resolve(self.settings.models)),
                         "planner_model": self.planner_model()},
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
                     owner=None, of=None, generated_from=None) -> dict:
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise NotFound(f"No asset {asset_id!r}.")
        if generated_from is not None:
            self._correct_provenance(asset, generated_from)
        if owner:  # the character who made it, in a group chat: {"name", "member", "session"}
            asset["by"] = {k: str(owner.get(k) or "")[:80] for k in ("name", "member", "session")}
            # A group shot's photographer is drawn at random from the people in it, so it is a real answer
            # but not a fact to file under: no "by" tag, or a guess becomes a filter everyone trusts.
            if owner.get("name") and owner.get("how") != "group":
                add_tags = list(add_tags or []) + [f"by {owner['name']}"]
        if of:  # who is in it, which for a candid is nobody who took it
            asset["of"] = [{"name": str(p.get("name") or "")[:80], "member": str(p.get("member") or "")[:80]}
                           for p in of][:8]
            add_tags = list(add_tags or []) + [f"of {p['name']}" for p in asset["of"] if p["name"]]
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

    def provenance(self, asset_id: str) -> dict:
        """Where an image came from, as the content checks see it.

        Fetched on demand rather than folded into asset_view: the walk is a lookup per ancestor, and a
        listing of a thousand assets should not pay for it a thousand times over.
        """
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise NotFound(f"No asset {asset_id!r}.")
        chain, seen, queue = [], set(), [asset]
        while queue:
            current = queue.pop(0)
            if current["id"] in seen or len(chain) > 40:
                continue
            seen.add(current["id"])
            source = current.get("source") or {}
            chain.append({"id": current["id"], "filename": current.get("filename", ""),
                          "type": source.get("type", ""), "generator": source.get("generator", ""),
                          "corrected": bool(source.get("corrected_at")),
                          "deleted": "deleted_at" in current, "restored": bool(source.get("restored_at"))})
            for ref_id in source.get("references") or []:
                ref = self.store.lineage(str(ref_id))
                if ref is None:
                    chain.append({"id": str(ref_id), "filename": "", "type": "missing", "generator": "",
                                  "corrected": False})
                else:
                    queue.append(ref)
        return {"id": asset["id"],
                "from_upload": local_images.from_upload(asset, self.store.lineage),
                # what the refusal is actually pointing at, which the message never used to name
                "upload_roots": [row["id"] for row in chain if row["type"] in ("upload", "missing")],
                "corrected_from": ((asset.get("source") or {}).get("corrected_from") or None),
                "chain": chain}

    def _correct_provenance(self, asset: dict, source_id: str) -> None:
        """Record that this file is a copy of an asset made here, not something brought in from outside.

        A client that could not reach a generated image on disk used to download it and upload it back. The
        copy arrives with no history, so the upload rules -- which exist because an uploaded photo may show a
        real person -- then apply to an image this pod drew from a prompt. Restoring lost files from the Drive
        export is what stops that happening; this is for the copies made before it did.

        It is an override of a content guardrail, so it leaves a trail rather than quietly rewriting history:
        the previous source is kept under "corrected_from", the change is stamped, and the server logs it.
        Passing "" puts the asset back the way it was.
        """
        previous = asset.get("source") or {}
        if not str(source_id or "").strip():  # undo
            restored = previous.get("corrected_from")
            if restored is None:
                raise RequestError(f"{asset['id']} has no corrected provenance to undo.")
            asset["source"] = restored
            log.warning("hawk_api: provenance correction on %s undone", asset["id"])
            return
        origin = self.store.get_asset(str(source_id).strip())
        if origin is None:
            raise RequestError(f"Unknown generated_from {source_id!r}; it must be an asset in this library.")
        if origin["id"] == asset["id"]:
            raise RequestError("An asset cannot be a copy of itself.")
        if local_images.from_upload(origin, self.store.lineage):
            # Otherwise the correction launders the very thing it is meant to undo: pointing at an asset that
            # is itself an upload, or descends from one, only moves the upload one step further away.
            raise RequestError(
                f"{origin['id']} is an upload or was made from one, so it cannot be the origin of a "
                "generated image. Name the image this file was copied from.")
        if self._descends_from(origin, asset["id"]):
            raise RequestError(f"{origin['id']} was made from {asset['id']}, so it cannot also be its origin.")
        asset["source"] = {"type": "generated",
                           "engine": (origin.get("source") or {}).get("engine", ""),
                           "generator": (origin.get("source") or {}).get("generator", ""),
                           "references": [origin["id"]],
                           "corrected_at": time.time(),
                           "corrected_from": previous}
        log.warning("hawk_api: provenance of %s corrected to a copy of %s (was %r) -- upload rules no longer "
                    "apply to it or to anything made from it", asset["id"], origin["id"], previous.get("type"))

    async def verify_ai_credential(self, asset_id: str, made_from: list[str] | None = None) -> dict:
        """Record the true origin of an AI image made elsewhere, from the C2PA credential its maker signed.

        An image Nano Banana made outside this server and uploaded here counts as an upload, which may show a
        real person. When its embedded credential validates, is signed by a trusted issuer and says the whole
        picture came from a generative model, the asset is recorded as generated by that issuer instead. The
        credential is kept on the asset as evidence; nothing is inferred that the file does not carry.

        ``made_from`` names the library images it was made from, when the credential says it had inputs: each
        one must itself count as generated (verify it first), or an AI edit of a real photo would pass. Like
        generated_from, it leaves a trail: the previous source is kept under "corrected_from", the change is
        logged, and PATCH generated_from="" undoes it.
        """
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise NotFound(f"No asset {asset_id!r}.")
        if asset["kind"] != "image":
            raise RequestError(f"{asset['filename']} is {asset['kind']}; content credentials are checked on images.")
        previous = asset.get("source") or {}
        if previous.get("type") == "generated":
            return {"id": asset["id"], "verified": True, "already": True,
                    "reason": "Already recorded as generated; nothing to change.", "provenance": self.provenance(asset_id)}
        inputs = list(dict.fromkeys(str(ref).strip() for ref in made_from or [] if str(ref).strip()))
        for ref_id in inputs:
            ref = self.store.get_asset(ref_id)
            if ref is None:
                raise RequestError(f"Unknown made_from {ref_id!r}; it must be an asset in this library.")
            if ref["id"] == asset["id"]:
                raise RequestError("An image cannot be made from itself.")
            if local_images.from_upload(ref, self.store.lineage):
                raise RequestError(f"{ref['filename']} ({ref_id}) counts as an upload, so it cannot vouch for what "
                                   "went into this image. Verify its own credential first.")
            if self._descends_from(ref, asset["id"]):
                raise RequestError(f"{ref_id} was made from {asset['id']}, so it cannot also be its input.")
        data = await self.asset_bytes(asset)
        try:
            result = await asyncio.to_thread(content_credentials.check, data, _image_type(data)[1], len(inputs))
        except RuntimeError as exc:  # the c2pa library is not installed on this pod
            raise RequestError(str(exc)) from None
        if not result.ok:
            raise RequestError(f"{asset['filename']}: {result.reason}")
        asset["source"] = {"type": "generated", "engine": "external", "generator": result.generator or result.issuer,
                           "references": inputs, "credential": result.details, "corrected_at": time.time(),
                           "corrected_from": previous}
        self.store.add_asset(asset)
        log.warning("hawk_api: %s recorded as AI-generated by %s from its content credential (was %r, inputs %s) "
                    "-- upload rules no longer apply to it", asset["id"], result.issuer, previous.get("type"), inputs)
        return {"id": asset["id"], "verified": True, "reason": result.reason, "provenance": self.provenance(asset_id)}

    def _descends_from(self, asset: dict, ancestor_id: str, depth: int = 0) -> bool:
        """Whether this asset was made from that one, however far back. Guards against a cycle in the chain."""
        if depth > 20:
            return True
        for ref_id in (asset.get("source") or {}).get("references") or []:
            if str(ref_id) == ancestor_id:
                return True
            ref = self.store.lineage(str(ref_id))
            if ref is not None and self._descends_from(ref, ancestor_id, depth + 1):
                return True
        return False

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

    def restore_record(self, asset_id: str, undo: bool = False) -> dict:
        """Rebuild the history of a generated image that was deleted before deletes kept one.

        Its descendants still name it, and with no record it counts as an upload. It is restored only on the
        evidence this server itself left: an asset that still references it, and its copy in the Drive image
        export, which only ever receives images generated here and names them by id. What it was made from is
        not known, so the record says so; this is a content-guardrail override and leaves a trail like
        generated_from does. ``undo`` removes a record restored this way.
        """
        asset_id = str(asset_id or "").strip()
        if undo:
            record = self.store.get_history(asset_id)
            if not record or not (record.get("source") or {}).get("restored_at"):
                raise RequestError(f"{asset_id} has no restored record to undo.")
            self.store.drop_history(asset_id)
            log.warning("hawk_api: restored record of deleted asset %s removed", asset_id)
            return {"id": asset_id, "restored": False}
        if self.store.get_asset(asset_id) is not None:
            raise RequestError(f"{asset_id} is still in the library; nothing to restore.")
        if self.store.get_history(asset_id) is not None:
            raise RequestError(f"{asset_id} already has a record.")
        children = self.store.referenced_by(asset_id)
        if not children:
            raise RequestError(f"No asset in the library was made from {asset_id}, so there is nothing to restore.")
        exporter = self.drive_exporter
        if exporter is None or not exporter.browser.available:
            raise Unavailable("Google Drive is not mounted, and the record is restored from the Drive image export.")
        root = os.path.join(exporter.browser.root, exporter.settings()["image_folder"])
        found = sorted(glob.glob(os.path.join(root, "*", f"*_{asset_id[:8]}.*")))
        if len(found) != 1:
            raise RequestError(f"Expected one copy of {asset_id} in the Drive image export, found {len(found)}.")
        path = found[0]
        with open(path, "rb") as handle:
            digest = _hash_file(handle)
        relative = exporter.browser.relative(path)
        record = {"id": asset_id, "filename": os.path.basename(path), "kind": "image", "sha256": digest,
                  "created_at": os.path.getmtime(path),
                  "source": {"type": "generated", "engine": "", "generator": "", "references": [],
                             "restored_at": time.time(), "restored_from": relative,
                             "note": "Deleted before deletes kept history; rebuilt from this server's own Drive "
                                     "export. What it was made from is unknown."}}
        self.store.put_history(record)
        log.warning("hawk_api: record of deleted asset %s restored as generated from Drive export %s "
                    "(referenced by %s) -- upload rules no longer apply to images made from it",
                    asset_id, relative, ", ".join(children))
        return {"id": asset_id, "restored": True, "from": relative, "referenced_by": children}

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

    def list_assets(self, limit: int = 100, by: str = "", session: str = "") -> list[dict]:
        return self.store.list_assets(limit, by=by, session=session)

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

    def drive_asset_path(self, asset: dict) -> str | None:
        """A generated image's exported copy in the mounted Drive, when the original is gone.

        After a Colab runtime is restored from a snapshot the database is back but ComfyUI's input folder
        is empty, so the Drive export holds the only surviving pixels. export_assets names them
        <slug>_<asset_id[:8]>.<ext> under a dated folder, so the id alone is enough to find one.

        Deliberately not folded into local_asset_path: export_assets and delete_asset call that one, and
        an export that found its own output would copy a file onto itself.
        """
        exporter = self.drive_exporter
        if exporter is None or asset.get("kind") != "image" or not asset.get("id"):
            return None
        try:
            if not exporter.browser.available:
                return None
            folder = exporter.settings()["image_folder"]
        except Exception:
            return None
        stamp = asset.get("created_at") or time.time()
        root = os.path.join(exporter.browser.root, folder)
        # The export folder is dated, and a restore can land either side of midnight.
        for offset in (0, -86400, 86400):
            day = time.strftime("%Y-%m-%d", time.localtime(stamp + offset))
            found = glob.glob(os.path.join(root, day, f"*_{asset['id'][:8]}.*"))
            if found:
                return found[0]
        return None

    async def ensure_asset_on_disk(self, asset: dict) -> bool:
        """Put a reference's file back under its recorded path when a restored runtime has lost it.

        ComfyUI loads a reference by the relative path stored on the asset, so an edit of an image whose
        file is gone fails in the loader. After a Colab restore the database is back but ComfyUI's input
        folder is empty, and the Drive export holds the only surviving pixels -- copying one back means the
        asset id keeps working instead of the edit failing.

        This is what stops a client falling back to downloading the image and uploading it again. That
        workaround makes a *new* asset marked as an upload, and an upload is treated as a photograph that
        may show a real person, so an image this pod generated from a prompt ends up under the rules meant
        for real people. Restoring the file keeps the provenance the library already recorded.

        False only when the file is known to be missing and no export was found to restore it from.
        """
        input_dir = self.settings.comfy_input_dir
        if not input_dir or not os.path.isdir(input_dir):
            return True  # ComfyUI is on another machine; its own input folder is the authority
        if self.local_asset_path(asset):
            return True
        source = self.drive_asset_path(asset)
        if not source:
            return False
        target = os.path.join(input_dir, asset["path"])
        await asyncio.to_thread(os.makedirs, os.path.dirname(target), exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source, target)
        return True

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
        local = self.local_asset_path(asset) or self.drive_asset_path(asset)
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
        negative: str = "",
        cfg: float | None = None,
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

        # Every rung is on Atlas and the pod has no Atlas key: say so here rather than letting the request
        # reach an endpoint it cannot answer and come back as a status code for the agent to interpret.
        rungs = [spec for spec in (image_engines.get(rung) for rung in ladder) if spec]
        if rungs and all(spec.google for spec in rungs) and not self.google.configured:
            raise RequestError(
                f"{', '.join(spec.label for spec in rungs)} runs on Google, and this pod has no Google API key. "
                "Add one under Settings -> LLM Routing, or as the GOOGLE_API_KEY Colab secret.")
        if rungs and all(spec.atlas for spec in rungs) and not self.image_atlas.configured:
            named = ", ".join(spec.label for spec in rungs)
            here = await self.ready_image_engines(action)
            raise RequestError(
                f"{named} runs on Atlas Cloud, and this pod has no Atlas key. Add one under Settings -> Connect, "
                + (f"or use an engine on this GPU: {', '.join(here)}." if here
                   else "or install an image engine on this GPU."))

        # An engine that only makes images from text still has somewhere to go when given references.
        # A pinned local engine steps sideways to another local one: sending it to Seedream instead would
        # spend the user's money on an engine they didn't ask for. A pinned Atlas engine is already paid for.
        if sources and len(ladder) == 1:
            only = image_engines.get(ladder[0])
            if only and not only.edit and only.local:
                sideways = next((e for e in self.image_ladder("edit") if image_engines.get(e).local), "")
                if not sideways:
                    raise RequestError(
                        f"{only.label} can't use reference images, and no local engine here can edit. "
                        f'Drop the references, or use engine "seedream" to edit on Atlas.')
                notes.append(f"{only.label} can't use reference images, so {image_engines.get(sideways).label} made this one.")
                ladder = [sideways]
            elif only and not only.edit:
                notes.append(f"{only.label} can't use reference images, so Seedream edit made this one.")
                ladder, atlas_override = ["seedream"], IMAGE_EDIT_MODEL

        if raw == "auto" and loras:
            ladder = await self._lora_ladder(ladder, action, loras, notes)

        references: list[str] = []

        async def atlas_references() -> list[str]:
            """The sources as data URIs, built once and only if an Atlas engine actually runs."""
            if sources and not references:
                for asset in sources:
                    mime = mimetypes.guess_type(asset["filename"])[0] or "image/png"
                    references.append(f"data:{mime};base64," + base64.b64encode(await self.asset_bytes(asset)).decode("ascii"))
            return references

        google_refs: list[tuple[bytes, str]] = []

        async def google_references() -> list[tuple[bytes, str]]:
            """The sources as (bytes, mime), built once and only if a Google engine is actually reached."""
            if sources and not google_refs:
                for asset in sources:
                    mime = mimetypes.guess_type(asset["filename"])[0] or "image/png"
                    google_refs.append((await self.asset_bytes(asset), mime))
            return google_refs


        # One wait budget for the whole walk, as a deadline: three local rungs must not each wait the full
        # time on the same busy ComfyUI. Nothing to wait for when no local engine is in the ladder.
        wait_until = 0.0
        if any(image_engines.get(e).local for e in ladder):
            budget = self.image_engines.wait_seconds()
            wait_until = time.monotonic() + budget if budget else 0.0

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
                        ref_boost=ref_boost, max_adult_loras=max_adult_loras, negative=negative, cfg=cfg,
                        wait_seconds=max(0.0, wait_until - time.monotonic()) if wait_until else 0.0)
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
                                                extra={"loras": local.loras, "seconds": local.seconds,
                                                       **({"enhanced_prompt": local.enhanced_prompt}
                                                          if local.enhanced_prompt else {})})

            if spec.google:
                if loras and not said_loras:
                    notes.append("Image LoRAs only apply to the local engines; ignored here.")
                    said_loras = True
                if not self.google.configured:
                    why = (f"{spec.label} needs a Google API key: add one under Settings -> LLM Routing, or as the "
                           "GOOGLE_API_KEY Colab secret.")
                else:
                    # The fixed platform rules, exactly as the local engines apply them. A refusal is final: it is
                    # never handed on to another engine.
                    try:
                        local_images.check_prompt(prompt)
                        local_images.check_edit(prompt, sources, [], lookup=self.store.get_asset)
                    except LocalImageError as exc:
                        raise RequestError(" ".join([str(exc), *notes]).strip()) from None
                    try:
                        used, images, cost = await self._google_images(
                            spec, prompt, await google_references(), size, n, notes)
                    except google_images.GoogleError as exc:
                        why = f"{spec.label}: {exc}"
                    else:
                        return await self._image_result(
                            prompt, images, used, spec.tag_for(action), notes, tried, reference_asset_ids,
                            engine_id=spec.id, extra={"cost_usd": round(cost, 4)})
                # No key, or Google declined or failed: the next engine gets the request.
                last_error = RequestError(" ".join([why, *notes]).strip())
                if not more:
                    raise last_error
                tried.append({"engine": engine_id, "skipped": why[:300]})
                continue

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
                                            engine_id=image_engines.id_for_tag(tag),
                                            extra={"cost_usd": round(cost, 4)})
        if last_error:
            raise last_error
        raise RequestError("No image engine could make this image. " + (
            "Tried: " + "; ".join(f"{t['engine']}: {t.get('skipped') or t.get('error')}" for t in tried)
            if tried else "No engine is enabled for this in Studio, under Images."))

    async def _lora_ladder(self, ladder: list[str], action: str, loras: list[dict], notes: list) -> list[str]:
        """With LoRAs named, "auto" walks only the local engines that could load them.

        Naming a LoRA names the engine in all but words: an engine that does not have the file refuses the
        request outright instead of stepping aside, which is right for a typo and wrong when the file
        belongs to the rung below. A lead engine with no LoRAs of its own would otherwise turn every LoRA
        request into a refusal.

        Narrowed only when an engine that qualifies is also installed. Otherwise the ladder is left exactly
        as it was, so the walk still reaches an engine that can explain itself -- the alternative is a
        request for a LoRA quietly answered by a paid engine that ignores LoRAs entirely.
        """
        try:
            families = await self.local_images.families_for_loras([(spec or {}).get("name") for spec in loras])
            able = [e for e in ladder
                    if image_engines.get(e).local and image_engines.get(e).lora_family in families]
            if not able or not set(able) & set(await self.ready_image_engines(action)):
                return ladder
        except Exception:  # ComfyUI down; the walk is about to fail on something louder than this
            return ladder
        kept = [e for e in ladder if e in able or not image_engines.get(e).local]
        skipped = [image_engines.get(e).label for e in ladder if e not in kept]
        if skipped:
            notes.append(f"{', '.join(skipped)} {'do' if len(skipped) > 1 else 'does'} not have the LoRAs "
                         f"asked for, so {'they were' if len(skipped) > 1 else 'it was'} skipped.")
        return kept

    async def ready_image_engines(self, action: str = "generate") -> list[str]:
        """Enabled engines for this action that could actually run right now, best first.

        Naming an engine pins it, so a caller that steps up onto one whose weights are missing gets an error
        rather than a fallback. The agent asks this first so a failed take never steps onto a dead rung.
        """
        ready = []
        for engine_id in self.image_ladder(action):
            spec = image_engines.get(engine_id)
            if spec.google:
                # ready means "could run": whether a given request passes the SFW check is only known per request
                if self.google.configured:
                    ready.append(engine_id)
                continue
            if not spec.local:
                if self.image_atlas.configured:
                    ready.append(engine_id)
                continue
            if spec.id not in local_images.WIRED_ENGINES:
                continue
            try:
                status = await self.local_images.status(spec.id)
            except Exception:  # ComfyUI down: not a rung the agent can use
                continue
            if status.get("installed") and (action != "edit" or (status.get("edit") or {}).get("installed")):
                ready.append(engine_id)
        return ready

    def image_ladder(self, action: str = "generate") -> list[str]:
        """Engine ids to try for this action, best first, as Studio has them ordered."""
        return self.image_engines.order(action)

    #: Reported as the result's "model". Deliberately not "z-image/..." -- that prefix means the Atlas engine
    #: to _text_only_image_model, and a local id must never be mistaken for it.
    LOCAL_MODEL_NAMES = {"krea2": "krea2/turbo", "qwen21": "qwen-image/2.1", "zimage": "zimage/turbo"}

    async def _local_image(self, spec, action: str, prompt: str, sources: list, **kwargs):
        """Run one local engine. Returns its result and the model name to report."""
        if action == "edit":
            if spec.id == "qwen21":
                local = await self.local_images.edit_qwen21(
                    prompt, sources, size=kwargs["size"], n=kwargs["n"], seed=kwargs["seed"], loras=kwargs["loras"],
                    steps=kwargs["steps"], max_adult_loras=kwargs["max_adult_loras"],
                    wait_seconds=kwargs["wait_seconds"], negative=kwargs["negative"], cfg=kwargs["cfg"])
                return local, "qwen-image/2.1-edit"
            local = await self.local_images.edit(
                prompt, sources, size=kwargs["size"], n=kwargs["n"], seed=kwargs["seed"], loras=kwargs["loras"],
                steps=kwargs["steps"], ref_boost=kwargs["ref_boost"], max_adult_loras=kwargs["max_adult_loras"],
                wait_seconds=kwargs["wait_seconds"], cfg=kwargs["cfg"])
            return local, "krea2/identity-edit"
        local = await self.local_images.generate(
            prompt, size=kwargs["size"], n=kwargs["n"], seed=kwargs["seed"], loras=kwargs["loras"],
            steps=kwargs["steps"], max_adult_loras=kwargs["max_adult_loras"], engine=spec.id,
            wait_seconds=kwargs["wait_seconds"], negative=kwargs["negative"], cfg=kwargs["cfg"])
        return local, self.LOCAL_MODEL_NAMES[spec.id]

    def _ladder_view(self, action: str, settings: dict, local: dict) -> list[dict]:
        rows = []
        for row in settings[action]:
            spec = image_engines.get(row["engine"])
            if spec.google:
                ready, why = self.google.configured, "" if self.google.configured else (
                    f"{spec.label} runs on Google. This pod has no Google API key: add one under Settings -> LLM "
                    "Routing, or as the GOOGLE_API_KEY Colab secret.")
            elif not spec.local:
                ready, why = self.image_atlas.configured, "" if self.image_atlas.configured else (
                    f"{spec.label} runs on Atlas Cloud, whichever service answers chat. This pod has no Atlas key: "
                    "add one under Settings -> Connect, or use one of the engines on this GPU.")
            elif spec.id not in local_images.WIRED_ENGINES:
                ready, why = False, f"{spec.label} has no graph on this build yet."
            else:
                status = local if spec.id == "krea2" else (local.get("engines") or {}).get(spec.id) or {}
                edit = status.get("edit") or {}
                if not status.get("installed"):
                    ready, why = False, "Its model files are not on this pod: " + ", ".join(status.get("missing") or [])
                elif action == "edit" and not edit.get("installed"):
                    ready, why = False, "Its edit nodes or LoRA are missing: " + ", ".join(edit.get("missing") or [])
                else:
                    ready, why = True, ""
            rows.append({"engine": spec.id, "label": spec.label, "where": spec.where, "enabled": row["enabled"],
                         "cost_usd": IMAGE_PRICES.get(spec.price_key, 0.0), "max_refs": spec.max_refs,
                         "lora_family": spec.lora_family, "ready": ready, "why_not": why,
                         **({"sfw_only": True} if spec.google else {})})
        return rows

    async def image_options(self) -> dict:
        local = await self.local_images.status()
        # every wired local engine but Krea 2, whose status is the top-level one this merges into
        local["engines"] = {name: await self.local_images.status(name)
                            for name in local_images.LOCAL_MODELS if name != "krea2"}
        # The LoRA listing is passed in so view() can warn about an always-on LoRA that is configured but
        # not on the pod. It attaches nothing and raises nothing, so this is the only place it can show.
        try:
            installed = await self.local_images.lora_basenames()
        except Exception:  # ComfyUI down; "local" above already says so, and a second complaint helps nobody
            installed = None
        engines = self.image_engines.view(installed)
        ladders = {action: self._ladder_view(action, engines, local) for action in ("generate", "edit")}
        return {
            "default_engine": self.settings.image_engine,
            "generate_ladder": ladders["generate"],
            "edit_ladder": ladders["edit"],
            "would_use": {action: ((await self.ready_image_engines(action)) or [None])[0]
                          for action in ("generate", "edit")},
            "busy": engines["busy"],
            "engine_warnings": engines["warnings"],
            "system": await self.comfy.system_stats(),
            "local": local,
            "atlas": {"configured": self.image_atlas.configured, "text_to_image": self.settings.image_model,
                      "quality": IMAGE_MODEL, "edit": IMAGE_EDIT_MODEL, "lite": IMAGE_LITE_MODEL,
                      "prices_usd": {"z-image/turbo": IMAGE_PRICES["z-image"], "seedream 1.5K (up to 2.36 MP)": IMAGE_PRICES["pro-1.5k"],
                                     "seedream 2K": IMAGE_PRICES["pro-2k"], "seedream-lite (2K+)": IMAGE_PRICES["lite"]}},
            "google": {"configured": self.google.configured, "sfw_only": True,
                       "models": {e.id: e.google_model for e in image_engines.ENGINES.values() if e.google},
                       # the vision role, on the chat provider: the check never goes to Google
                       "sfw_check_model": self.llm().agent_vision_model_override or self.settings.agent_vision_model,
                       "prices_usd": google_images.PRICES},
            "sizes": ["1024x1024", "1024x1536", "1536x1024", "896x1600", "1600x896"],
        }

    async def sfw_check(self, prompt: str, action: str, sources: list[dict]) -> google_images.GateVerdict:
        """Whether a request may go to Google, decided on this server's own vision model -- never on Google.

        The check exists because the Nano Banana engines are for SFW work only, so the request it judges may
        well be NSFW. Sending that to Google to be judged would put exactly the content the gate is there to
        keep away from Google into the account's API traffic. So it runs on the vision role -- the same models
        inspect_image uses, on the chat provider in force -- and Google only ever sees requests already
        judged SFW.

        Never raises. No vision model, no provider key, every model failing or answering in a shape that
        cannot be read: each one is an nsfw verdict, so Google is skipped and the ladder moves on.
        """
        llm = self.llm()
        try:
            models = models_llm.resolve_many("vision", llm.agent_vision_model_override or self.settings.agent_vision_model,
                                             await self.model_catalogue())
        except Exception:  # an unreadable chain is no reason to raise out of a check that must not
            models = []
        if not models or not self.atlas.configured:
            return google_images.GateVerdict(False, "No vision model is available on the chat provider to run the "
                                                    "SFW check, so Google was not used.")
        parts: list[dict] = [{"type": "text", "text": (
            f"Request type: {action}\nReference images: {len(sources)}\nPrompt:\n{prompt.strip()}")}]
        for index, asset in enumerate(sources, 1):
            data, mime = await self.asset_file(asset["id"], 640)  # a 640 px copy is plenty to judge, as in inspect
            parts.append({"type": "text", "text": f"Reference image {index}:"})
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
        messages = [{"role": "system", "content": google_images.gate_instruction(self.prompts.get("sfw_gate"))},
                    {"role": "user", "content": parts}]
        failures = []
        for model in models[:3]:
            try:
                text, _ = await self.atlas.chat(model, messages, json_mode=True, max_tokens=400, temperature=0,
                                                max_retries=1)
            except AtlasError as exc:  # a refusal or an outage: the next model in the chain may answer
                failures.append(f"{model}: {str(exc)[:120]}")
                continue
            if google_images.readable_verdict(text):
                return google_images.parse_verdict(text)
            failures.append(f"{model}: no verdict in its reply")
        return google_images.GateVerdict(False, "The SFW check could not get a verdict from the vision model ("
                                                + "; ".join(failures)[:300] + "), so Google was not used.")

    async def _google_images(self, spec, prompt: str, references: list[tuple[bytes, str]], size,
                             n: int, notes: list) -> tuple[str, list[bytes], float]:
        """(model used, images, estimated USD). Gemini makes one image per request, so n runs in parallel."""
        max_tier = google_images.MAX_TIER.get(spec.id, "4K")
        _, tier = google_images.image_config(size, max_tier)
        if tier != google_images.image_config(size)[1]:
            notes.append(f"{spec.label} makes 1K images only, so this one is 1K rather than the size asked for.")
        batches = await asyncio.gather(*(self.google.generate(spec.google_model, prompt, references, size, max_tier)
                                         for _ in range(max(1, n or 1))))
        images = [image for batch in batches for image in batch]
        return spec.google_model, images, google_images.price(spec.id, tier) * len(images)

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
            batches = await asyncio.gather(*(self.image_atlas.generate_image(body) for body in payloads))
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
                  "prompt": prompt.strip()[:PROMPT_RECORD_LIMIT], "references": list(reference_asset_ids or [])}
        if extra and extra.get("loras"):
            source["loras"] = extra["loras"]
        if extra and extra.get("sfw_check_failed"):
            # Kept on the asset so a retake read back from it (agent._retake_args) knows Google is not an option.
            source["sfw_check"] = {"verdict": "nsfw", "reason": extra["sfw_check_failed"]}
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

    async def _refs(self, references: list[ReferenceIn]) -> list[graphs.Ref]:
        """The references as the graph wants them, restoring any file a restarted runtime has lost.

        ComfyUI loads a reference by its recorded path, and after a Colab restore the database is back
        while ComfyUI's input folder is empty -- so an asset the library still lists reaches the loader as
        "Invalid image file", which says nothing about why. Restoring from the Drive export first turns the
        common case into a working render, and names the files when it cannot.
        """
        refs, missing = [], []
        for reference in references:
            asset = self.store.get_asset(reference.asset_id)
            if asset is None:
                raise RequestError(f"Unknown asset_id {reference.asset_id!r}. Upload the file first.")
            if not await self.ensure_asset_on_disk(asset):
                missing.append(asset.get("filename") or asset["id"])
            refs.append(graphs.Ref(asset["id"], asset["kind"], asset["path"], reference.role, reference.label, reference.for_video))
        if missing:
            raise RequestError(self._lost_files_message(missing))
        return refs

    @staticmethod
    def _lost_files_message(names: list[str]) -> str:
        """Why a file the library lists is not on disk. ComfyUI would say "Invalid image file" and stop there."""
        return ("The file is gone from ComfyUI's input folder for " + ", ".join(names) +
                ", and no Drive export was found to restore it from. The library still lists it because the "
                "database survived the restart; the pixels did not. Re-upload the file, or use an asset made "
                "since the restart.")

    # ------------------------------------------------------------ jobs

    async def _planner_args(self, options: PlannerOptions, seed: int) -> dict:
        return {
            "story": options.story,
            "segment_count": options.segment_count,
            "segment_seconds": options.segment_seconds,
            "aspect_ratio": options.aspect_ratio,
            # Resolved, never the raw chain: the node takes one id, and a comma-separated string reaches the
            # provider as a model that does not exist.
            "model": await self._planner_model_id(options.model),
            "seed": seed,
            "temperature": options.temperature,
            # Blank keeps the node's built-in guide; an edited planner prompt gets the platform rules appended.
            "system_prompt": (f"{custom.rstrip()}\n\n{PLATFORM_RULES}" if (custom := self.prompts.custom("planner")) else ""),
        }

    async def _planner_model_id(self, requested: str = "", *, needs_vision: bool = False) -> str:
        """One model id for the planner: what the caller asked for, else the configured chain, resolved here.

        A request naming a model still goes through the resolver, so "xai/grok-4.6" works on a pod switched to
        OpenRouter, where the same model is spelled "x-ai/grok-4.6".

        ``needs_vision`` is set when the plan has reference photos, because those are sent as image parts and a
        text-only model is refused by the provider rather than merely doing the job badly. It is per call, not
        per role: a plan with no references is still free to use the cheaper text-only model at the head of the
        chain. This is how a chain like "deepseek-v4-pro, xai/grok-4.6" does the right thing in both cases.
        """
        return (await self._planner_model_ids(requested, needs_vision=needs_vision) or [""])[0]

    async def _planner_model_ids(self, requested: str = "", *, needs_vision: bool = False) -> list[str]:
        """Every planner id worth trying, best first. A model the caller named is tried alone; the configured
        chain is tried in full, so one model rate-limited upstream no longer fails the plan while the next
        candidate is free."""
        catalogue = await self.model_catalogue()
        if str(requested or "").strip():
            one = (models_llm.resolve("planner", requested, catalogue, vision=needs_vision)
                   or ("" if needs_vision else str(requested).strip()))
            return [one] if one else []
        llm = self.llm()
        configured = llm.planner_model_override or self.settings.planner_model
        return [m for m in models_llm.resolve_many("planner", configured, catalogue, vision=needs_vision) if m]

    def planner_system_prompt(self) -> str:
        """The planner's instructions: the user's edited prompt with the platform rules, else the node's own.

        Read from hawk_h3's own prompt file rather than duplicated, so the gateway and the node cannot drift.
        hawk_h3.planner itself imports ComfyUI, which the gateway has no business importing, but the prompt is
        just a file.
        """
        custom = self.prompts.custom("planner")
        if custom:
            return f"{custom.rstrip()}\n\n{PLATFORM_RULES}"
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hawk_h3", "prompts", "planner_system.md")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError as exc:
            raise Unavailable(f"The planner's instructions are missing ({path}): {exc}") from None

    async def _planner_message(self, options: PlannerOptions, refs: list) -> tuple[str, list[str]]:
        """The planner's user message and the images attached to it, in the order the text refers to them.

        This mirrors hawk_h3.planner.build_request, which cannot be reused directly because it works on
        decoded tensors inside ComfyUI while the gateway has assets on disk.
        """
        images: list[str] = []
        lines: list[str] = []
        counts = {"picture": 0, "pose": 0, "video": 0, "audio": 0}

        async def attach(asset: dict) -> None:
            data, mime = await self.asset_file(asset["id"], PLANNER_IMAGE_SIDE)
            images.append(f"data:{mime};base64," + base64.b64encode(data).decode("ascii"))

        for ref in refs:
            if ref.role not in counts:
                continue
            counts[ref.role] += 1
            number = counts[ref.role]
            asset = self.store.get_asset(ref.asset_id) or {}
            label = f" -- {ref.label}" if ref.label else ""
            if ref.role in ("picture", "pose"):
                await attach(asset)
                kind = ", a POSE reference (body pose only)" if ref.role == "pose" else ""
                lines.append(f"<{ref.role.capitalize()} {number}> = attached image {len(images)}{kind}{label}")
            elif ref.role == "video":
                first = len(images) + 1
                stamps = []
                for seconds in PLANNER_VIDEO_STAMPS:
                    frame = await self._video_frame(asset, seconds, PLANNER_IMAGE_SIDE)
                    if frame is None:
                        continue
                    images.append("data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii"))
                    stamps.append(f"{seconds:g}s")
                if stamps:
                    lines.append(f"<Video {number}> = clip; attached images {first}-{len(images)} are its frames "
                                 f"at {', '.join(stamps)}{label}")
                else:
                    # ffmpeg missing or the clip would not decode. Naming it unattached is honest: the planner
                    # can still write around a reference it was told about but cannot see.
                    lines.append(f"<Video {number}> = clip (frames not attached){label}")
            else:
                lines.append(f"<Audio {number}> = audio clip (not attached){label}")

        cap = max_segment_seconds()
        count = (f"exactly {options.segment_count} segment(s)" if options.segment_count > 0
                 else "as many segments as the story needs (usually 2-8)")
        available = {"Picture": counts["picture"], "Pose": counts["pose"],
                     "Video": counts["video"], "Audio": counts["audio"]}
        text = "\n".join([
            "BRIEF:", options.story.strip(), "",
            "REFERENCES (global numbering):", *(lines or ["(none -- this is a text-only film)"]), "",
            "CONSTRAINTS:",
            f"- Write {count}.",
            f"- Target about {min(options.segment_seconds, cap):g} seconds per segment (each 5-{cap:g}s; never longer "
            f"than {cap:g}s, even where your instructions allow 15: split a longer beat into two segments).",
            f"- Frame: {options.aspect_ratio}.",
            f"- {reference_counts_line(available)}",
            "- Return only the JSON object described in your instructions.",
        ])
        # A sexual brief gets the position guide: the planner otherwise writes "they have sex in the full nelson"
        # and H3, which knows no position names, draws whatever it guesses.
        note = pose_guide.planner_note(options.story)
        if note:
            text = f"{text}\n\n{note}"
        return text, images

    async def _video_frame(self, asset: dict, seconds: float, width: int) -> bytes | None:
        """One JPEG frame from a video, or None when it cannot be taken. Never raises: a reference the planner
        cannot see is a weaker plan, not a failed render."""
        if not asset or not shutil.which("ffmpeg"):
            return None
        source = self.local_asset_path(asset)
        temp = None
        if source is None:
            try:
                temp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(asset.get("filename", ""))[1] or ".mp4", delete=False)
                temp.write(await self.asset_bytes(asset))
                temp.close()
                source = temp.name
            except (OSError, NotFound):
                return None
        out = os.path.join(tempfile.gettempdir(), f"hawk_plan_{asset['id']}_{seconds:g}.jpg")
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{seconds:g}", "-i", source, "-frames:v", "1",
                "-vf", f"scale={width}:-2", out,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await process.wait()
            if process.returncode == 0 and os.path.isfile(out):
                with open(out, "rb") as handle:
                    return handle.read()
            return None
        finally:
            for path in (temp.name if temp is not None else None, out):
                if path and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    async def plan_text(self, options: PlannerOptions, refs: list) -> tuple[str, str, dict]:
        """Write the film's script here in the gateway, on the provider the settings actually name.

        The planner used to run inside ComfyUI, where the graph hardcoded Atlas's URL and an empty key, so the
        node fell back to ATLAS_API_KEY in ComfyUI's own environment: choosing OpenRouter in Studio set the
        planner's model and silently left its provider on Atlas, and none of its tokens were ever counted.
        Calling it from here fixes all three, and keeps the key out of a graph that gets stored in job records
        and ComfyUI's history.
        """
        # The message is built first because whether it carries photos is what decides which models can serve
        # it: with references this call is multimodal, and a text-only model is rejected outright.
        text, images = await self._planner_message(options, refs)
        models = await self._planner_model_ids(options.model, needs_vision=bool(images))
        if not models:
            raise Unavailable(
                "No planner model that can read reference photos: the provider in force lists none of the "
                "configured planner ids with image support. Put a model that can see at the head of the "
                "planner chain, or plan without references."
                if images else
                "No planner model: the provider in force lists none of the configured ids.")
        content: list[dict] = [{"type": "text", "text": text}]
        for uri in images:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        messages = [{"role": "system", "content": self.planner_system_prompt()},
                    {"role": "user", "content": content if images else text}]
        failures: list[str] = []
        for model in models:
            try:
                reply, usage = await self.atlas.chat(model, messages, json_mode=True, max_retries=2,
                                                    max_tokens=PLANNER_MAX_TOKENS, temperature=options.temperature)
                if not (reply or "").strip():
                    raise Unavailable(f"{model} returned an empty plan.")
                # "Not empty" was the only thing ever checked here, so a plan cut off mid-sentence was stored as
                # a finished one: the job said done with no error, Studio drew a half-written script, and the
                # first thing to actually parse it was the render, at _validate_script, minutes later. Parsing it
                # here turns that into a failure the next model in the chain gets to fix.
                try:
                    parse_script(reply)
                except ScriptError as exc:
                    raise Unavailable(f"{model} returned a plan that does not parse: {exc}") from None
                return reply, model, usage or {}
            except (AtlasError, Unavailable) as exc:
                failures.append(str(exc))
                if model != models[-1]:
                    log.warning("Planner %s failed (%s); trying the next model in the chain.", model, str(exc)[:200])
        if len(failures) == 1:
            raise Unavailable(failures[0])
        raise Unavailable("Every planner model failed: " + " | ".join(f[:300] for f in failures))

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
        """Start a plan. The script is written by the gateway, not by a node inside ComfyUI.

        A plan job therefore has no ComfyUI prompt and never reaches the queue; it runs as a task so the
        request returns at once and clients keep polling get_job exactly as before.
        """
        refs = await self._refs(request.references)
        seed = request.seed if request.seed is not None else random.randrange(1, 2**31)
        # Validated before anything is queued, so a bad wiring is still a 4xx rather than a failed job.
        try:
            wiring = graphs.reference_wiring(refs)
        except graphs.GraphError as exc:
            raise RequestError(str(exc)) from None
        job = self._new_job(
            "plan",
            request=request.model_dump(),
            refs=[dataclasses.asdict(ref) for ref in refs],
            graph={},
            nodes={},
            available=wiring.available,
            video_has_audio=wiring.video_has_audio,
            seed=seed,
        )
        job["status"] = "planning"  # the status a plan has always reported while the LLM is writing
        job["started_at"] = time.time()
        self.store.save_job(job)
        task = asyncio.create_task(self._run_plan(job["id"], request, refs))
        self._plan_tasks.add(task)
        task.add_done_callback(self._plan_tasks.discard)
        return self.store.get_job(job["id"]) or job

    async def _run_plan(self, job_id: str, request: PlanRequest, refs: list) -> None:
        """Write the script and finish the job. Every failure ends as a failed job, never an unretrieved
        exception: the caller has already been given a job id and is polling it."""
        try:
            script, model, usage = await self.plan_text(request, refs)
        except asyncio.CancelledError:
            job = self.store.get_job(job_id)
            if job and job["status"] not in ("done", "failed"):
                job.update(status="failed", error="The plan was interrupted by a server restart.", resumable=True)
                self.store.save_job(job)
            raise
        except (AtlasError, RequestError, Unavailable, NotFound) as exc:
            job = self.store.get_job(job_id)
            if job:
                job.update(status="failed", error=str(exc), resumable=True)
                self.store.save_job(job)
            return
        except Exception as exc:  # a planner bug must not leave a job running for ever
            log.exception("hawk_api: plan %s failed", job_id)
            job = self.store.get_job(job_id)
            if job:
                job.update(status="failed", error=f"The planner failed: {exc}", resumable=True)
                self.store.save_job(job)
            return
        job = self.store.get_job(job_id)
        if job is None or job["status"] == "failed":
            return
        job["script"] = script
        job["planner_model"] = model
        # The planner's tokens used to vanish: the call was made inside ComfyUI, where nothing counts them.
        job["usage"] = {"model": model, "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
                        "completion_tokens": (usage or {}).get("completion_tokens", 0)}
        job["status"] = "done"
        job["updated_at"] = time.time()
        self.store.save_job(job)

    async def create_video(self, request: VideoRequest) -> dict:
        settings = request.settings
        references = request.references
        script_text: str | None = None
        planner_model, planner_usage = "", {}

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

        refs = await self._refs(references)
        # The base model first: a TURBO checkpoint decides both which LoRAs are left off and the step count.
        defaults = self.render_models.resolve(self.settings.models)
        models = dataclasses.replace(
            defaults,
            attention=settings.attention or defaults.attention,
            unet_name=await self.choose_model(settings.unet_name, "diffusion_models", defaults.unet_name),
            clip_name=await self.choose_model(settings.clip_name, "text_encoders", defaults.clip_name),
            video_vae=await self.listed_name("vae", defaults.video_vae),
            audio_vae=await self.listed_name("vae", defaults.audio_vae),
        )
        loras, warnings = await self.resolve_loras(settings)
        loras, baked = drop_baked_turbo(loras, models.unet_name)
        warnings += baked
        steps, steps_reason = choose_steps(loras, settings.steps, models.unet_name)
        seed = settings.seed if settings.seed is not None else random.randrange(1, 2**48)
        job_id = str(uuid.uuid4())
        music_path = None
        if settings.music_asset_id:
            music = self.store.get_asset(settings.music_asset_id)
            if music is None:
                raise RequestError(f"Unknown music_asset_id {settings.music_asset_id!r}. Upload the track first.")
            if music["kind"] != "audio":
                raise RequestError(f"music_asset_id must be an audio file (mp3, wav, m4a...); {music['filename']} is {music['kind']}.")
            if not await self.ensure_asset_on_disk(music):  # loaded from the same folder, lost the same way
                raise RequestError(self._lost_files_message([music.get("filename") or music["id"]]))
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
        if request.story is not None:
            # Planned here rather than by a node in the graph, so the provider chosen in Studio is the one that
            # writes the film and its tokens are counted. The render then takes the finished script, which is
            # the path render_graph already had for a plan the user edited by hand.
            script_text, planner_model, planner_usage = await self.plan_text(request.story, refs)

        try:
            built, wiring = graphs.render_graph(refs, models, loras, params, script=script_text)
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
            # Recorded on the render itself when it planned its own script, because the planner is a paid call
            # made here in the gateway and used to be billed invisibly inside ComfyUI.
            planner_model=planner_model,
            planner_usage=planner_usage,
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
        if job["kind"] == "plan":
            # A plan has no graph to resubmit: it is an LLM call made here, so retrying means running it again.
            request = PlanRequest(**job["request"])
            refs = [graphs.Ref(**ref) for ref in job["refs"]]
            job.update(status="planning", error=None, resumable=False, script=None, started_at=time.time(),
                       updated_at=time.time())
            self.store.save_job(job)
            task = asyncio.create_task(self._run_plan(job["id"], request, refs))
            self._plan_tasks.add(task)
            task.add_done_callback(self._plan_tasks.discard)
            return self.store.get_job(job["id"]) or job
        # Only a job recorded before planning moved into the gateway still has a planner node in its graph.
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
            # The planner is a paid LLM call the gateway makes, so what it cost belongs in the job. It used to
            # happen inside ComfyUI, where nothing counted it.
            "planner_model": job.get("planner_model") or None,
            "usage": job.get("usage") or job.get("planner_usage") or None,
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
