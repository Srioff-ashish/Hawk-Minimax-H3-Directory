"""Job store (SQLite) and HawkService -- the one place REST and MCP both call."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import mimetypes
import os
import random
import re
import sqlite3
import tempfile
import threading
import time
import uuid

import httpx

from hawk_h3.script import ScriptError, build_jobs, parse_script

from . import graph as graphs
from .auth import sign_path
from .comfy_client import ComfyClient, ComfyError, ComfyNotFound, ComfyValidationError
from .config import ModelSettings, Settings
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
#: models folder -> (file-name family the Director can use, label for errors)
MODEL_FAMILIES = {"diffusion_models": ("ref2va", "Base model"), "text_encoders": ("qwen3vl", "Text encoder")}


def model_family(folder: str, files: list[str]) -> list[str]:
    family = MODEL_FAMILIES[folder][0]
    return [name for name in files if family in os.path.basename(name).lower()]


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
    def __init__(self, settings: Settings, store: Store | None = None, comfy: ComfyClient | None = None):
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self.comfy = comfy or ComfyClient(settings.comfy_url)
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

    async def available_loras(self, refresh: bool = False) -> list[str]:
        return await self.available_models("loras", refresh)

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
                raise RequestError(str(exc), exc.details) from None

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
        return {
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

    async def add_asset(self, filename: str, fileobj, content_type: str | None = None, size: int | None = None) -> dict:
        kind = asset_kind(filename, content_type)
        asset_id = uuid.uuid4().hex[:12]
        base = os.path.basename(filename.replace("\\", "/"))
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._") or f"file{os.path.splitext(base)[1]}"
        mime = content_type or mimetypes.guess_type(safe)[0] or "application/octet-stream"
        try:
            uploaded = await self.comfy.upload(fileobj, safe, f"hawk_api/{asset_id}", mime)
        except ComfyError as exc:
            raise Unavailable(str(exc)) from exc
        subfolder = uploaded.get("subfolder") or ""
        asset = {
            "id": asset_id,
            "kind": kind,
            "filename": base,
            "path": f"{subfolder}/{uploaded['name']}" if subfolder else uploaded["name"],
            "size": size,
            "created_at": time.time(),
        }
        self.store.add_asset(asset)
        return asset

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
            return await self.add_asset(name, buffer, content_type, size)

    def list_assets(self, limit: int = 100) -> list[dict]:
        return self.store.list_assets(limit)

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
            applied = [pair for node in job["nodes"].get("lora_stacks", []) for pair in job["loras_applied_by_node"].get(node, [])]
            job["loras_applied"] = [{"file": name, "strength": strength} for name, strength in applied]
            if job["nodes"].get("lora_stacks") or job["loras"]:
                job["warnings"].extend(compare_applied(job["loras"], applied))
            if job.get("segments_total"):
                job["segments_done"] = job["segments_total"]
        job["status"] = "done"
        self.store.save_job(job)

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
                video_url=None,
                segment_urls=[],
            )
            if job["status"] == "done" and job["outputs"].get("video"):
                view["video_url"] = base + sign_path(token, f"/v1/jobs/{job['id']}/video", ttl)
            done = job.get("segments_done") or 0
            view["segment_urls"] = [
                base + sign_path(token, f"/v1/jobs/{job['id']}/segments/{number}", ttl) for number in range(1, done + 1)
            ]
        return view

    async def open_video(self, job_id: str):
        job = self.get_job(job_id)
        video = job["outputs"].get("video")
        if job["kind"] != "render" or job["status"] != "done" or not video:
            raise NotFound(f"Job {job_id} has no finished video yet.")
        return await self._view(video["filename"], video["subfolder"], video.get("type", "output"))

    async def open_segment(self, job_id: str, number: int):
        job = self.get_job(job_id)
        if job["kind"] != "render" or number < 1:
            raise NotFound("No such segment.")
        return await self._view(f"segment_{number:03d}.mp4", f"hawk_h3/{job['run_name']}")

    async def _view(self, filename: str, subfolder: str, type_: str = "output"):
        try:
            return await self.comfy.view(filename, subfolder, type_)
        except ComfyNotFound as exc:
            raise NotFound(str(exc)) from None
        except ComfyError as exc:
            raise Unavailable(str(exc)) from None

    def upload_page_link(self) -> str:
        return self.settings.public_base_url + sign_path(self.settings.token, "/upload", self.settings.link_ttl_seconds)
