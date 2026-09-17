"""Asset library helpers: browsing a mounted Google Drive and bulk imports.

On Colab, Google Drive is mounted into the runtime (``drive.mount``), so the API reads
the files directly and copies them into ComfyUI's input folder: no tunnel, no upload
size limit. Imports run in the background and report progress.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import shutil
import time
import uuid

from .jobs import HawkService, RequestError

MEDIA_KINDS = ("image", "audio", "video")
_EXTRA_TYPES = {".webp": "image/webp", ".m4a": "audio/mp4", ".opus": "audio/ogg", ".mkv": "video/x-matroska", ".heic": "image/heic"}


def media_kind(filename: str) -> str | None:
    extension = os.path.splitext(filename)[1].lower()
    guess = _EXTRA_TYPES.get(extension) or mimetypes.guess_type(filename)[0] or ""
    kind = guess.split("/")[0]
    return kind if kind in MEDIA_KINDS else None


class DriveBrowser:
    def __init__(self, root: str):
        self.root = os.path.realpath(root) if root else ""

    @property
    def available(self) -> bool:
        return bool(self.root) and os.path.isdir(self.root)

    def resolve(self, relative: str) -> str:
        if not self.available:
            raise RequestError(
                "Google Drive is not mounted on the server. In the Colab notebook run: "
                "from google.colab import drive; drive.mount('/content/drive')"
            )
        path = os.path.realpath(os.path.join(self.root, (relative or "").lstrip("/")))
        if path != self.root and not path.startswith(self.root + os.sep):
            raise RequestError("That path is outside your Drive.")
        if not os.path.exists(path):
            raise RequestError(f"{relative!r} does not exist in your Drive.")
        return path

    def relative(self, path: str) -> str:
        return os.path.relpath(path, self.root).replace(os.sep, "/") if path != self.root else ""

    def browse(self, relative: str = "") -> dict:
        path = self.resolve(relative)
        if not os.path.isdir(path):
            raise RequestError(f"{relative!r} is a file, not a folder.")
        folders, files = [], []
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
        except OSError as exc:
            raise RequestError(f"Could not read {relative or 'My Drive'}: {exc}") from None
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                folders.append({"name": entry.name, "path": self.relative(entry.path)})
            else:
                kind = media_kind(entry.name)
                if kind:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = None
                    files.append({"name": entry.name, "path": self.relative(entry.path), "kind": kind, "size": size})
        rel = self.relative(path)
        return {
            "available": True,
            "path": rel,
            "parent": None if not rel else self.relative(os.path.dirname(path)),
            "folders": folders,
            "files": files,
        }

    def collect(self, relatives: list[str], recursive: bool, limit: int) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for relative in relatives:
            path = self.resolve(relative)
            if os.path.isfile(path):
                candidates = [path]
            else:
                candidates = []
                for folder, dirs, names in os.walk(path):
                    dirs[:] = sorted(d for d in dirs if not d.startswith(".")) if recursive else []
                    candidates.extend(os.path.join(folder, name) for name in sorted(names) if not name.startswith("."))
            for candidate in candidates:
                if candidate not in seen and media_kind(candidate):
                    seen.add(candidate)
                    found.append(candidate)
                    if len(found) > limit:
                        raise RequestError(f"More than {limit} media files selected; import a smaller folder.")
        return found


class ImportManager:
    """Background Drive imports with progress, kept in memory for the session."""

    def __init__(self, service: HawkService, browser: DriveBrowser):
        self.service = service
        self.browser = browser
        self.imports: dict[str, dict] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def start(self, paths: list[str], *, recursive: bool = True, collection: str | None = None, tags: list[str] | None = None) -> dict:
        if not paths:
            raise RequestError("Choose at least one Drive file or folder.")
        files = await asyncio.to_thread(self.browser.collect, paths, recursive, self.service.settings.max_import_files)
        if not files:
            raise RequestError("No images, audio or video files found there.")
        if not (collection or "").strip():
            first = self.browser.resolve(paths[0])
            collection = os.path.basename(first if os.path.isdir(first) else os.path.dirname(first)) or "Google Drive"
        state = {
            "id": uuid.uuid4().hex[:12],
            "status": "running",
            "collection": collection.strip(),
            "tags": tags or [],
            "total": len(files),
            "done": 0,
            "imported": 0,
            "duplicates": 0,
            "failed": 0,
            "asset_ids": [],
            "errors": [],
            "created_at": time.time(),
            "finished_at": None,
        }
        self.imports[state["id"]] = state
        self._tasks[state["id"]] = asyncio.create_task(self._run(state, files))
        return state

    async def _run(self, state: dict, files: list[str]) -> None:
        try:
            for path in files:
                relative = self.browser.relative(path)
                try:
                    with open(path, "rb") as handle:
                        asset = await self.service.add_asset(
                            os.path.basename(path), handle, None, os.path.getsize(path),
                            collection=state["collection"], tags=state["tags"],
                            source={"type": "drive", "path": relative}, local_path=path,
                        )
                    if asset.get("duplicate"):
                        state["duplicates"] += 1
                    else:
                        state["imported"] += 1
                    state["asset_ids"].append(asset["id"])
                except Exception as exc:  # one bad file never stops the import
                    state["failed"] += 1
                    if len(state["errors"]) < 50:
                        state["errors"].append(f"{relative}: {exc}")
                state["done"] += 1
            state["status"] = "done"
        except asyncio.CancelledError:
            state["status"] = "cancelled"
            raise
        finally:
            state["finished_at"] = time.time()
            self._tasks.pop(state["id"], None)

    def get(self, import_id: str) -> dict:
        state = self.imports.get(import_id)
        if state is None:
            from .jobs import NotFound

            raise NotFound(f"No import {import_id!r} (imports are forgotten when the API restarts).")
        return state

    def recent(self) -> list[dict]:
        return sorted(self.imports.values(), key=lambda s: s["created_at"], reverse=True)[:20]

    async def wait(self, import_id: str, seconds: float) -> dict:
        deadline = time.monotonic() + seconds
        while self.get(import_id)["status"] == "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        return self.get(import_id)

    async def stop(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)


# --------------------------------------------------------------------------- export

#: Colab's Drive mount exposes each file's Drive id as an extended attribute.
DRIVE_ID_ATTRS = ("user.drive.id", "user.drive.item_id", "user.drive.file_id")
DRIVE_ID_WAIT_SECONDS = 1800.0  # big videos can take a while to upload from the mount
DRIVE_ID_POLL_SECONDS = 10.0
EXPORT_DEFAULTS = {"enabled": True, "folder": "Hawk H3/Videos", "segments": False}


def drive_file_id(path: str) -> str | None:
    """The Drive file id of a file in the mounted Drive, once Drive has synced it."""
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:  # not Linux
        return None
    names = list(DRIVE_ID_ATTRS)
    try:
        names += [name for name in os.listxattr(path) if "id" in name.lower() and name not in names]
    except OSError:
        pass
    for name in names:
        try:
            value = getxattr(path, name).decode("utf-8", "ignore").strip()
        except OSError:
            continue
        # Drive's mount says "local-<n>" until the upload finishes; only a real id opens in Drive.
        if value and not value.lower().startswith("local"):
            return value
    return None


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:50] or "video"


class DriveExporter:
    """Copies finished renders into the mounted Google Drive so they play and download from
    Google's servers instead of through the tunnel."""

    def __init__(self, service: HawkService, browser: DriveBrowser):
        self.service = service
        self.browser = browser
        self.path = os.path.join(service.settings.data_dir, "drive_export.json")
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    def settings(self) -> dict:
        data = dict(EXPORT_DEFAULTS)
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            data.update({k: stored[k] for k in EXPORT_DEFAULTS if k in stored})
        except (OSError, ValueError):
            pass
        return {**data, "mounted": self.browser.available, "root": self.browser.root}

    def save_settings(self, *, enabled: bool | None = None, folder: str | None = None, segments: bool | None = None) -> dict:
        current = {k: v for k, v in self.settings().items() if k in EXPORT_DEFAULTS}
        if enabled is not None:
            current["enabled"] = bool(enabled)
        if folder is not None:
            clean = "/".join(part for part in folder.replace("\\", "/").split("/") if part and part not in (".", ".."))
            if not clean:
                raise RequestError("Give a Drive folder, e.g. Hawk H3/Videos.")
            current["folder"] = clean
        if segments is not None:
            current["segments"] = bool(segments)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2)
        return self.settings()

    def schedule(self, job_id: str, force: bool = False) -> bool:
        config = self.settings()
        if not force and not config["enabled"]:
            return False
        if not self.browser.available:
            if force:
                raise RequestError("Google Drive is not mounted on the server. Run drive.mount('/content/drive') in the notebook.")
            return False
        task = asyncio.create_task(self.export(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    def _set(self, job_id: str, **fields) -> None:
        job = self.service.store.get_job(job_id)
        if job is None:
            return
        drive = dict(job.get("drive") or {})
        drive.update(fields, updated_at=time.time())
        job["drive"] = drive
        self.service.store.save_job(job)

    async def export(self, job_id: str) -> None:
        try:
            async with self._lock:  # one copy at a time; waiting for Drive's id happens outside
                target = await self._export_files(job_id)
            self._set(job_id, status="syncing")
            deadline = time.monotonic() + DRIVE_ID_WAIT_SECONDS
            while time.monotonic() < deadline:
                file_id = await asyncio.to_thread(drive_file_id, target)
                if file_id and not file_id.lower().startswith("local"):
                    self._set(job_id, status="ready", file_id=file_id)
                    return
                await asyncio.sleep(DRIVE_ID_POLL_SECONDS)
            self._set(job_id, status="copied")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._set(job_id, status="failed", error=str(exc)[:300])

    async def _export_files(self, job_id: str) -> str:
        job = self.service.get_job(job_id)
        filename, subfolder, type_ = self.service.video_location(job_id)
        config = self.settings()
        from .jobs import job_title

        day = time.strftime("%Y-%m-%d", time.localtime(job["created_at"]))
        folder = f"{config['folder']}/{day}"
        name = f"{_slug(job_title(job))}_{job_id[:8]}.mp4"
        target_dir = os.path.join(self.browser.root, folder)
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, name)
        self._set(job_id, status="copying", path=f"{folder}/{name}", folder=folder, file_id=None, error=None)
        await self._copy(filename, subfolder, type_, target)
        if config["segments"]:
            segments_dir = os.path.join(target_dir, f"{name[:-4]}_segments")
            os.makedirs(segments_dir, exist_ok=True)
            for number in range(1, (job.get("segments_done") or 0) + 1):
                seg_name, seg_folder, seg_type = self.service.segment_location(job_id, number)
                await self._copy(seg_name, seg_folder, seg_type, os.path.join(segments_dir, seg_name))
        return target

    async def _copy(self, filename: str, subfolder: str, type_: str, target: str) -> None:
        local = self.service.local_output(filename, subfolder, type_)
        if local:
            await asyncio.to_thread(shutil.copyfile, local, target)
            return
        _, _, body = await self.service._view(filename, subfolder, type_)
        with open(target, "wb") as handle:
            async for chunk in body:
                await asyncio.to_thread(handle.write, chunk)

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
