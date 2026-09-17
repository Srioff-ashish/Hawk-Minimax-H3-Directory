"""Asset library helpers: browsing a mounted Google Drive and bulk imports.

On Colab, Google Drive is mounted into the runtime (``drive.mount``), so the API reads
the files directly and copies them into ComfyUI's input folder: no tunnel, no upload
size limit. Imports run in the background and report progress.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
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
