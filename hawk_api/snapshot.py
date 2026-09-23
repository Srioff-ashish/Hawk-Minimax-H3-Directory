"""Periodic copies of jobs.sqlite3 into the mounted Google Drive.

A Colab runtime is erased when it ends, and DATA_DIR lives on /content, so every chat, persona, memory and
asset record would go with it. The database is never *run* from Drive: a FUSE mount has no working advisory
locks and no fsync ordering, so SQLite corrupts there. Instead sqlite3's own backup API takes a consistent
copy onto local disk, and only that finished file is moved across.

Two files are kept on Drive -- the live snapshot and one previous generation -- so a copy taken while the
database was in a bad state can never be the only one left.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sqlite3
import time

log = logging.getLogger(__name__)

SNAPSHOT_NAME = "jobs.sqlite3"
PREV_SUFFIX = ".prev"
PART_SUFFIX = ".part"
MIN_MINUTES, MAX_MINUTES = 1.0, 120.0


def snapshot_db(db_path: str, local_tmp: str) -> int:
    """A consistent copy of a live database at local_tmp (local disk). Returns its size in bytes.

    Its own read-only connection: the two live ones (jobs.Store and agent.AgentStore, on this same file)
    are each behind a threading.Lock and each cover only half the tables. A separate connection is what
    SQLite's file locking is for -- both writers commit per statement, so the backup either reads a
    committed state or waits on a writer.
    """
    os.makedirs(os.path.dirname(os.path.abspath(local_tmp)), exist_ok=True)
    if os.path.exists(local_tmp):
        os.unlink(local_tmp)
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(local_tmp)
        try:
            # One step, not a paged loop: SQLite restarts a backup from scratch when a writer intervenes
            # mid-step, so paging a busy database can livelock instead of being gentle.
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    return os.path.getsize(local_tmp)


def _rotate(drive_dir: str, part: str) -> str:
    """Put a finished copy in place, keeping the one before it. Returns the live path."""
    live = os.path.join(drive_dir, SNAPSHOT_NAME)
    if os.path.exists(live):
        os.replace(live, live + PREV_SUFFIX)
    os.replace(part, live)
    return live


class DbSnapshotter:
    """DATA_DIR/jobs.sqlite3 -> <Drive>/<folder>/jobs.sqlite3, every few minutes, only when it changed."""

    def __init__(self, exporter):
        self.exporter = exporter  # owns drive_export.json, and the mounted-Drive check
        self.service = exporter.service
        self.browser = exporter.browser
        self._task: asyncio.Task | None = None
        self._last: tuple[int, int] | None = None  # (st_mtime_ns, st_size) when we last copied
        self.last_at: float = 0.0  # when the last copy finished, so Studio can show it worked
        self.last_path: str = ""
        self.last_size: int = 0
        self.last_error: str = ""

    # ------------------------------------------------------------------ settings

    def settings(self) -> dict:
        stored = self.exporter.settings()
        minutes = float(stored.get("snapshot_minutes") or 10)
        return {"enabled": bool(stored.get("snapshots", True)),
                "folder": str(stored.get("snapshot_folder") or "Hawk H3/Backups"),
                "minutes": max(MIN_MINUTES, min(MAX_MINUTES, minutes))}

    # ------------------------------------------------------------------ the copy

    def _stat(self) -> tuple[int, int] | None:
        try:
            info = os.stat(self.service.settings.db_path)
        except OSError:
            return None
        return (info.st_mtime_ns, info.st_size)

    def _write(self, folder: str) -> tuple[str, int]:
        """Back up locally, then move the finished file into Drive. Runs in a worker thread."""
        db_path = self.service.settings.db_path
        local_tmp = db_path + ".snapshot"
        size = snapshot_db(db_path, local_tmp)
        drive_dir = os.path.join(self.browser.root, folder)
        os.makedirs(drive_dir, exist_ok=True)
        part = os.path.join(drive_dir, SNAPSHOT_NAME + PART_SUFFIX)
        try:
            shutil.copyfile(local_tmp, part)
            _rotate(drive_dir, part)
        finally:
            for leftover in (local_tmp, part):
                if os.path.exists(leftover):
                    os.unlink(leftover)
        return os.path.join(folder, SNAPSHOT_NAME), size

    async def snapshot(self, force: bool = False) -> str | None:
        """Copy the database to Drive. Returns the Drive-relative path, or None when there was nothing to do."""
        config = self.settings()
        if not force and not config["enabled"]:
            return None
        if not self.browser.available:
            return None
        current = self._stat()
        if current is None:
            return None
        if not force and current == self._last:
            return None  # nothing was written since the last copy, so Drive already has it
        try:
            relative, size = await asyncio.to_thread(self._write, config["folder"])
        except Exception as exc:
            self.last_error = str(exc) or type(exc).__name__
            raise
        self._last = current
        self.last_at, self.last_path, self.last_size, self.last_error = time.time(), relative, size, ""
        log.info("hawk_api: snapshot %s (%.1f MB)", relative, size / 1_000_000)
        return relative

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        # Seed from the database as it stands, so a session that is restored and then left alone
        # re-uploads an identical file exactly never.
        self._last = self._stat()
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings()["minutes"] * 60)
            try:
                await self.snapshot()
            except asyncio.CancelledError:
                raise
            except Exception:  # a Drive hiccup must never take the API down with it
                log.exception("hawk_api: snapshot failed")

    def state(self) -> dict:
        """What Studio shows so the user can see the copy is actually happening."""
        config = self.settings()
        return {"snapshots": config["enabled"], "snapshot_folder": config["folder"],
                "snapshot_minutes": config["minutes"], "snapshot_at": self.last_at or None,
                "snapshot_path": self.last_path, "snapshot_size": self.last_size,
                "snapshot_error": self.last_error, "snapshot_ready": bool(self.browser.available)}

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        try:
            await self.snapshot()  # one last copy, now that nothing else is writing
        except Exception:
            log.exception("hawk_api: final snapshot failed")
