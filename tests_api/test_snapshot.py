"""The database copy that survives a Colab runtime ending (hawk_api/snapshot.py).

    python -m unittest tests_api.test_snapshot
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hawk_api.snapshot import DbSnapshotter, PREV_SUFFIX, SNAPSHOT_NAME  # noqa: E402


class Snapshot(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="hawk_data_")
        self.drive = tempfile.mkdtemp(prefix="hawk_drive_")
        self.addCleanup(shutil.rmtree, self.data, True)
        self.addCleanup(shutil.rmtree, self.drive, True)
        self.db = os.path.join(self.data, SNAPSHOT_NAME)
        # Two connections on one file, the way jobs.Store and agent.AgentStore both open db_path.
        self.jobs = sqlite3.connect(self.db, check_same_thread=False)
        self.jobs.executescript("CREATE TABLE assets(id TEXT PRIMARY KEY);")
        self.chats = sqlite3.connect(self.db, check_same_thread=False)
        self.chats.executescript("CREATE TABLE agent_sessions(id TEXT PRIMARY KEY);")
        self.addCleanup(self.jobs.close)
        self.addCleanup(self.chats.close)

        drive, db = self.drive, self.db
        browser = type("Browser", (), {"root": drive, "available": True})()
        service = type("Service", (), {"settings": type("Settings", (), {"db_path": db})()})()
        exporter = type("Exporter", (), {
            "browser": browser, "service": service,
            "settings": lambda _self: {"snapshots": True, "snapshot_folder": "Backups", "snapshot_minutes": 10},
        })()
        self.snapshots = DbSnapshotter(exporter)
        self.folder = os.path.join(self.drive, "Backups")

    def add_chat(self, chat_id):
        self.chats.execute("INSERT INTO agent_sessions VALUES(?)", (chat_id,))
        self.chats.commit()
        os.utime(self.db, None)

    def chats_in(self, path):
        db = sqlite3.connect(path)
        try:
            return sorted(row[0] for row in db.execute("SELECT id FROM agent_sessions"))
        finally:
            db.close()

    async def test_a_live_database_copies_across_whole(self):
        self.add_chat("s1")
        self.jobs.execute("INSERT INTO assets VALUES('asset_1')")
        self.jobs.commit()
        self.assertIsNotNone(await self.snapshots.snapshot(force=True))
        copy = os.path.join(self.folder, SNAPSHOT_NAME)
        self.assertEqual(self.chats_in(copy), ["s1"])
        db = sqlite3.connect(copy)
        self.assertEqual([r[0] for r in db.execute("SELECT id FROM assets")], ["asset_1"],
                         "both stores share the one file, so both their tables have to come across")
        db.close()

    async def test_an_idle_session_writes_nothing(self):
        self.add_chat("s1")
        await self.snapshots.snapshot(force=True)
        self.assertIsNone(await self.snapshots.snapshot(),
                          "nothing was written, so Drive already has it")

    async def test_the_copy_before_this_one_is_kept(self):
        self.add_chat("s1")
        await self.snapshots.snapshot(force=True)
        self.add_chat("s2")
        await self.snapshots.snapshot()
        self.assertEqual(sorted(os.listdir(self.folder)), [SNAPSHOT_NAME, SNAPSHOT_NAME + PREV_SUFFIX],
                         "two files, and no half-written .part left behind")
        self.assertEqual(self.chats_in(os.path.join(self.folder, SNAPSHOT_NAME)), ["s1", "s2"])
        self.assertEqual(self.chats_in(os.path.join(self.folder, SNAPSHOT_NAME + PREV_SUFFIX)), ["s1"],
                         "the older copy is what makes a bad one recoverable")

    async def test_snapshots_never_grow_past_two_files(self):
        for n in range(5):
            self.add_chat(f"s{n}")
            await self.snapshots.snapshot()
        self.assertEqual(len(os.listdir(self.folder)), 2, "five copies still leave two files")

    async def test_an_unmounted_drive_is_simply_skipped(self):
        self.snapshots.browser = type("Browser", (), {"root": self.drive, "available": False})()
        self.assertIsNone(await self.snapshots.snapshot(force=True))


if __name__ == "__main__":
    unittest.main()
