"""One shared SQLite connection per file, and boards moved from storyboards.json into the database."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hawk_api import db
from hawk_api.storyboard import StoryboardStore


class Shared(unittest.TestCase):
    def test_the_same_file_shares_one_connection_in_wal_mode(self):
        with tempfile.TemporaryDirectory() as data:
            path = os.path.join(data, "jobs.sqlite3")
            a, lock_a = db.connect(path)
            b, lock_b = db.connect(path)
            self.assertIs(a, b)
            self.assertIs(lock_a, lock_b)
            self.assertEqual(a.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
            db.forget(path)

    def test_a_closed_connection_is_reopened(self):
        with tempfile.TemporaryDirectory() as data:
            path = os.path.join(data, "jobs.sqlite3")
            first, _ = db.connect(path)
            first.close()
            second, _ = db.connect(path)
            self.assertIsNot(first, second)
            second.execute("SELECT 1")
            db.forget(path)


class Boards(unittest.TestCase):
    def test_boards_move_from_the_json_file_once_with_a_backup(self):
        with tempfile.TemporaryDirectory() as data:
            old = {"b1": {"id": "b1", "title": "One", "status": "done", "created_at": 1.0, "updated_at": 2.0},
                   "b2": {"id": "b2", "title": "Two", "status": "review", "created_at": 1.0, "updated_at": 3.0}}
            with open(os.path.join(data, "storyboards.json"), "w") as handle:
                json.dump(old, handle)
            store = StoryboardStore(data)
            self.assertEqual([b["id"] for b in store.all()], ["b2", "b1"])
            self.assertEqual(store.get("b1")["title"], "One")
            self.assertFalse(os.path.exists(os.path.join(data, "storyboards.json")))
            self.assertTrue(os.path.exists(os.path.join(data, "storyboards.json.imported")))
            backups = [d for d in os.listdir(data) if d.startswith("backup-")]
            self.assertTrue(os.path.exists(os.path.join(data, backups[0], "storyboards.json")))
            # A second start imports nothing again, and a board saved since is kept.
            store.save({"id": "b1", "title": "Edited", "status": "done"})
            again = StoryboardStore(data)
            self.assertEqual(again.get("b1")["title"], "Edited")
            db.forget(os.path.join(data, "jobs.sqlite3"))

    def test_save_get_delete(self):
        with tempfile.TemporaryDirectory() as data:
            store = StoryboardStore(data)
            store.save({"id": "x", "title": "T", "status": "draft"})
            self.assertEqual(store.get("x")["title"], "T")
            store.delete("x")
            self.assertIsNone(store.get("x"))
            db.forget(os.path.join(data, "jobs.sqlite3"))


if __name__ == "__main__":
    unittest.main()
