"""Shots as units (insert, delete and move keep every panel on its own shot), the planner told the whole frame, and
a second Make film reusing every part that did not change."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api import storyboard as sb  # noqa: E402
from hawk_api.characters import CharacterManager  # noqa: E402
from test_film_engine import B5, B7, EngineService, settle  # noqa: E402
from test_storyboard import png, sheet_png  # noqa: E402


class Shots(unittest.TestCase):
    def setUp(self):
        self.service = EngineService(png(10, 10))
        self.boards = sb.StoryboardManager(self.service)
        board = self.boards.create(title="T", identity_asset_ids=["face"], shots=["one", "two", "three"],
                                   motions=["m1", "m2", "m3"], modes=["frame", "look", "frame"], draw=False)
        board = self.boards.board(board["id"])
        board.update(panels=["p1", "p2", "p3"], dialogue=[["a"], ["b"], ["c"]],
                     checks={"1": {"ok": False, "issues": "bad", "dropped": True, "mode_was": "look"}})
        self.boards.store.save(board)
        self.id = board["id"]

    def rows(self):
        board = self.boards.board(self.id)
        return [(s.split("Action: ")[-1] if "Action: " in s else s, p, m, d[0] if d else "")
                for s, p, m, d in zip(board["shots"], board["panels"], board["modes"], board["dialogue"])]

    def test_insert_keeps_every_panel_on_its_shot(self):
        self.boards.insert_shot(self.id, 1, "new", motion="mn", lines=["n"])
        rows = self.rows()
        self.assertEqual([r[1] for r in rows], ["p1", None, "p2", "p3"])
        self.assertEqual([r[3] for r in rows], ["a", "n", "b", "c"])
        self.assertEqual(list(self.boards.board(self.id)["checks"]), ["2"], "the check followed its shot")

    def test_delete_and_move(self):
        self.boards.delete_shot(self.id, 0)
        self.assertEqual([r[1] for r in self.rows()], ["p2", "p3"])
        self.assertEqual(list(self.boards.board(self.id)["checks"]), ["0"])
        self.boards.move_shot(self.id, 0, 1)
        self.assertEqual([r[1] for r in self.rows()], ["p3", "p2"])
        self.assertEqual(list(self.boards.board(self.id)["checks"]), ["1"])

    def test_changing_a_drawn_shots_text_says_so(self):
        board = self.boards.board(self.id)
        shots = list(board["shots"])
        shots[1] = "something else entirely"
        self.boards.update(self.id, shots=shots)
        self.assertIn("Shot 2 changed after the panel was drawn", self.boards.board(self.id)["note"])

    def test_dialogue_can_be_edited(self):
        self.boards.update(self.id, dialogue=[["Rebel: hi"], [], ["  "]])
        self.assertEqual(self.boards.board(self.id)["dialogue"], [["Rebel: hi"], [], []])

    def test_the_planner_hears_the_frame_not_only_the_motion(self):
        board = self.boards.board(self.id)
        beat = self.boards._beat(board, 0)
        self.assertTrue(beat.startswith("m1. In frame:"), beat)
        self.assertIn("one", beat)


class Reuse(unittest.IsolatedAsyncioTestCase):
    async def test_a_second_film_reuses_unchanged_parts(self):
        service = EngineService(sheet_png(4, 1, 2048, 1152, border=40))
        boards = sb.StoryboardManager(service)
        boards.poll = 0.01
        boards.characters = CharacterManager(service)

        async def fake_join(board_id):
            boards._set(board_id, status="done")

        boards._join = fake_join
        original_wait = boards._wait

        async def done_wait(job_id):  # the fake renders finish at once
            job = service.store.get_job(job_id) or {}
            job["status"] = "done"
            service.store.jobs[job_id] = job if hasattr(service.store, "jobs") else job
            return job

        boards._wait = done_wait
        service.store.get_job = getattr(service.store, "get_job", lambda job_id: {"id": job_id, "status": "done"})
        board = boards.create(title="Club", identity_asset_ids=["face"], shots=["arrives", "poses", "dances", "waves"],
                              modes=["frame", "frame", "look", "frame"], check=False,
                              then_render={"segment_seconds": 5})
        board = await settle(boards, board["id"])
        self.assertEqual(board["status"], "done", board.get("error"))
        first = len(service.videos)
        self.assertEqual(first, 3)
        # nothing changed: Make film again pays for nothing
        original_get = service.store.get_job
        service.store.get_job = lambda job_id: {"id": job_id, "status": "done"}
        boards.start_render(board["id"], segment_seconds=5)
        board = await settle(boards, board["id"])
        self.assertEqual(len(service.videos), first, "every part reused")
        # one shot's motion changes: only its part renders again
        board = boards.board(board["id"])
        motions = list(board.get("motions") or ["", "", "", ""])
        motions += [""] * (4 - len(motions))
        motions[2] = "she spins faster"
        boards.update(board["id"], motions=motions)
        boards.start_render(board["id"], segment_seconds=5)
        board = await settle(boards, board["id"])
        self.assertEqual(len(service.videos), first + 1, "only the changed part")
        service.store.get_job = original_get
        boards._wait = original_wait


if __name__ == "__main__":
    unittest.main()
