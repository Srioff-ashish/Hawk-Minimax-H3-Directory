"""The film engine: characters, panel modes, parts by base model, the check, review, music."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image

    from hawk_api import storyboard as sb
    from hawk_api.characters import CharacterManager
    from hawk_api.jobs import NotFound, RequestError
    from hawk_api.prompts import PromptStore
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"engine dependencies missing: {exc}")

from tests_api.test_storyboard import FakeService, png, sheet_png

B5, B7 = "10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors", "10Eros_Max_h3_8stepTURBO_beta7.safetensors"


class EngineService(FakeService):
    """The storyboard fake, plus what the engine's check, models and music call."""

    def __init__(self, sheet: bytes, verdicts=None):
        super().__init__(sheet)
        self.prompts = PromptStore(tempfile.mkdtemp())
        self.verdicts = list(verdicts or [])
        self.chats, self.media = [], []
        service = self

        class Chat:
            async def chat(self, model, messages, **kwargs):
                service.chats.append(messages)
                reply = service.verdicts.pop(0) if service.verdicts else {"panels": []}
                return json.dumps(reply), {}

        class Atlas:
            async def generate_media(self, payload, **kwargs):
                service.media.append((payload, kwargs))
                return [b"ID3fake-mp3"]

        self.atlas, self.image_atlas = Chat(), Atlas()

    async def model_catalogue(self):
        return []

    async def asset_file(self, asset_id, width=0):
        return png(20, 30), "image/png"

    async def available_models(self, folder, refresh=False):
        return [B7, B5]


async def settle(boards, board_id):
    for _ in range(100):
        task = boards._tasks.get(board_id)
        if task is None:
            return boards.board(board_id)
        await task
        await asyncio.sleep(0)
    raise AssertionError("still running")


class Rules(unittest.TestCase):
    def test_models_by_kind(self):
        self.assertEqual(sb.pick_models([B7, B5, "other.safetensors"]), {"frame": B5, "action": B7})
        self.assertEqual(sb.pick_models([B7]), {"frame": B7, "action": B7}, "one installed: it does both")
        self.assertEqual(sb.pick_models([]), {"frame": "", "action": ""}, "none: the server default")

    def test_parts_follow_the_model_and_the_picture_limit(self):
        modes = ["frame", "frame", "look", "look", "none", "frame"]
        parts = sb.plan_parts(modes, 1, {"frame": B5, "action": B7})
        self.assertEqual([(p["shots"], p["unet"]) for p in parts], [([0, 1], B5), ([2, 3, 4], B7), ([5], B5)])
        same = sb.plan_parts(["frame"] * 10, 1, {"frame": B5, "action": B5})
        self.assertEqual([len(p["shots"]) for p in same], [8, 2], "8 panels + identity = 9 pictures")
        no_panels = sb.plan_parts(["none"] * 11, 3, {"frame": "", "action": ""})
        self.assertEqual(len(no_panels), 1, "shots without panels take no picture slots")

    def test_the_brief_says_what_each_picture_is(self):
        brief = sb.film_brief(["she poses", "a quick beat", "she dances"], "Rebel.", identities=1,
                              modes=["frame", "none", "look"], dialogue=[["Rebel: 'Hello, darling.'"], [], []],
                              music_over=True)
        self.assertIn("<Picture 3> is the identity reference", brief, "two panels, then the identity")
        self.assertIn("Segment 1: <Picture 1> is this segment's first frame", brief)
        self.assertIn("Segment 2: this segment has no picture of its own", brief)
        self.assertIn("Segment 3: <Picture 2> is only a look reference", brief)
        self.assertIn("NOT its first frame", brief)
        self.assertIn("Rebel: 'Hello, darling.'", brief)
        self.assertIn("No music in any segment", brief)

    def test_platform_rules_reach_the_drawing(self):
        self.assertIn("fully and modestly clothed", sb.platform_style("warm light", "instagram"))
        self.assertEqual(sb.platform_style("warm light", "fanvue"), "warm light")


class Characters(unittest.TestCase):
    def setUp(self):
        self.service = EngineService(png(10, 10))
        for n in range(3):
            self.service.store.assets[f"id{n}"] = {"id": f"id{n}", "kind": "image", "filename": f"i{n}.png"}
        self.chars = CharacterManager(self.service)

    def test_saved_once_found_by_name_and_minors_refused(self):
        rebel = self.chars.create("Rebel", "fair-skinned Indian woman, 45, very petite, enormous breasts", ["id0", "id1"],
                                  notes="teardrop breasts with natural sag", platform="fanvue")
        self.assertEqual(self.chars.find("rebel")["id"], rebel["id"])
        with self.assertRaises(RequestError):
            self.chars.create("Rebel", "duplicate")
        with self.assertRaises(RequestError):
            self.chars.create("Kid", "a schoolgirl of 14")
        with self.assertRaises(NotFound):
            self.chars.find("Nobody")

    def test_several_people_resolve_to_one_text_and_one_picture_each_first(self):
        self.chars.create("Rebel", "Indian woman, 45", ["id0", "id1"], platform="fanvue")
        self.chars.create("Blaze", "Korean woman, 45", ["id2"])
        both = self.chars.resolve(["Rebel", "Blaze"])
        self.assertEqual(both["identity_asset_ids"], ["id0", "id2", "id1"], "each person's first picture before extras")
        self.assertIn("Rebel: Indian woman, 45", both["character"])
        self.assertIn("Blaze: Korean woman, 45", both["character"])
        self.assertEqual(both["platform"], "fanvue")


class Engine(unittest.IsolatedAsyncioTestCase):
    def manager(self, service):
        boards = sb.StoryboardManager(service)
        boards.poll = 0.01
        boards.characters = CharacterManager(service)
        return boards

    async def test_a_character_board_checks_redraws_a_rejected_panel_and_films_by_model(self):
        service = EngineService(sheet_png(4, 1, 2048, 1152, border=40), verdicts=[
            {"panels": [{"shot": 1, "ok": True}, {"shot": 2, "ok": False, "issues": "six fingers"}, {"shot": 3, "ok": True}]},
            {"panels": [{"shot": 4, "ok": True}]},
            {"panels": [{"shot": 2, "ok": True}]},
        ])
        boards = self.manager(service)
        boards.characters.create("Rebel", "fair-skinned Indian woman, 45", ["face"], platform="instagram")
        joined = []

        async def fake_join(board_id):
            joined.append(board_id)
            boards._set(board_id, status="done")

        boards._join = fake_join
        board = boards.create(title="Club", characters=["Rebel"], shots=["arrives", "poses", "dances", "waves"],
                              modes=["frame", "frame", "look", "frame"], then_render={"segment_seconds": 5})
        self.assertEqual((board["platform"], board["identity_asset_ids"]), ("instagram", ["face"]))
        board = await settle(boards, board["id"])
        self.assertEqual(board["status"], "done", board.get("error"))
        sheet_prompt = service.image_calls[0][0]
        self.assertIn("fully and modestly clothed", sheet_prompt, "the platform rule is drawn in")
        redo_prompt = service.image_calls[1][0]
        self.assertIn("Fix: six fingers", redo_prompt, "the rejected panel is drawn again with what was wrong")
        self.assertTrue(board["checks"]["1"]["ok"], "and passes the second look")
        self.assertIn("Platform rule:", service.chats[0][1]["content"][0]["text"])
        units = [v.settings.unet_name for v in service.videos]
        self.assertEqual(units, [B5, B7, B5], "frames on beta5, the dance on beta7, in film order")
        self.assertIn("is only a look reference", service.videos[1].story.story)
        self.assertEqual(joined, [board["id"]])

    async def test_review_stops_before_the_film_until_approved(self):
        service = EngineService(sheet_png(2, 1, 2048, 2048, border=40))
        boards = self.manager(service)
        boards._join = lambda board_id: asyncio.sleep(0)
        board = boards.create(title="Ask", character="Rebel.", identity_asset_ids=["face"], shots=["a", "b"],
                              review="ask", then_render={"segment_seconds": 4})
        board = await settle(boards, board["id"])
        self.assertEqual((board["status"], service.videos), ("review", []), "nothing rendered before approval")
        boards.approve(board["id"])
        await settle(boards, board["id"])
        self.assertEqual(len(service.videos), 1)
        with self.assertRaises(sb.StoryboardError):
            boards.approve(board["id"])

    async def test_one_music_track_is_made_once_for_the_whole_film(self):
        service = EngineService(sheet_png(2, 1, 2048, 2048, border=40))
        boards = self.manager(service)
        boards._join = lambda board_id: asyncio.sleep(0)
        board = boards.create(title="Music", character="Rebel.", identity_asset_ids=["face"],
                              shots=[f"s{n}" for n in range(10)], music_prompt="slow sultry R&B, 85 bpm",
                              check=False, then_render={"segment_seconds": 4})
        board = await settle(boards, board["id"])
        self.assertEqual(len(service.media), 1, "one track for two render parts")
        self.assertEqual(service.media[0][0]["model"], "minimax/music-3.0")
        self.assertTrue(service.media[0][0]["is_instrumental"])
        self.assertTrue(board["music"]["asset_id"])
        self.assertTrue(all("No music in any segment" in v.story.story for v in service.videos))


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class MusicMix(unittest.TestCase):
    def test_the_track_is_laid_under_the_whole_film(self):
        work = tempfile.mkdtemp()
        film, music, out = (os.path.join(work, n) for n in ("film.mp4", "music.mp3", "out.mp4"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=3",
                        "-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-shortest", film], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=600:duration=1",
                        music], check=True)
        self.assertTrue(sb.lay_music(film, music, out, 3.0))
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                               capture_output=True, text=True).stdout.strip()
        self.assertAlmostEqual(float(probe), 3.0, delta=0.3, msg="the 1 s track loops under the 3 s film")


if __name__ == "__main__":
    unittest.main()
