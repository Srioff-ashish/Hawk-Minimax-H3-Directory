"""What the user wrote survives to the model: roster descriptions over the catalogue's guess, a scene's camera,
and Qwen-only pictures that still get made on a pod without Qwen Image 2.1."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api import manhwa  # noqa: E402
from hawk_api.story import DRAFT_PROMPT, clean_scene  # noqa: E402


class RosterWins(unittest.TestCase):
    roster = {"Mina": {"appearance": "short black bob, red coat", "user_edited": True},
              "Jun": {"appearance": "catalogue guess for Jun"}}

    def test_the_users_description_beats_the_tile(self):
        self.assertEqual(manhwa.look_for({"name": "Mina", "appearance": "long brown hair"}, self.roster),
                         "short black bob, red coat")

    def test_without_a_user_edit_the_tile_still_leads(self):
        self.assertEqual(manhwa.look_for({"name": "Jun", "appearance": "tile says blue shirt"}, self.roster),
                         "tile says blue shirt")
        self.assertEqual(manhwa.look_for({"name": "Jun", "appearance": ""}, self.roster), "catalogue guess for Jun")

    def test_photo_prompt_and_film_frame_use_it(self):
        tile = {"status": "ok", "scene": "she waves", "characters": [{"name": "Mina", "appearance": "long brown hair"}]}
        self.assertIn("short black bob", manhwa.real_prompt(tile, roster=self.roster))
        self.assertNotIn("long brown hair", manhwa.real_prompt(tile, roster=self.roster))
        frame = manhwa.film_frame(tile, self.roster)
        self.assertIn("short black bob", str(frame))


class Camera(unittest.TestCase):
    def test_a_scene_keeps_its_camera(self):
        scene = clean_scene({"description": "She turns.", "setting": "rooftop", "camera": "low-angle close-up"})
        self.assertEqual(scene["camera"], "low-angle close-up")
        self.assertIn('"camera"', DRAFT_PROMPT)


class QwenFallback(unittest.IsolatedAsyncioTestCase):
    async def test_without_qwen_the_ladder_makes_it_and_says_so(self):
        calls = []

        class Local:
            async def status(self, engine):
                return {"installed": False}

        class Service:
            local_images = Local()

            async def generate_images(self, prompt, **kwargs):
                calls.append(kwargs)
                return {"assets": [{"id": "x"}]}

        manager = manhwa.ManhwaManager.__new__(manhwa.ManhwaManager)
        manager.service = Service()
        manager.busy_wait = 0
        result = await manager._convert_panel("panel", "a photo", [{"name": "Anime2Real.safetensors", "strength": 0.8}])
        self.assertEqual(calls[0]["engine"], "auto")
        self.assertNotIn("loras", calls[0], "Qwen-only LoRAs are not sent to another engine")
        self.assertIn("not installed", result["warnings"][0])

    async def test_with_qwen_nothing_changes(self):
        calls = []

        class Local:
            async def status(self, engine):
                return {"installed": True, "edit": {"installed": True}}

        class Service:
            local_images = Local()

            async def generate_images(self, prompt, **kwargs):
                calls.append(kwargs)
                return {"assets": []}

        manager = manhwa.ManhwaManager.__new__(manhwa.ManhwaManager)
        manager.service = Service()
        manager.busy_wait = 0
        await manager._convert_panel("panel", "a photo", [{"name": "Anime2Real.safetensors", "strength": 0.8}])
        self.assertEqual(calls[0]["engine"], "qwen21")
        self.assertEqual(calls[0]["loras"][0]["name"], "Anime2Real.safetensors")


if __name__ == "__main__":
    unittest.main()
