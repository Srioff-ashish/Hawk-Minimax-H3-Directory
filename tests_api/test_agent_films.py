"""make_film as the chat agent calls it: one base model for the whole film and look shots by default."""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from hawk_api import mcp_server, storyboard
from hawk_api.schemas import RenderSettings

POD_UNET = "10Eros_Max_h3_8stepTURBO_beta7.safetensors"


def fake_service(unet: str = POD_UNET):
    models = SimpleNamespace(unet_name=unet)
    return SimpleNamespace(settings=SimpleNamespace(models=models),
                           render_models=SimpleNamespace(resolve=lambda defaults: defaults))


class MakeFilmDefaults(unittest.IsolatedAsyncioTestCase):
    async def call(self, film: dict, unet: str = POD_UNET) -> dict:
        captured = {}

        async def fake_make_film(boards, *, settings=None, **fields):
            captured.update(fields, settings=settings)
            return {"id": "board1"}

        with mock.patch.object(storyboard, "make_film", fake_make_film):
            mcp = mcp_server.build_mcp(fake_service(unet), boards=object())
            await mcp.call_tool("make_film", {"film": film})
        return captured

    async def test_written_shots_use_the_pod_model_and_look_mode(self):
        got = await self.call({"idea": "a beach drive", "identity_asset_ids": ["face1"], "shot_count": 4})
        self.assertEqual(got["settings"].unet_name, POD_UNET)
        self.assertEqual(got["settings"].continuity, "off")
        self.assertEqual(got["modes"], ["look"] * 4)
        self.assertNotIn("megapixels", got["settings"].model_dump(exclude_unset=True),
                         "the quality preset must still choose the size")

    async def test_caller_choices_win(self):
        got = await self.call({"idea": "x", "identity_asset_ids": ["face1"], "shot_count": 2,
                               "modes": ["frame", "look"],
                               "settings": {"unet_name": "10Eros_Max_h3_TURBO-hybrid_beta5.safetensors"}})
        self.assertEqual(got["settings"].unet_name, "10Eros_Max_h3_TURBO-hybrid_beta5.safetensors")
        self.assertEqual(got["modes"], ["frame", "look"])

    async def test_own_panels_keep_their_modes(self):
        got = await self.call({"identity_asset_ids": ["face1"], "panel_asset_ids": ["p1", "p2"]})
        self.assertIsNone(got["modes"])
        self.assertEqual(got["settings"].unet_name, POD_UNET)

    async def test_no_pod_model_leaves_settings_alone(self):
        got = await self.call({"idea": "x", "identity_asset_ids": ["face1"]}, unet="")
        self.assertIsNone(got["settings"])


class FollowSettings(unittest.TestCase):
    def test_only_given_fields_are_stored(self):
        stored = RenderSettings(unet_name=POD_UNET, continuity="off").model_dump(exclude_unset=True)
        rebuilt = RenderSettings(**json.loads(json.dumps(stored)))
        self.assertNotIn("megapixels", rebuilt.model_fields_set)


if __name__ == "__main__":
    unittest.main()
