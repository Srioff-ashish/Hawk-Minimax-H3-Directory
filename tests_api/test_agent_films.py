"""make_film as the chat agent calls it: one base model for the whole film and look shots by default."""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from mcp.server.mcpserver.exceptions import ToolError

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


class MakeVideo(unittest.IsolatedAsyncioTestCase):
    """The direct route: one render from the reference pictures, no storyboard."""

    async def call(self, args: dict, unet: str = POD_UNET):
        made = {}
        service = fake_service(unet)

        async def create_video(request):
            made["request"] = request
            return {"id": "job1"}

        service.create_video = create_video
        service.job_view = lambda job: job
        mcp = mcp_server.build_mcp(service, boards=None)
        result = await mcp.call_tool("make_video", args)
        return made.get("request"), result

    async def test_it_plans_and_renders_in_one_job(self):
        request, _ = await self.call({"identity_asset_ids": ["face", "body"], "character": "Rebel in a pink skirt.",
                                      "shots": ["She walks into the sea.", "She looks back."],
                                      "look_asset_ids": ["look"], "music_asset_id": "track"})
        self.assertIsNone(request.script, "the planner writes the segments, not a template")
        self.assertEqual(request.story.segment_count, 2)
        self.assertIn("1. She walks into the sea.", request.story.story)
        self.assertIn("2. She looks back.", request.story.story)
        self.assertIn("<Picture 3>", request.story.story, "each shot names its look reference")
        self.assertEqual([r.asset_id for r in request.references], ["face", "body", "look"])
        self.assertIn("shots 1, 2", request.references[2].label)
        self.assertEqual(request.settings.unet_name, POD_UNET)
        self.assertEqual(request.settings.megapixels, 0.6)
        self.assertEqual(request.settings.music_asset_id, "track")
        self.assertEqual(request.settings.continuity, "tail_22", "a scene flows from segment to segment by default")

    async def test_a_look_per_shot_cuts_and_no_looks(self):
        request, _ = await self.call({"identity_asset_ids": ["face"], "character": "Rebel.", "continuity": "off",
                                      "shots": ["a", "b", "c"], "look_asset_ids": ["l1", "l2", "l1"]})
        self.assertEqual([r.asset_id for r in request.references], ["face", "l1", "l2"])
        self.assertIn("shots 1, 3", request.references[1].label)
        self.assertEqual(request.settings.continuity, "off")
        self.assertIn("hard cuts", request.story.story)
        plain, _ = await self.call({"identity_asset_ids": ["face"], "character": "Rebel.", "shots": ["a"]})
        self.assertNotIn("look reference", plain.story.story)

    async def test_bad_calls_are_refused(self):
        for args in ({"identity_asset_ids": [], "character": "x", "shots": ["a"]},
                     {"identity_asset_ids": ["f"], "character": "x", "shots": []},
                     {"identity_asset_ids": ["f"], "character": "x", "shots": ["a", "b", "c"], "look_asset_ids": ["l1", "l2"]},
                     {"identity_asset_ids": ["f1", "f2"], "character": "x", "shots": [str(i) for i in range(8)],
                      "look_asset_ids": [f"l{i}" for i in range(8)]}):
            with self.assertRaises(ToolError, msg=args):
                await self.call(args)


class GetJobKnowsStoryboards(unittest.IsolatedAsyncioTestCase):
    async def test_a_storyboard_id_says_what_it_is(self):
        from hawk_api.jobs import NotFound

        def get_job(job_id):
            raise NotFound(f"No job {job_id!r}.")

        service = fake_service()
        service.get_job = get_job
        service.job_view = lambda job: job
        service.store = SimpleNamespace(get_asset=lambda asset_id: {"id": asset_id} if asset_id == "film1" else None)
        service.asset_view = lambda asset: {"file_url": f"https://pod/{asset['id']}.mp4"}
        boards = SimpleNamespace(board=lambda board_id: {"id": board_id, "status": "done",
                                                         "render": {"film_asset_id": "film1", "jobs": ["j1"]}})
        mcp = mcp_server.build_mcp(service, boards=boards)
        result = await mcp.call_tool("get_job", {"job_id": "board1"})
        content = getattr(result, "structured_content", None)
        if not isinstance(content, dict):
            content = json.loads(result.content[0].text)
        self.assertEqual(content["kind"], "storyboard")
        self.assertEqual(content["film_url"], "https://pod/film1.mp4")


class FollowSettings(unittest.TestCase):
    def test_only_given_fields_are_stored(self):
        stored = RenderSettings(unet_name=POD_UNET, continuity="off").model_dump(exclude_unset=True)
        rebuilt = RenderSettings(**json.loads(json.dumps(stored)))
        self.assertNotIn("megapixels", rebuilt.model_fields_set)


if __name__ == "__main__":
    unittest.main()
