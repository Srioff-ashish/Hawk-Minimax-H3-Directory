"""The planner hears voice and music references when a model that can hear is available, and every other model gets
the plan request it always got, with the audio "not attached"."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_gateway  # noqa: E402  (not `from ... import Gateway`: unittest would run its tests here too)

LISTENER = {"id": "google/gemini-3-flash-preview", "name": "Gemini 3 Flash", "input_modalities": ["text", "image", "audio"],
            "output_modalities": ["text"], "pricing": {"prompt": "0.0000005", "completion": "0.000003"}}


def _audio_parts(request: dict) -> list[dict]:
    content = request["messages"][-1]["content"]
    return [part for part in content if isinstance(part, dict) and part.get("type") == "input_audio"] \
        if isinstance(content, list) else []


def _text(request: dict) -> str:
    content = request["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if part.get("type") == "text")


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class PlannerHearsAudio(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_gateway.Gateway.asyncSetUp
    asyncTearDown = test_gateway.Gateway.asyncTearDown
    wait = test_gateway.Gateway.wait

    async def _voice(self) -> str:
        work = tempfile.mkdtemp()
        try:
            path = os.path.join(work, "voice.mp3")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=4",
                            "-c:a", "libmp3lame", path], check=True)
            data = open(path, "rb").read()
        finally:
            shutil.rmtree(work, ignore_errors=True)
        response = await self.http.post("/v1/assets", files={"files": ("voice.mp3", data, "audio/mpeg")})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["assets"][0]["id"]

    async def _plan(self, references) -> dict:
        plan = (await self.http.post("/v1/plans", json={"story": "She sings on a rooftop", "references": references,
                                                         "segment_count": 2})).json()
        return await self.wait(plan["id"])

    async def test_a_model_that_can_hear_gets_the_clip_first(self):
        self.fake.llm_models_data.append(LISTENER)
        voice = await self._voice()
        plan = await self._plan([{"asset_id": voice, "role": "audio", "label": "her voice"}])
        self.assertEqual(plan["status"], "done", plan)
        first = self.fake.chat_requests[0]
        self.assertEqual(first["model"], LISTENER["id"], "the listener is added ahead of a chain that cannot hear")
        parts = _audio_parts(first)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["input_audio"]["format"], "mp3")
        self.assertTrue(parts[0]["input_audio"]["data"])
        self.assertIn("<Audio 1> = audio clip, attached as sound 1", _text(first))
        self.assertNotIn("(not attached)", _text(first))

    async def test_when_the_listener_fails_the_chain_plans_without_the_clip(self):
        self.fake.llm_models_data.append(LISTENER)
        self.fake.chat_status_by_model[LISTENER["id"]] = 403  # e.g. it refuses the brief
        voice = await self._voice()
        plan = await self._plan([{"asset_id": voice, "role": "audio"}])
        self.assertEqual(plan["status"], "done", plan)
        second = self.fake.chat_requests[1]
        self.assertNotEqual(second["model"], LISTENER["id"])
        self.assertEqual(_audio_parts(second), [], "a model that cannot hear is never sent sound")
        self.assertIn("<Audio 1> = audio clip (not attached)", _text(second))

    async def test_without_a_listener_nothing_changes(self):
        voice = await self._voice()
        plan = await self._plan([{"asset_id": voice, "role": "audio"}])
        self.assertEqual(plan["status"], "done", plan)
        self.assertTrue(all(not _audio_parts(r) for r in self.fake.chat_requests))
        self.assertIn("(not attached)", _text(self.fake.chat_requests[0]))


if __name__ == "__main__":
    unittest.main()
