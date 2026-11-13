"""Each person in a video gets their own face (shaped on a live three-woman gym video: the pictures sent were the
first woman's two and the second's one, so the second was drawn with the first's face and the third had none)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_agent import AgentHarness, tiny_png  # noqa: E402


class VideoFaces(AgentHarness):
    async def upload(self, n: int) -> list[str]:
        ids = []
        for i in range(n):
            r = await self.http.post("/v1/assets", files={"files": (f"p{i}.png", tiny_png((i * 30, 80, 40)), "image/png")})
            ids.append(r.json()["assets"][0]["id"])
        return ids

    async def test_three_people_one_face_each_in_the_order_the_script_names_them(self):
        r1, r2, v1, v2, m1, m2 = await self.upload(6)
        chat = await self.new_chat(cast=[{"name": "Randitya", "persona": "R.", "refs": [r1, r2]},
                                         {"name": "Raveen", "persona": "V.", "refs": [v1, v2]},
                                         {"name": "Meu", "persona": "M.", "refs": [m1, m2]}])
        character = ("Meu, the woman whose face is <Picture 1>, in green shorts; Randitya, whose face is <Picture 2>, "
                     "in burgundy; Raveen, whose face is <Picture 3>, in grey.")
        calls = iter([[{"tool": "make_video", "args": {"identity_asset_ids": [r1, r2, v1], "character": character,
                                                        "shots": ["They work out."]}}]])

        def reply(body):
            actions = next(calls, [])
            return json.dumps({"lines": [{"speaker": "Randitya", "say": "Ok."}], "actions": actions, "done": not actions})

        self.atlas.reply = reply
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "video of all three of you at the gym"})
        view = await self.settle(chat)
        tool = next(m["content"] for m in view["messages"] if m["role"] == "tool" and m["content"]["tool"] == "make_video")
        self.assertEqual(tool["args"]["identity_asset_ids"], [m1, r1, v1], "one each, in the script's order")
        self.assertIn("(<Picture 1> is Meu; <Picture 2> is Randitya; <Picture 3> is Raveen.)", tool["args"]["character"])

    async def test_one_person_keeps_up_to_three_pictures(self):
        a, b = await self.upload(2)
        chat = await self.new_chat(cast=[{"name": "Tara", "persona": "T.", "refs": [a, b]}])
        calls = iter([[{"tool": "make_video", "args": {"identity_asset_ids": [a], "character": "Tara, face <Picture 1>.",
                                                        "shots": ["She walks."]}}]])

        def reply(body):
            actions = next(calls, [])
            return json.dumps({"say": "Ok.", "actions": actions, "done": not actions})

        self.atlas.reply = reply
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "a video of you walking"})
        view = await self.settle(chat)
        tool = next(m["content"] for m in view["messages"] if m["role"] == "tool" and m["content"]["tool"] == "make_video")
        self.assertEqual(tool["args"]["identity_asset_ids"], [a, b])
        self.assertIn("<Picture 1> to <Picture 2> are Tara", tool["args"]["character"])
