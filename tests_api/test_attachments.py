"""What the user attaches is understood, not guessed at: pictures and video frames are looked at once when they
arrive, and the characters are told plainly that they cannot hear audio (a live chat described how an mp3 sounded,
and its file was not even on the server)."""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.agent import LOOK_PROMPT, attachment_line  # noqa: E402
from test_agent import AgentHarness, tiny_png  # noqa: E402


class Lines(unittest.TestCase):
    def test_each_kind(self):
        self.assertIn("What it shows: a woman on a rooftop",
                      attachment_line({"asset_id": "a", "kind": "image", "filename": "x.png", "seen": "a woman on a rooftop"}))
        self.assertIn("call inspect_image", attachment_line({"asset_id": "a", "kind": "image", "filename": "x.png"}))
        self.assertIn("not the motion or the sound",
                      attachment_line({"asset_id": "v", "kind": "video", "filename": "x.mp4", "seen": "a beach"}))
        self.assertIn("cannot watch", attachment_line({"asset_id": "v", "kind": "video", "filename": "x.mp4"}))
        self.assertIn("never describe how it sounds", attachment_line({"asset_id": "m", "kind": "audio", "filename": "beach.mp3"}))
        self.assertIn("missing", attachment_line({"asset_id": "m", "kind": "image", "filename": "x.png", "seen": "",
                                                  "missing": "gone"}))


class InTheChat(AgentHarness):
    async def test_an_attached_picture_is_looked_at_and_the_character_gets_it(self):
        looks, director = [], []

        def reply(body):
            first = body["messages"][0]["content"]
            if isinstance(first, list) and LOOK_PROMPT in str(first[0].get("text")):
                looks.append(body)
                return json.dumps({"items": [{"asset_id": "whatever", "seen": "A woman in a red saree on a rooftop at dusk."}]})
            director.append(body)
            return json.dumps({"say": "That's me on the rooftop!", "actions": [], "done": True})

        self.atlas.reply = reply
        uploaded = await self.http.post("/v1/assets", files={"files": ("rooftop.png", tiny_png(), "image/png")})
        asset = uploaded.json()["assets"][0]["id"]
        chat = await self.new_chat(persona="You are Tara.", name="Tara")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "What am I wearing here?", "attachments": [asset]})
        view = await self.settle(chat)
        self.assertEqual(len(looks), 1, "looked at once")
        user = next(m for m in view["messages"] if m["role"] == "user")
        self.assertEqual(user["content"]["attachments"][0]["seen"], "A woman in a red saree on a rooftop at dusk.",
                         "a made-up id still lands on the picture sent")
        prompt = json.dumps(director[0]["messages"])
        self.assertIn("What it shows: A woman in a red saree", prompt)

    async def test_audio_is_not_looked_at_and_the_character_is_told_it_cannot_hear(self):
        bodies = []

        def reply(body):
            bodies.append(body)
            return json.dumps({"say": "I can't hear it, but the name says beach.", "actions": [], "done": True})

        self.atlas.reply = reply
        uploaded = await self.http.post("/v1/assets", files={"files": ("beach.mp3", b"ID3" + b"\x00" * 64, "audio/mpeg")})
        asset = uploaded.json()["assets"][0]["id"]
        chat = await self.new_chat(persona="You are Kabir.", name="Kabir")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "What kind of music is this?", "attachments": [asset]})
        await self.settle(chat)
        self.assertFalse(any(LOOK_PROMPT in json.dumps(b) for b in bodies), "no vision call for audio")
        self.assertIn("never describe how it sounds", json.dumps(bodies[0]["messages"]))


if __name__ == "__main__":
    unittest.main()
