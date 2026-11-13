"""In "let them talk", a character who promises a picture asks for it (a live talk ended on "give me a few minutes to
bake the render" and nothing was ever made)."""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.agent import promises_a_make  # noqa: E402
from test_agent import AgentHarness  # noqa: E402


class Detect(unittest.TestCase):
    def test_promises(self):
        for say in ("Give me a few minutes to bake the full-res render.", "Let's do the full quality render with the music",
                    "main abhi tasveer bhejti hoon", "I'll send you a selfie in a sec", "Ruko, video bana rahi hoon"):
            with self.subTest(say=say):
                self.assertTrue(promises_a_make(say))
        for say in ("I loved that photo", "video achha tha", "the render looked great yesterday", "Lo-fi it is."):
            with self.subTest(say=say):
                self.assertFalse(promises_a_make(say))


class Talk(AgentHarness):
    async def test_a_promise_without_make_is_sent_back_once(self):
        asks = []

        def reply(body):
            text = json.dumps(body["messages"])
            if "THE CONVERSATION SO FAR" not in text:  # the director making what was asked
                return json.dumps({"say": "Done.", "actions": [], "done": True})
            asks.append(body)
            if "gets anything made" in text:
                return json.dumps({"say": "Here it comes.", "to": "all", "make": "Kabir's photo of Tara at sunset on Juhu beach",
                                   "act": "snap", "of": ["Tara"]})
            return json.dumps({"say": "Give me a few minutes to bake the full-res render.", "to": "all"})

        self.atlas.reply = reply
        chat = await self.new_chat(cast=[{"name": "Kabir", "persona": "K."}, {"name": "Tara", "persona": "T."}])
        response = await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 1, "makes": 1})
        self.assertEqual(response.status_code, 202, response.text)
        view = await self.settle(chat)
        makes = [m["content"] for m in view["messages"] if m["role"] == "note" and m["content"].get("kind") == "make"]
        self.assertTrue(makes, "the promise turned into a real make")

    async def test_no_nudge_when_makes_are_off(self):
        asks = []

        def reply(body):
            asks.append(body)
            return json.dumps({"say": "Give me a few minutes to bake the full-res render.", "to": "all"})

        self.atlas.reply = reply
        chat = await self.new_chat(cast=[{"name": "Kabir", "persona": "K."}, {"name": "Tara", "persona": "T."}])
        await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 1, "makes": 0})
        await self.settle(chat)
        self.assertFalse(any("gets anything made" in json.dumps(b) for b in asks))
