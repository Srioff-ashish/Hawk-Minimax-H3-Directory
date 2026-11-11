"""Long-term memory: each character's canon and the chat's story, captured every turn by their own call, shown to
every model, never condensed away -- and every picture of a character carrying that character's identity pictures."""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api import agent as agent_module  # noqa: E402
from hawk_api.agent import CANON_MAX, clean_canon  # noqa: E402
from test_agent import AgentHarness, is_summary, tiny_png  # noqa: E402

SAREE_RULE = "Never wears a saree again; only western, skimpy clothes."


def is_memory(body: dict) -> bool:
    return "PERMANENT MEMORY KEEPER" in str((body.get("messages") or [{}])[0].get("content") or "")


def is_fold(body: dict) -> bool:
    return str((body.get("messages") or [{}])[0].get("content") or "").startswith("Fold these story beats")


def all_text(body: dict) -> str:
    return "\n".join(str(m.get("content") or "") for m in body.get("messages") or [])


class LongTermMemory(AgentHarness):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        agent_module.MEMORY_CAPTURE = True

    async def chat(self, memory_reply, director=None) -> str:
        """A one-character chat whose memory keeper answers memory_reply (a value or a function of the call)."""
        chat = await self.new_chat(persona="Rebel, a 45-year-old woman next door.", name="Rebel")
        director = director or (lambda body: json.dumps({"say": "theek hai", "actions": [], "done": True}))

        def reply(body):
            if is_memory(body):
                return memory_reply(body) if callable(memory_reply) else memory_reply
            if is_fold(body):
                return json.dumps({"story": ["Rebel's old life ended and her new one began."]})
            if is_summary(body):
                return json.dumps({"summary": "s", "facts": []})
            return director(body)

        self.atlas.reply = reply
        return chat

    async def say(self, chat: str, text: str) -> dict:
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": text})
        return await self.settle(chat)

    async def memory(self, chat: str) -> dict:
        return (await self.http.get(f"/v1/agent/sessions/{chat}/memory")).json()

    async def upload(self) -> str:
        """A distinct image each time: identical bytes are stored once, under one id."""
        self.uploads = getattr(self, "uploads", 0) + 1
        png = tiny_png(color=(20 * self.uploads % 256, 90, 160))
        return (await self.http.post("/v1/assets", files={"files": ("p.png", png, "image/png")})).json()["assets"][0]["id"]

    async def test_a_rule_is_kept_and_reaches_the_model_that_answers(self):
        chat = await self.chat(json.dumps({
            "canon": [{"speaker": "Rebel", "kind": "rule", "text": SAREE_RULE, "replaces": None}],
            "story": ["Rebel burned all her sarees."]}))
        await self.say(chat, "Burn all your sarees. From now on never wear a saree, only western skimpy clothes.")
        memory = await self.memory(chat)
        rebel = memory["characters"][0]
        self.assertEqual([c["text"] for c in rebel["canon"]], [SAREE_RULE])
        self.assertEqual(rebel["canon"][0]["kind"], "rule")
        self.assertEqual(memory["story"], ["Rebel burned all her sarees."])
        directors = [r for r in self.atlas.requests if not (is_memory(r) or is_summary(r) or is_fold(r))]
        self.assertTrue(directors)
        self.assertIn(SAREE_RULE, all_text(directors[-1]),
                      "captured before the director is asked, so the very reply that follows already holds it")
        self.assertIn("PERMANENT MEMORY", all_text(directors[-1]))
        captured = memory_note = [m for m in (await self.http.get(f"/v1/agent/sessions/{chat}")).json()["messages"]
                                  if m["role"] == "note" and m["content"].get("kind") == "canon"]
        self.assertEqual([m["content"]["text"] for m in memory_note], [SAREE_RULE], captured)

    async def test_a_newer_item_replaces_the_one_it_contradicts(self):
        chat = await self.chat(json.dumps({"canon": [], "story": []}))
        await self.http.put(f"/v1/agent/sessions/{chat}/memory", json={"add": ["Wears cotton sarees at home."]})
        self.atlas.reply = (lambda old: lambda body: json.dumps({
            "canon": [{"speaker": "Rebel", "kind": "rule", "text": SAREE_RULE, "replaces": 1}], "story": []})
            if is_memory(body) else old(body))(self.atlas.reply)
        await self.say(chat, "No more sarees, ever.")
        self.assertEqual([c["text"] for c in (await self.memory(chat))["characters"][0]["canon"]], [SAREE_RULE])

    async def test_an_unusable_capture_is_read_again_next_turn(self):
        answers = iter(["not json at all", json.dumps({
            "canon": [{"speaker": "Rebel", "kind": "rule", "text": SAREE_RULE}], "story": []})])
        seen = []

        def memory_reply(body):
            seen.append(body["messages"][1]["content"])
            return next(answers, json.dumps({"canon": [], "story": []}))

        chat = await self.chat(memory_reply)
        await self.say(chat, "Never wear a saree again.")
        self.assertEqual((await self.memory(chat))["characters"][0]["canon"], [])
        self.assertEqual((await self.memory(chat))["captured_upto"], 0, "a failed capture keeps its place")
        await self.say(chat, "Aur haan, collar hamesha pehna rehna.")
        self.assertIn("Never wear a saree again.", seen[1], "the window the failed call missed is read again")
        self.assertEqual([c["text"] for c in (await self.memory(chat))["characters"][0]["canon"]], [SAREE_RULE])

    async def test_the_story_is_folded_not_dropped_when_it_grows_long(self):
        saved = agent_module.STORY_MAX, agent_module.STORY_FOLD
        agent_module.STORY_MAX, agent_module.STORY_FOLD = 3, 2
        try:
            beats = iter([["She met Aarav.", "She fixed the Wi-Fi."], ["She posted her first reel.", "She burned her sarees."]])
            chat = await self.chat(lambda body: json.dumps({"canon": [], "story": next(beats, [])}))
            await self.say(chat, "pehla")
            await self.say(chat, "doosra")
            story = (await self.memory(chat))["story"]
        finally:
            agent_module.STORY_MAX, agent_module.STORY_FOLD = saved
        self.assertEqual(story[0], "Rebel's old life ended and her new one began.")
        self.assertEqual(story[1:], ["She posted her first reel.", "She burned her sarees."])

    async def test_cast_edits_keep_the_canon_and_the_pictures(self):
        chat = await self.chat(json.dumps({"canon": [], "story": []}))
        face = await self.upload()
        await self.http.put(f"/v1/agent/sessions/{chat}/memory", json={"add": [SAREE_RULE], "refs": [face]})
        session = (await self.http.get(f"/v1/agent/sessions/{chat}?limit=1")).json()["session"]
        cast = [{"id": m["id"], "name": m["name"], "persona": m["persona"] + " Bold now."} for m in session["cast"]]
        await self.http.patch(f"/v1/agent/sessions/{chat}", json={"cast": cast})
        rebel = (await self.memory(chat))["characters"][0]
        self.assertEqual([c["text"] for c in rebel["canon"]], [SAREE_RULE])
        self.assertEqual(rebel["refs"], [face])
        bad = await self.http.put(f"/v1/agent/sessions/{chat}/memory", json={"refs": ["no-such-asset"]})
        self.assertGreaterEqual(bad.status_code, 400)

    async def test_every_picture_of_her_carries_her_identity_pictures(self):
        face, body, outfit = await self.upload(), await self.upload(), await self.upload()
        calls = []

        def director(request):
            calls.append(request)
            if len(calls) == 1:
                return json.dumps({"say": "ek second", "done": False, "actions": [{"tool": "generate_image", "args": {
                    "prompt": "Rebel by the pool in a red bikini, the same soft rounded face.",
                    "reference_asset_ids": [outfit]}}]})
            return json.dumps({"say": "ye lo", "actions": [], "done": True})

        chat = await self.chat(json.dumps({"canon": [], "story": []}), director)
        await self.http.put(f"/v1/agent/sessions/{chat}/memory", json={"refs": [face, body]})
        view = await self.say(chat, "pool pe ek photo")
        tool = next(m for m in view["messages"] if m["role"] == "tool" and m["content"]["tool"] == "generate_image")
        args = tool["content"]["args"]
        self.assertEqual(args["reference_asset_ids"], [face, body, outfit],
                         "her own pictures first, labelled, then the reference the model chose")
        self.assertIn("<image1> to <image2> are Rebel", args["prompt"])

    async def test_the_agent_can_set_a_characters_pictures(self):
        face = await self.upload()
        calls = []

        def director(request):
            calls.append(request)
            if len(calls) == 1:
                return json.dumps({"say": "ok", "done": False,
                                   "actions": [{"tool": "set_references", "args": {"asset_ids": [face]}}]})
            return json.dumps({"say": "done", "actions": [], "done": True})

        chat = await self.chat(json.dumps({"canon": [], "story": []}), director)
        await self.say(chat, "use this as your reference")
        self.assertEqual((await self.memory(chat))["characters"][0]["refs"], [face])


class TheCanonOnItsOwn(unittest.TestCase):
    def test_rules_outlive_the_cap(self):
        items = [{"text": "Kneels and greets him as Master every evening.", "kind": "rule"}] + [
            {"text": f"fact {i}", "kind": "fact"} for i in range(CANON_MAX + 5)]
        kept = clean_canon(items)
        self.assertEqual(len(kept), CANON_MAX)
        self.assertEqual(kept[0]["text"], "Kneels and greets him as Master every evening.")
        self.assertEqual(kept[-1]["text"], f"fact {CANON_MAX + 4}")

    def test_strings_are_rules_and_repeats_are_dropped(self):
        kept = clean_canon(["Never wears a saree.", {"text": "never wears a saree.", "kind": "fact"}, ""])
        self.assertEqual(kept, [{"text": "Never wears a saree.", "kind": "rule", "at": 0.0}])


if __name__ == "__main__":
    unittest.main()
