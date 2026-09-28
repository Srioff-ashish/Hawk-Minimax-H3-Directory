"""Two models in one chat, and something that can still see the image.

python -m unittest discover -s tests_api -p 'test_agent_models.py'
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_agent import DEEPSEEK, MODELS, AgentHarness, tiny_png  # noqa: E402  (fixtures, reused not copied)

PROSE = "deepseek-ai/deepseek-v4.1-flash"


class TwoModelsOneChat(AgentHarness):
    """The director keeps its own model; only the characters move to the prose one."""

    def character_calls(self, model: str | None = None) -> list[dict]:
        calls = [r for r in self.atlas.requests
                 if re.match(r"You are (\w|\s)+, one of the characters", str(r["messages"][0]["content"]))]
        return [c for c in calls if model is None or c["model"] == model]

    async def talk_once(self, chat: str) -> None:
        """One round of the characters talking. Driven straight through /talk rather than through a director
        tool call, so nothing here depends on the director's own behaviour."""
        started = await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 1})
        self.assertEqual(started.status_code, 202, started.text)
        await self.settle(chat)

    async def test_the_characters_speak_with_the_prose_model_and_the_director_keeps_its_own(self):
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}],
            "prose_model": PROSE})).json()
        self.assertEqual(chat["prose_model"], PROSE, "the chat should remember the model its characters speak with")

        def reply(body):
            if re.match(r"You are (\w|\s)+, one of the characters", str(body["messages"][0]["content"])):
                return json.dumps({"say": "arre wah", "to": "all"})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        # A talk round deliberately bypasses the director, so a plain message is sent too: the point of this
        # test is that the two roles use two models in the same chat.
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "hello"})
        await self.settle(chat["id"])
        await self.talk_once(chat["id"])

        spoken = self.character_calls()
        self.assertTrue(spoken, "the characters should have spoken")
        self.assertEqual({c["model"] for c in spoken}, {PROSE},
                         "every character turn should use the prose model")
        director = [r for r in self.atlas.requests if str(r["messages"][0]["content"]).startswith("You are Hawk")]
        self.assertTrue(director, "the director should still have run")
        self.assertEqual({c["model"] for c in director}, {"xai/grok-4.6"},
                         "the director needs strict tool JSON and must not follow the characters to a prose model")

    async def test_a_chat_made_before_the_split_falls_back_to_the_pods_own_model(self):
        # The migration story: the key is simply absent on every chat that already exists.
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}]})).json()
        self.assertEqual(chat.get("prose_model", ""), "", "a chat with no prose model of its own is the default")

        def reply(body):
            if re.match(r"You are (\w|\s)+, one of the characters", str(body["messages"][0]["content"])):
                return json.dumps({"say": "theek hai", "to": "all"})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        await self.talk_once(chat["id"])
        self.assertEqual({c["model"] for c in self.character_calls()}, {"xai/grok-4.6"},
                         "with nothing set anywhere the characters use what they always used")

    async def test_a_prose_model_that_writes_no_json_still_says_something(self):
        # The risk the whole split creates: a model chosen for its voice is often worse at formatting, and a
        # character that says nothing is indistinguishable from a broken chat.
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}],
            "prose_model": PROSE})).json()

        def reply(body):
            if re.match(r"You are (\w|\s)+, one of the characters", str(body["messages"][0]["content"])):
                return "Maya: yaar, main aa rahi hoon."   # prose, no JSON anywhere
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        await self.talk_once(chat["id"])
        view = (await self.http.get(f"/v1/agent/sessions/{chat['id']}")).json()
        spoken = [m["content"] for m in view["messages"] if m["role"] == "assistant" and m["content"].get("talk")]
        self.assertTrue(spoken, "a character that will not write JSON should still be heard")
        self.assertIn("aa rahi hoon", json.dumps(spoken), "the words the model wrote are what gets said")
        self.assertNotIn("Maya:", spoken[0]["lines"][0]["say"],
                         "the speaker's own name is a prefix, not part of the line")

    async def test_a_prose_model_that_fails_is_retried_on_the_directors_model(self):
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}],
            "prose_model": PROSE})).json()

        def reply(body):
            if re.match(r"You are (\w|\s)+, one of the characters", str(body["messages"][0]["content"])):
                # The prose model never manages it; the director's model does.
                return "{{ not json" if body["model"] == PROSE else json.dumps({"say": "ho gaya", "to": "all"})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        await self.talk_once(chat["id"])
        self.assertTrue(self.character_calls("xai/grok-4.6"),
                        "after the prose model's attempts are spent the turn should be retried on the director's")
        view = (await self.http.get(f"/v1/agent/sessions/{chat['id']}")).json()
        self.assertIn("ho gaya", json.dumps([m["content"] for m in view["messages"]]))


class SomethingThatCanSeeTheImage(AgentHarness):
    """inspect_image used to offer only two hardcoded Atlas ids, so it broke on any other provider."""

    async def inspect(self, chat: str, asset_id: str) -> dict:
        reviewers = []

        def reply(body):
            first = body["messages"][0]["content"]
            if isinstance(first, list):
                reviewers.append(body["model"])
                return json.dumps({"images": [{"asset_id": asset_id, "score": 8, "issues": [], "verdict": "ok"}],
                                   "best": asset_id, "advice": "Fine."})
            if "TOOL RESULT" in str(body["messages"][-1]["content"]):
                return json.dumps({"say": "Looks good.", "actions": [], "done": True})
            return json.dumps({"say": "", "actions": [{"tool": "inspect_image", "args": {"asset_ids": [asset_id]}}]})

        self.atlas.reply = reply
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "look at it"})
        await self.settle(chat)
        view = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()
        results = [m["content"] for m in view["messages"] if m["role"] == "tool"]
        return {"reviewers": reviewers, "result": results[-1] if results else {}}

    async def test_a_chat_on_a_blind_model_falls_through_to_one_that_can_see(self):
        blind = dict(DEEPSEEK, id="vendor/blind-but-lovely", input_modalities=["text"])
        self.atlas.model_list = MODELS + [blind]
        asset = (await self.http.post("/v1/assets", files={"files": ("face.png", tiny_png(), "image/png")})).json()["assets"][0]["id"]
        chat = (await self.http.post("/v1/agent/sessions", json={"model": blind["id"]})).json()["id"]
        seen = await self.inspect(chat, asset)
        self.assertTrue(seen["result"].get("ok"), seen["result"])
        self.assertNotIn(blind["id"], seen["reviewers"],
                         "a model the provider lists as text-only should never be asked to look")
        self.assertTrue(seen["reviewers"], "some listed vision model should have been tried")

    async def test_a_provider_that_spells_the_ids_differently_still_has_something_that_sees(self):
        # OpenRouter namespaces Grok "x-ai/", which is why the old hardcoded pair 404ed there.
        self.atlas.model_list = [dict(m, id=m["id"].replace("xai/", "x-ai/")) for m in MODELS]
        asset = (await self.http.post("/v1/assets", files={"files": ("face.png", tiny_png(), "image/png")})).json()["assets"][0]["id"]
        chat = (await self.http.post("/v1/agent/sessions", json={"model": "x-ai/grok-4.6"})).json()["id"]
        seen = await self.inspect(chat, asset)
        self.assertTrue(seen["result"].get("ok"), seen["result"])
        for model in seen["reviewers"]:
            self.assertIn(model, [m["id"] for m in self.atlas.model_list],
                          "every model tried must be one this provider actually lists")


if __name__ == "__main__":
    unittest.main()


class ACacheablePrefix(AgentHarness):
    """The instruction block is ~6k tokens and is re-sent on every call of a chat, up to eight times per user
    message. A provider caches it by its leading bytes and stops at the first byte that differs, so anything
    volatile sitting inside it costs full price for everything below it."""

    async def systems_after(self, chat: str, text: str) -> list[str]:
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": text})
        await self.settle(chat)
        return [m["content"] for m in self.atlas.requests[-1]["messages"] if m["role"] == "system"]

    async def test_the_prefix_is_byte_identical_when_only_growth_changed(self):
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}],
            "adaptive": True})).json()["id"]
        grew = iter([True, False])

        def reply(body):
            if next(grew, False):
                return json.dumps({"lines": [{"speaker": "Maya", "say": "haan boss"}], "actions": [], "done": True,
                                   "grow": [{"speaker": "Maya", "about": "user", "note": "Calls the user boss now."}]})
            return json.dumps({"lines": [{"speaker": "Maya", "say": "theek hai"}], "actions": [], "done": True})

        self.atlas.reply = reply
        first = await self.systems_after(chat, "hello")
        second = await self.systems_after(chat, "again")
        self.assertEqual(first[0], second[0],
                         "a growth note must not change one byte of the prefix, or the whole block is re-read")
        self.assertIn("Calls the user boss now.", "\n".join(second),
                      "and the note must still reach the model, below the prefix")

    async def test_the_volatile_block_is_sent_below_the_history(self):
        """Measured on the pod, and the reason this test exists: a prompt cache matches the longest identical
        prefix and re-reads from the first byte that differs. With the character state and recalled facts sent
        between the summary and the history, a turn that changed them re-read the whole conversation -- the
        cached share alternated between 98% and 1%. Below the history they invalidate only themselves."""
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Riya", "persona": "director"}, {"name": "Tiya", "persona": "stylist"}],
            "adaptive": True})).json()["id"]
        session = self.agent.get_session(chat)
        self.agent._seed_graph(session)
        session = self.agent.get_session(chat)
        self.agent._write_fact(session, "", "Riya", "made", "a1b2c3d4e5f6", "Riya made the rooftop portrait.", 1.0, 1)

        self.atlas.reply = lambda body: json.dumps({"lines": [{"speaker": "Riya", "say": "ok"}], "actions": [], "done": True})
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Riya, rooftop ready?"})
        await self.settle(chat)

        sent = self.atlas.requests[-1]["messages"]
        recalled = [i for i, m in enumerate(sent) if "Riya made the rooftop portrait." in str(m["content"])]
        self.assertTrue(recalled, "the fact should have been recalled at all")
        last_user_at = max(i for i, m in enumerate(sent) if m["role"] == "user")
        self.assertGreater(recalled[0], last_user_at,
                           "recall must sit below the history, or a turn that changes it re-reads every "
                           "message above it at full price")
        self.assertEqual(sent[0]["role"], "system", "the stable instruction block still comes first")

    async def test_a_compaction_does_not_shrink_the_tool_catalogue(self):
        # _tools_in_use used to read only the messages after the summary watermark, so compacting a chat took a
        # tool's schema back out of the catalogue: the prefix shrank, and the agent was told to ask once.
        chat = (await self.http.post("/v1/agent/sessions", json={"persona": "You are Maya."})).json()["id"]
        asked = iter([True])

        def reply(body):
            if next(asked, False):
                return json.dumps({"say": "", "actions": [{"tool": "describe_tool", "args": {"name": "render_film"}}]})
            return json.dumps({"say": "ok", "actions": [], "done": True})

        self.atlas.reply = reply
        before = (await self.systems_after(chat, "what can you do"))[0]
        self.assertIn("render_film", before)
        session = self.agent.get_session(chat)
        self.assertIn("render_film", session.get("tools_seen") or [],
                      "the tool is remembered on the session, not inferred from messages that a summary replaces")

        # A compaction only drops messages beyond agent_keep_messages, so there has to be a chat to compact.
        for n in range(12):
            await self.systems_after(chat, f"filler {n}")
        self.assertEqual((await self.http.post(f"/v1/agent/sessions/{chat}/compact")).status_code, 200)
        compacted = self.agent.get_session(chat)
        self.assertTrue(compacted.get("summary"), "the compaction has to have happened for this test to mean anything")
        self.assertGreater(compacted.get("summary_upto", 0), 0,
                           "the watermark past which messages are dropped is what used to shrink the catalogue")
        after = (await self.systems_after(chat, "and now"))[0]
        self.assertEqual(before, after, "compacting must not change the cacheable prefix at all")
