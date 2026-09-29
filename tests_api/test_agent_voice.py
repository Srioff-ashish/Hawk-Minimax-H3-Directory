"""Which model answers in a chat with a character, and what happens when it repeats itself.

A chat with one character never reached _character_turn -- that path needs two, because it is the characters
talking to each other -- so every word a lone character said came from the director model and the prose
setting did nothing, though Studio offers it. The director call also sent no temperature, so an 800-message
roleplay ran at the 0.4 meant for tool orchestration: once two identical replies were in the history they
were the clearest pattern in the prompt, and the model reproduced them word for word whatever was asked.

python -m unittest discover -s tests_api -p 'test_agent_voice.py'
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_agent import AgentHarness, last_user  # noqa: E402  (fixtures, reused not copied)

PROSE = "deepseek-ai/deepseek-v4.1-flash"
DIRECTOR = "xai/grok-4.3"


class WhichModelSpeaks(AgentHarness):
    """The prose model says what the character says; the director reads the tool results."""

    def calls(self) -> list[dict]:
        """Every chat call that is a director/character turn, in order: (model, temperature)."""
        return [(r["model"], r.get("temperature")) for r in self.atlas.requests
                if not str((r["messages"] or [{}])[0].get("content", "")).startswith("Summarise the conversation")]

    async def test_a_chat_with_one_character_answers_with_the_prose_model(self):
        chat = (await self.http.post("/v1/agent/sessions", json={
            "persona": "A blunt Mumbai ad-film director, 32.", "model": DIRECTOR, "prose_model": PROSE})).json()
        before = len(self.atlas.requests)
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "kaisi ho?"})
        await self.settle(chat["id"])
        spoke = [(m, t) for m, t in self.calls()[before:]]
        self.assertEqual(spoke[0][0], PROSE,
                         "a lone character is still a character: its voice is the prose model, not the director")
        self.assertEqual(spoke[0][1], 0.8,
                         "and it speaks at the temperature the characters already use with each other")

    async def test_a_chat_with_no_persona_stays_on_the_director(self):
        chat = (await self.http.post("/v1/agent/sessions", json={"model": DIRECTOR, "prose_model": PROSE})).json()
        before = len(self.atlas.requests)
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "render me a chai ad"})
        await self.settle(chat["id"])
        spoke = self.calls()[before:]
        self.assertEqual(spoke[0][0], DIRECTOR,
                         "nobody is in character here, so this is someone making a video and it is the director's")
        self.assertEqual(spoke[0][1], 0.4, "tool work wants the JSON right more than it wants the wording fresh")

    async def test_the_tool_steps_of_a_character_chat_go_back_to_the_director(self):
        # The character speaks once; reading what a tool returned is the director's job whoever the chat is with.
        def reply(body):
            if not any(m["role"] == "assistant" for m in body["messages"]):
                return json.dumps({"say": "Dekhti hoon.", "actions": [{"tool": "list_options", "args": {}}]})
            return json.dumps({"say": "Ho gaya.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = (await self.http.post("/v1/agent/sessions", json={
            "persona": "A stylist.", "model": DIRECTOR, "prose_model": PROSE})).json()
        before = len(self.atlas.requests)
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "kya options hain?"})
        await self.settle(chat["id"])
        spoke = self.calls()[before:]
        self.assertEqual(spoke[0][0], PROSE, "the first step is the answer to the user, so it is the character")
        self.assertEqual([m for m, _ in spoke[1:]], [DIRECTOR] * len(spoke[1:]),
                         "every step after it is reading a tool result, which is the director's")

    async def test_a_prose_model_the_provider_drops_does_not_end_the_turn(self):
        self.atlas.fail_models = {PROSE}
        chat = (await self.http.post("/v1/agent/sessions", json={
            "persona": "A stylist.", "model": DIRECTOR, "prose_model": PROSE})).json()
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "hello"})
        view = await self.settle(chat["id"])
        said = [m for m in view["messages"] if m["role"] == "assistant"]
        self.assertTrue(said, "a prose model that 404s must not end the turn the way it used to")
        self.assertNotEqual(said[-1]["content"]["usage"]["model"], PROSE, "the dead model cannot be the one that answered")

    async def test_the_director_chain_is_the_backstop_when_the_whole_prose_chain_is_down(self):
        # resolve_many already gives a role its own failover, so the appended director chain only earns its
        # keep here: every id the prose role would reach is refused, and only the director's own is left.
        self.atlas.fail_models = {"xai/grok-4.6", "xai/grok-4.3"}
        chat = (await self.http.post("/v1/agent/sessions", json={
            "persona": "A stylist.", "model": PROSE, "prose_model": "xai/grok-4.6"})).json()
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "hello"})
        view = await self.settle(chat["id"])
        said = [m for m in view["messages"] if m["role"] == "assistant"]
        self.assertTrue(said, "without the director chain behind it this turn would die with a model error")
        self.assertEqual(said[-1]["content"]["usage"]["model"], PROSE,
                         "a model picked for voice may be the worse one at JSON, so the director is the backstop")


class AReplyThatRepeatsItself(AgentHarness):
    """Two identical replies are self-reinforcing, so the second one is asked for again instead of stored."""

    async def test_an_identical_reply_is_asked_for_again_rather_than_kept(self):
        # The same words twice is what locks a long chat: the pair becomes the clearest pattern in the prompt.
        seen = {"n": 0}

        def reply(body):
            seen["n"] += 1
            if seen["n"] <= 2:
                return json.dumps({"say": "Wahi baat phir se.", "actions": [], "done": True})
            return json.dumps({"say": "Achha, aage badhte hain.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = (await self.http.post("/v1/agent/sessions", json={"persona": "A stylist."})).json()
        for text in ("pehla", "doosra"):
            await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": text})
            await self.settle(chat["id"])
        says = [m["content"].get("say") for m in (await self.http.get(
            f"/v1/agent/sessions/{chat['id']}")).json()["messages"] if m["role"] == "assistant"]
        self.assertEqual(says, ["Wahi baat phir se.", "Achha, aage badhte hain."],
                         "the repeat is never stored, because storing it is what makes the next one identical too")

    async def test_the_retry_says_what_was_wrong_with_it(self):
        seen = {"n": 0}

        def reply(body):
            seen["n"] += 1
            if seen["n"] <= 2:
                return json.dumps({"say": "Wahi baat phir se.", "actions": [], "done": True})
            return json.dumps({"say": "Kuch naya.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = (await self.http.post("/v1/agent/sessions", json={"persona": "A stylist."})).json()
        for text in ("pehla", "doosra"):
            await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": text})
            await self.settle(chat["id"])
        nudged = [r for r in self.atlas.requests if "repeated the previous one word for word" in last_user(r)]
        self.assertTrue(nudged, "the retry has to tell the model what it did, or it just does it again")
        self.assertEqual(nudged[-1]["temperature"], 1.0,
                         "and asking the same question the same way gets the same answer")

    async def test_a_reply_that_repeats_but_calls_a_tool_is_left_alone(self):
        # Two identical "waiting for the render" lines are not a stuck chat: the work is what moved on.
        def reply(body):
            done = sum(1 for m in body["messages"] if m["role"] == "assistant")
            if done < 2:
                return json.dumps({"say": "Dekhti hoon.", "actions": [{"tool": "list_options", "args": {}}]})
            return json.dumps({"say": "Ho gaya.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = (await self.http.post("/v1/agent/sessions", json={"persona": "A stylist."})).json()
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "options?"})
        view = await self.settle(chat["id"])
        says = [m["content"].get("say") for m in view["messages"] if m["role"] == "assistant"]
        self.assertEqual(says.count("Dekhti hoon."), 2,
                         "a repeated line that carries an action is doing something, so it is kept")


if __name__ == "__main__":
    unittest.main()
