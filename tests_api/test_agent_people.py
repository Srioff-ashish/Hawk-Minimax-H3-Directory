"""People across chats: regenerate a reply, list everyone, bring a character into another chat, and identity
pictures that never push out the picture being edited.

python -m unittest discover -s tests_api -p 'test_agent_people.py'
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.agent import AgentService  # noqa: E402
from test_agent import AgentHarness  # noqa: E402


class People(AgentHarness):
    async def test_regenerate_hides_the_reply_and_answers_again(self):
        replies = iter(["First answer.", "Second answer."])
        self.atlas.reply = lambda body: json.dumps({"say": next(replies, "Again."), "actions": [], "done": True})
        chat = await self.new_chat(persona="You are Maya.")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Hi"})
        await self.settle(chat)
        response = await self.http.post(f"/v1/agent/sessions/{chat}/regenerate")
        self.assertEqual(response.status_code, 200, response.text)
        view = await self.settle(chat)
        said = [m["content"].get("say") for m in view["messages"] if m["role"] == "assistant" and not m.get("excluded")]
        self.assertEqual(said[-1], "Second answer.")
        hidden = [m for m in view["messages"] if m["role"] == "assistant" and m.get("excluded")]
        self.assertEqual([m["content"].get("say") for m in hidden], ["First answer."], "kept, greyed, restorable")

    async def test_regenerate_needs_a_message(self):
        chat = await self.new_chat(persona="You are Maya.")
        response = await self.http.post(f"/v1/agent/sessions/{chat}/regenerate")
        self.assertEqual(response.status_code, 404)

    async def test_people_groups_the_same_name_across_chats(self):
        first = await self.new_chat(persona="Maya, 27, a photographer in Mumbai.", name="Maya")
        second = await self.new_chat(persona="Maya again.", name="Maya")
        await self.http.put(f"/v1/agent/sessions/{first}/memory", json={"add": ["Calls the user boss."]})
        people = (await self.http.get("/v1/people")).json()["people"]
        maya = next(p for p in people if p["name"] == "Maya")
        self.assertEqual({c["session_id"] for c in maya["chats"]}, {first, second})
        self.assertEqual(sorted(c["canon"] for c in maya["chats"]), [0, 1])

    async def test_bring_in_carries_who_they_are_but_not_that_chats_feelings(self):
        source = await self.new_chat(persona="Maya, 27, a photographer.", name="Maya")
        await self.http.put(f"/v1/agent/sessions/{source}/memory", json={"add": ["Calls the user boss."]})
        session = self.agent.get_session(source)
        session["cast"][0]["growth"] = ["Hums when she cooks."]
        session["cast"][0]["feelings"] = {"user": ["annoyed about Goa"]}
        self.agent.store.save_session(session)

        target = await self.new_chat()
        response = await self.http.post(f"/v1/agent/sessions/{target}/bring_in",
                                        json={"from_session": source, "member": "Maya"})
        self.assertEqual(response.status_code, 200, response.text)
        cast = self.agent.get_session(target)["cast"]
        self.assertEqual(len(cast), 1, "an empty new chat's default persona is replaced, not joined")
        maya = cast[0]
        self.assertEqual(maya["name"], "Maya")
        self.assertIn("photographer", maya["persona"])
        self.assertEqual([c["text"] for c in maya["canon"]], ["Calls the user boss."])
        self.assertEqual(maya["growth"], ["Hums when she cooks."])
        self.assertEqual(maya["feelings"], {}, "feelings belong to the chat they grew in")

        again = await self.http.post(f"/v1/agent/sessions/{target}/bring_in",
                                     json={"from_session": source, "member": "Maya"})
        self.assertEqual(again.status_code, 409)


class Identity(unittest.TestCase):
    def test_identity_gives_way_to_the_picture_being_edited(self):
        identity = [("Maya", ["m1", "m2", "m3", "m4"]), ("Riya", ["r1", "r2", "r3", "r4"])]
        fitted = AgentService._fit_identity(identity, 6)
        self.assertEqual(sum(len(p) for _, p in fitted), 6)
        self.assertEqual([p[0] for _, p in fitted], ["m1", "r1"], "everyone keeps their first picture")
        squeezed = AgentService._fit_identity(identity, 1)
        self.assertEqual([p for _, p in squeezed], [["m1"], ["r1"]], "never below one each")

    def test_labels_can_start_after_the_models_own_references(self):
        text = AgentService._label_identity("Put the jacket from <image1> on her.", [("Maya", ["m1", "m2"])], start=2)
        self.assertIn("<image2> to <image3> are Maya", text)


if __name__ == "__main__":
    unittest.main()
