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


class FullMemory(AgentHarness):
    async def source_chat(self) -> str:
        """Maya and Riya in a chat that has a story, facts, a memory and feelings."""
        chat = await self.new_chat(cast=[{"name": "Maya", "persona": "Maya, 27, a photographer."},
                                         {"name": "Riya", "persona": "Riya, 26, her flatmate."}])
        session = self.agent.get_session(chat)
        maya, riya = session["cast"]
        maya.update(memory="I took Riya to Goa and we fought about the hotel.",
                    feelings={"user": ["missed you"], riya["id"]: ["still annoyed about Goa"]})
        session["story"] = ["Maya and Riya went to Goa.", "They argued about the hotel."]
        self.agent.store.save_session(session)
        node_m = self.agent.store.upsert_node(chat, "", f"member:{maya['id']}", "person", "Maya")
        node_g = self.agent.store.upsert_node(chat, "", "goa", "place", "Goa")
        self.agent.store.put_edge(chat, "", node_m, node_g, "owns", "owns", "Maya owns a flat in Goa.", 1.0, 1)
        secret = self.agent.store.upsert_node(chat, riya["id"], "secret", "thing", "secret")
        self.agent.store.put_edge(chat, riya["id"], secret, node_g, "owns", "owns", "Riya hides a ring.", 1.0, 1)
        return chat

    async def test_she_arrives_remembering_everything(self):
        source = await self.source_chat()
        target = await self.new_chat(cast=[{"name": "Riya", "persona": "Riya again."}, {"name": "Zoya", "persona": "Zoya."}])
        response = await self.http.post(f"/v1/agent/sessions/{target}/bring_in",
                                        json={"from_session": source, "member": "Maya", "full": True})
        self.assertEqual(response.status_code, 200, response.text)
        session = self.agent.get_session(target)
        maya = next(m for m in session["cast"] if m["name"] == "Maya")
        riya_here = next(m for m in session["cast"] if m["name"] == "Riya")
        past = maya["past"][0]["text"]
        self.assertIn("we fought about the hotel", past, "her own memory of that chat")
        self.assertIn("They argued about the hotel.", past, "what happened there")
        self.assertIn("Maya owns a flat in Goa.", past, "what it learned aloud")
        self.assertNotIn("Riya hides a ring", past, "another character's private fact stays private")
        self.assertEqual(maya["feelings"]["user"], ["missed you"])
        self.assertEqual(maya["feelings"][riya_here["id"]], ["still annoyed about Goa"],
                         "a feeling about Riya points at the Riya who is here")

        director = self.agent._memory_text(session)
        self.assertIn("MEMORIES FROM EARLIER CHATS", director)
        self.assertIn("only Maya knows this", director)
        self.assertIn("we fought", self.agent._memory_text(session, viewer=maya["id"]))
        self.assertNotIn("we fought", self.agent._memory_text(session, viewer=riya_here["id"]),
                         "the others do not know her past unless she tells them")

        memory = (await self.http.get(f"/v1/agent/sessions/{target}/memory")).json()
        self.assertEqual(next(c for c in memory["characters"] if c["name"] == "Maya")["past"][0]["from"],
                         self.agent.get_session(source)["title"])
        await self.http.put(f"/v1/agent/sessions/{target}/memory", json={"member": "Maya", "past": []})
        self.assertEqual(next(m for m in self.agent.get_session(target)["cast"] if m["name"] == "Maya")["past"], [])

    async def test_without_full_the_memories_stay_behind(self):
        source = await self.source_chat()
        target = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{target}/bring_in", json={"from_session": source, "member": "Maya"})
        maya = self.agent.get_session(target)["cast"][0]
        self.assertEqual((maya["past"], maya["feelings"]), ([], {}))


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
