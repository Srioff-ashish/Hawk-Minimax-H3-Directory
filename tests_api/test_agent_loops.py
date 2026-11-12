"""A character stops looping: near-copies of recent replies are caught, and permanent memory does not fill with
finished tasks and reworded duplicates (shaped on a live chat that looped for several turns)."""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.agent import is_finished_event, overlap, repeat_reason, tidy_canon  # noqa: E402
from test_agent import AgentHarness  # noqa: E402

A = "Toh kya hua? Main toh bas thoda aur maza lena chahti thi... theek hai, so jaate hain. Par yaad rakhna, subah sab baat karenge."
B = "Toh kya hua? Main toh bas thoda aur maza lena chahti thi... itna gussa kyun ho rahe ho? Subah dekhte hain, chai banaungi."
C = "Toh kya hua? Maine kaha tha na, subah uthte hi milna hi hai. Baaki sab baad mein kar lena, abhi toh mera mann hai."
FRESH = "Arre, tum uth gaye! Chalo pehle chai banati hoon, phir balcony mein baith kar baat karte hain aaj ke plan ke baare mein."


class Repeats(unittest.TestCase):
    def test_the_live_loop_is_caught(self):
        self.assertEqual(repeat_reason(A, [A]), "repeated the previous one word for word")
        self.assertEqual(repeat_reason(B, [A]), "opened exactly the way a previous reply did")
        self.assertTrue(repeat_reason(A.replace("theek hai", "accha"), [A]), "a word or two changed is still a copy")

    def test_a_new_reply_passes(self):
        self.assertEqual(repeat_reason(FRESH, [A, B, C]), "")
        self.assertEqual(repeat_reason("Haan.", ["Haan, bilkul."]), "", "short replies are not judged")

    def test_overlap(self):
        self.assertGreater(overlap(A, A.replace("so jaate hain", "chalo")), 0.5)
        self.assertLess(overlap(A, FRESH), 0.2)


class Memory(unittest.TestCase):
    def test_finished_tasks_are_story_not_canon(self):
        for text in ("Maya has filmed and delivered the one-minute video of herself dancing.",
                     "Maya must keep dancing and make an 11-minute video of it while he drinks his tea.",
                     "Arjun has ordered Maya to film a one-minute video of herself dancing.",
                     "Maya has delivered the 11-minute video of herself dancing."):
            with self.subTest(text=text):
                self.assertTrue(is_finished_event(text))
        for text in ("Maya always dresses before Arjun's father comes home.",
                     "Maya must vary her expressions and clothes in every photo session.",
                     "Arjun promised Maya a prize if she makes him laugh by modelling.",
                     "Arjun has given Maya free rein to do whatever she wants with him."):
            with self.subTest(text=text):
                self.assertFalse(is_finished_event(text))

    def test_tidy_merges_rewordings_and_moves_events(self):
        canon = [
            {"text": "Maya always dresses before Arjun's father comes home.", "kind": "rule"},
            {"text": "Maya has filmed and delivered the one-minute video of herself dancing.", "kind": "fact"},
            {"text": "Maya dances for Arjun the moment he wakes up in the morning, even before he has brushed.", "kind": "rule"},
            {"text": "Maya dances for Arjun the moment he wakes up in the morning, before brushing or washing.", "kind": "fact"},
        ]
        kept, events, merged = tidy_canon(canon)
        self.assertEqual(len(events), 1)
        self.assertEqual(merged, 1)
        self.assertEqual([k["kind"] for k in kept], ["rule", "rule"], "a rule absorbing a fact stays a rule")
        self.assertIn("before brushing or washing", kept[1]["text"], "the newer wording is kept")


class InTheChat(AgentHarness):
    async def test_a_near_copy_is_answered_again(self):
        replies = iter([A, B, FRESH])
        self.atlas.reply = lambda body: json.dumps({"say": next(replies, FRESH), "actions": [], "done": True})
        chat = await self.new_chat(persona="You are Maya.")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Good night"})
        await self.settle(chat)
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "*the next morning*"})
        view = await self.settle(chat)
        said = [m["content"].get("say") for m in view["messages"] if m["role"] == "assistant"]
        self.assertEqual(said, [A, FRESH], "the near-copy was not stored; the retry answered")
        notes = [m["content"]["text"] for m in view["messages"] if m["role"] == "note" and m["content"].get("kind") == "repair"]
        self.assertIn("opened exactly the way", notes[-1])

    async def test_a_character_does_not_open_the_way_another_just_did(self):
        lines = iter([json.dumps({"say": "It feels like a dream, Sam, honestly so light today.", "to": "all"}),
                      json.dumps({"say": "It feels like a dream, Sam, I agree with every word of that.", "to": "all"}),
                      json.dumps({"say": "No way, I'm exhausted. Can we order food first?", "to": "all"})])
        asks = []

        def reply(body):
            asks.append(body)
            return next(lines, json.dumps({"say": "Something else entirely now.", "to": "all"}))

        self.atlas.reply = reply
        chat = await self.new_chat(cast=[{"name": "Maya", "persona": "Maya."}, {"name": "Riya", "persona": "Riya."}])
        response = await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 1, "makes": 0})
        self.assertEqual(response.status_code, 202, response.text)
        view = await self.settle(chat)
        said = [line["say"] for m in view["messages"] if m["role"] == "assistant" for line in m["content"].get("lines") or []]
        self.assertEqual(len(said), 2)
        self.assertIn("exhausted", said[1], "the second character was asked again and said something new")
        self.assertTrue(any("opened with the same words" in str(b["messages"][-1]["content"]) for b in asks))

    async def test_tidy_cleans_an_existing_memory(self):
        chat = await self.new_chat(persona="You are Maya.", name="Maya")
        await self.http.put(f"/v1/agent/sessions/{chat}/memory", json={"canon": [
            {"text": "Maya has filmed and delivered the one-minute video of herself dancing.", "kind": "fact"},
            {"text": "Maya dances for Arjun the moment he wakes up, even before he has brushed.", "kind": "rule"},
            {"text": "Maya dances for Arjun the moment he wakes up, before brushing or anything.", "kind": "rule"},
            {"text": "Maya always dresses before Arjun's father comes home.", "kind": "rule"}]})
        out = (await self.http.post(f"/v1/agent/sessions/{chat}/memory/tidy")).json()
        self.assertEqual(out["tidied"][0], {"name": "Maya", "before": 4, "after": 2, "moved_to_story": 1, "merged": 1})
        self.assertIn("Maya has filmed and delivered the one-minute video of herself dancing.", out["story"])


if __name__ == "__main__":
    unittest.main()
