"""What a chat knows, as entities and relations: built free, kept across a restart, and readable.

python -m unittest discover -s tests_api -p 'test_agent_memory.py'
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.agent import AgentStore, read_summary_and_facts  # noqa: E402
from test_agent import AgentHarness, is_summary  # noqa: E402


def setting(harness, name, value):
    """Change one pod setting for the length of a test. Settings is a frozen dataclass on purpose, so this goes
    round the front door and puts the old value back."""
    settings = harness.agent.service.settings
    before = getattr(settings, name)
    object.__setattr__(settings, name, value)
    harness.addCleanup(object.__setattr__, settings, name, before)


class AGraphOfWhoIsWho(unittest.TestCase):
    """The store on its own: identity, contradiction and the walk. No HTTP, no model."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hawk_graph_test_")
        self.store = AgentStore(os.path.join(self.dir, "jobs.sqlite3"))

    def node(self, key, kind="person", label="", owner=""):
        return self.store.upsert_node("s", owner, key, kind, label or key)

    def test_a_new_fact_supersedes_the_old_one_and_the_old_one_is_kept(self):
        riya, tiya = self.node("member:1", label="Riya"), self.node("member:2", label="Tiya")
        self.store.put_edge("s", "", riya, tiya, "feels_toward", "feeling", "Riya trusts Tiya.", 0.8, 5)
        self.store.put_edge("s", "", riya, tiya, "relation_to", "relation", "Riya is Tiya's assistant.", 1.0, 6)
        self.store.put_edge("s", "", riya, tiya, "status_of", "feeling", "Riya no longer trusts Tiya.", -0.7, 9)

        live = {row["rel"]: row for row in self.store.all_edges("s", live_only=True)}
        self.assertIn("status_of", live, "the newest fact in a family is the live one")
        self.assertNotIn("feels_toward", live, "the one it contradicts is no longer live")
        self.assertIn("relation_to", live, "a fact in another family is untouched: they do not contradict")
        stale = [r for r in self.store.all_edges("s") if r["state"] == "stale"]
        self.assertEqual([r["fact"] for r in stale], ["Riya trusts Tiya."],
                         "history is kept so a character can say she used to trust me, and so it can be undone")

    def test_restating_a_fact_updates_it_rather_than_duplicating_it(self):
        riya, tiya = self.node("member:1", label="Riya"), self.node("member:2", label="Tiya")
        self.store.put_edge("s", "", riya, tiya, "feels_toward", "feeling", "Riya likes Tiya.", 0.5, 1)
        self.store.put_edge("s", "", riya, tiya, "feels_toward", "feeling", "Riya adores Tiya.", 0.9, 4)
        rows = self.store.all_edges("s")
        self.assertEqual(len(rows), 1, "the same triple twice is one fact, restated")
        self.assertEqual((rows[0]["fact"], rows[0]["weight"]), ("Riya adores Tiya.", 0.9))

    def test_one_characters_private_view_is_not_visible_to_another(self):
        mine = self.node("member:2", label="Tiya", owner="m1")
        me = self.node("member:1", label="Riya", owner="m1")
        self.store.put_edge("s", "m1", me, mine, "feels_toward", "feeling", "Riya secretly resents Tiya.", -0.8, 3)
        self.assertTrue(self.store.edges_around("s", ("", "m1"), [me]), "the owner sees her own view")
        self.assertEqual(self.store.edges_around("s", ("", "m2"), [me]), [],
                         "another character must not read a private feeling that was never said aloud")

    def test_forgetting_a_summary_drops_its_facts_and_revives_what_they_replaced(self):
        riya, tiya = self.node("member:1", label="Riya"), self.node("member:2", label="Tiya")
        self.store.put_edge("s", "", riya, tiya, "feels_toward", "feeling", "Riya trusts Tiya.", 0.8, 5)
        self.store.put_edge("s", "", riya, tiya, "status_of", "feeling", "Riya distrusts Tiya.", -0.7, 20)
        self.assertEqual(self.store.drop_facts_since("s", 20), 1)
        live = self.store.all_edges("s", live_only=True)
        self.assertEqual([r["fact"] for r in live], ["Riya trusts Tiya."],
                         "a fact from a forgotten reply goes, and its family is not left empty")

    def test_a_database_from_before_the_graph_still_opens(self):
        # The tables are created with IF NOT EXISTS on every open, so an existing pod gains them on next start.
        path = os.path.join(self.dir, "older.sqlite3")
        import sqlite3
        old = sqlite3.connect(path)
        old.executescript("CREATE TABLE agent_sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL,"
                          " created_at REAL NOT NULL, updated_at REAL NOT NULL);")
        old.commit()
        old.close()
        store = AgentStore(path)
        self.assertEqual(store.all_edges("s"), [], "a database written before the graph existed opens and is empty")
        self.assertTrue(store.upsert_node("s", "", "user", "person", "the user"), "and can then be written to")


class FactsFromCompaction(AgentHarness):
    """Extraction rides in the call the summariser already made: no extra request, and no new failure mode."""

    async def busy_chat(self, summary_reply) -> str:
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Riya", "persona": "director"}, {"name": "Tiya", "persona": "stylist"}],
            "adaptive": True})).json()["id"]
        self.agent.compact_tokens, self.agent.keep_messages = 1, 2

        def reply(body):
            if is_summary(body):
                return summary_reply
            return json.dumps({"lines": [{"speaker": "Riya", "say": "chalo"}], "actions": [], "done": True})

        self.atlas.reply = reply
        for text in ("pehla", "doosra", "teesra"):
            await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": text})
            await self.settle(chat)
        return chat

    async def test_a_compaction_writes_the_summary_and_the_facts_in_one_call(self):
        chat = await self.busy_chat(json.dumps({
            "summary": "Riya and Tiya argued over the rooftop shoot.",
            "facts": [{"s": "Riya", "r": "feels_toward", "o": "Tiya", "v": -0.6,
                       "f": "Riya resents how Tiya took over the shoot."}]}))
        session = self.agent.get_session(chat)
        self.assertEqual(session["summary"], "Riya and Tiya argued over the rooftop shoot.")
        facts = (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]
        self.assertIn("Riya resents how Tiya took over the shoot.", [f["fact"] for f in facts])
        summaries = [r for r in self.atlas.requests if is_summary(r)]
        self.assertTrue(summaries, "a summary should have been written")
        self.assertEqual(len({id(r) for r in summaries}), len(summaries))
        for request in summaries:
            self.assertIn("facts", request["messages"][0]["content"],
                          "the facts are asked for in the summariser's own call, not a second one")

    async def test_a_summary_model_that_ignores_the_json_still_produces_a_summary(self):
        # The property that makes this safe to ship: the old behaviour is the fallback, exactly.
        chat = await self.busy_chat("SUMMARY: they argued about the rooftop shoot.")
        session = self.agent.get_session(chat)
        self.assertEqual(session["summary"], "SUMMARY: they argued about the rooftop shoot.",
                         "an unparseable reply is the prose summary, as it always was")
        view = (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()
        invented = [f for f in view["facts"] if "argued" in f["fact"]]
        self.assertEqual(invented, [], "and it yields no facts rather than a broken one")

    async def test_a_fact_naming_a_relation_that_does_not_exist_is_dropped(self):
        chat = await self.busy_chat(json.dumps({
            "summary": "A summary.",
            "facts": [{"s": "Riya", "r": "vibes_with", "o": "Tiya", "f": "Riya vibes with Tiya."},
                      {"s": "Riya", "r": "relation_to", "o": "Tiya", "f": "Riya is Tiya's older sister."}]}))
        facts = [f["fact"] for f in (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]]
        self.assertNotIn("Riya vibes with Tiya.", facts, "the vocabulary is closed, so nothing contradicts nothing")
        self.assertIn("Riya is Tiya's older sister.", facts)

    async def test_a_fact_whose_object_is_the_wrong_kind_is_refused(self):
        # The fix for what the pod actually produced: "located_in :: Asset c7eab94fc3ea, portrait.png" and
        # "relation_to :: The plan has talent walking the parapet". The sentence reads fine and the relation is
        # real, so only the object's kind can catch it -- and the relation picks the family, which is what makes
        # a later fact supersede this one, so filing it wrong means it is never contradicted.
        chat = await self.busy_chat(json.dumps({
            "summary": "A summary.",
            "facts": [{"s": "Riya", "r": "relation_to", "o": "the rooftop parapet plan",
                       "f": "Riya is to the rooftop parapet plan."},
                      {"s": "Riya", "r": "made", "o": "c7eab94fc3ea", "f": "Riya made asset c7eab94fc3ea."}]}))
        facts = [f["fact"] for f in (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]]
        self.assertNotIn("Riya is to the rooftop parapet plan.", facts,
                         "relation_to is between people; a plan is not one, so the fact is refused not refiled")
        self.assertIn("Riya made asset c7eab94fc3ea.", facts, "and made does take an asset")

    async def test_a_characters_own_memory_does_not_write_a_private_copy_of_what_everyone_heard(self):
        # Measured on the pod: a three-message chat produced 32 facts, of which 20 were per-character copies of
        # the 12 that were said aloud. A character privately knowing what was said in front of everyone is not
        # private knowledge, it is just the conversation.
        chat = await self.busy_chat(json.dumps({
            "summary": "A summary.",
            "facts": [{"s": "Riya", "r": "relation_to", "o": "Tiya", "f": "Riya is Tiya's older sister."}]}))
        facts = (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]
        same = [f for f in facts if f["fact"] == "Riya is Tiya's older sister."]
        self.assertEqual(len(same), 1, "one fact said aloud should be stored once, not once per listener")
        self.assertEqual(same[0]["private_to"], "", "and it belongs to the room, not to one character")
        memory_calls = [r for r in self.atlas.requests
                        if "Write your own memory of the conversation" in str(r["messages"][0]["content"])]
        for call in memory_calls:
            self.assertNotIn('"facts"', str(call["messages"][0]["content"]),
                             "a character's own memory is prose again; it asks for no facts at all")

    async def test_the_fact_cap_holds_even_when_the_model_ignores_it(self):
        many = [{"s": "Riya", "r": "made", "o": "%012x" % n, "f": f"Riya made asset {n}."} for n in range(60)]
        chat = await self.busy_chat(json.dumps({"summary": "A summary.", "facts": many}))
        facts = (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]
        self.assertLessEqual(len([f for f in facts if "Riya made asset" in f["fact"]]), 20,
                             "a cheap model will invent facts given room, so the cap is enforced here too")

    async def test_a_wrong_fact_can_be_deleted_by_hand(self):
        chat = await self.busy_chat(json.dumps({
            "summary": "A summary.",
            "facts": [{"s": "Riya", "r": "relation_to", "o": "Tiya", "f": "Riya is Tiya's sister."}]}))
        facts = (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]
        wrong = next(f for f in facts if "sister" in f["fact"])
        left = (await self.http.delete(f"/v1/agent/sessions/{chat}/graph/facts/{wrong['id']}")).json()
        self.assertNotIn("Riya is Tiya's sister.", [f["fact"] for f in left["facts"]])
        self.assertEqual((await self.http.delete(f"/v1/agent/sessions/{chat}/graph/facts/{wrong['id']}")).status_code, 404)


class RecalledFactsInThePrompt(AgentHarness):
    async def test_growth_becomes_a_fact_without_any_extra_model_call(self):
        # A growth note already says who felt what about whom, so the relation is free.
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Riya", "persona": "director"}, {"name": "Tiya", "persona": "stylist"}],
            "adaptive": True})).json()["id"]
        grew = iter([True])

        def reply(body):
            if next(grew, False):
                return json.dumps({"lines": [{"speaker": "Riya", "say": "hmph"}], "actions": [], "done": True,
                                   "grow": [{"speaker": "Riya", "about": "Tiya", "note": "Tired of Tiya deciding."}]})
            return json.dumps({"lines": [{"speaker": "Riya", "say": "theek"}], "actions": [], "done": True})

        self.atlas.reply = reply
        calls_before = len(self.atlas.requests)
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "kya hua"})
        await self.settle(chat)
        facts = [f["fact"] for f in (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]]
        self.assertTrue(any("Tired of Tiya deciding." in f for f in facts), facts)
        self.assertEqual(len(self.atlas.requests) - calls_before, 1,
                         "one director call and nothing else: the graph write costs no tokens")

    async def test_a_private_feeling_never_reaches_another_characters_prompt(self):
        # The leak this very nearly shipped with: a growth note about another character is stored in the
        # character's own private feelings, so writing it as a fact "said aloud" would hand Riya's private view
        # of Tiya to Tiya through recall. Owned rows are the guard, and this is the test that holds it shut.
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Riya", "persona": "director"}, {"name": "Tiya", "persona": "stylist"}],
            "adaptive": True})).json()["id"]
        grew = iter([True])

        def reply(body):
            if next(grew, False):
                return json.dumps({"lines": [{"speaker": "Riya", "say": "hmph"}], "actions": [], "done": True,
                                   "grow": [{"speaker": "Riya", "about": "Tiya", "note": "Sick of Tiya deciding."}]})
            return json.dumps({"lines": [{"speaker": "Riya", "say": "ok"}], "actions": [], "done": True})

        self.atlas.reply = reply
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "kya hua"})
        await self.settle(chat)
        facts = (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]
        secret = next(f for f in facts if "Sick of Tiya deciding." in f["fact"])
        self.assertTrue(secret["private_to"], "a feeling about another character belongs to whoever feels it")

        session = self.agent.get_session(chat)
        others = [m["id"] for m in session["cast"] if m["name"] == "Tiya"]
        for member_id in others:
            self.assertNotIn("Sick of Tiya deciding.",
                             self.agent._recall_text(session, member_id, "Riya Tiya"),
                             "Tiya must not be told what Riya privately thinks of her")
        self.assertNotIn("Sick of Tiya deciding.", self.agent._recall_text(session, "", "Riya Tiya"),
                         "and the director already gets it through CHARACTER STATE, so recall must not repeat it")

    async def test_only_facts_about_who_the_turn_mentions_are_injected(self):
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Riya", "persona": "director"}, {"name": "Tiya", "persona": "stylist"}]})).json()["id"]
        session = self.agent.get_session(chat)
        self.agent._seed_graph(session)
        session = self.agent.get_session(chat)
        self.agent._write_fact(session, "", "Riya", "made", "a1b2c3d4e5f6", "Riya made the rooftop portrait.", 1.0, 1)
        self.agent._write_fact(session, "", "Tiya", "made", "f6e5d4c3b2a1", "Tiya made the lehenga still.", 1.0, 1)

        self.atlas.reply = lambda body: json.dumps({"lines": [{"speaker": "Riya", "say": "ok"}], "actions": [], "done": True})
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Riya, rooftop ready?"})
        await self.settle(chat)
        systems = "\n".join(m["content"] for m in self.atlas.requests[-1]["messages"] if m["role"] == "system")
        self.assertIn("Riya made the rooftop portrait.", systems, "a fact about who the turn names is recalled")
        self.assertNotIn("Tiya made the lehenga still.", systems,
                         "and one about someone the turn never mentions is not, or recall costs more than it saves")

    async def test_recall_stays_inside_its_token_budget(self):
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Riya", "persona": "director"}, {"name": "Tiya", "persona": "stylist"}]})).json()["id"]
        session = self.agent.get_session(chat)
        self.agent._seed_graph(session)
        session = self.agent.get_session(chat)
        for n in range(80):
            self.agent._write_fact(session, "", "Riya", "made", "%012x" % n,
                                   f"Riya made asset {n}, a long-ish sentence about it for the budget.", 1.0, n)
        setting(self, "agent_recall_tokens", 120)
        text = self.agent._recall_text(self.agent.get_session(chat), "", "Riya")
        self.assertTrue(text, "there is plenty to recall")
        self.assertLessEqual(len(text) // 4, 160, "the block must stay near its budget, not grow with the graph")

    async def test_recall_can_be_turned_off_without_losing_what_was_collected(self):
        chat = (await self.http.post("/v1/agent/sessions", json={"persona": "You are Maya."})).json()["id"]
        session = self.agent.get_session(chat)
        self.agent._seed_graph(session)
        session = self.agent.get_session(chat)
        self.agent._write_fact(session, "", "Maya", "made", "a1b2c3d4e5f6", "Maya made the rooftop portrait.", 1.0, 1)
        setting(self, "agent_graph_recall", False)
        self.assertEqual(self.agent._recall_text(self.agent.get_session(chat), "", "Maya rooftop"), "",
                         "the kill switch stops injection")
        setting(self, "agent_graph_recall", True)
        facts = (await self.http.get(f"/v1/agent/sessions/{chat}/graph")).json()["facts"]
        self.assertTrue(facts, "but nothing is thrown away, so turning it back on has data to work with")


class TheParserOnItsOwn(unittest.TestCase):
    def test_an_empty_reply_is_an_empty_summary_and_no_facts(self):
        self.assertEqual(read_summary_and_facts(""), ("", []))
        self.assertEqual(read_summary_and_facts(None), ("", []))

    def test_a_json_object_without_a_summary_is_treated_as_prose(self):
        raw = '{"facts": [{"s": "a", "r": "knows_about", "o": "b", "f": "c"}]}'
        self.assertEqual(read_summary_and_facts(raw), (raw, []),
                         "no summary key means the model was not answering our format, so nothing is trusted")


if __name__ == "__main__":
    unittest.main()
