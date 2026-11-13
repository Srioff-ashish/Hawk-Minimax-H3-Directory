"""Asking a character to add someone to the chat adds them (shaped on a live chat where "include her in chat
group" was answered in voice, saved to memory as done, and a later set_persona overwrote the lead)."""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.agent import asks_cast_change, is_chat_mechanics, named_engine, tidy_canon  # noqa: E402
from test_agent import AgentHarness, tiny_png  # noqa: E402

PROSE = "deepseek-ai/deepseek-v4.1-flash"
DIRECTOR = "xai/grok-4.6"


class Detect(unittest.TestCase):
    def test_cast_requests(self):
        for text in ("i fuck her for 60 min ..include her i chat group", "creaye persona ,add character ,set persona",
                     "set avataar correctly", "rebel avaaar theek kar apna", "add Priya to the chat",
                     "usko group mein daal do", "remove Sakshi from the chat", "2310bb217002 is rebel avatar..correct it",
                     "change you name in chat Anjali", "update persoa", "update your ame in persona Anal",
                     "rename Mrs Khanna to Anjali"):
            with self.subTest(text=text):
                self.assertTrue(asks_cast_change(text))
        for text in ("next day .i wake up", "tu bata teri hawas khatam hui", "bring me tea", "toh kitne lekar aayi",
                     "Mrs khanna what is your first name??"):
            with self.subTest(text=text):
                self.assertFalse(asks_cast_change(text))

    def test_chat_mechanics_are_not_memory(self):
        for text in ("Mrs. Khanna has now been added to the chat group.", "Her avatar asset is 25309665bef5.",
                     "Her name in chat is now Anal, replacing Anjali.",
                     "Her avatar is now created and set in the group alongside Rebel's and Sakshi's avatars.",
                     "Her avatar is the image the user attached in chat.", "Her image in the group picture is 2e8f250a35d0.",
                     "Her real, unedited face is now her profile photo in the mohalla group."):
            with self.subTest(text=text):
                self.assertTrue(is_chat_mechanics(text))
        for text in ("Must bring Mrs. Khanna to the master's bed.", "Calls the user boss.",
                     "Rebel's best friend Sakshi lives next door.", "Her chinal colour is Royal Cobalt.",
                     "Every morning at 7 she sends a selfie.", "Was born in 1980 in Ludhiana."):
            with self.subTest(text=text):
                self.assertFalse(is_chat_mechanics(text))
        kept, _, _ = tidy_canon([{"text": "Mrs. Khanna has now been added to the chat group.", "kind": "fact"},
                                 {"text": "Calls the user boss.", "kind": "rule"}])
        self.assertEqual([k["text"] for k in kept], ["Calls the user boss."])


class Engines(unittest.TestCase):
    def test_named_engines(self):
        for text, engine in (("Use engine krea2.", "krea2"), ("with engine krea2", "krea2"), ("use Krea 2 for it", "krea2"),
                             ("Same scene. Use engine zimage.", "zimage"), ("z-image se banao", "zimage"),
                             ("Use engine seedream.", "seedream"), ("seedream flash please", "seedream-flash"),
                             ("engine: turbo", "turbo"), ("try qwen 2.1", "qwen21"), ("make it on ph3", "ph3")):
            with self.subTest(text=text):
                self.assertEqual(named_engine(text), engine)
        for text in ("make the best photo", "a fast one", "local market at night", "turbo base video", "next morning"):
            with self.subTest(text=text):
                self.assertEqual(named_engine(text), "")


class InTheChat(AgentHarness):
    async def test_a_new_name_joins_and_the_lead_stays(self):
        def reply(body):
            if not any(m["role"] == "assistant" for m in body["messages"]):
                return json.dumps({"say": "Aao Khanna ji.", "actions": [
                    {"tool": "set_persona", "args": {"name": "Mrs. Khanna", "persona": "A neighbour."}}]})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Rebel", "persona": "Rebel."}, {"name": "Sakshi", "persona": "Sakshi."}],
            "prose_model": PROSE, "model": DIRECTOR})).json()["id"]
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "include her in the chat group"})
        view = await self.settle(chat)
        self.assertEqual([m["display_name"] for m in view["session"]["cast"]], ["Rebel", "Sakshi", "Mrs. Khanna"])
        self.assertEqual(view["session"]["cast"][0]["persona"], "Rebel.", "the lead was not overwritten")
        first = next(r for r in self.atlas.requests if str(r["messages"][0]["content"]).startswith("You are Hawk"))
        self.assertEqual(first["model"], DIRECTOR, "a cast change goes to the model that calls tools")

    async def test_ordinary_talk_still_uses_the_prose_model(self):
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Rebel", "persona": "Rebel."}], "prose_model": PROSE, "model": DIRECTOR})).json()["id"]
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "next day, i wake up"})
        await self.settle(chat)
        first = next(r for r in self.atlas.requests if str(r["messages"][0]["content"]).startswith("You are Hawk"))
        self.assertEqual(first["model"], PROSE)

    async def test_editing_the_cast_without_an_avatar_keeps_it(self):
        chat = await self.new_chat(cast=[{"name": "Rebel", "persona": "Rebel."}])
        view = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()["session"]
        rebel = view["cast"][0]["id"]
        uploaded = await self.http.post("/v1/assets", files={"files": ("face.png", tiny_png(), "image/png")})
        self.agent_avatar = uploaded.json()["assets"][0]["id"]
        await self.http.patch(f"/v1/agent/sessions/{chat}", json={"cast": [
            {"id": rebel, "name": "Rebel", "persona": "Rebel.", "avatar_asset_id": self.agent_avatar}]})
        await self.http.patch(f"/v1/agent/sessions/{chat}", json={"cast": [
            {"id": rebel, "name": "Rebel", "persona": "Rebel."}, {"name": "Sakshi", "persona": "S."}]})
        view = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()["session"]
        self.assertEqual(view["cast"][0]["avatar_asset_id"], self.agent_avatar)
        await self.http.patch(f"/v1/agent/sessions/{chat}", json={"cast": [
            {"id": rebel, "name": "Rebel", "persona": "Rebel.", "avatar_asset_id": ""}]})
        view = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()["session"]
        self.assertEqual(view["cast"][0]["avatar_asset_id"], "", "an empty id still removes it")

    async def test_pictures_without_an_owner_do_not_land_on_the_lead(self):
        uploaded = await self.http.post("/v1/assets", files={"files": ("face.png", tiny_png(), "image/png")})
        meu_face = uploaded.json()["assets"][0]["id"]
        calls = iter([[{"tool": "set_references", "args": {"asset_ids": [meu_face]}}],
                      [{"tool": "set_references", "args": {"asset_ids": [meu_face], "speaker": "Meu"}}]])

        def reply(body):
            actions = next(calls, [])
            return json.dumps({"say": "Ok.", "actions": actions, "done": not actions})

        self.atlas.reply = reply
        chat = await self.new_chat(cast=[{"name": "Randitya", "persona": "R."}, {"name": "Meu", "persona": "M."}])
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "set the reference pictures"})
        view = await self.settle(chat)
        cast = {m["display_name"]: m for m in view["session"]["cast"]}
        self.assertEqual(cast["Randitya"].get("refs") or [], [], "the lead keeps his own pictures")
        self.assertEqual(cast["Meu"]["refs"], [meu_face])
        errors = [m["content"].get("error") for m in view["messages"] if m["role"] == "tool"]
        self.assertIn("pass speaker", errors[0])

    async def test_a_renamed_character_keeps_their_former_names(self):
        chat = await self.new_chat(cast=[{"name": "Rebel", "persona": "R."}, {"name": "Mrs. Khanna", "persona": "K."}])
        khanna = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()["session"]["cast"][1]["id"]
        for new in ("Anjali", "Anal"):
            await self.http.patch(f"/v1/agent/sessions/{chat}", json={"cast": [
                {"id": "main", "name": "Rebel", "persona": "R."}, {"id": khanna, "name": new, "persona": "K."}]})
        member = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()["session"]["cast"][1]
        self.assertEqual(member["display_name"], "Anal")
        self.assertEqual(member["aka"], ["Mrs. Khanna", "Anjali"], "her old lines still find her face")
        from hawk_api.agent import find_member
        self.assertEqual(find_member([{"name": "Rebel"}, member], "Mrs. Khanna"), 1)

    async def test_a_named_engine_is_the_one_that_runs(self):
        calls = iter([[{"tool": "generate_image", "args": {"prompt": "a street at night", "engine": "auto"}}]])

        def reply(body):
            actions = next(calls, [])
            return json.dumps({"say": "Ok.", "actions": actions, "done": not actions})

        self.atlas.reply = reply
        chat = await self.new_chat(persona="You are Tara.", name="Tara")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "A street at night. Use engine zimage."})
        view = await self.settle(chat)
        tool = next(m["content"] for m in view["messages"] if m["role"] == "tool" and m["content"]["tool"] == "generate_image")
        self.assertEqual(tool["args"]["engine"], "zimage", "the director sent auto; the user's engine ran")

    async def test_renaming_takes_the_current_name(self):
        def reply(body):
            if not any(m["role"] == "assistant" for m in body["messages"]):
                return json.dumps({"say": "Ok.", "actions": [
                    {"tool": "set_persona", "args": {"speaker": "Rebel", "name": "Rebecca", "persona": "Rebel."}}]})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat(cast=[{"name": "Rebel", "persona": "Rebel."}, {"name": "Sakshi", "persona": "S."}])
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "change your persona name to Rebecca"})
        view = await self.settle(chat)
        self.assertEqual([m["display_name"] for m in view["session"]["cast"]], ["Rebecca", "Sakshi"])


if __name__ == "__main__":
    unittest.main()
