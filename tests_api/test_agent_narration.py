"""Narration: a line about the scene rather than a line spoken to the characters.

"we got drunk and we were intoxicated" and "you went to sleep and wake up next morning" state what has
happened. Every user message used to reach the model as the same "USER: ..." line, so a character answered
those as though they had been said to it -- asking whether you really got drunk instead of carrying on from
the morning after.

python -m unittest discover -s tests_api -p 'test_agent_narration.py'
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_agent import AgentHarness, last_user  # noqa: E402  (fixtures, reused not copied)


class NarrationRatherThanSpeech(AgentHarness):
    """What the model is handed for a narrated message, and what still counts as ordinary speech."""

    async def say(self, chat: str, text: str, **body) -> tuple[dict, str]:
        """Send one message, let the turn finish, and return (what was stored, the newest user turn sent)."""
        before = len(self.atlas.requests)
        sent = await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": text, **body})
        self.assertEqual(sent.status_code, 202, sent.text)
        await self.settle(chat)
        calls = self.atlas.requests[before:]
        self.assertTrue(calls, "the message should have reached the model")
        return sent.json()["message"]["content"], last_user(calls[0])

    async def test_a_message_wrapped_in_asterisks_is_the_scene(self):
        chat = await self.new_chat(persona="A blunt Mumbai ad-film director.")
        stored, prompt = await self.say(chat, "*we got drunk and we were intoxicated*")
        self.assertTrue(stored["narration"], "the wrapper marks it as the scene")
        self.assertEqual(stored["text"], "we got drunk and we were intoxicated",
                         "the asterisks are the marker, so they are not part of what was narrated")
        self.assertIn("NARRATION (", prompt, "and the model is told which kind of line this is")
        self.assertIn("we got drunk and we were intoxicated", prompt, "with the narration itself intact")
        self.assertNotIn("USER: we got drunk", prompt, "sent as speech it reads as something to answer")

    async def test_the_model_is_told_not_to_answer_it(self):
        # The instruction is the whole point: without it "NARRATION" is just a different prefix on a line
        # the model still treats as its cue to reply.
        chat = await self.new_chat()
        _, prompt = await self.say(chat, "*you went to sleep and woke up next morning*")
        for phrase in ("true from now on", "Do not answer it as though the user said it to you",
                       "carry on in character from the new situation"):
            self.assertIn(phrase, prompt, f"the narration line should say {phrase!r}")

    async def test_the_composer_toggle_narrates_without_asterisks(self):
        chat = await self.new_chat()
        stored, prompt = await self.say(chat, "you went to sleep and wake up next morning", narration=True)
        self.assertTrue(stored["narration"], "the toggle is the other way to say it")
        self.assertEqual(stored["text"], "you went to sleep and wake up next morning",
                         "nothing was wrapped, so nothing is stripped")
        self.assertIn("NARRATION (", prompt, "and it reaches the model as the scene either way")

    async def test_an_asterisk_inside_a_sentence_is_still_speech(self):
        # Starring single words is an ordinary roleplay habit, and reading it as the scene would silently
        # stop the character answering perfectly normal messages.
        chat = await self.new_chat()
        for text in ("*sigh* theek hai, aur tum?", "2 * 3 is 6", "a * b * c"):
            stored, prompt = await self.say(chat, text)
            self.assertNotIn("narration", stored, f"{text!r} is a line spoken to the character")
            self.assertEqual(stored["text"], text, "and it reaches the model exactly as typed")
            self.assertIn(f"USER: {text}", prompt, f"{text!r} should be sent as speech")

    async def test_a_message_that_is_only_a_marker_is_left_alone(self):
        chat = await self.new_chat()
        stored, _ = await self.say(chat, "**")
        self.assertNotIn("narration", stored, "there is no scene between those asterisks")
        self.assertEqual(stored["text"], "**", "and nothing was stripped out of it")

    async def test_a_whisper_is_speech_even_when_it_is_wrapped(self):
        # "@Name ..." is a line said to one character in private. Reading it as the scene as well would lose
        # the one thing that makes it private.
        chat = (await self.http.post("/v1/agent/sessions", json={
            "cast": [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}],
            "whispers": True})).json()
        members = {m["name"]: m["id"] for m in chat["cast"]}
        stored, prompt = await self.say(chat["id"], "@Maya *emerald best hai*")
        self.assertEqual(stored["private_to"], members["Maya"], "it is still a whisper to Maya")
        self.assertNotIn("narration", stored, "and a whisper is something said, not something that happened")
        self.assertIn("whispering privately to Maya", prompt, "so it is sent as the private line it is")


if __name__ == "__main__":
    unittest.main()
