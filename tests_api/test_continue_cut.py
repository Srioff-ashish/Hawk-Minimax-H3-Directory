"""A JSON reply cut off at the token cap is kept and continued, instead of thrown away whole."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.atlas import AtlasClient, AtlasError, AtlasTruncated, CONTINUE_PROMPT, _stitch  # noqa: E402


def _prose(seed: int, words: int) -> str:
    """Varied words, like a real plan: a repeated phrase would be (rightly) taken for a model stuck in a loop."""
    import random
    rng = random.Random(seed)
    vocab = ("she turns toward the window light falls across rain street market steam rises camera slowly pushes "
             "close on hands coffee cup trembles neon sign flickers crowd parts footsteps echo night city").split()
    return " ".join(rng.choice(vocab) + str(rng.randint(0, 99)) for _ in range(words))


def _reply(content: str, finish: str = "stop", tokens: int = 10) -> dict:
    return {"id": "g", "model": "m", "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": tokens}}


class ContinueCutTest(unittest.TestCase):
    def _run(self, replies: list[dict], **kwargs):
        sent: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=replies[len(sent) - 1])

        async def go():
            client = AtlasClient("https://openrouter.ai/api/v1", "k")
            client._pool = (asyncio.get_running_loop(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            return await client.chat("m", [{"role": "user", "content": "plan"}], **kwargs)

        return asyncio.run(go()), sent

    def test_a_cut_off_plan_is_continued_and_joined(self):
        whole = json.dumps({"segments": [{"prompt": f"shot {i} " + _prose(i, 40)} for i in range(30)]})
        cut = len(whole) // 2
        (text, usage), sent = self._run(
            [_reply(whole[:cut], "length", 100), _reply("```json\n" + whole[cut - 30:] + "\n```", "stop", 90)],
            continue_cut=2)
        self.assertEqual(json.loads(text), json.loads(whole), "the pieces join into the plan, overlap and fence dropped")
        self.assertEqual(usage["completion_tokens"], 190, "both calls are counted")
        second = sent[1]
        self.assertEqual(second["messages"][-2], {"role": "assistant", "content": whole[:cut]},
                         "the model sees what it already wrote")
        self.assertEqual(second["messages"][-1]["content"], CONTINUE_PROMPT)
        self.assertNotIn("response_format", second, "JSON mode would make it start a new object")

    def test_a_reply_cut_twice_is_continued_twice(self):
        whole = json.dumps({"text": _prose(99, 400), "n": 1})
        a, b = len(whole) // 3, 2 * len(whole) // 3
        (text, _), sent = self._run(
            [_reply(whole[:a], "length"), _reply(whole[a:b], "length"), _reply(whole[b:])], continue_cut=2)
        self.assertEqual(json.loads(text), json.loads(whole))
        self.assertEqual(len(sent), 3)

    def test_without_continue_cut_it_fails_as_before(self):
        with self.assertRaises(AtlasTruncated) as caught:
            self._run([_reply('{"a": "unfinis', "length")])
        self.assertIn("cut off at the token cap", str(caught.exception))
        self.assertEqual(caught.exception.partial, '{"a": "unfinis')

    def test_a_reply_that_is_looping_is_not_continued(self):
        loop = '{"say": "' + "haha " * 3000
        with self.assertRaises(AtlasTruncated) as caught:
            self._run([_reply(loop, "length")], continue_cut=2)
        self.assertIn("repeating itself", str(caught.exception))

    def test_still_cut_after_every_continuation_moves_on(self):
        with self.assertRaises(AtlasTruncated) as caught:
            self._run([_reply('{"a": "one ' + "x" * 10, "length"), _reply(" two three", "length"),
                       _reply(" four five", "length")], continue_cut=2)
        self.assertIn("after 2 continuations", str(caught.exception))
        self.assertIn("cut off at the token cap", str(caught.exception), "the router still never moves it")

    def test_a_continuation_that_does_not_join_is_an_error(self):
        with self.assertRaises(AtlasError):
            self._run([_reply('{"a": "one two', "length"), _reply("Sorry, here is a summary instead.")],
                      continue_cut=2)

    def test_a_model_that_starts_over_with_a_complete_object_is_accepted(self):
        whole = json.dumps({"a": "full answer"})
        (text, _), _ = self._run([_reply('{"a": "full ans', "length"), _reply(whole)], continue_cut=1)
        self.assertEqual(json.loads(text), {"a": "full answer"})

    def test_stitch_without_overlap_just_joins(self):
        self.assertEqual(_stitch('{"a": "on', 'e"}'), '{"a": "one"}')


if __name__ == "__main__":
    unittest.main()
