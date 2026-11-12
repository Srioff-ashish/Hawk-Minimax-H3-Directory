"""Chat across Atlas and OpenRouter: provider prefixes, id spelling, failover on a provider's own failure."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hawk_api.atlas import AtlasClient, AtlasError
from hawk_api.llm_router import ChatRouter, split_provider


class Fake(AtlasClient):
    def __init__(self, url, models, fail=None):
        super().__init__(url, "key")
        self.served = models
        self.fail = fail or {}
        self.calls = []

    async def list_models(self, refresh=False):
        return [{"id": m, "vision": False} for m in self.served]

    async def chat(self, model, messages, **kwargs):
        self.calls.append(model)
        if model in self.fail:
            raise AtlasError(self.fail[model])
        return f"{self.service}:{model}", {}


def pair(atlas_fail=None, router_fail=None):
    atlas = Fake("https://api.atlascloud.ai/v1", ["deepseek-ai/deepseek-v4.1-flash", "xai/grok-4.6"], atlas_fail)
    openrouter = Fake("https://openrouter.ai/api/v1", ["deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash:free",
                                                       "x-ai/grok-4.6"], router_fail)
    return atlas, openrouter


class Router(unittest.IsolatedAsyncioTestCase):
    def test_split(self):
        self.assertEqual(split_provider("openrouter:x-ai/grok-4.6"), ("openrouter", "x-ai/grok-4.6"))
        self.assertEqual(split_provider(" Atlas : xai/grok-4.6"), ("atlas", "xai/grok-4.6"))
        self.assertEqual(split_provider("deepseek/deepseek-v4-pro"), ("", "deepseek/deepseek-v4-pro"))

    async def test_an_id_is_sent_in_the_providers_own_spelling(self):
        atlas, openrouter = pair()
        router = ChatRouter(openrouter, atlas)
        text, _ = await router.chat("deepseek-ai/deepseek-v4.1-flash", [])
        self.assertEqual(text, "openrouter:deepseek/deepseek-v4.1-flash", "the plain id, not the :free variant")

    async def test_a_prefix_picks_the_provider(self):
        atlas, openrouter = pair()
        router = ChatRouter(openrouter, atlas)
        text, _ = await router.chat("atlas:x-ai/grok-4.6", [])
        self.assertEqual(text, "atlas:xai/grok-4.6")

    async def test_a_rate_limit_moves_the_call_to_the_other_provider(self):
        atlas, openrouter = pair(router_fail={"deepseek/deepseek-v4.1-flash": "OpenRouter error 429 for x: slow down"})
        router = ChatRouter(openrouter, atlas)
        text, _ = await router.chat("deepseek-ai/deepseek-v4.1-flash", [])
        self.assertEqual(text, "atlas:deepseek-ai/deepseek-v4.1-flash")

    async def test_a_refusal_or_truncation_is_not_moved(self):
        for message in ("The model refused: no", "The model's reply was cut off at the token cap after 9 characters",
                        "OpenRouter error 400 for x: bad request"):
            with self.subTest(message=message):
                atlas, openrouter = pair(router_fail={"x-ai/grok-4.6": message})
                router = ChatRouter(openrouter, atlas)
                with self.assertRaises(AtlasError):
                    await router.chat("x-ai/grok-4.6", [])
                self.assertEqual(atlas.calls, [])

    async def test_no_second_key_means_no_failover(self):
        atlas, openrouter = pair(router_fail={"x-ai/grok-4.6": "OpenRouter error 503 for x: down"})
        atlas.api_key = ""
        router = ChatRouter(openrouter, atlas)
        with self.assertRaises(AtlasError):
            await router.chat("x-ai/grok-4.6", [])

    async def test_a_model_the_other_provider_does_not_serve_is_not_moved(self):
        atlas, openrouter = pair(router_fail={"x-ai/grok-4.6": "OpenRouter error 429"})
        atlas.served = ["moonshotai/kimi-k2.6"]
        router = ChatRouter(openrouter, atlas)
        with self.assertRaises(AtlasError):
            await router.chat("x-ai/grok-4.6", [])

    async def test_errors_name_the_right_provider(self):
        client = AtlasClient("https://openrouter.ai/api/v1", "")
        with self.assertRaises(AtlasError) as caught:
            await client.chat("x", [])
        self.assertIn("OpenRouter", str(caught.exception))

    async def test_other_attributes_read_from_the_primary(self):
        atlas, openrouter = pair()
        router = ChatRouter(openrouter, atlas)
        self.assertIn("openrouter", router.base_url)
        self.assertEqual(router.service, "openrouter")


if __name__ == "__main__":
    unittest.main()
