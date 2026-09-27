"""Which service answers a chat call, and what a blank field means.

python -m unittest discover -s tests_api -p 'test_llm_settings.py'
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.config import LLMSettings, LLMSettingsStore, Settings  # noqa: E402

TOKEN = "t" * 24


class TheStoredChoice(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.store = LLMSettingsStore(self.dir)

    def test_a_blank_value_defers_rather_than_overwriting(self):
        # Studio sends every field, so a box the user never filled arrives blank. Kept as a value it would
        # replace the default it was meant to defer to: an empty URL reaching AtlasClient becomes Atlas's
        # own, which is how "OpenRouter" ends up calling Atlas with an OpenRouter key.
        self.store.save({"llm_provider": "openrouter", "openrouter_url": "   ", "agent_model_override": ""})
        self.assertEqual(self.store.stored(), {"llm_provider": "openrouter"})
        resolved = self.store.resolve(LLMSettings())
        self.assertEqual(resolved.openrouter_url, LLMSettings.openrouter_url, "the default survives a blank")
        self.assertEqual(resolved.llm_provider, "openrouter")

    def test_a_blank_clears_a_value_that_was_there(self):
        self.store.save({"agent_model_override": "deepseek/deepseek-chat"})
        self.assertEqual(self.store.stored()["agent_model_override"], "deepseek/deepseek-chat")
        self.store.save({"agent_model_override": ""})
        self.assertNotIn("agent_model_override", self.store.stored(), '"" means back to the pod\'s own choice')

    def test_an_unknown_setting_is_named_rather_than_ignored(self):
        with self.assertRaises(ValueError) as caught:
            self.store.save({"llm_privider": "atlas"})
        self.assertIn("llm_privider", str(caught.exception))

    def test_a_hint_identifies_a_key_without_being_one(self):
        # The panel has to show that a key is stored without being able to send it back.
        self.store.save({"openrouter_api_key": "sk-or-v1-abcdefghijkl"})
        hint = self.store.hints()["openrouter_api_key_hint"]
        self.assertNotIn("sk-or-v1-abcdefghijkl", hint, "a hint must not contain the key")
        self.assertTrue(hint.startswith("sk-") and hint.endswith("jkl"), hint)
        self.assertEqual(self.store.hints()["atlas_api_key_override_hint"], "", "no key, nothing to hint at")

    def test_every_field_round_trips(self):
        values = {name: "x" for name in LLMSettingsStore.FIELDS if name != "llm_provider"}
        values["llm_provider"] = "openrouter"
        self.store.save(values)
        self.assertEqual(LLMSettingsStore(self.dir).stored(), values, "including agent_summary_model_override")


class WhichServiceAnswers(unittest.IsolatedAsyncioTestCase):
    """HawkService.atlas, which is read fresh on every call so Studio needs no restart."""

    def service(self, **over):
        from hawk_api.jobs import HawkService

        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        settings = Settings(token=TOKEN, data_dir=self.dir, **over)
        return HawkService(settings, store=object(), comfy=object())

    def test_atlas_by_default(self):
        service = self.service(atlas_api_key="atlas-key")
        self.assertEqual(service.atlas.api_key, "atlas-key")
        self.assertIn("atlascloud", service.atlas.base_url)

    def test_choosing_openrouter_switches_url_and_key(self):
        service = self.service(atlas_api_key="atlas-key")
        service.llm_settings.save({"llm_provider": "openrouter", "openrouter_api_key": "sk-or-1"})
        self.assertEqual(service.atlas.api_key, "sk-or-1")
        self.assertIn("openrouter", service.atlas.base_url)

    def test_an_openrouter_only_pod_does_not_call_atlas_with_no_key(self):
        # A runtime given OPENROUTER_API_KEY and no ATLAS_API_KEY used to reach Atlas unauthenticated and
        # then report itself as having no key at all.
        service = self.service(openrouter_api_key="sk-or-2")
        self.assertEqual(service.atlas.api_key, "sk-or-2")
        self.assertIn("openrouter", service.atlas.base_url)
        self.assertTrue(service.atlas.configured)

    def test_a_new_key_is_not_served_the_old_accounts_models(self):
        # One client per (url, key): a client caches its model list, so rewriting the key on a live one
        # went on answering with the previous account's models until that cache expired.
        service = self.service(atlas_api_key="first")
        before = service.atlas
        before._models = (9e9, [{"id": "first-account-model"}])
        service.llm_settings.save({"atlas_api_key_override": "second"})
        service.forget_llm_clients()
        after = service.atlas
        self.assertEqual(after.api_key, "second")
        self.assertIsNot(after, before, "a different key is a different client")
        self.assertEqual(after._models[1], [], "and it starts with no cached models")

    def test_the_same_settings_reuse_one_client(self):
        service = self.service(atlas_api_key="atlas-key")
        self.assertIs(service.atlas, service.atlas, "a client per call would drop every cache each time")


class TheModelListFromEitherProvider(unittest.IsolatedAsyncioTestCase):
    """list_models reads one shape for Atlas and another for OpenRouter, and must not lose either."""

    #: OpenRouter's /models, trimmed to the fields that are read. Its modalities live under "architecture"
    #: and its prices are strings of USD per token.
    OPENROUTER = {"data": [
        {"id": "anthracite-org/magnum-v4-72b", "name": "Magnum v4 72B", "context_length": 32768,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "pricing": {"prompt": "0.0000019", "completion": "0.0000022"}},
        {"id": "google/gemini-pro-1.5", "name": "Gemini Pro 1.5", "context_length": 2000000,
         "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
         "pricing": {"prompt": "0.00000125", "completion": "0.000005", "input_cache_read": "0.0000003"}},
        {"id": "some/thing-draws-pictures", "name": "Image only", "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
         "pricing": {"prompt": "0.00001", "completion": "0"}},
    ]}
    #: Atlas's /models: the same information, one level up.
    ATLAS = {"data": [
        {"id": "xai/grok-4.6", "name": "Grok 4.6", "context_length": 256000,
         "input_modalities": ["text", "image"], "output_modalities": ["text"],
         "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
    ]}

    async def models(self, payload):
        import httpx

        from hawk_api.atlas import AtlasClient

        client = AtlasClient("https://openrouter.ai/api/v1", "sk-or-test")

        class Response:
            status_code = 200

            @staticmethod
            def json():
                return payload

        class Http:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def get(self, *_args, **_kwargs):
                return Response()

        original = httpx.AsyncClient
        httpx.AsyncClient = lambda *a, **k: Http()
        try:
            return await client.list_models()
        finally:
            httpx.AsyncClient = original

    async def test_an_openrouter_vision_model_is_not_reported_blind(self):
        # The one that mattered: inspect_image only offers a model it believes can see, and every
        # OpenRouter model looked blind because the modalities are nested a level down.
        found = {m["id"]: m for m in await self.models(self.OPENROUTER)}
        self.assertTrue(found["google/gemini-pro-1.5"]["vision"], "its inputs include image")
        self.assertFalse(found["anthracite-org/magnum-v4-72b"]["vision"], "and this one's do not")

    async def test_a_model_that_does_not_answer_in_text_is_left_out(self):
        found = {m["id"] for m in await self.models(self.OPENROUTER)}
        self.assertNotIn("some/thing-draws-pictures", found, "a chat model has to reply in text")

    async def test_prices_come_through_as_dollars_per_token(self):
        found = {m["id"]: m for m in await self.models(self.OPENROUTER)}
        gemini = found["google/gemini-pro-1.5"]
        self.assertEqual((gemini["price_in"], gemini["price_out"]), (0.00000125, 0.000005))
        self.assertEqual(gemini["price_cache"], 0.0000003, "a cached prefix bills lower")
        self.assertEqual(found["anthracite-org/magnum-v4-72b"]["price_cache"], 0.0000019,
                         "no cache price means cached tokens bill in full")

    async def test_the_atlas_shape_still_reads(self):
        found = {m["id"]: m for m in await self.models(self.ATLAS)}
        self.assertTrue(found["xai/grok-4.6"]["vision"], "top-level modalities must keep working")
        self.assertEqual(found["xai/grok-4.6"]["context"], 256000)


class WhichServiceServesTheModel(unittest.IsolatedAsyncioTestCase):
    """OpenRouter's provider routing: naming a model there does not say who runs it, or at what weights."""

    def setUp(self):
        from hawk_api import atlas

        self.atlas = atlas

    def test_the_default_preset_buys_quality_first_then_price(self):
        # The order matters: sorting by price across every weight is how a call lands on an int4 copy.
        # The quantisation filter is the floor, and "sort" only chooses among what already cleared it.
        block = self.atlas.routing_block("balanced")
        self.assertEqual(block["sort"], "price")
        self.assertEqual(block["quantizations"], ["bf16", "fp16", "fp8"])
        self.assertNotIn("int4", block["quantizations"])
        self.assertTrue(block["require_parameters"], "a service that drops response_format answers in prose")
        self.assertTrue(block["allow_fallbacks"], "one service being down must not fail the call")

    def test_a_typo_falls_back_to_filtering_rather_than_to_none(self):
        self.assertEqual(self.atlas.routing_block("chepest"), self.atlas.routing_block("balanced"),
                         "a mistyped setting must not quietly re-open the cheapest weights")

    def test_asking_for_openrouters_own_choice_sends_nothing(self):
        self.assertIsNone(self.atlas.routing_block("default"), "no block means no provider field at all")

    async def test_atlas_is_never_sent_a_routing_block(self):
        # Atlas serves its own models, so there is nothing to choose between and the field would be noise
        # on a request it cannot act on.
        service = WhichServiceAnswers.service(self, atlas_api_key="atlas-key")
        self.assertIsNone(service.atlas.routing)

    async def test_openrouter_is(self):
        service = WhichServiceAnswers.service(self, atlas_api_key="atlas-key")
        service.llm_settings.save({"llm_provider": "openrouter", "openrouter_api_key": "sk-or-1"})
        self.assertEqual(service.atlas.routing, self.atlas.routing_block("balanced"))

    async def test_changing_the_routing_is_a_different_client(self):
        # The routing travels in the request, not the connection, but a client caches a model list and the
        # cache key has to tell two settings apart or a change would not take.
        service = WhichServiceAnswers.service(self, atlas_api_key="atlas-key")
        service.llm_settings.save({"llm_provider": "openrouter", "openrouter_api_key": "sk-or-1"})
        balanced = service.atlas
        service.llm_settings.save({"openrouter_routing": "quality"})
        service.forget_llm_clients()
        self.assertEqual(service.atlas.routing["quantizations"], ["bf16", "fp16"])
        self.assertIsNot(service.atlas, balanced)

    async def test_the_block_rides_on_the_request_and_is_dropped_when_it_matches_nothing(self):
        import httpx

        sent, replies = [], [(404, '{"error":{"message":"No allowed providers are available"}}'),
                             (200, '{"choices":[{"message":{"content":"{}"}}],"usage":{}}')]

        class Response:
            def __init__(self, status, text):
                self.status_code, self.text = status, text

            def json(self):
                return json.loads(self.text)

        class Http:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def post(self, _url, headers=None, json=None):
                sent.append(dict(json))
                return Response(*replies.pop(0))

        client = self.atlas.AtlasClient("https://openrouter.ai/api/v1", "sk-or-test",
                                        routing=self.atlas.routing_block("quality"))
        original = httpx.AsyncClient
        httpx.AsyncClient = lambda *a, **k: Http()
        try:
            await client.chat("some/model", [{"role": "user", "content": "hi"}])
        finally:
            httpx.AsyncClient = original
        self.assertEqual(sent[0]["provider"]["quantizations"], ["bf16", "fp16"], "filtered on the first try")
        self.assertNotIn("provider", sent[1],
                         "nothing cleared the filter, so the retry runs unfiltered rather than failing")
