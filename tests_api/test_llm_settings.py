"""Which service answers a chat call, and what a blank field means.

python -m unittest discover -s tests_api -p 'test_llm_settings.py'
"""

from __future__ import annotations

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
