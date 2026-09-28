"""A model id that survives switching provider.

python -m unittest discover -s tests_api -p 'test_models_llm.py'
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api import models_llm  # noqa: E402

#: The two providers as their /models calls describe them: the same models, spelled differently.
ATLAS = [
    {"id": "xai/grok-4.6", "vision": True, "price_in": 3e-6},
    {"id": "xai/grok-4.3", "vision": True, "price_in": 1e-6},
    {"id": "deepseek-ai/deepseek-v4.1-flash", "vision": False, "price_in": 1e-7},
]
OPENROUTER = [
    {"id": "x-ai/grok-4.6", "vision": True, "price_in": 3e-6},
    {"id": "deepseek/deepseek-v4.1-flash", "vision": False, "price_in": 1e-7},
    {"id": "anthracite-org/magnum-v4-72b", "vision": False, "price_in": 2e-6},
    {"id": "google/gemini-pro-1.5", "vision": True, "price_in": 1e-6},
]


class AModelIdOnEitherProvider(unittest.TestCase):
    def test_an_atlas_id_resolves_to_the_openrouter_spelling_of_the_same_model(self):
        # The whole defect in one line: these are one model, and only the vendor prefix differs.
        self.assertEqual(models_llm.resolve("director", "xai/grok-4.6", OPENROUTER), "x-ai/grok-4.6",
                         "an Atlas id should find the same model in OpenRouter's namespace")
        self.assertEqual(models_llm.resolve("director", "x-ai/grok-4.6", ATLAS), "xai/grok-4.6",
                         "and back again, so switching either way works")

    def test_a_vendor_prefix_that_differs_by_a_hyphen_is_still_the_same_model(self):
        self.assertEqual(models_llm.resolve("summary", "deepseek-ai/deepseek-v4.1-flash", OPENROUTER),
                         "deepseek/deepseek-v4.1-flash", "deepseek-ai/ and deepseek/ name one model")

    def test_a_role_can_be_given_the_image_requirement_for_one_call(self):
        # The planner is a text role right up until the plan has reference photos, and then the very same
        # call is multimodal. Asking a text-only model to read an image is not a weaker answer: the provider
        # refuses the request outright, which is how every plan with references came to fail.
        chain = "anthracite-org/magnum-v4-72b, x-ai/grok-4.6"
        self.assertEqual(models_llm.resolve("planner", chain, OPENROUTER, vision=True), "x-ai/grok-4.6",
                         "with photos attached the chain should skip the text-only head and take one that sees")

    def test_without_the_requirement_the_same_chain_keeps_its_cheaper_text_model(self):
        # The other half of the bargain: a plan with no references must not be pushed onto a vision model
        # it does not need, or the requirement would just be a more expensive default.
        chain = "anthracite-org/magnum-v4-72b, x-ai/grok-4.6"
        self.assertEqual(models_llm.resolve("planner", chain, OPENROUTER), "anthracite-org/magnum-v4-72b",
                         "with no photos the head of the chain is still the right answer")

    def test_a_variant_suffix_does_not_stop_a_match(self):
        listed = [{"id": "x-ai/grok-4.6:free", "vision": True, "price_in": 0.0}]
        self.assertEqual(models_llm.resolve("director", "xai/grok-4.6", listed), "x-ai/grok-4.6:free",
                         "a chain written without :free should still find the model")

    def test_a_chain_falls_through_to_its_second_id_when_the_first_is_not_served(self):
        # This is how the prose model gets an Atlas fallback: Magnum is OpenRouter-only.
        magnum = "anthracite-org/magnum-v4-72b, xai/grok-4.6"
        self.assertEqual(models_llm.resolve("prose", magnum, OPENROUTER), "anthracite-org/magnum-v4-72b",
                         "on OpenRouter the uncensored model is available and should win")
        self.assertEqual(models_llm.resolve("prose", magnum, ATLAS), "xai/grok-4.6",
                         "on Atlas the chain should fall through rather than leave the chat with no model")

    def test_a_slug_matching_two_listed_models_resolves_to_neither(self):
        # Ambiguity names nobody, the same rule cast_talk uses for a first name two characters share: a wrong
        # model is a silent quality change, where falling through to the next candidate is correctable.
        twins = [{"id": "vendor-a/grok-4.6", "vision": True, "price_in": 5e-6},
                 {"id": "vendor-b/grok-4.6", "vision": True, "price_in": 4e-6},
                 {"id": "xai/grok-4.3", "vision": True, "price_in": 1e-6}]
        self.assertEqual(models_llm.resolve("director", "xai/grok-4.6, xai/grok-4.3", twins), "xai/grok-4.3",
                         "the ambiguous candidate should be skipped for the next one in the chain")

    def test_a_single_id_is_read_as_a_one_element_chain(self):
        # Every setting already stored is a single id, so this is the whole backward-compatibility story.
        self.assertEqual(models_llm.chain("xai/grok-4.6"), ["xai/grok-4.6"])
        self.assertEqual(models_llm.chain(" a/one ,, b/two , a/one "), ["a/one", "b/two"],
                         "blanks and repeats should be dropped, order kept")

    def test_a_vision_role_never_returns_a_model_that_cannot_see(self):
        blind = [{"id": "vendor/text-only", "vision": False, "price_in": 1e-9}]
        self.assertEqual(models_llm.resolve_many("vision", "vendor/text-only", blind), [],
                         "asking a blind model to inspect an image wastes a call and returns nonsense")
        self.assertEqual(models_llm.resolve("vision", "", OPENROUTER), "x-ai/grok-4.6",
                         "the built-in vision chain should resolve on OpenRouter, where it used to 404")

    def test_the_vision_ladder_is_capped_so_a_refusal_loop_cannot_bill_forever(self):
        many = [{"id": f"vendor/sees-{n}", "vision": True, "price_in": n * 1e-7} for n in range(9)]
        self.assertLessEqual(len(models_llm.resolve_many("vision", ", ".join(m["id"] for m in many), many)),
                             models_llm.MAX_VISION_CANDIDATES,
                             "a model still bills for refusing to look, so the ladder must be bounded")

    def test_an_unreachable_catalogue_returns_the_configured_id_unchanged(self):
        # The property that keeps a provider outage from also being a resolver outage: with no catalogue,
        # behaviour is exactly what it was before this module existed.
        for empty in ([], None):
            self.assertEqual(models_llm.resolve("planner", "xai/grok-4.6, xai/grok-4.3", empty), "xai/grok-4.6",
                             "a /models call that failed must not stop the call it was meant to inform")

    def test_the_cheapest_listed_model_is_reached_only_when_every_candidate_is_exhausted(self):
        odd = [{"id": "vendor/dear", "vision": False, "price_in": 9e-6},
               {"id": "vendor/cheap", "vision": False, "price_in": 1e-9}]
        self.assertEqual(models_llm.resolve("director", "made-up/model", odd), "vendor/cheap",
                         "a feature should degrade to a listed model rather than fail outright")
        self.assertEqual(models_llm.resolve("director", "xai/grok-4.6", ATLAS), "xai/grok-4.6",
                         "and a configured model that is served must never be displaced by a cheaper one")

    def test_the_report_says_when_a_configured_model_is_not_the_one_that_will_be_used(self):
        # A typo, or a model that only the other provider has, must not look like it took effect.
        card = models_llm.report({"director": "xai/grok-4.6", "summary": "made-up/model"}, ATLAS)
        self.assertTrue(card["director"]["honoured"], "a served model the user chose is honoured")
        self.assertEqual(card["director"]["resolved"][0], "xai/grok-4.6",
                         "the configured model comes first; the built-in chain follows it as backup")
        self.assertFalse(card["summary"]["honoured"],
                         "the user asked for a model Atlas does not serve and should be told so")
        self.assertFalse(card["summary"]["fallback"],
                         "but the role still works, because its own chain was served -- not a cheapest-rung landing")
        self.assertEqual(card["summary"]["resolved"][0], "deepseek-ai/deepseek-v4.1-flash")
        self.assertEqual(set(card), set(models_llm.ROLES), "every role belongs in the report")

    def test_the_report_flags_a_role_left_on_a_model_nobody_chose(self):
        odd = [{"id": "vendor/cheap", "vision": True, "price_in": 1e-9}]
        card = models_llm.report({}, odd)
        self.assertTrue(card["director"]["fallback"],
                        "landing on the cheapest listed model should be flagged, not hidden")
        self.assertEqual(card["director"]["resolved"], ["vendor/cheap"])

    def test_every_role_lands_on_a_listed_model_on_both_providers(self):
        # The end-to-end promise: flip the provider, change nothing else, and nothing is left pointing at an
        # id that does not exist.
        for name, catalogue in (("atlas", ATLAS), ("openrouter", OPENROUTER)):
            listed = {entry["id"] for entry in catalogue}
            for role in models_llm.ROLES:
                got = models_llm.resolve(role, "", catalogue)
                self.assertIn(got, listed, f"{role} resolved to something {name} does not serve")


if __name__ == "__main__":
    unittest.main()
