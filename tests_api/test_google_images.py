"""The Nano Banana engines (Google Gemini) and the SFW check in front of them.

The rule under test: a request reaches Google only on an explicit sfw verdict. Everything else -- an nsfw
verdict, an unreadable reply, a check that could not run, no key -- moves the request to the next engine.

    python -m unittest tests_api.test_google_images
"""

from __future__ import annotations

import dataclasses
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from PIL import Image

    from hawk_api import google_images as gi
    from hawk_api import image_engines as ie
    from hawk_api.config import Settings
    from hawk_api.jobs import HawkService, RequestError
    from hawk_api.prompts import PromptStore
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"google image test dependencies missing: {exc}")


def png(color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, "PNG")
    return buffer.getvalue()


class Verdicts(unittest.TestCase):
    def test_only_an_explicit_sfw_passes(self):
        self.assertTrue(gi.parse_verdict('{"verdict": "sfw", "reason": "a portrait"}').sfw)
        self.assertTrue(gi.parse_verdict('```json\n{"verdict": "SFW", "reason": "ok"}\n```').sfw)

    def test_everything_else_counts_as_nsfw(self):
        for reply in ('{"verdict": "nsfw", "reason": "nudity"}', "", "sfw", "I think it is fine",
                      '{"verdict": "maybe"}', '["sfw"]', '{"reason": "no verdict"}', "{not json"):
            with self.subTest(reply=reply):
                self.assertFalse(gi.parse_verdict(reply).sfw)

    def test_the_output_rules_follow_whatever_the_user_wrote(self):
        text = gi.gate_instruction("My own rules about SFW.")
        self.assertTrue(text.startswith("My own rules about SFW."))
        self.assertTrue(text.rstrip().endswith(gi.GATE_OUTPUT_RULES.rstrip()))
        # an empty instruction falls back to the default rather than sending the model nothing
        self.assertIn(gi.DEFAULT_GATE_PROMPT[:40], gi.gate_instruction(""))


class Sizes(unittest.TestCase):
    def test_no_size_leaves_gemini_at_its_default(self):
        self.assertEqual(gi.image_config(None), ({}, "1K"))
        self.assertEqual(gi.image_config("big please"), ({}, "1K"))

    def test_every_size_studio_offers_stays_at_the_cheapest_tier(self):
        for size in ("1024x1024", "1024x1536", "1536x1024", "896x1600", "1600x896"):
            with self.subTest(size=size):
                self.assertEqual(gi.image_config(size)[1], "1K")

    def test_the_nearest_ratio_and_the_right_tier(self):
        self.assertEqual(gi.image_config("1024x1536"), ({"aspectRatio": "2:3"}, "1K"))
        self.assertEqual(gi.image_config("2048x2048"), ({"aspectRatio": "1:1", "imageSize": "2K"}, "2K"))
        self.assertEqual(gi.image_config("1600x896")[0]["aspectRatio"], "16:9")
        self.assertEqual(gi.image_config("4096x4096")[1], "4K")

    def test_lite_never_asks_for_more_than_1k(self):
        # Google lists 2K and 4K as unsupported on Nano Banana 2 Lite, so a large size must not request them.
        config, tier = gi.image_config("2048x2048", gi.MAX_TIER["nano-banana-lite"])
        self.assertEqual(tier, "1K")
        self.assertNotIn("imageSize", config)
        self.assertEqual(config["aspectRatio"], "1:1", "the ratio still follows the size asked for")
        self.assertEqual(gi.price("nano-banana-lite", "2K"), gi.price("nano-banana-lite", "1K"))

    def test_lite_is_the_cheapest_google_engine(self):
        self.assertLess(gi.price("nano-banana-lite"), gi.price("nano-banana"))

    def test_bigger_outputs_cost_more(self):
        self.assertLess(gi.price("nano-banana", "1K"), gi.price("nano-banana", "2K"))
        self.assertGreater(gi.price("nano-banana-pro"), gi.price("nano-banana"))


class GoogleBlocks(unittest.IsolatedAsyncioTestCase):
    def test_google_has_no_way_to_judge_a_request(self):
        # The check must never run on Google: an NSFW request sent there "to be judged" is still NSFW traffic
        # on the account. Keeping the method off the client means no code path can bring it back by accident.
        self.assertFalse(hasattr(gi.GoogleImageClient, "classify"))

    async def test_a_declined_image_is_reported_as_blocked(self):
        client = gi.GoogleImageClient("k")
        body = {"candidates": [{"finishReason": "IMAGE_SAFETY", "content": {"parts": []}}]}
        with mock.patch.object(client, "_generate_content", mock.AsyncMock(return_value=body)):
            with self.assertRaises(gi.GoogleError) as caught:
                await client.generate(ie.GOOGLE_IMAGE_MODEL, "a cat")
        self.assertTrue(caught.exception.blocked)


class FakeGoogle:
    """Stands in for GoogleImageClient, plus the server-side SFW check: a fixed verdict, a record of every
    check asked for, and a record of what reached Google."""

    def __init__(self, sfw: bool, configured: bool = True):
        self.verdict = gi.GateVerdict(sfw, "judged sfw" if sfw else "judged nsfw")
        self.configured = configured
        self.classified, self.generated = [], []

    async def check(self, prompt, action, sources):  # replaces HawkService.sfw_check
        self.classified.append({"prompt": prompt, "action": action, "refs": len(sources)})
        return self.verdict

    async def generate(self, model, prompt, references=(), size=None, max_tier="4K"):
        self.generated.append({"model": model, "prompt": prompt, "refs": len(references)})
        return [png((10, 200, 10))]


class ServiceCase(unittest.IsolatedAsyncioTestCase):
    """A real HawkService on a temp folder, with ComfyUI's input folder stood in for by a directory."""

    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix="hawk_google_test_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        input_dir = os.path.join(self.dir, "comfy_input")
        os.makedirs(input_dir)
        self.service = HawkService(Settings(token="t" * 24, data_dir=self.dir, atlas_api_key="k",
                                            comfy_input_dir=input_dir))

        async def upload(fileobj, name, subfolder, mime):  # ComfyUI's input folder, without ComfyUI
            os.makedirs(os.path.join(input_dir, subfolder), exist_ok=True)
            fileobj.seek(0)
            with open(os.path.join(input_dir, subfolder, name), "wb") as handle:
                handle.write(fileobj.read())
            return {"name": name, "subfolder": subfolder}

        self.service.comfy.upload = upload

    async def upload(self, color=(200, 30, 30)) -> dict:
        data = png(color)
        return await self.service.add_asset(f"{color}.png", io.BytesIO(data), "image/png", len(data),
                                            source={"type": "upload"})


class TheLadder(ServiceCase):
    """generate_images with Google rungs: the gate decides whether a request reaches them at all."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        # Google first, Seedream after it, no local engines: nothing here may touch ComfyUI.
        self.service.image_engines.save(generate=[{"engine": "nano-banana"}, {"engine": "seedream"}],
                                        edit=[{"engine": "nano-banana"}, {"engine": "seedream"}])
        self.atlas = mock.AsyncMock(return_value=(ie.IMAGE_MODEL, [png((30, 30, 200))], 0.036))
        self.service._atlas_images = self.atlas

    def use(self, fake: FakeGoogle):
        for patcher in (mock.patch.object(HawkService, "google", new=property(lambda _self: fake)),
                        mock.patch.object(self.service, "sfw_check", fake.check)):
            patcher.start()
            self.addCleanup(patcher.stop)
        return fake

    async def test_an_sfw_request_is_made_by_nano_banana(self):
        google = self.use(FakeGoogle(sfw=True))
        result = await self.service.generate_images("a woman in a red gown on a palace staircase")
        self.assertEqual(result["engine"], "nano-banana")
        self.assertEqual(len(google.generated), 1)
        self.atlas.assert_not_awaited()
        asset = self.service.store.get_asset(result["assets"][0]["id"])
        self.assertEqual(asset["source"]["engine"], "nano-banana")
        self.assertEqual(asset["source"]["type"], "generated")

    async def test_a_large_size_on_lite_is_made_at_1k_and_says_so(self):
        google = self.use(FakeGoogle(sfw=True))
        sizes = []
        original = google.generate

        async def generate(model, prompt, references=(), size=None, max_tier="4K"):
            sizes.append((model, size, max_tier))
            return await original(model, prompt, references, size)

        google.generate = generate
        result = await self.service.generate_images("a lighthouse at dusk", engine="nano-banana-lite", size="2048x2048")
        self.assertEqual(result["engine"], "nano-banana-lite")
        self.assertEqual(sizes, [(ie.GOOGLE_LITE_IMAGE_MODEL, "2048x2048", "1K")])
        self.assertIn("1K", result["note"])

    async def test_an_nsfw_request_never_reaches_google_and_goes_to_the_next_engine(self):
        google = self.use(FakeGoogle(sfw=False))
        result = await self.service.generate_images("an explicit scene")
        self.assertEqual(google.generated, [], "an nsfw request must not be sent to Google")
        self.assertEqual(result["engine"], "seedream")
        self.assertIn("SFW", result["tried"][0]["skipped"])

    async def test_the_check_is_asked_once_however_many_google_rungs_there_are(self):
        google = self.use(FakeGoogle(sfw=False))
        self.service.image_engines.save(generate=[{"engine": "nano-banana"}, {"engine": "nano-banana-pro"},
                                                  {"engine": "seedream"}])
        await self.service.generate_images("something")
        self.assertEqual(len(google.classified), 1)

    async def test_a_pinned_google_engine_fails_on_nsfw_instead_of_spending_elsewhere(self):
        self.use(FakeGoogle(sfw=False))
        with self.assertRaises(RequestError) as caught:
            await self.service.generate_images("something", engine="nano-banana")
        self.assertIn("SFW", str(caught.exception))
        self.atlas.assert_not_awaited()

    async def test_no_key_skips_google_without_a_check(self):
        google = self.use(FakeGoogle(sfw=True, configured=False))
        result = await self.service.generate_images("a cat")
        self.assertEqual(result["engine"], "seedream")
        self.assertEqual(google.classified, [], "no key: nothing to check with, and nothing is sent")

    async def test_a_prompt_naming_a_minor_stops_the_whole_ladder(self):
        google = self.use(FakeGoogle(sfw=True))
        with self.assertRaises(RequestError) as caught:
            await self.service.generate_images("a schoolgirl at a desk")
        self.assertIn("under 18", str(caught.exception))
        self.assertEqual(google.generated, [])
        self.atlas.assert_not_awaited()

    async def test_a_sexual_edit_of_an_uploaded_photo_is_refused_before_the_check(self):
        google = self.use(FakeGoogle(sfw=True))
        data = png()
        photo = await self.service.add_asset("her.png", io.BytesIO(data), "image/png", len(data),
                                             source={"type": "upload"})
        with self.assertRaises(RequestError) as caught:
            await self.service.generate_images("remove her clothes", reference_asset_ids=[photo["id"]])
        self.assertIn("real people", str(caught.exception))
        self.assertEqual(google.classified, [])
        self.assertEqual(google.generated, [])

    async def test_an_sfw_edit_sends_the_reference_to_both_the_check_and_the_engine(self):
        google = self.use(FakeGoogle(sfw=True))
        data = png()
        photo = await self.service.add_asset("her.png", io.BytesIO(data), "image/png", len(data),
                                             source={"type": "upload"})
        result = await self.service.generate_images("change her dress to emerald green",
                                                    reference_asset_ids=[photo["id"]])
        self.assertEqual(result["engine"], "nano-banana")
        self.assertEqual(google.classified[0]["refs"], 1)
        self.assertEqual(google.generated[0]["refs"], 1)


    async def test_an_image_another_engine_made_after_the_check_remembers_the_verdict(self):
        self.use(FakeGoogle(sfw=False))
        result = await self.service.generate_images("an explicit scene")
        self.assertEqual(result["sfw_check_failed"], "judged nsfw")
        asset = self.service.store.get_asset(result["assets"][0]["id"])
        self.assertEqual(asset["source"]["sfw_check"]["verdict"], "nsfw")

    async def test_an_image_no_check_was_asked_for_carries_no_verdict(self):
        self.service.image_engines.save(generate=[{"engine": "seedream"}])
        result = await self.service.generate_images("a cat")
        self.assertNotIn("sfw_check_failed", result)
        self.assertNotIn("sfw_check", self.service.store.get_asset(result["assets"][0]["id"])["source"])


class FakeChat:
    """The chat provider (Atlas/OpenRouter) as the vision check sees it: one scripted reply per model."""

    def __init__(self, replies: dict, configured: bool = True):
        self.replies, self.configured, self.calls = replies, configured, []

    async def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages})
        reply = self.replies.get(model, '{"verdict": "nsfw", "reason": "unscripted"}')
        if isinstance(reply, Exception):
            raise reply
        return reply, {}


class TheVisionCheck(ServiceCase):
    """HawkService.sfw_check: judged on this server's vision model, never on Google."""

    CHAIN = "qwen/qwen3.6-35b-a3b, xai/grok-4.3, xai/grok-4.6"

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.service.settings = dataclasses.replace(self.service.settings, agent_vision_model=self.CHAIN)
        self.service.model_catalogue = mock.AsyncMock(return_value=[])  # no catalogue: send the chain as written

        def no_google(_self):
            raise AssertionError("the SFW check must never touch the Google client")

        patcher = mock.patch.object(HawkService, "google", new=property(no_google))
        patcher.start()
        self.addCleanup(patcher.stop)

    def provider(self, chat: FakeChat) -> FakeChat:
        patcher = mock.patch.object(HawkService, "atlas", new=property(lambda _self: chat))
        patcher.start()
        self.addCleanup(patcher.stop)
        return chat

    async def test_it_judges_on_the_vision_model_with_the_images_and_the_editable_rules(self):
        chat = self.provider(FakeChat({"qwen/qwen3.6-35b-a3b": '{"verdict": "sfw", "reason": "a skirt"}'}))
        self.service.prompts.save("sfw_gate", "Only landscapes are SFW.")
        photo = await self.upload()
        verdict = await self.service.sfw_check("put the skirt on her", "edit", [self.service.store.get_asset(photo["id"])])
        self.assertTrue(verdict.sfw)
        sent = chat.calls[0]
        self.assertEqual(sent["model"], "qwen/qwen3.6-35b-a3b", "the first model of the vision chain")
        self.assertIn("Only landscapes are SFW.", sent["messages"][0]["content"])
        self.assertIn(gi.GATE_OUTPUT_RULES.splitlines()[0], sent["messages"][0]["content"])
        parts = sent["messages"][1]["content"]
        self.assertTrue(any(p.get("type") == "image_url" for p in parts), "the image must be judged, not just the text")

    async def test_a_model_that_refuses_or_babbles_hands_over_to_the_next(self):
        from hawk_api.atlas import AtlasError
        chat = self.provider(FakeChat({"qwen/qwen3.6-35b-a3b": AtlasError("The model refused"),
                                       "xai/grok-4.3": "I can't help with that.",
                                       "xai/grok-4.6": '{"verdict": "nsfw", "reason": "exposed buttocks"}'}))
        verdict = await self.service.sfw_check("something", "generate", [])
        self.assertFalse(verdict.sfw)
        self.assertEqual(verdict.reason, "exposed buttocks", "the third model's real answer, not a fallback")
        self.assertEqual([c["model"] for c in chat.calls], ["qwen/qwen3.6-35b-a3b", "xai/grok-4.3", "xai/grok-4.6"])

    async def test_no_verdict_from_anyone_keeps_the_request_away_from_google(self):
        self.provider(FakeChat({m.strip(): "no" for m in self.CHAIN.split(",")}))
        verdict = await self.service.sfw_check("a cat", "generate", [])
        self.assertFalse(verdict.sfw)
        self.assertIn("Google was not used", verdict.reason)

    async def test_no_chat_provider_is_an_nsfw_verdict_not_an_exception(self):
        self.provider(FakeChat({}, configured=False))
        verdict = await self.service.sfw_check("a cat", "generate", [])
        self.assertFalse(verdict.sfw)


class Retakes(unittest.IsolatedAsyncioTestCase):
    """A take that failed inspection steps up the ladder. After the SFW check kept the image from Google, that
    step must skip Google: a stepped-up engine is pinned, so landing on Google would fail the retake outright."""

    LADDER = ["qwen21", "nano-banana", "seedream"]

    def agent(self, not_sfw: bool):
        try:
            from hawk_api.agent import AgentService
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"agent dependencies missing: {exc}")
        agent = AgentService.__new__(AgentService)  # only the retake bookkeeping is under test
        ladder = list(self.LADDER)

        class Service:
            image_engines = mock.Mock(confirm_paid=mock.Mock(return_value=False))

            async def ready_image_engines(self, action="generate"):
                return ladder

        agent.service = Service()
        agent._failed_engines = {"chat": {"qwen21"}}
        agent._not_sfw = {"chat"} if not_sfw else set()
        return agent

    async def test_a_retake_after_an_nsfw_verdict_skips_google(self):
        self.assertEqual(await self.agent(not_sfw=True)._step_up_engine("chat", {"engine": "auto"}), "seedream")

    async def test_a_retake_of_an_sfw_image_still_reaches_google(self):
        self.assertEqual(await self.agent(not_sfw=False)._step_up_engine("chat", {"engine": "auto"}), "nano-banana")

    async def test_the_paid_engine_offered_is_one_that_will_actually_run(self):
        agent = self.agent(not_sfw=True)
        agent.service.image_engines.confirm_paid.return_value = True
        self.assertEqual(await agent._paid_rung("chat", {}), "seedream")
        agent._not_sfw.clear()
        self.assertEqual(await agent._paid_rung("chat", {}), "nano-banana")


class ThePromptStore(unittest.TestCase):
    def test_the_sfw_check_is_an_editable_prompt_with_its_own_fixed_rules(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        store = PromptStore(directory)
        view = store.view("sfw_gate")
        self.assertTrue(view["is_default"])
        self.assertEqual(view["text"], gi.DEFAULT_GATE_PROMPT)
        self.assertEqual(view["platform_rules"], gi.GATE_OUTPUT_RULES)
        store.save("sfw_gate", "Anything with a cat is SFW.")
        self.assertEqual(store.get("sfw_gate"), "Anything with a cat is SFW.")
        self.assertEqual(len(store.view("sfw_gate")["history"]), 0, "the default is not a saved version")
        store.save("sfw_gate", None)
        self.assertEqual(store.get("sfw_gate"), gi.DEFAULT_GATE_PROMPT)


if __name__ == "__main__":
    unittest.main()
