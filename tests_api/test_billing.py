"""The Billing pages: Atlas and OpenRouter read from their billing APIs, Google from this server's own record.

The provider responses below are trimmed copies of real ones (October 2026), so a change in the parsing
shows up here rather than as an empty page.

    python -m unittest tests_api.test_billing
"""

from __future__ import annotations

import datetime as dt
import io
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import httpx
    from PIL import Image

    from hawk_api import billing
    from hawk_api.config import Settings
    from hawk_api.jobs import HawkService, RequestError
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"billing test dependencies missing: {exc}")

TODAY = dt.date(2026, 10, 5)


def money(value):
    return {"value": f"{value:.6f}", "currency": "usd"}


ATLAS_BALANCE = {"object": "balance", "available": money(9.392241), "cash": money(9.392241), "bonus": money(0),
                 "subscription_bonus": money(0), "frozen": money(0.5)}


def atlas_result(name, kind, **extra):
    return {"model_type": kind, "model": {"id": "ms-x", "name": name, "type": kind}, **extra}


ATLAS_COSTS = [
    {"date": "2026-10-03", "results": [atlas_result("xai/grok-4.3", "text", amount=money(0.196035)),
                                       atlas_result("alibaba/wan-3.0-prime/image-to-video", "video", amount=money(1.224))]},
    {"date": "2026-10-05", "results": [atlas_result("bytedance/seedream-v5.0-pro/edit", "image", amount=money(0.3048))]},
]
ATLAS_USAGE = [
    {"date": "2026-10-03", "results": [
        atlas_result("xai/grok-4.3", "text", usage={"requests": 35, "tokens": {"input": 98571, "output": 23302,
                                                                              "cache_read": 72832}}),
        atlas_result("alibaba/wan-3.0-prime/image-to-video", "video",
                     usage={"requests": 2, "tokens": None, "images": None, "video": {"seconds": 20}})]},
    {"date": "2026-10-05", "results": [
        atlas_result("bytedance/seedream-v5.0-pro/edit", "image",
                     usage={"requests": 8, "tokens": None, "images": {"count": 8}, "video": None})]},
]


def atlas_transport(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path.endswith("/balance"):
            return httpx.Response(200, json=ATLAS_BALANCE)
        data = ATLAS_COSTS if path.endswith("/model-costs") else ATLAS_USAGE
        # Two pages, to prove the cursor is followed.
        if request.url.params.get("page") == "p2":
            return httpx.Response(200, json={"data": data[1:], "has_more": False, "next_page": None})
        return httpx.Response(200, json={"data": data[:1], "has_more": True, "next_page": "p2"})
    return httpx.MockTransport(handler)


OR_CREDITS = {"data": {"total_credits": 40, "total_usage": 12.637883484}}
OR_KEY = {"data": {"label": "sk-or-v1-bf6...149", "limit": 100, "limit_remaining": 87.36, "usage": 12.63,
                   "usage_daily": 0.0628, "usage_weekly": 0.0628, "usage_monthly": 6.27}}
OR_ACTIVITY = {"data": [
    {"date": "2026-10-04", "model": "deepseek/deepseek-v4-pro", "provider_name": "DeepSeek", "usage": 0.5,
     "byok_usage_inference": 0, "requests": 10, "prompt_tokens": 1000, "completion_tokens": 200},
    {"date": "2026-10-04", "model": "deepseek/deepseek-v4-pro", "provider_name": "Novita", "usage": 0.25,
     "requests": 4, "prompt_tokens": 400, "completion_tokens": 100},
    {"date": "2026-08-01", "model": "old/model", "usage": 9.0, "requests": 1},  # outside a 7-day period
]}


def openrouter_transport(seen: list, management_ok=True):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers["authorization"]))
        if request.url.path.endswith("/credits"):
            return httpx.Response(200, json=OR_CREDITS)
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json=OR_KEY)
        if request.url.path.endswith("/activity"):
            if not management_ok:
                return httpx.Response(403, json={"error": {"message": "Only management keys can fetch activity"}})
            return httpx.Response(200, json=OR_ACTIVITY)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


class Atlas(unittest.IsolatedAsyncioTestCase):
    async def test_balance_spend_and_models(self):
        seen = []
        async with httpx.AsyncClient(transport=atlas_transport(seen)) as http:
            report = await billing.atlas_report("apikey-x", 7, http=http, today=TODAY)
        self.assertTrue(report["configured"])
        self.assertAlmostEqual(report["balance"]["available"], 9.392241)
        self.assertAlmostEqual(report["balance"]["frozen"], 0.5)
        self.assertAlmostEqual(report["total"], 0.196035 + 1.224 + 0.3048, places=6)
        self.assertEqual([d["date"] for d in report["daily"]][0], "2026-09-29")
        self.assertEqual(len(report["daily"]), 7, "one row per day, zeros included")
        self.assertEqual(report["models"][0]["model"], "alibaba/wan-3.0-prime/image-to-video", "biggest first")
        grok = next(m for m in report["models"] if m["model"] == "xai/grok-4.3")
        self.assertEqual((grok["requests"], grok["input_tokens"], grok["output_tokens"]), (35, 98571 + 72832, 23302))
        seedream = next(m for m in report["models"] if m["type"] == "image")
        self.assertEqual(seedream["images"], 8)
        self.assertEqual(next(iter(report["by_type"])), "video")
        costs = [r for r in seen if r.url.path.endswith("/model-costs")]
        self.assertEqual(len(costs), 2, "followed next_page")
        self.assertEqual(costs[0].url.params["start_date"], "2026-09-29")
        self.assertEqual(costs[0].url.params["end_date"], "2026-10-06", "end_date is exclusive")
        self.assertEqual(costs[0].url.params["scope"], "account")

    async def test_no_key_is_a_report_not_an_error(self):
        report = await billing.atlas_report("", 30)
        self.assertFalse(report["configured"])
        self.assertIn("ATLAS_API_KEY", report["message"])

    async def test_a_refusal_is_a_billing_error_without_the_key(self):
        transport = httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
        async with httpx.AsyncClient(transport=transport) as http:
            with self.assertRaises(billing.BillingError) as caught:
                await billing.atlas_report("apikey-secret", 30, http=http, today=TODAY)
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("apikey-secret", str(caught.exception))


class OpenRouter(unittest.IsolatedAsyncioTestCase):
    async def test_without_a_management_key_totals_only(self):
        seen = []
        async with httpx.AsyncClient(transport=openrouter_transport(seen)) as http:
            report = await billing.openrouter_report("sk-or-plain", "", 30, http=http, today=TODAY)
        self.assertAlmostEqual(report["balance"]["available"], 40 - 12.637883484, places=5)
        self.assertAlmostEqual(report["key"]["month"], 6.27)
        self.assertIsNone(report["models"])
        self.assertIn("management key", report["notes"][0])
        self.assertNotIn("/api/v1/activity", [path for path, _ in seen])

    async def test_with_a_management_key_spend_per_model(self):
        seen = []
        async with httpx.AsyncClient(transport=openrouter_transport(seen)) as http:
            report = await billing.openrouter_report("sk-or-plain", "sk-or-mgmt", 7, http=http, today=TODAY)
        self.assertAlmostEqual(report["total"], 0.75)
        [model] = report["models"]
        self.assertEqual((model["requests"], model["providers"]), (14, ["DeepSeek", "Novita"]))
        self.assertIn(("/api/v1/activity", "Bearer sk-or-mgmt"), seen)
        self.assertIn(("/api/v1/key", "Bearer sk-or-plain"), seen, "the key's own usage comes from the plain key")

    async def test_period_is_capped_at_thirty_days(self):
        async with httpx.AsyncClient(transport=openrouter_transport([])) as http:
            report = await billing.openrouter_report("k", "m", 180, http=http, today=TODAY)
        self.assertEqual(report["period"]["days"], 30)

    async def test_a_rejected_management_key_says_so(self):
        async with httpx.AsyncClient(transport=openrouter_transport([], management_ok=False)) as http:
            report = await billing.openrouter_report("k", "m", 30, http=http, today=TODAY)
        self.assertIsNone(report["models"])
        self.assertIn("403", report["notes"][0])


class Google(unittest.TestCase):
    def record(self, day, engine="nano-banana", cost=None, generator="gemini-3.1-flash-image"):
        created = dt.datetime.combine(day, dt.time(12), dt.timezone.utc).timestamp()
        source = {"type": "generated", "engine": engine, "generator": generator}
        if cost is not None:
            source["cost_usd"] = cost
        return {"id": f"a{created}{engine}", "created_at": created, "source": source}

    def test_recorded_and_estimated(self):
        records = [self.record(TODAY, cost=0.101), self.record(TODAY - dt.timedelta(days=1)),
                   self.record(TODAY, engine="nano-banana-pro", cost=0.24, generator="gemini-3-pro-image"),
                   self.record(TODAY - dt.timedelta(days=40), cost=5.0),  # outside the period
                   self.record(TODAY, engine="seedream", cost=1.0)]  # not Google
        report = billing.google_report(records, 30, today=TODAY)
        self.assertAlmostEqual(report["total"], 0.101 + 0.067 + 0.24)
        self.assertEqual(report["estimated_images"], 1)
        self.assertEqual(report["models"][0]["model"], "gemini-3-pro-image")
        self.assertIsNone(report["balance"])
        self.assertIn("aistudio.google.com", report["links"]["AI Studio billing"])


def png(color=(1, 2, 3)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, "PNG")
    return buffer.getvalue()


class Service(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix="hawk_billing_test_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        input_dir = os.path.join(self.dir, "comfy_input")
        os.makedirs(input_dir)
        self.service = HawkService(Settings(token="t" * 24, data_dir=self.dir, comfy_input_dir=input_dir,
                                            google_api_key="g"))

        async def upload(fileobj, name, subfolder, mime):
            return {"name": name, "subfolder": subfolder}

        self.service.comfy.upload = upload

    async def test_a_google_image_keeps_its_price_and_counts_after_delete(self):
        result = await self.service._image_result("a red boot", [png(), png((4, 5, 6))], "gemini-3.1-flash-image",
                                                  "nano-banana", [], [], [], engine_id="nano-banana",
                                                  extra={"cost_usd": 0.202})
        first, second = (asset["id"] for asset in result["assets"])
        self.assertEqual(self.service.store.get_asset(first)["source"]["cost_usd"], 0.101, "per image")
        self.service.store.delete_asset(second)
        report = await self.service.billing("google", 30)
        self.assertAlmostEqual(report["total"], 0.202, msg="a deleted image was still paid for")
        self.assertEqual(report["estimated_images"], 0)
        self.assertTrue(report["configured"])

    async def test_unknown_provider(self):
        with self.assertRaises(RequestError):
            await self.service.billing("aws")

    async def test_reports_are_cached_until_refresh(self):
        calls = []

        async def fake(key, days):
            calls.append(key)
            return {"provider": "atlas", "configured": True}

        original = billing.atlas_report
        billing.atlas_report = fake
        self.addCleanup(setattr, billing, "atlas_report", original)
        self.service.settings = Settings(token="t" * 24, data_dir=self.dir, atlas_api_key="a1")
        await self.service.billing("atlas", 30)
        await self.service.billing("atlas", 30)
        self.assertEqual(calls, ["a1"])
        await self.service.billing("atlas", 30, refresh=True)
        self.assertEqual(calls, ["a1", "a1"])
        self.service.llm_settings.save({"atlas_api_key_override": "a2"})
        await self.service.billing("atlas", 30)
        self.assertEqual(calls, ["a1", "a1", "a2"], "a new key is never answered from the old account's figures")


if __name__ == "__main__":
    unittest.main()
