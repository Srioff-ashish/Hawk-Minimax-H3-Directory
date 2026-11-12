"""URL imports only reach public addresses, including after a redirect, and a cancelled plan stays cancelled."""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx

from hawk_api import netguard


class PublicOnly(unittest.IsolatedAsyncioTestCase):
    async def test_local_and_private_addresses_are_refused(self):
        for url in ("http://127.0.0.1:8188/view", "http://localhost/x", "http://169.254.169.254/latest/meta-data",
                    "http://10.0.0.5/a", "http://192.168.1.2/a", "http://[::1]/a", "http://0.0.0.0/a",
                    "http://[::ffff:127.0.0.1]/a", "http://comfy.localhost/a"):
            with self.subTest(url=url):
                with self.assertRaises(netguard.BlockedAddress):
                    await netguard.check_url(url)

    async def test_a_public_address_passes(self):
        await netguard.check_url("https://93.184.216.34/image.png")

    async def test_a_name_that_resolves_inward_is_refused(self):
        async def fake(host):
            return ["127.0.0.1"]
        with mock.patch.object(netguard, "_addresses", fake):
            with self.assertRaises(netguard.BlockedAddress):
                await netguard.check_url("https://looks-public.example/a.png")

    async def test_an_unresolvable_name_is_left_to_the_connection(self):
        async def fake(host):
            return []
        with mock.patch.object(netguard, "_addresses", fake):
            await netguard.check_url("https://no-such-host.invalid/a.png")

    async def test_a_redirect_to_a_local_address_is_refused(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "93.184.216.34":
                return httpx.Response(302, headers={"location": "http://127.0.0.1:8188/history"})
            return httpx.Response(200, content=b"secret")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True,
                                     event_hooks=netguard.hooks()) as client:
            with self.assertRaises(netguard.BlockedAddress):
                await client.get("https://93.184.216.34/start")


class FromUrl(unittest.IsolatedAsyncioTestCase):
    async def test_the_service_refuses_a_local_url(self):
        from hawk_api.config import Settings
        from hawk_api.jobs import HawkService, RequestError

        with tempfile.TemporaryDirectory() as data:
            settings = Settings(token="t", comfy_url="http://127.0.0.1:1", data_dir=data)
            service = HawkService(settings)
            try:
                with self.assertRaises(RequestError) as caught:
                    await service.add_asset_from_url("http://127.0.0.1:8188/view?filename=x.png")
                self.assertIn("private or local", str(caught.exception))
            finally:
                await service.comfy.close()


class CancelPlan(unittest.IsolatedAsyncioTestCase):
    async def test_a_cancelled_plan_stays_cancelled(self):
        from hawk_api.config import Settings
        from hawk_api.jobs import HawkService
        from hawk_api.schemas import PlanRequest

        with tempfile.TemporaryDirectory() as data:
            settings = Settings(token="t", comfy_url="http://127.0.0.1:1", data_dir=data)
            service = HawkService(settings)
            started = asyncio.Event()

            async def slow_plan(request, refs):
                started.set()
                await asyncio.sleep(30)
                return "never", "m", {}

            try:
                with mock.patch.object(service, "plan_text", slow_plan):
                    job = service._new_job("plan", request=PlanRequest(story="a walk").model_dump(), refs=[],
                                           graph={}, nodes={}, available={}, video_has_audio=[], seed=1)
                    job["status"] = "planning"
                    service.store.save_job(job)
                    task = service._start_plan_task(job["id"], PlanRequest(story="a walk"), [])
                    await started.wait()
                    result = await service.cancel(job["id"])
                    self.assertEqual(result["status"], "cancelled")
                    await asyncio.gather(task, return_exceptions=True)
                    self.assertTrue(task.cancelled())
                    self.assertEqual(service.store.get_job(job["id"])["status"], "cancelled")
            finally:
                await service.comfy.close()


if __name__ == "__main__":
    unittest.main()
