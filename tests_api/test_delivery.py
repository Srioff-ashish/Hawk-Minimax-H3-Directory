"""Video delivery: light job polling, gzip, byte ranges from a local output folder,
thumbnails, and copying finished renders into a (temp stand-in for the) mounted Google Drive.

    python -m unittest tests_api.test_delivery
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests_api"))

try:
    import httpx
    import uvicorn
    from aiohttp import web

    import hawk_api.library as library_module
    from hawk_api.app import create_app
    from hawk_api.auth import sign_path
    from hawk_api.config import Settings
    from test_gateway import TOKEN, FakeComfy, free_port
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"delivery test dependencies missing: {exc}")


class Delivery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hawk_delivery_test_")
        self.output_dir = os.path.join(self.tmp, "comfy_output")
        self.drive = os.path.join(self.tmp, "MyDrive")
        os.makedirs(self.drive)
        self.fake = FakeComfy()
        self.fake.output_dir = self.output_dir
        self.fake.final_bytes = bytes(range(256)) * 40  # 10 KB
        self.runner = web.AppRunner(self.fake.app)
        await self.runner.setup()
        comfy_port, api_port = free_port(), free_port()
        await web.TCPSite(self.runner, "127.0.0.1", comfy_port).start()

        self._patched = (library_module.drive_file_id, library_module.DRIVE_ID_POLL_SECONDS)
        polls: dict[str, int] = {}

        def fake_drive_id(path):  # like the mount: a local placeholder id until the upload finishes
            polls[path] = polls.get(path, 0) + 1
            return None if not os.path.isfile(path) else "local-214" if polls[path] < 3 else "DRIVEID123"

        library_module.drive_file_id = fake_drive_id
        library_module.DRIVE_ID_POLL_SECONDS = 0.05

        self.base = f"http://127.0.0.1:{api_port}"
        self.settings = Settings(
            token=TOKEN, comfy_url=f"http://127.0.0.1:{comfy_port}", public_base_url=self.base,
            data_dir=os.path.join(self.tmp, "data"), lora_cache_seconds=0, reconcile_seconds=0.3,
            comfy_output_dir=self.output_dir, drive_root=self.drive,
        )
        self.server = uvicorn.Server(uvicorn.Config(create_app(self.settings), host="127.0.0.1", port=api_port, log_level="warning"))
        self.server_task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)
        self.http = httpx.AsyncClient(base_url=self.base, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20)
        await asyncio.sleep(0.2)

    async def asyncTearDown(self):
        library_module.drive_file_id, library_module.DRIVE_ID_POLL_SECONDS = self._patched
        await self.http.aclose()
        self.server.should_exit = True
        await self.server_task
        await self.runner.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def render(self, script: str = "A walk in the rain") -> dict:
        job = (await self.http.post("/v1/videos", json={"script": script, "settings": {"use_default_loras": False}})).json()
        for _ in range(400):
            job = (await self.http.get(f"/v1/jobs/{job['id']}")).json()
            if job["status"] in ("done", "failed", "cancelled"):
                self.assertEqual(job["status"], "done", job)
                return job
            await asyncio.sleep(0.03)
        self.fail("render did not finish")

    async def drive_ready(self, job_id: str) -> dict:
        for _ in range(200):
            job = (await self.http.get(f"/v1/jobs/{job_id}")).json()
            if (job.get("drive") or {}).get("status") in ("ready", "copied", "failed"):
                return job
            await asyncio.sleep(0.05)
        self.fail("Drive export did not finish")

    async def test_summary_polling_and_gzip(self):
        job = await self.render()
        await self.drive_ready(job["id"])  # the Drive copy updates the job; let it settle before testing "since"
        full = await self.http.get("/v1/jobs")
        summary = await self.http.get("/v1/jobs", params={"view": "summary"})
        self.assertEqual(summary.headers.get("content-encoding"), None if len(summary.content) < 1024 else "gzip")
        item = summary.json()["jobs"][0]
        self.assertEqual((item["id"], item["title"]), (job["id"], "A walk in the rain"))
        self.assertNotIn("script", item)
        self.assertNotIn("final_prompts", item)
        self.assertIn("script", full.json()["jobs"][0])
        self.assertTrue(item["thumb_url"] and item["download_url"].endswith("&download=1"))

        later = summary.json()["server_time"]
        self.assertEqual((await self.http.get("/v1/jobs", params={"view": "summary", "since": later})).json()["jobs"], [])
        self.assertEqual(len((await self.http.get("/v1/jobs", params={"since": later - 3600})).json()["jobs"]), 1)

        big = await self.http.get("/v1/jobs", headers={"Accept-Encoding": "gzip"})
        if len(big.content) >= 1024:
            self.assertEqual(big.headers.get("content-encoding"), "gzip")

    async def test_ranges_download_and_stable_links(self):
        job = await self.render()
        data = self.fake.final_bytes
        async with httpx.AsyncClient() as browser:
            whole = await browser.get(job["video_url"], headers={"Accept-Encoding": "gzip"})
            self.assertEqual((whole.status_code, whole.content), (200, data))
            self.assertIsNone(whole.headers.get("content-encoding"), "video is never gzipped")
            self.assertEqual(whole.headers["accept-ranges"], "bytes")
            self.assertIn("max-age", whole.headers["cache-control"])
            self.assertTrue(whole.headers["content-disposition"].startswith("inline"))

            part = await browser.get(job["video_url"], headers={"Range": "bytes=100-199"})
            self.assertEqual((part.status_code, part.content, part.headers["content-range"]), (206, data[100:200], f"bytes 100-199/{len(data)}"))

            download = await browser.get(job["download_url"])
            self.assertTrue(download.headers["content-disposition"].startswith("attachment"))

            # Not on local disk any more: proxied from ComfyUI instead.
            shutil.rmtree(self.output_dir)
            remote = await browser.get(job["video_url"])
            self.assertEqual((remote.status_code, remote.content), (200, data))

        # Day-long links round their expiry to the day, so the URL (and browser cache) is stable.
        self.assertEqual(sign_path(TOKEN, "/x", 7 * 86400, now=1000.0), sign_path(TOKEN, "/x", 7 * 86400, now=2000.0))
        again = (await self.http.get(f"/v1/jobs/{job['id']}")).json()
        self.assertEqual(again["video_url"], job["video_url"])

    async def test_drive_export(self):
        settings = (await self.http.get("/v1/drive/export")).json()
        self.assertEqual((settings["enabled"], settings["folder"], settings["mounted"]), (True, "Hawk H3/Videos", True))

        job = await self.drive_ready((await self.render("Chai ad, rooftop"))["id"])
        drive = job["drive"]
        self.assertEqual((drive["status"], drive["file_id"]), ("ready", "DRIVEID123"), drive)
        self.assertTrue(drive["path"].startswith("Hawk H3/Videos/" + time.strftime("%Y-%m-%d")))
        self.assertTrue(drive["path"].endswith(f"Chai_ad_rooftop_{job['id'][:8]}.mp4"))
        with open(os.path.join(self.drive, drive["path"]), "rb") as handle:
            self.assertEqual(handle.read(), self.fake.final_bytes)
        self.assertEqual(drive["preview_url"], "https://drive.google.com/file/d/DRIVEID123/preview")
        self.assertIn("export=download", drive["download_url"])
        summary = (await self.http.get("/v1/jobs", params={"view": "summary"})).json()["jobs"][0]
        self.assertEqual(summary["drive"]["file_id"], "DRIVEID123")

        updated = (await self.http.put("/v1/drive/export", json={"enabled": False, "folder": "../Films//Out", "segments": True})).json()
        self.assertEqual((updated["enabled"], updated["folder"], updated["segments"]), (False, "Films/Out", True))
        second = await self.render("Second film")
        await asyncio.sleep(0.3)
        self.assertIsNone((await self.http.get(f"/v1/jobs/{second['id']}")).json()["drive"], "export is off")

        resent = await self.http.post(f"/v1/jobs/{second['id']}/drive")
        self.assertEqual(resent.status_code, 202, resent.text)
        second = await self.drive_ready(second["id"])
        self.assertTrue(second["drive"]["path"].startswith("Films/Out/"))
        segments = os.path.join(self.drive, second["drive"]["path"][:-4] + "_segments")
        self.assertEqual(sorted(os.listdir(segments)), ["segment_001.mp4"])

        self.assertEqual((await self.http.put("/v1/drive/export", json={"folder": "/.."})).status_code, 422)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    async def test_job_thumbnail(self):
        clip = os.path.join(self.tmp, "clip.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24", "-t", "2",
                        "-pix_fmt", "yuv420p", clip], check=True)
        with open(clip, "rb") as handle:
            self.fake.final_bytes = handle.read()
        job = await self.render()
        async with httpx.AsyncClient() as browser:
            thumb = await browser.get(job["thumb_url"])
        self.assertEqual((thumb.status_code, thumb.headers["content-type"], thumb.content[:3]), (200, "image/jpeg", b"\xff\xd8\xff"))


if __name__ == "__main__":
    unittest.main()
