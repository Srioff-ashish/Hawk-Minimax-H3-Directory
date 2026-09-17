"""Asset library: collections, tags, duplicates, bulk actions, Google Drive browse and
import (a temp folder stands in for the mounted Drive and ComfyUI's input folder).

    python -m unittest tests_api.test_library
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests_api"))

try:
    import httpx
    import uvicorn
    from aiohttp import web
    from mcp import Client

    from hawk_api.app import create_app
    from hawk_api.config import Settings
    from hawk_api.library import DriveBrowser
    from test_agent import FakeAtlas, tiny_png
    from test_gateway import TOKEN, FakeComfy, free_port
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"library test dependencies missing: {exc}")


def write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


class Library(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hawk_library_test_")
        self.drive = os.path.join(self.tmp, "MyDrive")
        self.input_dir = os.path.join(self.tmp, "comfy_input")
        os.makedirs(self.input_dir)
        write(os.path.join(self.drive, "Model Eunha", "face.png"), tiny_png((10, 20, 30)))
        write(os.path.join(self.drive, "Model Eunha", "poses", "arms_up.png"), tiny_png((40, 50, 60)))
        write(os.path.join(self.drive, "Model Eunha", "notes.txt"), b"not media")
        write(os.path.join(self.drive, "Model Eunha", ".hidden.png"), tiny_png((1, 1, 1)))
        write(os.path.join(self.drive, "Music", "beat.mp3"), b"ID3" + b"1" * 256)

        self.fake, self.atlas = FakeComfy(), FakeAtlas()
        self.runners = []
        comfy_port, atlas_port, api_port = free_port(), free_port(), free_port()
        for app, port in ((self.fake.app, comfy_port), (self.atlas.app, atlas_port)):
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            self.runners.append(runner)
        self.base = f"http://127.0.0.1:{api_port}"
        settings = Settings(
            token=TOKEN, comfy_url=f"http://127.0.0.1:{comfy_port}", public_base_url=self.base,
            data_dir=os.path.join(self.tmp, "data"), lora_cache_seconds=0, reconcile_seconds=0.3,
            atlas_url=f"http://127.0.0.1:{atlas_port}/v1", atlas_api_key="k",
            comfy_input_dir=self.input_dir, drive_root=self.drive,
        )
        self.server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=api_port, log_level="warning"))
        self.server_task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)
        self.http = httpx.AsyncClient(base_url=self.base, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20)

    async def asyncTearDown(self):
        await self.http.aclose()
        self.server.should_exit = True
        await self.server_task
        for runner in self.runners:
            await runner.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def upload(self, name: str, data: bytes, mime: str, **form) -> dict:
        response = await self.http.post("/v1/assets", files={"files": (name, data, mime)}, data=form)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["assets"][0]

    async def finish_import(self, import_id: str) -> dict:
        for _ in range(200):
            state = (await self.http.get(f"/v1/imports/{import_id}")).json()
            if state["status"] != "running":
                return state
            await asyncio.sleep(0.05)
        self.fail("import did not finish")

    async def test_collections_tags_duplicates_and_bulk(self):
        face = await self.upload("face.png", tiny_png((200, 0, 0)), "image/png", collection="Model Eunha", tags="Face, hero")
        self.assertEqual((face["collection"], face["tags"]), ("Model Eunha", ["face", "hero"]))
        again = await self.upload("copy_of_face.png", tiny_png((200, 0, 0)), "image/png")
        self.assertEqual((again["id"], again.get("duplicate")), (face["id"], True))
        beat = await self.upload("beat.mp3", b"ID3" + b"2" * 128, "audio/mpeg", collection="Music beds")
        loose = await self.upload("scene.png", tiny_png((0, 200, 0)), "image/png")
        self.assertEqual(loose["collection"], "Uploads")

        library = (await self.http.get("/v1/library")).json()
        self.assertEqual(library["total"], 3)
        self.assertEqual({c["name"]: c["count"] for c in library["collections"]}, {"Model Eunha": 1, "Music beds": 1, "Uploads": 1})
        self.assertEqual([t["name"] for t in library["tags"]], ["face", "hero"])
        self.assertEqual([a["id"] for a in (await self.http.get("/v1/library", params={"kind": "audio"})).json()["assets"]], [beat["id"]])
        self.assertEqual((await self.http.get("/v1/library", params={"tag": "hero"})).json()["total"], 1)
        self.assertEqual((await self.http.get("/v1/library", params={"q": "scene"})).json()["assets"][0]["id"], loose["id"])

        patched = (await self.http.patch(f"/v1/assets/{loose['id']}", json={"collection": "Locations", "add_tags": ["Studio"]})).json()
        self.assertEqual((patched["collection"], patched["tags"]), ("Locations", ["studio"]))
        bulk = (await self.http.post("/v1/assets/bulk", json={"ids": [face["id"], loose["id"]], "action": "tag", "tags": ["shoot-1"]})).json()
        self.assertEqual(len(bulk["done"]), 2)
        await self.http.post("/v1/assets/bulk", json={"ids": [face["id"]], "action": "untag", "tags": ["hero"]})
        moved = (await self.http.post("/v1/assets/bulk", json={"ids": [face["id"], "missing"], "action": "move", "collection": "Cast"})).json()
        self.assertEqual((moved["done"], len(moved["errors"])), ([face["id"]], 1))
        face_now = next(a for a in (await self.http.get("/v1/library")).json()["assets"] if a["id"] == face["id"])
        self.assertEqual((face_now["collection"], face_now["tags"]), ("Cast", ["face", "shoot-1"]))

        deleted = (await self.http.post("/v1/assets/bulk", json={"ids": [beat["id"]], "action": "delete"})).json()
        self.assertEqual(deleted["done"], [beat["id"]])
        self.assertEqual((await self.http.get("/v1/library")).json()["total"], 2)

    async def test_drive_browse_and_import(self):
        top = (await self.http.get("/v1/drive")).json()
        self.assertEqual([f["name"] for f in top["folders"]], ["Model Eunha", "Music"])
        model = (await self.http.get("/v1/drive", params={"path": "Model Eunha"})).json()
        self.assertEqual(([f["name"] for f in model["folders"]], [f["name"] for f in model["files"]]), (["poses"], ["face.png"]))
        self.assertEqual(model["parent"], "")
        escape = await self.http.get("/v1/drive", params={"path": "../comfy_input"})
        self.assertEqual(escape.status_code, 422)

        started = await self.http.post("/v1/drive/import", json={"paths": ["Model Eunha"], "tags": ["cast"]})
        self.assertEqual(started.status_code, 202, started.text)
        state = await self.finish_import(started.json()["id"])
        self.assertEqual((state["status"], state["total"], state["imported"], state["failed"], state["collection"]),
                         ("done", 2, 2, 0, "Model Eunha"))
        assets = (await self.http.get("/v1/library", params={"collection": "Model Eunha"})).json()["assets"]
        self.assertEqual(sorted(a["filename"] for a in assets), ["arms_up.png", "face.png"])
        for asset in assets:
            self.assertEqual((asset["source"]["type"], asset["tags"]), ("drive", ["cast"]))
            self.assertTrue(os.path.isfile(os.path.join(self.input_dir, asset["path"])), "copied into ComfyUI's input folder")
        async with httpx.AsyncClient() as browser:
            thumb = await browser.get(assets[0]["thumb_url"])
        self.assertEqual((thumb.status_code, thumb.headers["content-type"]), (200, "image/jpeg"))

        again = await self.finish_import((await self.http.post("/v1/drive/import", json={"paths": ["Model Eunha"]})).json()["id"])
        self.assertEqual((again["imported"], again["duplicates"]), (0, 2))
        flat = await self.finish_import((await self.http.post("/v1/drive/import", json={"paths": ["Model Eunha"], "recursive": False, "collection": "Faces"})).json()["id"])
        self.assertEqual(flat["total"], 1)
        empty = await self.http.post("/v1/drive/import", json={"paths": ["Model Eunha/notes.txt"]})
        self.assertEqual(empty.status_code, 422)

        deleted = assets[0]
        await self.http.delete(f"/v1/assets/{deleted['id']}")
        self.assertFalse(os.path.exists(os.path.join(self.input_dir, "hawk_api", deleted["id"])))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    async def test_video_thumbnail(self):
        clip = os.path.join(self.drive, "Clips", "walk.mp4")
        os.makedirs(os.path.dirname(clip))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24", "-t", "2", "-pix_fmt", "yuv420p", clip], check=True)
        state = await self.finish_import((await self.http.post("/v1/drive/import", json={"paths": ["Clips"]})).json()["id"])
        asset = (await self.http.get("/v1/library", params={"kind": "video"})).json()["assets"][0]
        self.assertEqual((state["imported"], asset["filename"]), (1, "walk.mp4"))
        async with httpx.AsyncClient() as browser:
            thumb = await browser.get(asset["thumb_url"])
        self.assertEqual((thumb.status_code, thumb.content[:3]), (200, b"\xff\xd8\xff"))

    async def test_mcp_library_tools(self):
        await self.upload("face.png", tiny_png((9, 9, 9)), "image/png", collection="Cast", tags="hero")
        async with Client(f"{self.base}/t/{TOKEN}/mcp") as client:
            async def call(name, args=None):
                result = await client.call_tool(name, args or {})
                self.assertFalse(result.is_error, result)
                return result.structured_content if result.structured_content is not None else json.loads(result.content[0].text)

            names = {tool.name for tool in (await client.list_tools()).tools}
            self.assertTrue({"list_collections", "organize_assets", "browse_drive", "import_from_drive", "get_import"} <= names)
            self.assertEqual((await call("list_references", {"collection": "Cast"}))["total"], 1)
            self.assertEqual((await call("list_collections"))["collections"][0]["name"], "Cast")
            self.assertEqual([f["name"] for f in (await call("browse_drive"))["folders"]], ["Model Eunha", "Music"])
            state = await call("import_from_drive", {"paths": ["Music"], "collection": "Beats", "wait_seconds": 20})
            self.assertEqual((state["status"], state["imported"], state["collection"]), ("done", 1, "Beats"))
            organised = await call("organize_assets", {"asset_ids": state["asset_ids"], "add_tags": ["edm"]})
            self.assertEqual(organised["assets"][0]["tags"], ["edm"])


class Browser(unittest.TestCase):
    def test_unmounted_drive(self):
        browser = DriveBrowser(os.path.join(tempfile.gettempdir(), "no-such-drive-here"))
        self.assertFalse(browser.available)
        with self.assertRaisesRegex(Exception, "not mounted"):
            browser.browse("")


if __name__ == "__main__":
    unittest.main()
