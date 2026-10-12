"""Manhwa catalogue: page parsing, next-chapter discovery, tiling, the vision ladder and FTS search.

A fake site (httpx.MockTransport) and a stand-in service with a scripted chat client; no network, no ComfyUI.

    python -m unittest tests_api.test_manhwa
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import httpx
    from PIL import Image

    from hawk_api import manhwa
    from hawk_api.atlas import AtlasError
    from hawk_api.manhwa import ManhwaManager, ManhwaStore, chapter_number, increment_url, parse_chapter, slice_image
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"manhwa test dependencies missing: {exc}")


def jpeg(size=(690, 2400), color=(240, 240, 240)) -> bytes:
    buffer = io.BytesIO()
    # noise keeps the file above MIN_IMAGE_BYTES, as a real panel would be
    image = Image.effect_noise(size, 60).convert("RGB")
    Image.blend(image, Image.new("RGB", size, color), 0.5).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def chapter_html(n: int, images: list[str], next_href: str | None) -> str:
    imgs = "".join(f'<img class="lazy" src="/spinner.gif" data-src="{src}">' for src in images)
    nav = f'<a class="btn" href="{next_href}">Next Chapter »</a>' if next_href else ""
    return (f"<html><head><title>Test Series Chapter {n}</title></head><body>"
            f'<img src="/static/logo.png"><a href="/series/test/chapter-{n - 1}">« Prev</a>'
            f'<div class="reader">{imgs}</div>{nav}</body></html>')


class Parsing(unittest.TestCase):
    def test_images_in_order_lazy_attrs_and_junk_dropped(self):
        html = ('<img src="/logo.png"><img data-src="a/1.jpg" src="x.gif"><img srcset="b/2-small.jpg 1x, b/2.jpg 2x">'
                '<img src="data:image/gif;base64,AAAA"><img data-lazy-src="https://cdn.example.com/3.webp">'
                '<img data-src="a/1.jpg"><img src="/icons/x.svg">')
        page = parse_chapter(html, "https://site.example/series/s/chapter-4/")
        self.assertEqual(page["images"], [
            "https://site.example/series/s/chapter-4/a/1.jpg",
            "https://site.example/series/s/chapter-4/b/2.jpg",
            "https://cdn.example.com/3.webp",
        ])

    def test_next_link_prefers_rel_next_and_ignores_prev_and_paging(self):
        html = ('<a href="/c/3">« Previous</a><a href="?page=2">Next page</a><a class="next-btn" href="/c/5">›</a>'
                '<link rel="next" href="/c/5-real"><img src="/p.jpg">')
        self.assertEqual(parse_chapter(html, "https://s.example/c/4")["next_url"], "https://s.example/c/5-real")
        html = '<a href="/c/3">Prev</a><a href="#">Next</a><a aria-label="next chapter" href="/c/5"><svg></svg></a>'
        self.assertEqual(parse_chapter(html, "https://s.example/c/4")["next_url"], "https://s.example/c/5")
        self.assertIsNone(parse_chapter('<a href="/c/3">Previous</a>', "https://s.example/c/4")["next_url"])

    def test_chapter_numbers_and_increment(self):
        self.assertEqual(chapter_number("https://s.example/series/x/chapter-12/"), "12")
        self.assertEqual(chapter_number("https://s.example/read?ep=7&lang=en"), "7")
        self.assertEqual(chapter_number("https://s.example/x/45", ""), "45")
        self.assertEqual(chapter_number("https://s.example/step-3/ch-8"), "8")
        self.assertEqual(chapter_number("https://s.example/x/", "My Series Chapter 12.5"), "12.5")
        self.assertEqual(increment_url("https://s.example/series/x/chapter-12/"), "https://s.example/series/x/chapter-13/")
        self.assertEqual(increment_url("https://s.example/read?ch=9"), "https://s.example/read?ch=10")
        self.assertEqual(increment_url("https://s.example/s2/x/45"), "https://s.example/s2/x/46")
        self.assertIsNone(increment_url("https://s.example/series/x/"))

    def test_tall_strip_is_tiled_with_overlap(self):
        tiles = slice_image(jpeg((2048, 9000)))  # downscaled to 1024 x 4500 first
        sizes = [Image.open(io.BytesIO(t)).size for t in tiles]
        self.assertTrue(all(w == manhwa.TILE_WIDTH for w, _ in sizes))
        self.assertEqual(len(tiles), 3)  # 0-1600, 1480-3080, 2960-4500
        self.assertEqual(sum(h for _, h in sizes) - 2 * manhwa.TILE_OVERLAP, 4500)
        self.assertEqual(len(slice_image(jpeg((800, 1200)))), 1)


class FakeChat:
    """Answers catalogue calls with one described tile per image sent; refuses for models in ``refuse``."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.refuse: set[str] = set()
        self.garble: set[str] = set()

    async def chat(self, model, messages, *, purpose="", **_kw):
        self.calls.append((model, purpose))
        content = messages[0]["content"]
        if purpose == "manhwa summary":
            return json.dumps({"summary": "Mina meets the stranger on the rooftop.",
                               "rename": {"unknown-1": "the stranger"},
                               "characters": [{"name": "Mina", "aliases": ["Min"], "appearance": "short black hair"}]}), {}
        if model in self.refuse:
            raise AtlasError("Atlas error 400: content policy")
        if model in self.garble:
            return "I cannot help with that.", {}
        images = [p for p in content if p.get("type") == "image_url"]
        tiles = [{"scene": f"Mina waits on the rooftop, tile {i}", "setting": "rooftop", "mood": "tense",
                  "characters": [{"name": "Mina", "appearance": "short black hair, school uniform", "action": "waits"},
                                 {"name": "unknown-1", "appearance": "hooded figure", "action": "watches"}],
                  "dialogue": [{"speaker": "Mina", "text": "늦었네", "lang": "ko", "text_en": "You're late"}],
                  "sfx": ["whoosh"], "tags": ["rooftop", "night"]} for i in range(len(images))]
        return json.dumps({"tiles": tiles, "summary": "They meet."}), {}


class FakeService:
    def __init__(self, tmp: str):
        self.settings = SimpleNamespace(db_path=os.path.join(tmp, "jobs.sqlite3"), agent_vision_model="vis-a, vis-b")
        self.atlas = FakeChat()
        self.assets: dict[str, dict] = {}
        self.store = SimpleNamespace(get_asset=self.assets.get)

    def llm(self):
        return SimpleNamespace(agent_vision_model_override="")

    async def model_catalogue(self):
        return []  # no catalogue: the configured chain goes out as-is

    async def model_for(self, role):
        return "summary-model"

    async def add_asset(self, filename, fileobj, content_type=None, size=None, *, collection=None, tags=None, source=None):
        asset_id = f"a{len(self.assets) + 1:03d}"
        asset = {"id": asset_id, "kind": "image", "filename": filename, "collection": collection, "tags": tags,
                 "source": source}
        self.assets[asset_id] = asset
        return asset

    def asset_view(self, asset):
        return dict(asset, thumb_url=f"/thumb/{asset['id']}", file_url=f"/file/{asset['id']}")


class Site:
    """Chapters 1..last at /series/test/chapter-N, two panels each; chapter 2 has no next button, so the run
    has to guess chapter-3 from the URL."""

    def __init__(self, last: int = 3):
        self.last = last
        self.requests: list[httpx.Request] = []
        self.panel = jpeg()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.startswith("/img/"):
            if request.headers.get("referer", "").startswith("https://read.example/series/test/chapter-"):
                return httpx.Response(200, content=self.panel, headers={"content-type": "image/jpeg"})
            return httpx.Response(403)
        if path in ("/static/logo.png", "/spinner.gif"):
            return httpx.Response(200, content=b"GIF89a", headers={"content-type": "image/gif"})
        if path.startswith("/series/test/chapter-"):
            n = int(path.rsplit("-", 1)[1])
            if n > self.last:
                return httpx.Response(404, text="not found", headers={"content-type": "text/html"})
            nxt = f"/series/test/chapter-{n + 1}" if n != 2 else None
            return httpx.Response(200, text=chapter_html(n, [f"/img/{n}-1.jpg", f"/img/{n}-2.jpg"], nxt),
                                  headers={"content-type": "text/html; charset=utf-8"})
        return httpx.Response(404)


class Runs(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hawk_manhwa_test_")
        self.service = FakeService(self.tmp)
        self.site = Site()
        self.manager = ManhwaManager(self.service, transport=httpx.MockTransport(self.site.handler), delay=(0, 0))

    async def asyncTearDown(self):
        await self.manager.stop()

    async def run_to_end(self, **kwargs) -> dict:
        state = await self.manager.start("https://read.example/series/test/chapter-1", "Test Series", **kwargs)
        return await self.manager.wait(state["id"], 30)

    async def test_follows_next_chapters_then_stops_on_404(self):
        run = await self.run_to_end(follow_next=True, max_chapters=10)
        self.assertEqual(run["status"], "done", run)
        self.assertEqual(run["chapters_done"], 3)  # 1, 2, then 3 guessed from the URL, then 4 is a 404
        self.assertIn("did not load", run["stop_reason"])
        chapters = self.manager.store.list_chapters("Test Series")
        self.assertEqual([c["chapter"] for c in chapters], ["1", "2", "3"])
        self.assertTrue(all(c["status"] == "done" for c in chapters))
        # two panels per chapter, logo and spinner never stored, panels fetched with the chapter as Referer
        self.assertEqual(len(self.service.assets), 6)
        self.assertEqual(self.service.assets["a001"]["collection"], "manhwa/Test Series/1")

    async def test_chapter_limit_and_resume_skips_done_chapters(self):
        run = await self.run_to_end(follow_next=True, max_chapters=2)
        self.assertEqual(run["chapters_done"], 2)
        self.assertEqual(run["stop_reason"], "Reached the chapter limit.")
        again = await self.run_to_end(follow_next=True, max_chapters=3)
        self.assertEqual([s["chapter"] for s in again["skipped"]], ["1", "2"])
        self.assertEqual(again["chapters_done"], 1)
        self.assertEqual(len(self.manager.store.list_chapters("Test Series")), 3)

    async def test_catalogue_search_and_roster(self):
        await self.run_to_end()
        chapter = self.manager.store.list_chapters("Test Series")[0]
        self.assertEqual(chapter["summary"], "Mina meets the stranger on the rooftop.")
        view = self.manager.chapter_view(chapter["id"])
        self.assertEqual(len(view["pages"]), 4)  # two 690x2400 panels -> two tiles each
        self.assertEqual(view["pages"][0]["thumb_url"], "/thumb/a001")
        # unknown-1 renamed by the wrap-up everywhere it appears
        names = {p["name"] for page in view["pages"] for p in page["characters"]}
        self.assertEqual(names, {"Mina", "the stranger"})

        hits = self.manager.search("late rooftop")  # English translation + tag
        self.assertEqual(len(hits), 4)
        self.assertEqual(hits[0]["chapter"], "1")
        self.assertEqual(len(self.manager.search("늦었네")), 4)  # original Korean text
        self.assertEqual(self.manager.search("dragon"), [])
        self.assertEqual(self.manager.search('"unbalanced'), [])  # FTS syntax never reaches the user as an error

        roster = {c["name"]: c for c in self.manager.characters("Test Series")}
        self.assertEqual(roster["Mina"]["chapters"], ["1"])
        self.assertEqual(roster["Mina"]["aliases"], ["Min"])
        self.assertIn("the stranger", roster)

    async def test_vision_ladder_and_failed_tiles(self):
        self.service.atlas.refuse.add("vis-a")
        await self.run_to_end()
        catalogue_calls = [m for m, purpose in self.service.atlas.calls if purpose == "manhwa catalogue"]
        self.assertEqual(catalogue_calls, ["vis-a", "vis-b"])  # one batch of 4 tiles: refused, then answered
        chapter = self.manager.store.list_chapters("Test Series")[0]
        self.assertEqual(chapter["models"], ["vis-b"])

        self.service.atlas.garble.add("vis-b")
        site2 = Site()
        manager2 = ManhwaManager(self.service, ManhwaStore(os.path.join(self.tmp, "other.sqlite3")),
                                 transport=httpx.MockTransport(site2.handler), delay=(0, 0))
        state = await manager2.start("https://read.example/series/test/chapter-1", "Test Series")
        run = await manager2.wait(state["id"], 30)
        self.assertEqual(run["status"], "done")
        chapter = manager2.store.list_chapters("Test Series")[0]
        self.assertEqual(chapter["status"], "done")
        self.assertEqual(chapter["failed_tiles"], 4)
        pages = manager2.store.pages(chapter["id"])
        self.assertTrue(all(p["status"] == "failed" and "no usable JSON" in p["error"] for p in pages))

    async def test_bad_input_and_page_without_panels(self):
        from hawk_api.jobs import RequestError

        with self.assertRaises(RequestError):
            await self.manager.start("ftp://x", "S")
        with self.assertRaises(RequestError):
            await self.manager.start("https://read.example/series/test/chapter-1", "  ")
        state = await self.manager.start("https://read.example/series/test/chapter-99", "Test Series")
        run = await self.manager.wait(state["id"], 10)
        self.assertEqual(run["chapters_done"], 0)
        self.assertIn("did not load", run["stop_reason"])

    async def test_cancel_stops_after_current_chapter(self):
        state = await self.manager.start("https://read.example/series/test/chapter-1", "Test Series",
                                         follow_next=True, max_chapters=10)
        self.manager.cancel(state["id"])
        run = await self.manager.wait(state["id"], 30)
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(run["chapters_done"], 1)


if __name__ == "__main__":
    unittest.main()
