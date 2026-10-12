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

    def test_thumbnails_and_small_declared_images_dropped(self):
        html = ('<div id="_imageList"><img class="_images" width="700" src="bg.png" data-url="https://cdn.example/p1.jpg">'
                '<img class="_images" width="700" src="bg.png" data-url="https://cdn.example/p2.jpg"></div>'
                '<img class="_thumbnailImages" width="92" data-url="https://cdn.example/ep2-thumb.jpg">'
                '<img class="author" width="40" src="https://cdn.example/a.jpg">')
        self.assertEqual(parse_chapter(html, "https://s.example/v?episode_no=1")["images"],
                         ["https://cdn.example/p1.jpg", "https://cdn.example/p2.jpg"])

    def test_chapter_number_from_query_parameter_first(self):
        url = "https://www.webtoons.com/en/fantasy/x/season-1-ep-0/viewer?title_no=95&episode_no=1"
        self.assertEqual(chapter_number(url, "[Season 1] Ep. 0 | X"), "1")
        self.assertEqual(increment_url(url), url.replace("episode_no=1", "episode_no=2"))

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
        self.delay = 0.0

    async def chat(self, model, messages, *, purpose="", **_kw):
        self.calls.append((model, purpose))
        if self.delay:
            await asyncio.sleep(self.delay)
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
        self.store = SimpleNamespace(get_asset=self.assets.get, all_assets=lambda: list(self.assets.values()))

    def delete_asset(self, asset_id):
        from hawk_api.jobs import NotFound

        if self.assets.pop(asset_id, None) is None:
            raise NotFound(asset_id)

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

    async def test_cancel_mid_chapter_keeps_described_tiles(self):
        self.site.panel = jpeg((690, 7000))  # 5 tiles a panel, 10 a chapter: three calls of up to 4 tiles
        self.service.atlas.delay = 0.3
        state = await self.manager.start("https://read.example/series/test/chapter-1", "Test Series",
                                         follow_next=True, max_chapters=10)
        for _ in range(100):  # until the first batch is described and the second is being asked for
            await asyncio.sleep(0.05)
            if (state["current"] or {}).get("tiles_done", 0) >= 4:
                break
        run = await self.manager.cancel_and_wait(state["id"])
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(run["stop_reason"], "Cancelled.")
        self.assertEqual(run["chapters_done"], 0)
        chapter = self.manager.store.list_chapters("Test Series")[0]
        self.assertEqual(chapter["status"], "cancelled")
        self.assertEqual(chapter["tiles"], 4)
        self.assertEqual(len(self.manager.store.pages(chapter["id"])), 4)
        calls = [p for _m, p in self.service.atlas.calls]
        self.assertNotIn("manhwa summary", calls)
        self.assertLessEqual(calls.count("manhwa catalogue"), 2)  # the call in flight when it was cancelled

    async def test_one_run_per_series_and_names_that_differ_only_in_spelling_are_one_series(self):
        self.service.atlas.delay = 0.3
        first = await self.manager.start("https://read.example/series/test/chapter-1", "Test Series")
        with self.assertRaises(Exception) as caught:  # same series, spelt as a slug
            await self.manager.start("https://read.example/series/test/chapter-1", "test-series")
        self.assertIn(first["id"], str(caught.exception))
        await self.manager.wait(first["id"], 30)
        again = await self.manager.start("https://read.example/series/test/chapter-2", "TEST-SERIES")
        self.assertEqual(again["series"], "Test Series")
        await self.manager.wait(again["id"], 30)
        self.assertEqual([s["series"] for s in self.manager.store.series()], ["Test Series"])
        # every panel stored once: two chapters x two panels
        self.assertEqual(len(self.service.assets), 4)

    async def test_delete_chapter_and_series_with_images(self):
        run = await self.run_to_end(follow_next=True, max_chapters=2)
        self.assertEqual(run["chapters_done"], 2)
        first, second = self.manager.store.list_chapters("Test Series")
        out = await self.manager.delete("test-series", first["id"])
        self.assertEqual(out, {"series": "Test Series", "chapters_deleted": 1, "images_deleted": 2, "runs_cancelled": []})
        self.assertEqual([c["id"] for c in self.manager.store.list_chapters("Test Series")], [second["id"]])
        self.assertEqual(len(self.service.assets), 2)
        self.assertEqual(len(self.manager.search("rooftop")), 4)  # only the second chapter's tiles are left
        self.assertEqual(self.manager.characters("Test Series")[0]["name"], "Mina")
        out = await self.manager.delete("Test Series")
        self.assertEqual((out["chapters_deleted"], out["images_deleted"]), (1, 2))
        self.assertEqual(self.manager.store.series(), [])
        self.assertEqual(self.manager.characters("Test Series"), [])
        self.assertEqual(self.manager.search("rooftop"), [])
        self.assertEqual(self.service.assets, {})

    async def test_delete_cancels_a_running_run_first(self):
        self.service.atlas.delay = 0.3
        state = await self.manager.start("https://read.example/series/test/chapter-1", "Test Series")
        for _ in range(100):
            await asyncio.sleep(0.05)
            if (state["current"] or {}).get("step") == "cataloguing":
                break
        out = await self.manager.delete("Test Series")
        self.assertEqual(out["runs_cancelled"], [state["id"]])
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(self.manager.store.series(), [])
        self.assertEqual(self.service.assets, {})

    async def test_deleting_another_chapter_leaves_the_run_alone(self):
        await self.run_to_end()  # chapter 1 catalogued
        done = self.manager.store.list_chapters("Test Series")[0]
        self.service.atlas.delay = 0.3
        state = await self.manager.start("https://read.example/series/test/chapter-2", "Test Series")
        for _ in range(100):
            await asyncio.sleep(0.05)
            if (state["current"] or {}).get("step") == "cataloguing":
                break
        out = await self.manager.delete("Test Series", done["id"])
        self.assertEqual(out["runs_cancelled"], [])
        self.assertEqual(state["status"], "running")
        busy = next(c for c in self.manager.store.list_chapters("Test Series") if c["chapter"] == "2")
        out = await self.manager.delete("Test Series", busy["id"])
        self.assertEqual(out["runs_cancelled"], [state["id"]], "the chapter being catalogued stops its run")


if __name__ == "__main__":
    unittest.main()


def strip_with_panels(heights=(900, 1200), gutter=80, width=800) -> bytes:
    """A webtoon strip: noisy panels separated by flat white gutters, with a short caption box between them."""
    total = gutter + sum(h + gutter for h in heights) + 60 + gutter
    page = Image.new("RGB", (width, total), (255, 255, 255))
    y = gutter
    for h in heights:
        page.paste(Image.effect_noise((width, h), 80).convert("RGB"), (0, y))
        y += h + gutter
    page.paste(Image.effect_noise((width, 60), 80).convert("RGB"), (0, y))  # a caption box: too short to keep
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return buffer.getvalue()


class Panels(unittest.TestCase):
    def test_panels_found_at_gutters_and_matched_to_tiles(self):
        panels, tiles = manhwa.find_panels(strip_with_panels())
        self.assertEqual(len(panels), 2, panels)
        (a_top, a_bottom), (b_top, b_bottom) = panels
        self.assertLess(abs(a_top - 80), 4)
        self.assertLess(abs((a_bottom - a_top) - 900), 6)
        self.assertLess(abs((b_bottom - b_top) - 1200), 6)
        self.assertEqual(tiles, len(slice_image(strip_with_panels())))
        cut = manhwa.cut_panels(strip_with_panels())
        self.assertEqual([part for part, _ in cut], [0, 1])  # the second panel's middle (1660) is past 1480 px

    def test_a_tall_panel_is_cut_into_parts(self):
        panels, _ = manhwa.find_panels(strip_with_panels(heights=(5000,)))
        self.assertEqual(len(panels), 3)
        self.assertTrue(all(b - t <= manhwa.MAX_PANEL for t, b in panels))

    def test_prompt_carries_the_catalogue(self):
        tile = {"status": "ok", "scene": "Mina waits on the rooftop.", "setting": "rooftop at dusk",
                "characters": [{"name": "Mina", "appearance": "short black hair, grey coat"}]}
        text = manhwa.real_prompt(tile, "soft light")
        for part in ("realistic photograph of real adult people", "Mina waits", "short black hair", "soft light"):
            self.assertIn(part, text)
        self.assertEqual(manhwa.real_prompt({"status": "failed"}), manhwa.REAL_PROMPT)


class Lora(SimpleNamespace):
    def __init__(self, **kw):
        super().__init__(**{"label": "", "kind": "other", "range": (0.0, 2.0), "notes": "", "trigger": "", **kw})


class ConvertService(FakeService):
    def __init__(self, tmp):
        super().__init__(tmp)
        self.data: dict[str, bytes] = {}
        self.calls: list[dict] = []
        self.refuse_prompt = ""
        self.busy = 0
        self.local_images = SimpleNamespace(catalogue=self.catalogue)

    async def catalogue(self, family=""):
        return [Lora(file="Qwen Image2.1_Anime2Real.safetensors", strength=1.0, installed=True),
                Lora(file="qwen21/Anime2Real_LAtest.safetensors", strength=0.0, installed=True),
                Lora(file="Qwen2.1_Anything2RealCharacters.safetensors", strength=0.7, installed=False),
                Lora(file="lenovo_qwen21.safetensors", strength=0.55, installed=True),
                Lora(file="PornMaster_QI2.1_Age_Slider_V1.safetensors", strength=2.0, installed=True)]

    async def add_asset(self, filename, fileobj, content_type=None, size=None, *, collection=None, tags=None, source=None):
        asset = await super().add_asset(filename, fileobj, content_type, size, collection=collection, tags=tags,
                                        source=source)
        fileobj.seek(0)
        self.data[asset["id"]] = fileobj.read()
        return asset

    async def asset_bytes(self, asset):
        return self.data[asset["id"]]

    def update_asset(self, asset_id, *, collection=None, add_tags=None, **_kw):
        self.assets[asset_id].update(collection=collection, tags=add_tags)

    async def generate_images(self, prompt, *, reference_asset_ids=(), engine=None, loras=None, negative="",
                              steps=None, cfg=None, **_kw):
        from hawk_api.jobs import RequestError

        if self.busy:
            self.busy -= 1
            raise RequestError("Qwen Image 2.1: ComfyUI is busy (1 job(s) running or queued, usually a video render).")
        self.calls.append({"prompt": prompt, "refs": list(reference_asset_ids), "engine": engine, "loras": loras,
                           "negative": negative, "steps": steps, "cfg": cfg})
        if self.refuse_prompt and self.refuse_prompt in prompt:
            raise RequestError("Refused: the prompt mentions 'teen'.")
        made = await self.add_asset("gen.png", io.BytesIO(b"png"), collection="Generated",
                                    source={"type": "generated", "references": list(reference_asset_ids)})
        return {"assets": [self.asset_view(made)]}


class Convert(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hawk_manhwa_convert_")
        self.service = ConvertService(self.tmp)
        self.manager = ManhwaManager(self.service, delay=(0, 0))
        self.manager.busy_wait = 0
        page = await self.service.add_asset("p001.png", io.BytesIO(strip_with_panels()), collection="manhwa/S/1",
                                            source={"type": "manhwa"})
        self.manager.store.save_chapter({"id": "ch1", "series": "S", "chapter": "1", "url": "https://x/1",
                                         "status": "done", "created_at": 1.0})
        self.manager.store.save_pages("ch1", "S", [
            {"id": "ch1-0000", "idx": 0, "asset_id": page["id"], "part": 0, "status": "ok",
             "scene": "Mina waits on the rooftop.", "characters": [{"name": "Mina", "appearance": "short black hair"}]},
            {"id": "ch1-0001", "idx": 1, "asset_id": page["id"], "part": 1, "status": "ok",
             "scene": "A stranger arrives.", "characters": []},
        ])

    async def asyncTearDown(self):
        await self.manager.stop()

    async def test_each_panel_becomes_a_photograph_with_the_latest_anime2real(self):
        state = await self.manager.convert("ch1")
        self.assertEqual(state["loras"], [{"name": "qwen21/Anime2Real_LAtest.safetensors", "strength": 1.0},
                                          {"name": "lenovo_qwen21.safetensors", "strength": 0.5},
                                          {"name": "PornMaster_QI2.1_Age_Slider_V1.safetensors", "strength": 1.0}])
        state = await self.manager.wait_conversion(state["id"], 30)
        self.assertEqual((state["status"], state["panels_total"], state["converted"]), ("done", 2, 2), state)
        first, second = self.service.calls
        self.assertEqual(first["engine"], "qwen21")
        self.assertIn("Mina waits on the rooftop", first["prompt"])
        self.assertIn("A stranger arrives", second["prompt"])
        self.assertIn("anime", first["negative"])
        panel = self.service.assets[first["refs"][0]]
        self.assertEqual((panel["source"]["type"], panel["collection"]), ("manhwa", "manhwa/S/1/panels"))
        made = self.service.assets[state["results"][0]["asset_ids"][0]]
        self.assertEqual(made["collection"], "manhwa/S/1/real")
        view = self.manager.chapter_view("ch1")
        self.assertEqual([len(p["real"]) for p in view["pages"]], [1, 1])

    async def test_refusals_are_counted_and_the_run_goes_on_and_busy_waits(self):
        self.service.refuse_prompt = "Mina"
        self.service.busy = 2
        state = await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        self.assertEqual((state["converted"], state["refused"], state["failed"]), (1, 1, 0), state)
        self.assertTrue(state["results"][0]["error"].startswith("Refused"))

    async def test_tile_ids_loras_and_one_at_a_time(self):
        mine = [{"name": "Qwen Image2.1_Anime2Real.safetensors", "strength": 0.9}]
        state = await self.manager.convert("ch1", tile_ids=["ch1-0001"], loras=mine)
        with self.assertRaises(Exception):
            await self.manager.convert("ch1")
        state = await self.manager.wait_conversion(state["id"], 30)
        self.assertEqual(state["panels_total"], 1)
        self.assertEqual(self.service.calls[0]["loras"], mine)

    async def test_no_anime2real_installed_says_so(self):
        from hawk_api.jobs import RequestError

        async def none(family=""):
            return []
        self.service.local_images.catalogue = none
        with self.assertRaises(RequestError):
            await self.manager.convert("ch1")


class PlanService(ConvertService):
    def __init__(self, tmp):
        super().__init__(tmp)
        self.plans = []

    async def create_plan(self, request):
        self.plans.append(request)
        return {"id": f"job{len(self.plans)}"}


class Plans(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hawk_manhwa_plan_")
        self.service = getattr(self, "service_class", PlanService)(self.tmp)
        self.manager = ManhwaManager(self.service, delay=(0, 0))
        self.manager.busy_wait = 0
        self.manager.store.save_chapter({"id": "ch1", "series": "S", "chapter": "1", "url": "https://x/1",
                                         "status": "done", "created_at": 1.0, "summary": "Mina meets a stranger."})
        pages = []
        for n in range(6):  # six tiles, two photographs each: twelve shots -> plans of 9 and 3
            photos = [(await self.service.add_asset(f"real{n}{k}.png", io.BytesIO(b"x"), source={"type": "generated"}))["id"]
                      for k in range(2)]
            pages.append({"id": f"ch1-{n:04d}", "idx": n, "asset_id": "page", "part": n, "status": "ok",
                          "scene": f"scene {n}", "characters": [{"name": "Mina", "appearance": "black hair"}],
                          "dialogue": [{"speaker": "Mina", "text": "늦었네", "text_en": f"line {n}"},
                                       {"speaker": "narration", "text": "Later.", "text_en": ""}],
                          "real": [{"panel_asset_id": f"p{n}{k}", "asset_ids": [photos[k]]} for k in range(2)]})
        self.manager.store.save_pages("ch1", "S", pages)

    async def test_one_plan_per_nine_photographs_with_dialogue_spoken_once(self):
        out = await self.manager.plan_chapter("ch1")
        self.assertEqual([p["segments"] for p in out["plans"]], [9, 3])
        first, second = self.service.plans
        self.assertEqual((first.segment_count, first.aspect_ratio, len(first.references)), (9, "9:16", 9))
        self.assertTrue(all(r.role == "picture" for r in first.references))
        self.assertEqual(first.references[0].asset_id, self.manager.shots("ch1")[0]["asset_id"])
        self.assertIn("part 1 of 2", first.story)
        self.assertEqual(first.story.count('Mina: "line 0"'), 1, "a tile's line goes on its first panel only")
        self.assertIn('Narration (voice-over): "Later."', first.story)
        self.assertIn("Mina meets a stranger.", first.story)
        self.assertIn('Mina: "line 5"', second.story)
        self.assertEqual(self.manager.store.get_chapter("ch1")["plans"][1]["job_id"], "job2")

    async def test_no_photographs_and_bad_settings_are_request_errors(self):
        from hawk_api.jobs import RequestError

        with self.assertRaises(RequestError):
            await self.manager.plan_chapter("ch1", aspect_ratio="2:7")
        self.manager.store.save_pages("ch1", "S", [{"id": "ch1-0000", "idx": 0, "asset_id": "page", "part": 0}])
        with self.assertRaises(RequestError):
            await self.manager.plan_chapter("ch1")

    async def test_a_conversion_ends_by_planning_the_chapter(self):
        page = await self.service.add_asset("p001.png", io.BytesIO(strip_with_panels()), source={"type": "manhwa"})
        self.manager.store.save_pages("ch1", "S", [
            {"id": "ch1-0000", "idx": 0, "asset_id": page["id"], "part": 0, "status": "ok", "scene": "a"},
            {"id": "ch1-0001", "idx": 1, "asset_id": page["id"], "part": 1, "status": "ok", "scene": "b"}])
        state = await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        self.assertEqual([p["segments"] for p in state["plans"]], [2], state)
        self.assertEqual(len(self.service.plans), 1)


class PanelChat:
    """Writes a panel prompt; models in ``refuse`` fail, and the panel numbered ``skip_call`` reads as a minor."""

    def __init__(self):
        self.calls, self.refuse, self.skip_call = [], set(), 0
        self.drawn, self.identifiable = True, False

    async def chat(self, model, messages, *, purpose="", **_kw):
        self.calls.append((model, purpose))
        if model in self.refuse:
            raise AtlasError("Atlas error 429: too many requests")
        written = sum(1 for _, p in self.calls if p == "panel prompt" and _ not in self.refuse)
        if written == self.skip_call:
            return json.dumps({"skip": "one person looks like a teenager"}), {}
        return json.dumps({"drawn": self.drawn, "identifiable": self.identifiable,
                           "prompt": f"Turn this manhwa panel into a realistic photograph. A woman in her 30s with "
                                     f"short black hair stands at a rooftop rail at dusk, written by {model}.",
                           "negative": "anime, text, extra fingers"}), {}


class AnalyzedConvert(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = Convert.asyncTearDown

    async def asyncSetUp(self):
        await Convert.asyncSetUp(self)  # the same chapter, without re-running Convert's tests
        self.chat = PanelChat()
        self.service.atlas = self.chat

    async def test_grok_writes_each_panel_prompt(self):
        state = await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        self.assertEqual(state["converted"], 2, state)
        self.assertEqual([m for m, p in self.chat.calls if p == "panel prompt"], ["xai/grok-4.3"] * 2)
        first = self.service.calls[0]
        self.assertIn("written by xai/grok-4.3", first["prompt"])
        self.assertEqual(first["negative"], "anime, text, extra fingers")
        self.assertEqual(state["results"][0]["prompt_by"], "xai/grok-4.3")

    async def test_the_vision_chain_takes_over_and_a_minor_is_skipped(self):
        self.chat.refuse = {"xai/grok-4.3"}
        self.chat.skip_call = 2
        state = await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        self.assertEqual((state["converted"], state["skipped"]), (1, 1), state)
        self.assertEqual(state["results"][0]["prompt_by"], "vis-a")
        self.assertEqual(len(self.service.calls), 1, "the skipped panel never reaches the image engine")
        self.assertIn("teenager", state["results"][1]["error"])

    async def test_no_answer_falls_back_to_the_template(self):
        self.chat.refuse = {"xai/grok-4.3", "vis-a", "vis-b"}
        state = await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        self.assertEqual(state["converted"], 2)
        self.assertTrue(self.service.calls[0]["prompt"].startswith(manhwa.REAL_PROMPT))
        self.assertIn("Template prompt used", state["results"][0]["prompt_note"])

    async def test_the_panel_prompt_call_is_the_person_check(self):
        state = await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        panel = self.service.assets[state["results"][0]["panel_asset_id"]]
        check = panel["source"]["person_check"]
        self.assertEqual((check["identifiable"], check["by"], check["model"]), (False, "panel prompt", "xai/grok-4.3"))
        self.assertNotIn("person check", [p for _, p in self.chat.calls])

    async def test_a_real_person_keeps_the_rules_and_an_unclear_answer_records_nothing(self):
        self.chat.identifiable = True
        state = await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        panel = self.service.assets[state["results"][0]["panel_asset_id"]]
        self.assertIs(panel["source"]["person_check"]["identifiable"], True)
        self.chat.identifiable, self.chat.drawn = False, None
        await self.manager.wait_conversion((await self.manager.convert("ch1", tile_ids=["ch1-0001"]))["id"], 30)
        fresh = [a for a in self.service.assets.values()
                 if (a.get("source") or {}).get("tile_id") == "ch1-0001"]
        self.assertTrue(fresh)
        self.assertTrue(all("person_check" not in a["source"] or a["source"]["person_check"]["identifiable"] is True
                            for a in fresh))

    async def test_another_model_can_analyse(self):
        await self.manager.wait_conversion(
            (await self.manager.convert("ch1", prompt_model="deepseek-ai/deepseek-v4.1-flash"))["id"], 30)
        self.assertEqual({m for m, p in self.chat.calls if p == "panel prompt"}, {"deepseek-ai/deepseek-v4.1-flash"})

    async def test_analyze_off_never_calls_a_model(self):
        await self.manager.wait_conversion((await self.manager.convert("ch1", analyze=False))["id"], 30)
        self.assertEqual(self.chat.calls, [])


class Settings(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await Convert.asyncSetUp(self)

    asyncTearDown = Convert.asyncTearDown

    async def test_saved_settings_govern_the_next_conversion(self):
        view = await self.manager.settings_view()
        self.assertIsNone(view["settings"]["loras"])
        self.assertEqual(view["auto_loras"][0]["name"], "qwen21/Anime2Real_LAtest.safetensors")
        self.assertIn("lenovo_qwen21.safetensors", [a["file"] for a in view["available"]])
        mine = [{"name": "Qwen Image2.1_Anime2Real.safetensors", "strength": 0.9}]
        self.manager.save_settings({"loras": mine, "steps": 20, "cfg": 1.5, "negative": "tattoos",
                                    "prompt": "soft light", "analyze": False, "plan_after": False})
        state = await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        self.assertEqual(state["loras"], mine)
        call = self.service.calls[0]
        self.assertEqual((call["loras"], call["steps"], call["cfg"]), (mine, 20, 1.5))
        self.assertTrue(call["negative"].endswith(", tattoos"))
        self.assertIn("soft light", call["prompt"])
        self.assertEqual(state["plans"], [])

    async def test_a_run_can_override_and_empty_loras_go_back_to_automatic(self):
        self.manager.save_settings({"steps": 20, "loras": [{"name": "x", "strength": 1}]})
        self.assertEqual(self.manager.save_settings({"loras": []})["loras"], None)
        state = await self.manager.wait_conversion((await self.manager.convert("ch1", steps=12, analyze=False))["id"], 30)
        self.assertEqual(self.service.calls[0]["steps"], 12)
        self.assertEqual(state["loras"][0]["name"], "qwen21/Anime2Real_LAtest.safetensors")

    async def test_the_settings_load_without_comfyui(self):
        async def down(family=""):
            raise OSError("ComfyUI is not reachable")
        self.service.local_images.catalogue = down
        view = await self.manager.settings_view()
        self.assertEqual((view["auto_loras"], view["available"]), ([], []))
        self.assertIn("not reachable", view["auto_error"])


class Context(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await Convert.asyncSetUp(self)

    asyncTearDown = Convert.asyncTearDown

    CONTEXT = {"characters": "Mina: early 30s, short black hair, grey coat", "setting": "a rooftop at dusk",
               "action": "Mina waits, then a stranger arrives", "one_scene": True}

    async def test_context_goes_into_every_panel_prompt(self):
        self.manager.set_context("ch1", self.CONTEXT)
        self.assertEqual(self.manager.chapter_view("ch1")["context"]["setting"], "a rooftop at dusk")
        chat = PanelChat()
        self.service.atlas = chat
        seen = []
        original = chat.chat

        async def spy(model, messages, **kw):
            seen.append(messages[0]["content"][0]["text"])
            return await original(model, messages, **kw)
        chat.chat = spy
        await self.manager.wait_conversion((await self.manager.convert("ch1"))["id"], 30)
        self.assertIn("one continuous scene", seen[0])
        self.assertIn("Mina: early 30s, short black hair, grey coat", seen[0])

    async def test_template_prompt_and_plan_brief_use_it(self):
        self.manager.set_context("ch1", self.CONTEXT)
        await self.manager.wait_conversion((await self.manager.convert("ch1", analyze=False))["id"], 30)
        self.assertIn("Setting: a rooftop at dusk.", self.service.calls[0]["prompt"])
        brief = manhwa.plan_brief(self.manager.store.get_chapter("ch1"), [{"scene": "x"}], 1, 1)
        self.assertIn("Action: Mina waits, then a stranger arrives", brief)
        self.assertIn("no cuts", brief)

    async def test_clearing_and_unknown_chapter(self):
        self.manager.set_context("ch1", self.CONTEXT)
        self.assertEqual(self.manager.set_context("ch1", {})["context"], {})
        self.assertEqual(manhwa.context_text({}), "")
        from hawk_api.jobs import NotFound
        with self.assertRaises(NotFound):
            self.manager.set_context("nope", self.CONTEXT)


class RenderService(PlanService):
    def __init__(self, tmp):
        super().__init__(tmp)
        self.jobs: dict[str, dict] = {}
        self.videos = []
        self.store.get_job = self.jobs.get

    async def create_plan(self, request):
        job = await super().create_plan(request)
        self.jobs[job["id"]] = {"id": job["id"], "kind": "plan", "status": "planning"}
        return job

    async def create_video(self, request):
        self.videos.append(request)
        job = {"id": f"vid{len(self.videos)}", "kind": "video", "status": "queued"}
        self.jobs[job["id"]] = job
        return job


class RenderAndDelete(unittest.IsolatedAsyncioTestCase):
    service_class = RenderService

    async def asyncSetUp(self):
        self.manager = None

    async def asyncTearDown(self):
        if self.manager is not None:
            await self.manager.stop()

    async def test_render_finished_plans_and_report_the_rest(self):
        await Plans.asyncSetUp(self)  # twelve photographs over six tiles -> plans of 9 and 3
        out = await self.manager.plan_chapter("ch1")
        with self.assertRaises(Exception):
            await self.manager.render_chapter("nope")
        first, second = (p["job_id"] for p in out["plans"])
        self.service.jobs[first]["status"] = "done"
        result = await self.manager.render_chapter("ch1", settings={"unet_name": "beta6"})
        self.assertEqual([r["part"] for r in result["started"]], [1])
        self.assertEqual(result["not_ready"][0]["status"], "planning")
        video = self.service.videos[0]
        self.assertEqual((video.plan_job_id, video.settings.aspect_ratio, video.settings.unet_name), (first, "9:16", "beta6"))
        self.service.jobs[second]["status"] = "done"
        again = await self.manager.render_chapter("ch1")
        self.assertEqual(([r["part"] for r in again["started"]], [r["part"] for r in again["already"]]), ([2], [1]))
        self.service.jobs["vid1"]["status"] = "failed"
        self.assertEqual([r["part"] for r in (await self.manager.render_chapter("ch1"))["started"]], [1])
        self.assertEqual(len(self.manager.store.get_chapter("ch1")["renders"]), 2)

    async def test_no_plans_is_a_request_error(self):
        from hawk_api.jobs import RequestError

        await Plans.asyncSetUp(self)
        self.manager.store.save_chapter({"id": "c0", "series": "S", "chapter": "0", "url": "u", "status": "done",
                                         "created_at": 1.0})
        with self.assertRaises(RequestError):
            await self.manager.render_chapter("c0")

    async def test_deleting_a_chapter_takes_its_photographs_and_panels_but_not_videos(self):
        await Convert.asyncSetUp(self)
        state = await self.manager.wait_conversion((await self.manager.convert("ch1", analyze=False,
                                                                               plan_after=False))["id"], 30)
        photo = state["results"][0]["asset_ids"][0]
        panel = state["results"][0]["panel_asset_id"]
        self.service.assets[photo]["collection"] = "Favourites"  # moved: still found through the catalogue
        video = await self.service.add_asset("film.mp4", io.BytesIO(b"v"), collection="manhwa/S/1/real",
                                             source={"type": "generated"})
        video["kind"] = "video"
        other = await self.service.add_asset("p.png", io.BytesIO(b"o"), collection="manhwa/S/10",
                                             source={"type": "manhwa"})
        out = await self.manager.delete("S", "ch1")
        self.assertNotIn(photo, self.service.assets)
        self.assertNotIn(panel, self.service.assets)
        self.assertIn(video["id"], self.service.assets, "rendered videos are kept")
        self.assertIn(other["id"], self.service.assets, "chapter 10 is not chapter 1")
        self.assertGreaterEqual(out["images_deleted"], 4)


class VideoSettings(unittest.IsolatedAsyncioTestCase):
    service_class = RenderService

    async def asyncSetUp(self):
        await Plans.asyncSetUp(self)

    async def asyncTearDown(self):
        await self.manager.stop()

    async def test_saved_video_settings_drive_plans_and_renders(self):
        self.manager.save_settings({"planner_model": "xai/grok-4.6", "aspect_ratio": "3:4", "segment_seconds": 8,
                                    "render": {"unet_name": "beta6", "steps": 10, "continuity": "tail_39",
                                               "loras": [{"name": "hmnsfw", "strength": 0.7}], "aspect_ratio": "16:9",
                                               "megapixels": None}})
        stored = self.manager.settings()["render"]
        self.assertNotIn("aspect_ratio", stored, "the aspect ratio is set once, outside render")
        self.assertNotIn("megapixels", stored, "blank means the server default")
        out = await self.manager.plan_chapter("ch1")
        self.assertEqual((self.service.plans[0].model, self.service.plans[0].aspect_ratio), ("xai/grok-4.6", "3:4"))
        self.service.jobs[out["plans"][0]["job_id"]]["status"] = "done"
        await self.manager.render_chapter("ch1", parts=[1], settings={"steps": 12})
        render = self.service.videos[0].settings
        self.assertEqual((render.unet_name, render.steps, render.continuity, render.aspect_ratio, render.default_seconds),
                         ("beta6", 12, "tail_39", "3:4", 8))
        self.assertEqual([(l.name, l.strength) for l in render.loras], [("hmnsfw", 0.7)])

    def test_bad_render_settings_are_refused(self):
        from hawk_api.schemas import ManhwaSettingsIn

        with self.assertRaises(ValueError):
            ManhwaSettingsIn(render={"continuity": "sideways"})
        self.assertEqual(ManhwaSettingsIn(render={"steps": 8}).render, {"steps": 8})
