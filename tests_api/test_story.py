"""Story page: the co-writer chat, chapter drafts, the cast's pictures, and chapters turned into one main film.

Reuses the stand-in services of test_manhwa (fake plan and render jobs, real one-second ffmpeg clips).

    python -m unittest tests_api.test_story
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from hawk_api.atlas import AtlasError
    from hawk_api.jobs import NotFound, RequestError, Unavailable
    from hawk_api.story import DEFAULT_MODELS, PLAN_MODELS, StoryManager, chapter_brief
    from test_manhwa import PipelineService, duration
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"story test dependencies missing: {exc}")

DRAFT = {"title": "The rooftop", "summary": "Mina waits; Jun arrives late.", "scenes": [
    {"description": "Mina leans on the rail at dusk, city lights behind her.", "setting": "rooftop, dusk",
     "characters": ["Mina"], "dialogue": [{"speaker": "narration", "text": "She had waited an hour."}]},
    {"description": "Jun steps out of the stairwell, out of breath.", "setting": "rooftop door",
     "characters": ["Mina", "Jun"], "dialogue": [{"speaker": "Jun", "text": "Sorry, the trains."},
                                                 {"speaker": "Mina", "text": "You always say that."}]}]}


class WriterChat:
    """The co-writer: models in ``down`` fail; chat answers in text, a draft request gets DRAFT."""

    def __init__(self):
        self.calls, self.down, self.sent = [], set(), []

    async def chat(self, model, messages, *, purpose="", json_mode=True, **_kw):
        self.calls.append((model, purpose))
        self.sent.append(messages)
        if model in self.down:
            raise AtlasError(f"{model} is unavailable")
        if purpose == "story chapter":
            return json.dumps(DRAFT), {}
        return f"Nice -- {model} says: what if Jun is late because he was hiding something?", {}


class Story(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hawk_story_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.service = PipelineService(self.tmp)
        self.chat = WriterChat()
        self.service.atlas = self.chat
        self.service.retry = self.service.retry
        self.manager = StoryManager(self.service)
        self.manager.poll = 0.01
        self.story = self.manager.create_story("Night Shift", "Two coworkers fall for each other.", "slow-burn, neon")

    async def asyncTearDown(self):
        await self.manager.stop()

    async def finish(self, chapter_id, seconds=30):
        for _ in range(int(seconds / 0.02)):
            chapter = self.manager.store.get_chapter(chapter_id)
            if chapter["status"] not in ("planning", "rendering", "appending"):
                return chapter
            await asyncio.sleep(0.02)
        self.fail(f"still working: {self.manager.store.get_chapter(chapter_id)}")

    async def drafted(self):
        await self.manager.chat(self.story["id"], "Mina waits on the roof for Jun.")
        return await self.manager.draft_chapter(self.story["id"])

    async def test_chat_uses_gemma_first_and_falls_back(self):
        self.chat.down = {"google/gemma-4-31b-it"}
        out = await self.manager.chat(self.story["id"], "Start on a rooftop at night.")
        self.assertEqual([m for m, _ in self.chat.calls], ["google/gemma-4-31b-it", "deepseek-ai/deepseek-v4.1-flash"])
        self.assertIn("deepseek", out["assistant"]["text"])
        self.assertEqual(out["assistant"]["model"], "deepseek-ai/deepseek-v4.1-flash")
        system = self.chat.sent[0][0]["content"]
        self.assertIn("Two coworkers fall for each other.", system)
        self.assertIn("adult", system)
        self.assertEqual([m["role"] for m in self.manager.messages(self.story["id"])], ["user", "assistant"])
        self.chat.down = set(DEFAULT_MODELS.split(", ")) | {"deepseek-ai/deepseek-v4.1-flash", "xai/grok-4.6", "xai/grok-4.3"}
        with self.assertRaises(Unavailable):
            await self.manager.chat(self.story["id"], "next")

    async def test_a_draft_from_the_chat_and_new_names_join_the_cast(self):
        with self.assertRaises(RequestError):
            await self.manager.draft_chapter(self.story["id"])
        chapter = await self.drafted()
        self.assertEqual((chapter["title"], chapter["status"], len(chapter["scenes"])), ("The rooftop", "draft", 2))
        self.assertEqual(chapter["scenes"][1]["dialogue"][0], {"speaker": "Jun", "text": "Sorry, the trains."})
        cast = {c["name"]: c for c in self.manager.view(self.story["id"])["characters"]}
        self.assertEqual({n: c["role"] for n, c in cast.items()}, {"Mina": "extra", "Jun": "extra"})
        prompt = self.chat.sent[-1][0]["content"]
        self.assertIn("USER: Mina waits on the roof for Jun.", prompt)
        # the next chapter is drafted from what was said after this one
        self.assertEqual(self.manager._since_last_chapter(self.story["id"]), [])

    async def test_pictures_cast_first_nine_at_most_and_the_rules(self):
        chapter = await self.drafted()
        sid = self.story["id"]
        art = [(await self.service.add_asset(f"a{n}.png", io.BytesIO(b"x%d" % n), source={"type": "generated"}))["id"]
               for n in range(12)]
        await self.manager.set_character_image(sid, "Mina", art[0])
        view = self.manager.view(sid)
        self.assertEqual(next(c for c in view["characters"] if c["name"] == "Mina")["role"], "cast",
                         "a face makes an extra cast")
        self.manager.edit_chapter(sid, chapter["id"], images=[{"asset_id": a, "label": f"place {n}"} for n, a in enumerate(art[1:10])])
        story = self.manager._story(sid)
        refs, labels = self.manager._references(story, self.manager.store.get_chapter(chapter["id"]))
        self.assertEqual(len(refs), 9)
        self.assertEqual(refs[0].asset_id, art[0])
        self.assertTrue(labels[0].startswith("Mina"))
        # the rules
        self.manager.add_character(sid, "Little Sora", "a child of eight", "extra")
        with self.assertRaises(RequestError):
            await self.manager.set_character_image(sid, "Little Sora", art[11])
        self.manager.add_character(sid, "Voice", "", "narrator")
        with self.assertRaises(RequestError):
            await self.manager.set_character_image(sid, "Voice", art[11])
        photo = await self.service.add_asset("star.jpg", io.BytesIO(b"p"),
                                             source={"type": "upload", "person_check": {"identifiable": True}})
        with self.assertRaises(RequestError):
            await self.manager.set_character_image(sid, "Jun", photo["id"])

    async def test_the_brief_times_scenes_by_their_lines(self):
        chapter = await self.drafted()
        story = self.manager._story(self.story["id"])
        brief, seconds = chapter_brief(story, self.manager.store.get_chapter(chapter["id"]), [], ["Mina -- dark hair"], 12)
        self.assertEqual(len(seconds), 2)
        self.assertIn("Write exactly 2 segments", brief)
        self.assertIn('Jun: "Sorry, the trains."', brief)
        self.assertIn('Narration (voice-over): "She had waited an hour."', brief)
        self.assertIn("<Picture 1> = Mina -- dark hair", brief)

    async def test_chapters_grow_one_film_and_a_redo_replaces_in_place(self):
        sid = self.story["id"]
        first = await self.drafted()
        self.manager.make_video(sid, first["id"])
        done = await self.finish(first["id"])
        self.assertEqual(done["status"], "done", done)
        plan = self.service.plans[0]
        self.assertEqual(plan.model, PLAN_MODELS, "plans use their own chain: Gemma, DeepSeek, Kimi")
        self.assertEqual(PLAN_MODELS.split(", "), ["google/gemma-4-31b-it", "deepseek-ai/deepseek-v4.1-flash",
                                                   "moonshotai/kimi-k2.6"])
        self.assertEqual(plan.segment_count, 2)
        film = self.manager.view(sid)["film"]
        self.assertEqual(film["chapters"], [first["id"]])

        await self.manager.chat(sid, "Next they go downstairs.")
        second = await self.manager.draft_chapter(sid)
        self.manager.make_video(sid, second["id"])
        await self.finish(second["id"])
        film = self.manager.view(sid)["film"]
        self.assertEqual(film["chapters"], [first["id"], second["id"]])
        asset = self.service.assets[film["asset_id"]]
        self.assertEqual((asset["kind"], asset["collection"]), ("video", "story/Night Shift"))

        with self.assertRaises(RequestError):
            self.manager.make_video(sid, first["id"])  # already has a video: redo it instead
        old_job = self.manager.store.get_chapter(first["id"])["render_job_id"]
        self.manager.make_video(sid, first["id"], redo=True)
        redone = await self.finish(first["id"])
        self.assertNotEqual(redone["render_job_id"], old_job)
        self.assertEqual(redone["history"][0]["render_job_id"], old_job)
        film = self.manager.view(sid)["film"]
        self.assertEqual(film["chapters"], [first["id"], second["id"]], "the redone chapter keeps its place")
        self.assertEqual(self.service.assets[film["asset_id"]]["source"]["jobs"][0], redone["render_job_id"])

    async def test_a_failed_plan_is_retried_once_then_reported(self):
        self.service.fail_plans = {1}
        chapter = await self.drafted()
        self.manager.make_video(self.story["id"], chapter["id"])
        done = await self.finish(chapter["id"])
        self.assertEqual((done["status"], len(self.service.retried)), ("done", 1))

    async def test_a_restart_carries_on(self):
        chapter = await self.drafted()
        self.service.finish_videos = False
        self.manager.make_video(self.story["id"], chapter["id"])
        for _ in range(500):
            if self.manager.store.get_chapter(chapter["id"]).get("render_job_id"):
                break
            await asyncio.sleep(0.02)
        await self.manager.stop()  # the runtime goes away mid-render
        self.assertEqual(self.manager.store.get_chapter(chapter["id"])["status"], "rendering")
        for job in self.service.jobs.values():
            job["status"] = "done"
        fresh = StoryManager(self.service)
        fresh.poll = 0.01
        self.manager = fresh
        self.assertEqual(await fresh.resume(wait_for_gpu=0), [chapter["id"]])
        done = await self.finish(chapter["id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(len(self.service.videos), 1, "the render was not started twice")

    async def test_editing_and_deleting(self):
        chapter = await self.drafted()
        sid = self.story["id"]
        edited = self.manager.edit_chapter(sid, chapter["id"], title="Rooftop, revised",
                                           scenes=[{"description": "Only Mina.", "characters": ["Mina"], "dialogue": []}])
        self.assertEqual((edited["title"], len(edited["scenes"])), ("Rooftop, revised", 1))
        with self.assertRaises(RequestError):
            self.manager.edit_chapter(sid, chapter["id"], scenes=[{}])
        self.manager.edit_character(sid, "Mina", new_name="Mina Park")
        self.assertEqual(self.manager.store.get_chapter(chapter["id"])["scenes"][0]["characters"], ["Mina Park"])
        self.manager.delete_chapter(sid, chapter["id"])
        self.assertEqual(self.manager.view(sid)["chapters"], [])
        await self.manager.delete_story(sid)
        with self.assertRaises(NotFound):
            self.manager.view(sid)


if __name__ == "__main__":
    unittest.main()
