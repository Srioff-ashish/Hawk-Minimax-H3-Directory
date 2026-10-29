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


OUTLINE = {"characters": [{"name": "Mina", "description": "late 20s, short black hair", "role": "cast"},
                          {"name": "Jun", "description": "early 30s, tall, tired eyes", "role": "cast"},
                          {"name": "Voice", "description": "warm, low", "role": "narrator"}],
           "chapters": [{"title": "Chapter 1: The rooftop", "beats": "Mina waits; Jun arrives late."},
                        {"title": "Downstairs", "beats": "They take the stairs down together."}]}


class WriterChat:
    """The co-writer: models in ``down`` fail; chat answers in text, a draft request gets DRAFT, an outline OUTLINE.
    ``delay`` makes every call slow, as a reasoning model is."""

    def __init__(self):
        self.calls, self.down, self.sent, self.delay, self.outline_size = [], set(), [], 0.0, None

    async def chat(self, model, messages, *, purpose="", json_mode=True, **_kw):
        self.calls.append((model, purpose))
        self.sent.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        if model in self.down:
            raise AtlasError(f"{model} is unavailable")
        if purpose == "story outline":
            if self.outline_size:  # a stingy model: this many chapters per reply, whatever was asked
                n = sum(1 for _, p in self.calls if p == purpose)
                return json.dumps({"characters": OUTLINE["characters"], "chapters": [
                    {"title": f"Part {n}.{i}", "beats": f"Beat {n}.{i}."} for i in range(self.outline_size)]}), {}
            return json.dumps(OUTLINE), {}
        if purpose == "story chapter":
            draft = dict(DRAFT, title="Chapter 9: " + DRAFT["title"]) if "PLAN FOR THIS CHAPTER" in messages[0]["content"] else DRAFT
            return json.dumps(draft), {}
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
        self.assertEqual([m for m, _ in self.chat.calls][:2], ["google/gemma-4-31b-it", "deepseek-ai/deepseek-v4.1-flash"])
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

    async def test_a_storyboard_opens_each_scene_on_its_panel(self):
        chapter = await self.drafted()
        sid = self.story["id"]
        face = (await self.service.add_asset("mina.png", io.BytesIO(b"mina"), source={"type": "generated"}))["id"]
        panels = [(await self.service.add_asset(f"p{n}.png", io.BytesIO(b"panel%d" % n), source={"type": "generated"}))["id"]
                  for n in range(2)]
        await self.manager.set_character_image(sid, "Mina", face)

        class Boards:
            made = {}

            def create(self, **kwargs):
                self.made = kwargs
                return {"id": "board1", "status": "drawing"}

            def board(self, board_id):
                return {"id": board_id, "status": "ready", "panels": panels}

        boards = Boards()
        self.manager.boards = boards
        made = self.manager.make_storyboard(sid, chapter["id"])
        self.assertEqual(made["chapter"]["storyboard_id"], "board1")
        self.assertEqual(boards.made["identity_asset_ids"], [face], "the cast's pictures keep their faces")
        self.assertEqual(len(boards.made["shots"]), 2, "one shot per scene")
        self.assertEqual(boards.made["link"], {"story_id": sid, "chapter_id": chapter["id"]})

        story = self.manager._story(sid)
        refs, labels = self.manager._references(story, self.manager.store.get_chapter(chapter["id"]))
        self.assertEqual([r.asset_id for r in refs], [face] + panels, "cast first, then the panels")
        self.assertEqual(labels[1:], ["first frame of scene 1 (storyboard)", "first frame of scene 2 (storyboard)"])
        brief, _ = chapter_brief(story, self.manager.store.get_chapter(chapter["id"]), [], labels, 12)
        self.assertIn("keyframe completion", brief)

        self.manager.drop_storyboard(sid, chapter["id"])
        refs, _ = self.manager._references(story, self.manager.store.get_chapter(chapter["id"]))
        self.assertEqual([r.asset_id for r in refs], [face], "dropped: planned from the text alone again")

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

    async def test_make_video_goes_through_the_film_engine_when_the_cast_has_pictures(self):
        from test_manhwa import clip

        sid = self.story["id"]
        chapter = await self.drafted()
        face = (await self.service.add_asset("mina.png", io.BytesIO(b"mina"), source={"type": "generated"}))["id"]
        await self.manager.set_character_image(sid, "Mina", face)
        path = clip(os.path.join(self.tmp, "film.mp4"))
        with open(path, "rb") as handle:
            film = (await self.service.add_asset("chapter_film.mp4", handle))["id"]

        class Boards:
            made, status = {}, "rendering"

            def create(self, **kwargs):
                self.made = kwargs
                return {"id": "film1", "status": "drawing"}

            def board(self, board_id):
                return {"id": board_id, "status": self.status, "error": "", "render": {"film_asset_id": film}}

        boards = Boards()
        self.manager.boards = boards
        self.manager.make_video(sid, chapter["id"])
        for _ in range(100):
            if boards.made:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(boards.made["identity_asset_ids"], [face], "the cast keeps their faces")
        self.assertEqual(len(boards.made["shots"]), 2, "a shot per scene")
        self.assertIn('Jun: "Sorry, the trains."', boards.made["dialogue"][1], "lines go through word for word")
        self.assertEqual((boards.made["review"], boards.made["link"]["chapter_id"]), ("auto", chapter["id"]))
        self.assertEqual(self.manager.store.get_chapter(chapter["id"])["board_id"], "film1")
        boards.status = "done"
        done = await self.finish(chapter["id"])
        self.assertEqual((done["status"], done["film_asset_id"]), ("done", film), done)
        self.assertEqual(self.service.plans, [], "no classic plan was made")
        view = self.manager.view(sid)
        self.assertEqual(view["film"]["chapters"], [chapter["id"]], "the chapter's film joins the main film")
        self.assertEqual(next(c for c in view["chapters"] if c["id"] == chapter["id"])["video_url"], f"/file/{film}")

        self.manager.update_story(sid, settings={"video_engine": "classic"})
        self.manager.make_video(sid, chapter["id"], redo=True)
        redone = await self.finish(chapter["id"])
        self.assertTrue(redone["render_job_id"], "classic still works when asked for")
        self.assertEqual(redone["history"][0]["board_id"], "film1")

    async def test_the_story_resolution_reaches_the_render(self):
        sid = self.story["id"]
        with self.assertRaises(RequestError):
            self.manager.update_story(sid, settings={"render": {"megapixels": 5}})
        self.manager.update_story(sid, settings={"render": {"megapixels": 0.4, "steps": None}})
        self.assertEqual(self.manager.view(sid)["settings"]["render"], {"megapixels": 0.4}, "blank keeps the default")
        chapter = await self.drafted()
        self.manager.make_video(sid, chapter["id"])
        await self.finish(chapter["id"])
        self.assertEqual(self.service.videos[0].settings.megapixels, 0.4)

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


    async def test_one_co_writer_call_at_a_time_and_replies_in_the_background(self):
        sid = self.story["id"]
        self.chat.delay = 0.2
        out = self.manager.start_chat(sid, "Rooftop, night.")
        self.assertTrue(out["pending"])
        with self.assertRaises(RequestError):
            self.manager.start_draft(sid)  # the reply is still being written: refused, not doubled
        with self.assertRaises(RequestError):
            self.manager.start_chat(sid, "again")
        self.assertEqual(self.manager.view(sid)["busy"]["what"], "chat")
        while self.manager.view(sid)["busy"]:
            await asyncio.sleep(0.02)
        self.assertEqual([m["role"] for m in self.manager.messages(sid)], ["user", "assistant"])
        self.manager.start_draft(sid)
        with self.assertRaises(RequestError):
            self.manager.start_draft(sid)  # the double press that made two Chapter 4s
        while self.manager.view(sid)["busy"]:
            await asyncio.sleep(0.02)
        self.assertEqual(len(self.manager.view(sid)["chapters"]), 1)

    async def test_a_failed_reply_is_said_in_the_chat(self):
        self.chat.down = {"google/gemma-4-31b-it", "deepseek-ai/deepseek-v4.1-flash", "moonshotai/kimi-k2.6",
                          "xai/grok-4.6", "xai/grok-4.3"}
        self.manager.start_chat(self.story["id"], "hello")
        while self.manager.view(self.story["id"])["busy"]:
            await asyncio.sleep(0.02)
        last = self.manager.messages(self.story["id"])[-1]
        self.assertEqual((last["role"], last.get("error")), ("event", True))
        self.assertIn("Reply failed", last["text"])

    async def test_chapters_are_numbered_in_order_with_clean_titles(self):
        sid = self.story["id"]
        for n in range(3):
            await self.manager.chat(sid, f"idea {n}")
            await self.manager.draft_chapter(sid)
        chapters = self.manager.store.chapters(sid)
        broken = chapters[2]
        broken["idx"] = 1  # as the overlapping drafts left them
        self.manager.store.save_chapter(broken)
        self.assertEqual([c["idx"] for c in self.manager.view(sid)["chapters"]], [0, 1, 2])
        await self.manager.chat(sid, "more")
        chapter = await self.manager.draft_chapter(sid, plan="Chapter 9: The rooftop: x")
        self.assertEqual((chapter["idx"], chapter["title"]), (3, "The rooftop"), "no 'Chapter 9:' in the title")

    async def test_autopilot_from_the_premise_to_the_main_film(self):
        sid = self.story["id"]
        pilot = self.manager.start_autopilot(sid, chapters=2)
        self.assertEqual(pilot["status"], "running")
        with self.assertRaises(RequestError):
            self.manager.start_autopilot(sid)
        for _ in range(1500):
            pilot = self.manager.view(sid)["autopilot"]
            if pilot["status"] != "running":
                break
            await asyncio.sleep(0.02)
        self.assertEqual(pilot["status"], "done", pilot)
        view = self.manager.view(sid)
        self.assertEqual([c["title"] for c in view["chapters"]], ["The rooftop", "The rooftop"])
        self.assertEqual([c["status"] for c in view["chapters"]], ["done", "done"])
        self.assertEqual(len(view["film"]["chapters"]), 2)
        cast = {c["name"]: c for c in view["characters"]}
        self.assertEqual(cast["Voice"]["role"], "narrator")
        self.assertEqual(len(cast["Mina"]["images"]) + len(cast["Jun"]["images"]), 2, "a picture for each cast member")
        self.assertEqual(cast["Voice"]["images"], [], "the narrator is a voice")
        portrait = [c for c in self.service.calls if "portrait photograph of Mina" in c["prompt"]]
        self.assertEqual(len(portrait), 1)
        self.assertIn("PLAN FOR THIS CHAPTER: The rooftop: Mina waits", json.dumps(self.chat.sent))

    async def test_an_outline_short_of_chapters_is_asked_again_for_the_rest(self):
        self.chat.outline_size = 2
        outline = await self.manager._outline(self.story["id"], 5)
        self.assertEqual([o["title"] for o in outline], ["Part 1.0", "Part 1.1", "Part 2.0", "Part 2.1", "Part 3.0"])
        asks = [m[0]["content"] for m in self.chat.sent]
        self.assertIn("Write exactly 5 chapters", asks[0])
        self.assertIn("Write exactly 1 chapter in", asks[2])
        self.assertIn("- Part 2.1: Beat 2.1.", asks[2], "the follow-up shows what is already planned")
        self.assertEqual(sum(1 for c in self.manager.view(self.story["id"])["characters"] if c["name"] == "Mina"), 1)

    async def test_autopilot_needs_a_premise_stops_and_resumes(self):
        bare = self.manager.create_story("No premise yet")
        with self.assertRaises(RequestError):
            self.manager.start_autopilot(bare["id"])
        sid = self.story["id"]
        self.service.finish_videos = False
        self.manager.start_autopilot(sid, chapters=2, portraits=False)
        for _ in range(1500):
            if (self.manager.view(sid)["autopilot"] or {}).get("stage", "").startswith("video"):
                break
            await asyncio.sleep(0.02)
        await self.manager.stop()  # a restart in the middle of the first video
        self.assertEqual(self.manager.view(sid)["autopilot"]["status"], "running")
        for job in self.service.jobs.values():
            job["status"] = "done"
        self.service.finish_videos = True
        fresh = StoryManager(self.service)
        fresh.poll = 0.01
        self.manager = fresh
        await fresh.resume(wait_for_gpu=0)
        for _ in range(1500):
            if fresh.view(sid)["autopilot"]["status"] != "running":
                break
            await asyncio.sleep(0.02)
        view = fresh.view(sid)
        self.assertEqual(view["autopilot"]["status"], "done", view["autopilot"])
        self.assertEqual(len(view["chapters"]), 2, "the resumed autopilot did not draft chapter 1 again")
        stopped = fresh.start_autopilot(sid, chapters=1, portraits=False, videos=False)
        await fresh.stop_autopilot(sid)
        self.assertEqual(fresh.view(sid)["autopilot"]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
