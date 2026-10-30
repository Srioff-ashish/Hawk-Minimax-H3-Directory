"""Storyboard sheets: the grid, the prompt, cutting a sheet, and the film that opens each segment on its panel."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image

    from hawk_api import storyboard as sb
except ImportError as exc:  # pragma: no cover - the gateway's dependencies are optional for the node tests
    raise unittest.SkipTest(f"storyboard dependencies missing: {exc}")


def sheet_png(columns: int, rows: int, width: int = 800, height: int = 800, border: int = 20, dark: bool = False) -> bytes:
    """A black sheet with columns x rows coloured panels, the way Seedream draws one."""
    image = Image.new("RGB", (width, height), (0, 0, 0))
    cell_w, cell_h = width // columns, height // rows
    for row in range(rows):
        for column in range(columns):
            colour = (10, 10, 12) if dark else (60 + 20 * column, 120 + 30 * row, 200)
            image.paste(colour, (column * cell_w + border, row * cell_h + border,
                                 (column + 1) * cell_w - border, (row + 1) * cell_h - border))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class Grid(unittest.TestCase):
    def test_four_tall_panels_go_two_by_two_on_a_tall_sheet(self):
        self.assertEqual(sb.grid_for(4, "9:16"), (2, 2, (1584, 2816)))

    def test_wide_panels_stack_instead(self):
        columns, rows, (width, height) = sb.grid_for(4, "16:9")
        panel = (width / columns) / (height / rows)
        self.assertLess(abs(panel - 16 / 9), 0.35, (columns, rows, width, height))

    def test_limits(self):
        with self.assertRaises(sb.StoryboardError):
            sb.grid_for(5, "9:16")  # past four panels the image model drifts
        with self.assertRaises(sb.StoryboardError):
            sb.grid_for(4, "5:4")

    def test_the_prompt_asks_for_a_cuttable_grid_and_every_beat(self):
        prompt = sb.sheet_prompt(["she wakes", "she showers"], "Rebel, petite.", 2, 1)
        self.assertIn("grid of 2 columns and 1 row", prompt)
        self.assertIn("No text", prompt)
        self.assertIn("Panel 1 -- she wakes.", prompt)
        self.assertIn("Panel 2 -- she showers.", prompt)

    def test_a_frame_is_described_in_full(self):
        frame = {"characters": ["Mi-ah", "Dae-ho"], "wearing": "white bikini; black swim shorts",
                 "setting": "sunny beach", "camera": "low-angle wide shot", "action": "she spikes the ball"}
        prompt = sb.sheet_prompt([frame], "", 1, 1)
        self.assertIn("Panel 1 -- Characters: Mi-ah, Dae-ho · Wearing: white bikini; black swim shorts · Setting: "
                      "sunny beach · Camera: low-angle wide shot · Action: she spikes the ball.", prompt)

    def test_image_prompts_name_no_relationships_and_add_no_ages(self):
        prompt = sb.sheet_prompt(["Ronny hugs his stepmother in the kitchen", "Mrs. Cha, Dae-ho's friend's mom, laughs",
                                  "Riya (Didi) and her sisters wave"], "Ronny; Riya.", 3, 1)
        for word in ("stepmother", "mom", "Didi", "sisters", "adult"):
            self.assertNotIn(word, prompt)
        self.assertIn("Ronny hugs the woman in the kitchen.", prompt)
        self.assertIn("Mrs. Cha laughs.", prompt)
        self.assertIn("Riya and the women wave.", prompt)


class Cut(unittest.TestCase):
    def test_the_borders_are_found_and_the_panels_come_in_reading_order(self):
        panels, found = sb.cut_grid(sheet_png(4, 2), 4, 2)
        self.assertTrue(found)
        self.assertEqual(len(panels), 8)
        first, fifth = (Image.open(io.BytesIO(p)) for p in (panels[0], panels[4]))
        self.assertLess(first.width, 160, "inside the border, not the whole cell")
        self.assertGreater(fifth.getpixel((first.width // 2, first.height // 2))[1],
                           first.getpixel((first.width // 2, first.height // 2))[1], "panel 5 is the second row")
        for panel in (first, fifth):  # no black border left on the edges
            self.assertGreater(sum(panel.getpixel((1, 1))), 60)

    def test_a_dark_band_lining_up_across_a_row_is_not_taken_for_a_border(self):
        image = Image.open(io.BytesIO(sheet_png(4, 2))).convert("RGB")
        image.paste((0, 0, 0), (0, 250, 800, 262))  # a bed frame at the same height in every top panel
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        boxes, found = sb.find_grid(buffer.getvalue(), 4, 2)
        self.assertTrue(found)
        self.assertLess(boxes[0][1], 40)
        self.assertGreater(boxes[0][3], 360, "the top row is one panel, band and all")

    def test_with_no_grid_given_the_panels_are_found_even_past_a_false_band(self):
        image = Image.open(io.BytesIO(sheet_png(4, 2))).convert("RGB")
        image.paste((0, 0, 0), (0, 250, 800, 262))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        boxes, _ = sb.find_grid(buffer.getvalue(), 0, 0)
        self.assertEqual(len(boxes), 8, "the band splits the top row unevenly, so it is merged back")

    def test_a_dark_sheet_is_cut_as_the_even_grid_it_was_asked_to_be(self):
        panels, found = sb.cut_grid(sheet_png(2, 2, dark=True), 2, 2)
        self.assertFalse(found)
        self.assertEqual(len(panels), 4)

    def test_the_film_brief_opens_every_segment_on_its_panel(self):
        brief = sb.film_brief(["wakes up", "stretches"], "Rebel.", music="slow R&B")
        self.assertIn("<Picture 3> is the identity reference", brief)
        self.assertIn("keyframe completion", brief)
        self.assertIn("Segment 2: <Picture 2> is this segment's first frame", brief)
        self.assertIn("stretches.", brief)
        self.assertIn("Music: slow R&B.", brief)


class FakeStore:
    def __init__(self):
        self.assets, self.jobs = {}, {}

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)

    def get_job(self, job_id):
        return self.jobs.get(job_id)


class FakeService:
    """Just what the storyboard steps call."""

    def __init__(self, sheet: bytes, data_dir: str = ""):
        import tempfile, types

        self.settings = types.SimpleNamespace(data_dir=data_dir or tempfile.mkdtemp())
        self.store = FakeStore()
        self.sheet = sheet
        self.image_calls, self.videos, self.tags = [], [], {}
        self.store.assets["face"] = {"id": "face", "kind": "image", "filename": "face.png", "collection": "Rebel"}

    async def generate_images(self, prompt, **kwargs):
        self.image_calls.append((prompt, kwargs))
        asset_id = "sheet1" if len(self.image_calls) == 1 else f"img{len(self.image_calls)}"
        asset = {"id": asset_id, "kind": "image", "filename": "sheet.jpg", "collection": "Generated"}
        self.store.assets[asset_id] = asset
        return {"assets": [asset], "cost_usd": 0.018, "model": "bytedance/seedream-v5.0-flash/edit"}

    def job_summary(self, job):
        return job

    def update_asset(self, asset_id, **kwargs):
        self.tags.setdefault(asset_id, []).extend(kwargs.get("add_tags") or [])

    async def asset_bytes(self, asset):
        return self.sheet

    async def add_asset(self, filename, fileobj, content_type, size, *, collection=None, tags=None, source=None):
        asset_id = f"panel{len(self.store.assets)}"
        asset = {"id": asset_id, "kind": "image", "filename": filename, "collection": collection, "tags": tags,
                 "source": source}
        self.store.assets[asset_id] = asset
        return asset

    def asset_view(self, asset):
        return asset

    async def create_video(self, request):
        self.videos.append(request)
        job = {"id": f"job{len(self.videos)}", "status": "done"}
        self.store.jobs[job["id"]] = job
        return job


class Steps(unittest.TestCase):
    def test_sheet_then_panels_then_a_film_that_opens_on_each(self):
        service = FakeService(sheet_png(2, 2, 1584, 2816, border=40))
        beats = [f"shot {n}" for n in range(1, 5)]

        made = asyncio.run(sb.make_sheet(service, beats, "Rebel, petite.", ["face"], "9:16"))
        prompt, kwargs = service.image_calls[0]
        self.assertEqual((kwargs["engine"], kwargs["size"], kwargs["reference_asset_ids"]),
                         ("seedream-flash", "1584x2816", ["face"]))
        self.assertEqual(made["grid"], {"columns": 2, "rows": 2, "panels": 4, "aspect_ratio": "9:16"})
        self.assertIn("storyboard-sheet", service.tags["sheet1"])

        cut = asyncio.run(sb.split_sheet(service, "sheet1", 2, 2, 4))
        self.assertTrue(cut["borders_found"])
        self.assertEqual([p["source"]["panel"] for p in cut["panels"]], list(range(1, 5)))
        self.assertEqual(cut["panels"][0]["source"]["sheet"], "sheet1")

        asyncio.run(sb.render_film(service, [p["id"] for p in cut["panels"]], "face", beats, "Rebel."))
        request = service.videos[0]
        self.assertEqual(len(request.references), 5, "4 panels and the identity picture")
        self.assertEqual(request.references[-1].asset_id, "face")
        self.assertEqual(request.story.segment_count, 4)
        self.assertIn("keyframe completion", request.story.story)

    def test_mismatched_beats_and_missing_identity_are_refused(self):
        service = FakeService(sheet_png(2, 1))
        with self.assertRaises(sb.StoryboardError):
            asyncio.run(sb.make_sheet(service, ["one"], "Rebel.", []))
        with self.assertRaises(sb.StoryboardError):
            asyncio.run(sb.render_film(service, ["a", "b"], "face", ["only one"], "Rebel."))


def png(width: int, height: int, colour=(200, 60, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


class Compose(unittest.TestCase):
    def test_four_tall_images_make_a_squarish_sheet_with_every_image_whole(self):
        data, grid = sb.compose_grid([png(450, 800)] * 4, border=20)
        self.assertEqual((grid["columns"], grid["rows"]), (2, 2), "full rows beat a squarer grid with empty cells")
        sheet = Image.open(io.BytesIO(data))
        self.assertEqual(list(sheet.size), grid["size"])
        self.assertEqual(sheet.getpixel((5, 5)), (0, 0, 0), "black border")
        self.assertEqual(sheet.getpixel((20 + 10, 20 + 10)), (200, 60, 60), "the image starts inside the border")

    def test_contain_keeps_a_wide_image_whole_and_cover_fills_the_cell(self):
        images = [png(400, 700), png(1600, 400, (20, 200, 20))]
        contain, grid = sb.compose_grid(images, columns=2, fit="contain", border=0)
        cell_w, cell_h = grid["cell"]
        sheet = Image.open(io.BytesIO(contain))
        self.assertEqual(sheet.getpixel((cell_w + cell_w // 2, 5)), (0, 0, 0), "bars above the letterboxed wide image")
        cover, _ = sb.compose_grid(images, columns=2, fit="cover", border=0)
        self.assertEqual(Image.open(io.BytesIO(cover)).getpixel((cell_w + cell_w // 2, 5)), (20, 200, 20), "filled, cropped")

    def test_the_sheet_stays_within_the_size_cap_and_the_limits(self):
        _, grid = sb.compose_grid([png(3000, 3000)] * 9)
        self.assertLessEqual(max(grid["size"]), sb.COMPOSE_LONG_SIDE)
        with self.assertRaises(sb.StoryboardError):
            sb.compose_grid([png(10, 10)] * 17)
        with self.assertRaises(sb.StoryboardError):
            sb.compose_grid([png(10, 10)], fit="stretch")


class Boards(unittest.IsolatedAsyncioTestCase):
    async def test_my_own_images_are_the_panels_and_nothing_is_drawn(self):
        service = FakeService(sheet_png(2, 1))
        for n in range(3):
            service.store.assets[f"mine{n}"] = {"id": f"mine{n}", "kind": "image", "filename": f"m{n}.png"}
        boards = sb.StoryboardManager(service)
        boards.poll = 0.01
        boards._join = lambda board_id: asyncio.sleep(0)
        board = boards.create(title="Mine", character="Rebel.", identity_asset_ids=["face"], shots=[],
                              motions=["turns", "", "laughs"], panel_asset_ids=["mine0", "mine1", "mine2"],
                              then_render={"segment_seconds": 5})
        self.assertEqual(service.image_calls, [], "no image model call")
        self.assertEqual(board["panels"], ["mine0", "mine1", "mine2"])
        await boards._tasks[board["id"]]
        request = service.videos[0]
        self.assertEqual([r.asset_id for r in request.references], ["mine0", "mine1", "mine2", "face"])
        self.assertIn("Segment 1: <Picture 1> is this segment's first frame", request.story.story)
        self.assertIn("and animate from there. turns.", request.story.story)
        self.assertIn("animate this frame", request.story.story, "a blank motion falls back to the shot")
        with self.assertRaises(sb.StoryboardError):
            boards.create(title="x", character="Rebel.", identity_asset_ids=["face"], shots=["one"],
                          panel_asset_ids=["mine0", "mine1"])

    async def test_a_merged_sheet_gives_back_its_images_and_any_other_sheet_is_cut(self):
        service = FakeService(sheet_png(4, 2, 2048, 2048, border=40))
        for n in range(3):
            service.store.assets[f"m{n}"] = {"id": f"m{n}", "kind": "image", "filename": f"m{n}.png"}
        merged = await sb.compose_sheet(service, ["m0", "m1", "m2"])
        boards = sb.StoryboardManager(service)
        board = boards.create(title="Merged", character="Rebel.", identity_asset_ids=["face"], shots=[],
                              sheet_asset_id=merged["sheet"]["id"], check=False)
        self.assertEqual((board["status"], board["panels"]), ("ready", ["m0", "m1", "m2"]), "the originals, not cut-outs")

        service.store.assets["their_sheet"] = {"id": "their_sheet", "kind": "image", "filename": "made_elsewhere.jpg"}
        board = boards.create(title="Theirs", character="Rebel.", identity_asset_ids=["face"], shots=[],
                              sheet_asset_id="their_sheet", then_render={"segment_seconds": 5})
        self.assertEqual(board["status"], "drawing", "cut in the background")
        for _ in range(50):
            task = boards._tasks.get(board["id"])
            if task is None:
                break
            await task
            await asyncio.sleep(0)
        board = boards.board(board["id"])
        self.assertEqual(len(board["panels"]), 8, "4 x 2 found from the borders")
        self.assertEqual(len(board["shots"]), 8)
        self.assertEqual(len(service.videos[-1].references), 9, "then rendered: 8 panels and the identity picture")

    async def test_compose_sheet_stores_one_asset_and_costs_nothing(self):
        service = FakeService(png(450, 800))
        for n in range(2):
            service.store.assets[f"m{n}"] = {"id": f"m{n}", "kind": "image", "filename": f"m{n}.png", "collection": "Rebel"}
        out = await sb.compose_sheet(service, ["m0", "m1"], title="Rebel views")
        self.assertEqual(out["cost_usd"], 0.0)
        self.assertEqual(out["sheet"]["source"]["images"], ["m0", "m1"])
        self.assertEqual(out["sheet"]["collection"], "Rebel")
        self.assertIn("composed-sheet", out["sheet"]["tags"])
        self.assertEqual(service.image_calls, [])

    async def test_ten_shots_are_three_sheets_two_renders_and_one_film(self):
        service = FakeService(sheet_png(2, 2, 1584, 2816, border=40))
        boards = sb.StoryboardManager(service)
        boards.poll = 0.01
        joined = []

        async def fake_join(board_id):
            joined.append(board_id)
            board = boards.board(board_id)
            board["render"]["film_asset_id"] = "film"
            board["status"] = "done"
            boards.store.save(board)

        boards._join = fake_join
        board = boards.create(title="Villa", character="Rebel.", identity_asset_ids=["face"],
                              shots=[f"shot {n}" for n in range(1, 11)])
        self.assertEqual(board["status"], "drawing")
        await boards._tasks[board["id"]]
        board = boards.board(board["id"])
        self.assertEqual(board["status"], "ready", board.get("error"))
        self.assertEqual([len(s["shots"]) for s in board["sheets"]], [4, 4, 2], "four shots to a sheet")
        self.assertTrue(all(board["panels"]))
        self.assertEqual(board["cost_usd"], 0.054)

        redone = await boards.redo_panel(board["id"], 3, shot="shot three, laughing")
        prompt, kwargs = service.image_calls[-1]
        self.assertEqual(kwargs["size"], "1584x2816", "one tall 9:16 panel, the biggest at the same price")
        self.assertIn(board["panels"][2], kwargs["reference_asset_ids"], "the old panel keeps the setting")
        self.assertIn("laughing", prompt)
        self.assertEqual(redone["shots"][2], "shot three, laughing")

        boards.start_render(board["id"], segment_seconds=5)
        await boards._tasks[board["id"]]
        self.assertEqual([len(v.references) for v in service.videos], [9, 3], "8 + identity, then 2 + identity")
        self.assertEqual(joined, [board["id"]])
        self.assertEqual(boards.board(board["id"])["status"], "done")

    async def test_a_one_call_film_renders_once_drawn(self):
        service = FakeService(sheet_png(2, 1, 2048, 2048, border=40))
        boards = sb.StoryboardManager(service)
        boards.poll = 0.01

        async def fake_join(board_id):
            boards._set(board_id, status="done")

        boards._join = fake_join
        board = boards.create(title="Two", character="Rebel.", identity_asset_ids=["face"], shots=["a", "b"],
                              then_render={"segment_seconds": 4})
        for _ in range(50):  # the drawing hands over to the render, then the render ends
            task = boards._tasks.get(board["id"])
            if task is None:
                break
            await task
            await asyncio.sleep(0)
        self.assertEqual(len(service.videos), 1)
        self.assertEqual(boards.board(board["id"])["status"], "done")

    async def test_rendering_waits_for_every_panel(self):
        service = FakeService(sheet_png(2, 1))
        boards = sb.StoryboardManager(service)
        board = boards.create(title="x", character="Rebel.", identity_asset_ids=["face"], shots=["a", "b"], draw=False)
        with self.assertRaises(sb.StoryboardError):
            boards.start_render(board["id"])


class Writer(unittest.IsolatedAsyncioTestCase):
    async def test_the_writer_reads_the_picture_and_falls_back_when_a_model_writes_too_few(self):
        import json
        import tempfile
        import types

        from hawk_api.prompts import PLATFORM_RULES, PromptStore

        service = FakeService(png(10, 10))
        service.prompts = PromptStore(tempfile.mkdtemp())
        calls = []

        class Chat:
            async def chat(self, model, messages, **kwargs):
                calls.append((model, messages))
                if len(calls) == 1:
                    return json.dumps({"shots": ["only one"]}), {}
                return json.dumps({"character": "Rebel, in her forties", "style": "warm boudoir light",
                                   "shots": [f"shot {n}" for n in range(1, 5)], "motions": ["turns", "smiles"]}), {}

        service.atlas = Chat()

        async def catalogue():
            return []

        async def asset_file(asset_id, width=0):
            return b"jpegbytes", "image/jpeg"

        service.model_catalogue, service.asset_file = catalogue, asset_file
        out = await sb.suggest_shots(service, "a lazy villa morning", ["face"], 4, "9:16",
                                     models="google/gemma-4-31b-it, deepseek-ai/deepseek-v4.1-flash")
        self.assertEqual(out["shots"], ["shot 1", "shot 2", "shot 3", "shot 4"])
        self.assertEqual(out["motions"], ["turns", "smiles", "", ""], "short motion lists are padded, never invented")
        self.assertEqual((out["character"], out["style"]), ("Rebel, in her forties", "warm boudoir light"))
        self.assertEqual(len(calls), 2, "the first model wrote 1 of 4, so the next one was asked")
        system, user = calls[0][1]
        self.assertIn('"shots"', system["content"])
        self.assertIn(PLATFORM_RULES, system["content"])
        self.assertTrue(any(part.get("type") == "image_url" for part in user["content"]), "the reference picture is sent")
        self.assertIn("exactly 4", user["content"][0]["text"])
        await sb.suggest_shots(service, "a villa morning", ["face"], 4, character="Rebel, fair-skinned Indian, 45, petite")
        self.assertIn("Rebel, fair-skinned Indian, 45, petite", calls[-1][1][1]["content"][0]["text"], "the user's facts go in as fixed")
        with self.assertRaises(sb.StoryboardError):
            await sb.suggest_shots(service, "", ["face"], 4)

    def test_the_writer_prompt_is_editable_like_the_others(self):
        import tempfile

        from hawk_api.prompts import PROMPT_NAMES, PromptStore

        store = PromptStore(tempfile.mkdtemp())
        self.assertIn("storyboard", PROMPT_NAMES)
        self.assertEqual(store.view("storyboard")["warnings"], [])
        self.assertTrue(store.view("storyboard")["is_default"])
        saved = store.save("storyboard", "Write shots. Reply as JSON.")
        self.assertIn('No "shots"', " ".join(saved["warnings"]))


if __name__ == "__main__":
    unittest.main()
