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
    def test_eight_tall_panels_go_four_across_on_a_square_sheet(self):
        self.assertEqual(sb.grid_for(8, "9:16"), (4, 2, (2048, 2048)), "the layout the villa sheet used")

    def test_wide_panels_stack_instead(self):
        columns, rows, (width, height) = sb.grid_for(4, "16:9")
        panel = (width / columns) / (height / rows)
        self.assertLess(abs(panel - 16 / 9), 0.35, (columns, rows, width, height))

    def test_limits(self):
        with self.assertRaises(sb.StoryboardError):
            sb.grid_for(9, "9:16")  # with the identity picture that is ten, past a render's nine
        with self.assertRaises(sb.StoryboardError):
            sb.grid_for(4, "5:4")

    def test_the_prompt_asks_for_a_cuttable_grid_and_every_beat(self):
        prompt = sb.sheet_prompt(["she wakes", "she showers"], "Rebel, petite.", 2, 1)
        self.assertIn("grid of 2 columns and 1 row", prompt)
        self.assertIn("No text", prompt)
        self.assertIn("Panel 1: she wakes.", prompt)
        self.assertIn("Panel 2: she showers.", prompt)


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

    def test_a_dark_sheet_is_cut_as_the_even_grid_it_was_asked_to_be(self):
        panels, found = sb.cut_grid(sheet_png(2, 2, dark=True), 2, 2)
        self.assertFalse(found)
        self.assertEqual(len(panels), 4)

    def test_the_film_brief_opens_every_segment_on_its_panel(self):
        brief = sb.film_brief(["wakes up", "stretches"], "Rebel.", music="slow R&B")
        self.assertIn("<Picture 3> is the identity reference", brief)
        self.assertIn("keyframe completion", brief)
        self.assertIn("Segment 2: <Picture 2> -- stretches.", brief)
        self.assertIn("Music: slow R&B", brief)


class FakeStore:
    def __init__(self):
        self.assets = {}

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)


class FakeService:
    """Just what the storyboard steps call."""

    def __init__(self, sheet: bytes):
        self.store = FakeStore()
        self.sheet = sheet
        self.image_calls, self.videos, self.tags = [], [], {}
        self.store.assets["face"] = {"id": "face", "kind": "image", "filename": "face.png", "collection": "Rebel"}

    async def generate_images(self, prompt, **kwargs):
        self.image_calls.append((prompt, kwargs))
        asset = {"id": "sheet1", "kind": "image", "filename": "sheet.jpg", "collection": "Generated"}
        self.store.assets["sheet1"] = asset
        return {"assets": [asset], "cost_usd": 0.018, "model": "bytedance/seedream-v5.0-flash/edit"}

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
        return {"id": "job1", "status": "planning"}


class Steps(unittest.TestCase):
    def test_sheet_then_panels_then_a_film_that_opens_on_each(self):
        service = FakeService(sheet_png(4, 2, 2048, 2048, border=40))
        beats = [f"shot {n}" for n in range(1, 9)]

        made = asyncio.run(sb.make_sheet(service, beats, "Rebel, petite.", ["face"], "9:16"))
        prompt, kwargs = service.image_calls[0]
        self.assertEqual((kwargs["engine"], kwargs["size"], kwargs["reference_asset_ids"]),
                         ("seedream-flash", "2048x2048", ["face"]))
        self.assertEqual(made["grid"], {"columns": 4, "rows": 2, "panels": 8, "aspect_ratio": "9:16"})
        self.assertIn("storyboard-sheet", service.tags["sheet1"])

        cut = asyncio.run(sb.split_sheet(service, "sheet1", 4, 2, 8))
        self.assertTrue(cut["borders_found"])
        self.assertEqual([p["source"]["panel"] for p in cut["panels"]], list(range(1, 9)))
        self.assertEqual(cut["panels"][0]["source"]["sheet"], "sheet1")

        asyncio.run(sb.render_film(service, [p["id"] for p in cut["panels"]], "face", beats, "Rebel."))
        request = service.videos[0]
        self.assertEqual(len(request.references), 9, "8 panels and the identity picture: a render's limit")
        self.assertEqual(request.references[-1].asset_id, "face")
        self.assertEqual(request.story.segment_count, 8)
        self.assertIn("keyframe completion", request.story.story)

    def test_mismatched_beats_and_missing_identity_are_refused(self):
        service = FakeService(sheet_png(2, 1))
        with self.assertRaises(sb.StoryboardError):
            asyncio.run(sb.make_sheet(service, ["one"], "Rebel.", []))
        with self.assertRaises(sb.StoryboardError):
            asyncio.run(sb.render_film(service, ["a", "b"], "face", ["only one"], "Rebel."))


if __name__ == "__main__":
    unittest.main()
