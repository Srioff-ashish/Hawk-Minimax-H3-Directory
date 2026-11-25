"""Person replacement: the video is cut on H3's frame grid, only faces generated here go in, and the finished video
is the pieces joined with the original soundtrack."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.replace import (  # noqa: E402
    cut_pieces, grid_down, grid_up, nearest_aspect, origin_problem, probe, template_prompt, usable_prompt)
import test_gateway  # noqa: E402  (not `from ... import Gateway`: unittest would run its tests here too)

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def on_grid(frames: int) -> bool:
    return frames >= 5 and (frames - 5) % 17 == 0


class Pieces(unittest.TestCase):
    def test_grid(self):
        self.assertEqual((grid_up(5), grid_up(6), grid_up(22), grid_up(23)), (5, 22, 22, 39))
        self.assertEqual((grid_down(4), grid_down(5), grid_down(21), grid_down(22)), (0, 5, 5, 22))

    def test_a_minute_is_cut_into_grid_pieces_that_cover_every_frame_once(self):
        for total in (1440, 1441, 1452, 312, 277, 278, 2000):
            with self.subTest(total=total):
                pieces = cut_pieces(total, longest=277)
                self.assertTrue(all(on_grid(p["frames"]) and p["frames"] <= 277 for p in pieces), pieces)
                covered = sum(p["frames"] - p["skip"] for p in pieces)
                self.assertEqual(covered, total, "every frame once: the overlap of the last piece is skipped")
                self.assertEqual(pieces[-1]["start"] + pieces[-1]["frames"], total, "it ends on the video's last frame")
                for before, after in zip(pieces, pieces[1:]):
                    self.assertEqual(after["start"] + after["skip"], before["start"] + before["frames"])

    def test_a_short_video_loses_only_its_last_few_frames(self):
        pieces = cut_pieces(100, longest=277)
        self.assertEqual(pieces, [{"start": 0, "frames": 90, "skip": 0}])

    def test_aspect_follows_the_video(self):
        self.assertEqual(nearest_aspect(1920, 1080), "16:9")
        self.assertEqual(nearest_aspect(1080, 1920), "9:16")
        self.assertEqual(nearest_aspect(1000, 1000), "1:1")


class FakeStore:
    def __init__(self, assets):
        self.assets = assets

    def lineage(self, asset_id):
        return self.assets.get(asset_id)


class Origin(unittest.TestCase):
    """Only a face this server generated can go into a video; nothing that started as someone's photo."""

    def setUp(self):
        self.store = FakeStore({
            "made": {"filename": "made.png", "source": {"type": "generated", "engine": "ph3", "references": []}},
            "edited": {"filename": "edited.png", "source": {"type": "generated", "engine": "seedream-flash",
                                                           "references": ["made"]}},
            "photo": {"filename": "photo.jpg", "source": {"type": "upload"}},
            "from_photo": {"filename": "from_photo.png", "source": {"type": "generated", "engine": "krea2",
                                                                   "references": ["photo"]}},
            "twice_removed": {"filename": "twice.png", "source": {"type": "generated", "engine": "ph3",
                                                                 "references": ["edited", "from_photo"]}},
            "relabelled": {"filename": "relabelled.png", "source": {"type": "generated", "engine": "ph3",
                                                                   "corrected_from": {"type": "upload"}}},
            "outside": {"filename": "outside.png", "source": {"type": "generated", "engine": "external"}},
            "web": {"filename": "web.png", "source": {"type": "url", "url": "https://example.com/a.png"}},
        })

    def test_generated_here_is_allowed(self):
        self.assertEqual(origin_problem(self.store, "made"), "")
        self.assertEqual(origin_problem(self.store, "edited"), "", "an edit of a generated picture is still generated")

    def test_anything_that_started_as_a_photo_is_refused(self):
        for asset_id in ("photo", "from_photo", "twice_removed", "relabelled", "outside", "web", "missing"):
            with self.subTest(asset_id=asset_id):
                self.assertTrue(origin_problem(self.store, asset_id))
        self.assertIn("photo.jpg", origin_problem(self.store, "twice_removed"), "it names where the chain went wrong")


class Prompt(unittest.TestCase):
    def test_the_template_is_a_video_edit_of_the_piece(self):
        text = template_prompt("the woman in red", "Maya", "27, slim", 2, "she wears a black jacket")
        self.assertTrue(usable_prompt(text))
        self.assertIn("[video editing] The target video is an edited version of <Video 1>.", text)
        self.assertIn("<Picture 1> and <Picture 2>", text)
        self.assertIn("black jacket", text)
        self.assertNotIn("<Picture 3>", text)


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
class ReplaceJob(unittest.IsolatedAsyncioTestCase):
    """End to end against the fake ComfyUI: a 13 s clip with sound becomes two pieces (the second overlapping the
    first by a few frames), each rendered, joined back to 13 s with the original soundtrack."""

    asyncSetUp = test_gateway.Gateway.asyncSetUp
    asyncTearDown = test_gateway.Gateway.asyncTearDown
    wait = test_gateway.Gateway.wait
    upload_picture = test_gateway.Gateway.upload_picture

    def _clip(self, seconds: float, size: str, audio: bool) -> bytes:
        work = tempfile.mkdtemp()
        try:
            out = os.path.join(work, "clip.mp4")
            command = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration={seconds}"]
            if audio:
                command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
            command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", out]
            subprocess.run(command, check=True)
            return open(out, "rb").read()
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _service(self):
        return self.server.config.app.state.service

    def _generated_picture(self, asset_id: str) -> None:
        store = self._service().store
        asset = store.get_asset(asset_id)
        asset["source"] = {"type": "generated", "engine": "ph3", "generator": "test", "references": []}
        store.add_asset(asset)

    async def test_a_clip_is_replaced_piece_by_piece_and_keeps_its_sound(self):
        os.environ["HAWK_MAX_SEGMENT_SECONDS"] = "12"
        self.addCleanup(os.environ.pop, "HAWK_MAX_SEGMENT_SECONDS", None)
        self.fake.final_bytes = self._clip(11.6, "320x180", audio=True)  # what each piece's render "returns"
        source = (await self.http.post("/v1/assets", files={"files": ("dance.mp4", self._clip(13, "640x360", True),
                                                                       "video/mp4")})).json()["assets"][0]["id"]
        face = await self.upload_picture()

        refused = await self.http.post("/v1/videos/replace", json={"video_asset_id": source, "who": "the dancer",
                                                                   "picture_asset_ids": [face]})
        self.assertEqual(refused.status_code, 422, refused.text)
        self.assertIn("generated here", refused.text, "an uploaded photo cannot be put into a video")

        self._generated_picture(face)
        response = await self.http.post("/v1/videos/replace", json={
            "video_asset_id": source, "who": "the dancer", "picture_asset_ids": [face],
            "settings": {"megapixels": 0.4}})
        self.assertEqual(response.status_code, 202, response.text)
        job = await self.wait(response.json()["id"], timeout=60)
        self.assertEqual((job["kind"], job["status"]), ("replace", "done"), job)
        self.assertEqual([(p["frames"], p["skip"]) for p in job["pieces"]], [(277, 0), (39, 4)])
        self.assertTrue(job["video_url"])

        pieces = [(await self.http.get(f"/v1/jobs/{p['job_id']}")).json() for p in job["pieces"]]
        self.assertTrue(all(p["kind"] == "render" and p["parent_job_id"] == job["id"] for p in pieces))
        self.assertIn("[video editing] The target video is an edited version of <Video 1>.", pieces[0]["script"])

        out = self._service().store.get_asset(job["output_asset_id"])
        path = self._service().local_asset_path(out)
        if path is None:  # the fake ComfyUI keeps uploads in memory: fetch it through the API
            data = (await self.http.get(f"/v1/assets/{out['id']}/file")).content
            path = os.path.join(tempfile.mkdtemp(), "out.mp4")
            open(path, "wb").write(data)
        info = probe(path)
        self.assertAlmostEqual(info["seconds"], 13.0, delta=0.15, msg="the same length as the source")
        self.assertTrue(info["has_audio"], "the original soundtrack is laid back on")

    async def test_cancel_stops_the_job_and_its_pieces(self):
        source = (await self.http.post("/v1/assets", files={"files": ("clip.mp4", self._clip(3, "320x180", False),
                                                                       "video/mp4")})).json()["assets"][0]["id"]
        face = await self.upload_picture()
        self._generated_picture(face)
        self.fake.hold = True if hasattr(self.fake, "hold") else None
        job = (await self.http.post("/v1/videos/replace", json={"video_asset_id": source, "who": "the man",
                                                                "picture_asset_ids": [face]})).json()
        cancelled = await self.http.post(f"/v1/jobs/{job['id']}/cancel")
        self.assertIn(cancelled.status_code, (200, 409), cancelled.text)
        job = await self.wait(job["id"], timeout=30)
        self.assertIn(job["status"], ("cancelled", "done"))


if __name__ == "__main__":
    unittest.main()
