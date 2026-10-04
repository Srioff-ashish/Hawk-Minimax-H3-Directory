"""The person check: an upload with no identifiable person in it stops counting as a photo of someone.

The rule under test: only a clear "not identifiable" from the vision model lifts the upload rules. A visible
face, an unreadable answer, a refusal or no vision model at all leaves the upload flagged, and so does
everything made from it.

    python -m unittest tests_api.test_person_check
"""

from __future__ import annotations

import dataclasses
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from PIL import Image

    from hawk_api import image_engines as ie
    from hawk_api import local_images as li
    from hawk_api.atlas import AtlasError
    from hawk_api.config import Settings
    from hawk_api.jobs import HawkService, RequestError
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"person check test dependencies missing: {exc}")


def png(color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, "PNG")
    return buffer.getvalue()


class FromUpload(unittest.TestCase):
    UNCHECKED = {"id": "u1", "source": {"type": "upload"}}
    FACE = {"id": "u2", "source": {"type": "upload", "person_check": {"identifiable": True}}}
    GARMENT = {"id": "u3", "source": {"type": "upload", "person_check": {"identifiable": False}}}

    def lookup(self, asset_id):
        return {a["id"]: a for a in (self.UNCHECKED, self.FACE, self.GARMENT)}.get(asset_id)

    def test_only_a_clear_no_lifts_the_rules(self):
        self.assertTrue(li.from_upload(self.UNCHECKED))
        self.assertTrue(li.from_upload(self.FACE))
        self.assertFalse(li.from_upload(self.GARMENT))

    def test_an_image_made_from_it_follows_it(self):
        made = lambda ref: {"id": "g", "source": {"type": "generated", "references": [ref]}}
        self.assertFalse(li.from_upload(made("u3"), self.lookup), "made from a faceless garment shot")
        self.assertTrue(li.from_upload(made("u2"), self.lookup), "made from a photo with a face")
        self.assertTrue(li.from_upload(made("u1"), self.lookup), "made from an upload nobody has checked")

    def test_one_face_anywhere_in_the_chain_still_counts(self):
        mixed = {"id": "g", "source": {"type": "generated", "references": ["u3", "u2"]}}
        self.assertTrue(li.from_upload(mixed, self.lookup))


class FakeChat:
    def __init__(self, reply, configured=True):
        self.reply, self.configured, self.calls = reply, configured, []

    async def chat(self, model, messages, **kwargs):
        if not self.configured:  # as AtlasClient.chat does with no key
            raise AtlasError("No Atlas API key on the server. Set ATLAS_API_KEY and restart the API.")
        self.calls.append(model)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, {}


class Service(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix="hawk_person_test_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        input_dir = os.path.join(self.dir, "comfy_input")
        os.makedirs(input_dir)
        self.service = HawkService(Settings(token="t" * 24, data_dir=self.dir, atlas_api_key="k",
                                            comfy_input_dir=input_dir,
                                            agent_vision_model="qwen/qwen3.6-35b-a3b, xai/grok-4.3"))

        async def upload(fileobj, name, subfolder, mime):
            os.makedirs(os.path.join(input_dir, subfolder), exist_ok=True)
            fileobj.seek(0)
            with open(os.path.join(input_dir, subfolder, name), "wb") as handle:
                handle.write(fileobj.read())
            return {"name": name, "subfolder": subfolder}

        self.service.comfy.upload = upload
        self.service.model_catalogue = mock.AsyncMock(return_value=[])

    def provider(self, chat: FakeChat) -> FakeChat:
        patcher = mock.patch.object(HawkService, "atlas", new=property(lambda _self: chat))
        patcher.start()
        self.addCleanup(patcher.stop)
        return chat

    async def add(self, color, source) -> dict:
        data = png(color)
        return await self.service.add_asset(f"{color}.png", io.BytesIO(data), "image/png", len(data), source=source)

    def flagged(self, asset_id) -> bool:
        return li.from_upload(self.service.store.get_asset(asset_id), self.service.store.lineage)

    async def test_a_faceless_garment_shot_is_cleared_and_the_answer_kept(self):
        chat = self.provider(FakeChat('{"identifiable": false, "reason": "a skirt on a body cropped below the chest"}'))
        skirt = await self.add((1, 2, 3), {"type": "upload"})
        result = await self.service.person_check(skirt["id"])
        self.assertIs(result["identifiable"], False)
        self.assertFalse(self.flagged(skirt["id"]))
        self.assertEqual(self.service.store.get_asset(skirt["id"])["source"]["person_check"]["model"],
                         "qwen/qwen3.6-35b-a3b")
        await self.service.person_check(skirt["id"])
        self.assertEqual(len(chat.calls), 1, "checked once, then read back")

    async def test_a_visible_face_stays_flagged(self):
        self.provider(FakeChat('{"identifiable": true, "reason": "a woman\'s face is visible"}'))
        photo = await self.add((4, 5, 6), {"type": "upload"})
        await self.service.person_check(photo["id"])
        self.assertTrue(self.flagged(photo["id"]))

    async def test_no_clear_answer_records_nothing_and_stays_flagged(self):
        for reply in ("I can't help with that.", '{"identifiable": "maybe"}', AtlasError("The model refused")):
            with self.subTest(reply=reply):
                self.provider(FakeChat(reply))
                photo = await self.add((7, 8, len(str(reply)) % 255), {"type": "upload"})
                result = await self.service.person_check(photo["id"])
                self.assertFalse(result["checked"])
                self.assertNotIn("person_check", self.service.store.get_asset(photo["id"])["source"])
                self.assertTrue(self.flagged(photo["id"]))

    async def test_no_vision_model_stays_flagged(self):
        self.provider(FakeChat('{"identifiable": false}', configured=False))
        photo = await self.add((9, 9, 9), {"type": "upload"})
        self.assertFalse((await self.service.person_check(photo["id"]))["checked"])
        self.assertTrue(self.flagged(photo["id"]))

    async def test_using_an_upload_as_a_reference_runs_the_check_first(self):
        chat = self.provider(FakeChat('{"identifiable": false, "reason": "a skirt, no face"}'))
        skirt = await self.add((10, 11, 12), {"type": "upload"})
        character = await self.add((13, 14, 15), {"type": "generated", "references": []})
        self.service.image_engines.save(edit=[{"engine": "seedream"}])
        self.service._atlas_images = mock.AsyncMock(return_value=(ie.IMAGE_EDIT_MODEL, [png((20, 20, 20))], 0.036))
        result = await self.service.generate_images("she wears the skirt from <image2>",
                                                    reference_asset_ids=[character["id"], skirt["id"]])
        self.assertEqual(chat.calls, ["qwen/qwen3.6-35b-a3b"])
        self.assertFalse(self.flagged(result["assets"][0]["id"]), "made from a generated person and a cleared garment")

    async def test_a_sexual_edit_with_a_face_upload_is_still_refused(self):
        self.provider(FakeChat('{"identifiable": true, "reason": "face visible"}'))
        photo = await self.add((16, 17, 18), {"type": "upload"})
        self.service.image_engines.save(edit=[{"engine": "nano-banana"}, {"engine": "seedream"}])
        google = mock.Mock(configured=True)
        with mock.patch.object(HawkService, "google", new=property(lambda _self: google)):
            with self.assertRaises(RequestError) as caught:
                await self.service.generate_images("make her nude", reference_asset_ids=[photo["id"]])
        self.assertIn("real people", str(caught.exception))

    async def test_only_uploads_are_checked(self):
        chat = self.provider(FakeChat('{"identifiable": false}'))
        made = await self.add((19, 20, 21), {"type": "generated", "references": []})
        self.assertFalse((await self.service.person_check(made["id"]))["checked"])
        self.assertEqual(chat.calls, [])


if __name__ == "__main__":
    unittest.main()
