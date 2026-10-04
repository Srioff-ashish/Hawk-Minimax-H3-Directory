"""Recording an uploaded AI image as generated, from the C2PA content credential its maker signed.

The rule under test: only a credential that validates, is signed by a trusted issuer, and says every step
came from a generative model moves an upload to "generated" -- and one that lists input images passes only when
those inputs are named and are themselves generated, or an AI edit of a real photo would get through.

    python -m unittest tests_api.test_content_credentials
"""

from __future__ import annotations

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

    from hawk_api import content_credentials as cc
    from hawk_api import local_images
    from hawk_api.config import Settings
    from hawk_api.jobs import HawkService, RequestError
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"content credential test dependencies missing: {exc}")

AI = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


def store(issuer="Google LLC", sources=(AI, AI), ingredients=0, state="Valid", codes=("signingCredential.untrusted",)):
    """A manifest store shaped like the one Gemini signs into its images."""
    actions = [{"action": "c2pa.created" if i == 0 else "c2pa.edited", "digitalSourceType": s}
               for i, s in enumerate(sources)]
    return {"active_manifest": "m", "validation_state": state,
            "validation_status": [{"code": c} for c in codes],
            "manifests": {"m": {"signature_info": {"issuer": issuer},
                                "claim_generator_info": [{"name": "Google C2PA Core Generator Library"}],
                                "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}],
                                "ingredients": [{"relationship": "inputTo"}] * ingredients}}}


class Evaluate(unittest.TestCase):
    def test_a_google_ai_credential_passes(self):
        result = cc.evaluate(store())
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.issuer, "Google LLC")

    def test_no_credential_fails(self):
        self.assertFalse(cc.evaluate(None).ok)
        self.assertFalse(cc.evaluate({}).ok)

    def test_a_credential_that_does_not_validate_fails(self):
        self.assertFalse(cc.evaluate(store(state="Invalid")).ok)
        self.assertFalse(cc.evaluate(store(codes=("assertion.dataHash.mismatch",))).ok,
                         "a file changed after signing must not pass")

    def test_another_issuer_is_not_trusted(self):
        self.assertFalse(cc.evaluate(store(issuer="Somebody Else")).ok)

    def test_a_camera_or_a_composite_step_fails(self):
        for source in ("http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
                       "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"):
            with self.subTest(source=source):
                self.assertFalse(cc.evaluate(store(sources=(AI, source))).ok)

    def test_inputs_must_be_named_and_counted(self):
        self.assertFalse(cc.evaluate(store(ingredients=1)).ok, "an unnamed input may have been a photo")
        self.assertFalse(cc.evaluate(store(ingredients=1), inputs=2).ok)
        self.assertTrue(cc.evaluate(store(ingredients=1), inputs=1).ok)
        self.assertFalse(cc.evaluate(store(), inputs=1).ok, "naming an input the credential never had")

    def test_a_plain_photo_has_no_credential(self):
        try:
            import c2pa  # noqa: F401
        except ImportError:
            self.skipTest("c2pa-python not installed")
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, "JPEG")
        self.assertFalse(cc.check(buffer.getvalue(), "image/jpeg").ok)

    def test_a_real_gemini_image_passes(self):
        # Set HAWK_C2PA_SAMPLE to a Nano Banana image to check the reader against a real signed file.
        path = os.environ.get("HAWK_C2PA_SAMPLE")
        if not path:
            self.skipTest("HAWK_C2PA_SAMPLE not set")
        with open(path, "rb") as handle:
            data = handle.read()
        result = cc.check(data, "image/png" if data[:4] == b"\x89PNG" else "image/jpeg")
        self.assertTrue(result.ok, result.reason)


def png(color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, "PNG")
    return buffer.getvalue()


class Service(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix="hawk_c2pa_test_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        input_dir = os.path.join(self.dir, "comfy_input")
        os.makedirs(input_dir)
        self.service = HawkService(Settings(token="t" * 24, data_dir=self.dir, comfy_input_dir=input_dir))

        async def upload(fileobj, name, subfolder, mime):
            os.makedirs(os.path.join(input_dir, subfolder), exist_ok=True)
            fileobj.seek(0)
            with open(os.path.join(input_dir, subfolder, name), "wb") as handle:
                handle.write(fileobj.read())
            return {"name": name, "subfolder": subfolder}

        self.service.comfy.upload = upload

    async def upload(self, color) -> dict:
        data = png(color)
        return await self.service.add_asset(f"{color}.png", io.BytesIO(data), "image/png", len(data),
                                            source={"type": "upload"})

    def credential(self, ok=True, inputs_needed=0):
        def check(data, mime, inputs=0):
            if not ok:
                return cc.CredentialCheck(False, "no credential")
            if inputs != inputs_needed:
                return cc.CredentialCheck(False, "wrong number of inputs")
            return cc.CredentialCheck(True, "signed by Google", "Google LLC", "Google C2PA", {"issuer": "Google LLC"})
        patcher = mock.patch.object(cc, "check", side_effect=check)
        patcher.start()
        self.addCleanup(patcher.stop)

    def counts_as_upload(self, asset_id) -> bool:
        return local_images.from_upload(self.service.store.get_asset(asset_id), self.service.store.lineage)

    async def test_a_verified_upload_counts_as_generated_and_keeps_the_evidence(self):
        self.credential()
        photo = await self.upload((1, 2, 3))
        result = await self.service.verify_ai_credential(photo["id"])
        self.assertTrue(result["verified"])
        self.assertFalse(self.counts_as_upload(photo["id"]))
        source = self.service.store.get_asset(photo["id"])["source"]
        self.assertEqual(source["credential"]["issuer"], "Google LLC")
        self.assertEqual(source["corrected_from"]["type"], "upload", "the old record is kept, for undo")

    async def test_a_failed_check_changes_nothing(self):
        self.credential(ok=False)
        photo = await self.upload((4, 5, 6))
        with self.assertRaises(RequestError):
            await self.service.verify_ai_credential(photo["id"])
        self.assertTrue(self.counts_as_upload(photo["id"]))

    async def test_an_input_that_is_itself_an_upload_is_refused(self):
        self.credential(inputs_needed=1)
        base, edit = await self.upload((7, 8, 9)), await self.upload((10, 11, 12))
        with self.assertRaises(RequestError) as caught:
            await self.service.verify_ai_credential(edit["id"], made_from=[base["id"]])
        self.assertIn("counts as an upload", str(caught.exception))
        self.assertTrue(self.counts_as_upload(edit["id"]))

    async def test_an_edit_of_a_verified_image_passes_once_its_input_is_verified(self):
        self.credential()
        base = await self.upload((13, 14, 15))
        await self.service.verify_ai_credential(base["id"])
        cc.check.side_effect = lambda data, mime, inputs=0: cc.CredentialCheck(
            inputs == 1, "ok" if inputs == 1 else "wrong number of inputs", "Google LLC", "g", {})
        edit = await self.upload((16, 17, 18))
        await self.service.verify_ai_credential(edit["id"], made_from=[base["id"]])
        self.assertFalse(self.counts_as_upload(edit["id"]))
        self.assertEqual(self.service.store.get_asset(edit["id"])["source"]["references"], [base["id"]])

    async def test_it_can_be_undone(self):
        self.credential()
        photo = await self.upload((19, 20, 21))
        await self.service.verify_ai_credential(photo["id"])
        self.service.update_asset(photo["id"], generated_from="")
        self.assertTrue(self.counts_as_upload(photo["id"]))


if __name__ == "__main__":
    unittest.main()
