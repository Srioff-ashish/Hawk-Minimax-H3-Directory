"""The notebook's setup helpers: secrets, model paths, LoRA checks, sync and removal receipts."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deploy", "colab")))

import hawk_setup  # noqa: E402


def fake_safetensors(path: str, size: int = 2_000_000) -> None:
    header = b'{"__metadata__":{}}'
    with open(path, "wb") as handle:
        handle.write(len(header).to_bytes(8, "little") + header + b"\0" * (size - 8 - len(header)))


class Secrets(unittest.TestCase):
    def test_a_missing_optional_secret_is_skipped(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "x"}, clear=False):
            os.environ.pop("GOOGLE_API_KEY", None)
            found = hawk_setup.load_secrets(required=("HF_TOKEN",), optional=("GOOGLE_API_KEY",))
        self.assertEqual(found, {"HF_TOKEN": True, "GOOGLE_API_KEY": False})

    def test_a_missing_required_secret_stops(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NO_SUCH_SECRET_HAWK", None)
            with self.assertRaises(SystemExit):
                hawk_setup.load_secrets(required=("NO_SUCH_SECRET_HAWK",))


class Files(unittest.TestCase):
    def test_model_path_uses_the_saved_name_and_subfolder(self):
        entry = ("Comfy-Org/MiniMax-H3", "text_encoders/q.safetensors", "text_encoders", "h3/q.safetensors", 27.1)
        self.assertEqual(hawk_setup.model_path("/m", entry), "/m/text_encoders/h3/q.safetensors")
        entry = ("r", "vae/v.safetensors", "vae", None, None)
        self.assertEqual(hawk_setup.model_path("/m", entry), "/m/vae/v.safetensors")

    def test_an_html_page_is_not_a_safetensors(self):
        with tempfile.TemporaryDirectory() as root:
            good, bad = f"{root}/a.safetensors", f"{root}/b.safetensors"
            fake_safetensors(good)
            with open(bad, "wb") as handle:
                handle.write(b"<html>" + b" " * 2_000_000)
            self.assertTrue(hawk_setup.is_safetensors(good))
            self.assertFalse(hawk_setup.is_safetensors(bad))

    def test_downloads_skip_what_is_already_there(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(f"{root}/vae")
            with open(f"{root}/vae/v.safetensors", "wb") as handle:
                handle.write(b"x" * 1000)
            calls = []
            with mock.patch.object(hawk_setup, "_aria2", lambda url, out, headers, connections: calls.append(out) or True), \
                 mock.patch.object(hawk_setup.shutil, "which", return_value="/usr/bin/aria2c"):
                failed = hawk_setup.download_models(root, [("r", "vae/v.safetensors", "vae", None, None),
                                                           ("r", "loras/l.safetensors", "loras", None, 1.0)])
            self.assertEqual(failed, [])
            self.assertEqual(calls, [f"{root}/loras/l.safetensors"])


class Loras(unittest.TestCase):
    def test_sync_moves_fetches_and_never_deletes_what_it_installs(self):
        with tempfile.TemporaryDirectory() as root:
            fake_safetensors(f"{root}/flat.safetensors")
            os.makedirs(f"{root}/klein")
            fake_safetensors(f"{root}/klein/old.safetensors")
            fetched = []

            def civitai(out, version, token):
                fake_safetensors(out)
                fetched.append(os.path.basename(out))
                return True

            loras = {"h3": {"flat.safetensors": 1, "new.safetensors": 2}}
            with mock.patch.object(hawk_setup, "_civitai", civitai):
                report = hawk_setup.sync_loras(root, loras, "tok", remove=["old.safetensors", "new.safetensors"],
                                               dry_run=False)
            self.assertEqual(report["moved"], 1)
            self.assertEqual(fetched, ["new.safetensors"])
            self.assertTrue(os.path.exists(f"{root}/h3/flat.safetensors"))
            self.assertTrue(os.path.exists(f"{root}/h3/new.safetensors"))  # on the install list: refused
            self.assertFalse(os.path.exists(f"{root}/klein/old.safetensors"))
            receipt = json.load(open(f"{root}/{hawk_setup.LORA_RECEIPT}"))
            self.assertIn("old.safetensors", receipt)

    def test_a_dry_run_deletes_nothing_and_writes_no_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            fake_safetensors(f"{root}/old.safetensors")
            with mock.patch.object(hawk_setup, "_civitai", lambda *a: True):
                hawk_setup.sync_loras(root, {"h3": {}}, "tok", remove=["old.safetensors"], dry_run=True)
            self.assertTrue(os.path.exists(f"{root}/old.safetensors"))
            self.assertFalse(os.path.exists(f"{root}/{hawk_setup.LORA_RECEIPT}"))


if __name__ == "__main__":
    unittest.main()
