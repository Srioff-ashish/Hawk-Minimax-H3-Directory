"""Pure helpers of the Colab launcher. Standard library only."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deploy", "colab"))

import hawk_colab  # noqa: E402


class Manifest(unittest.TestCase):
    def test_defaults(self):
        items = hawk_colab.manifest(
            "ref2va pruned int8 (21 GB, recommended)", "nvfp4 (16 GB, recommended on G4)"
        )
        self.assertEqual(
            [(i.folder, i.filename) for i in items],
            [
                ("diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
                ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
                ("vae", "minimax_h3_video_vae_fp16.safetensors"),
                ("vae", "minimax_h3_audio_vae_fp32.safetensors"),
                ("loras", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),
            ],
        )
        self.assertTrue(all(i.repo == "Comfy-Org/MiniMax-H3" and i.path == f"{i.folder}/{i.filename}" for i in items))
        self.assertAlmostEqual(hawk_colab.estimated_gb(items), 44.44, places=1)

    def test_defaults_match_the_api_model_settings(self):
        sys.path.insert(0, ROOT)
        from hawk_api.config import ModelSettings

        defaults = ModelSettings()
        unet = os.path.basename(hawk_colab.DIFFUSION_MODELS["ref2va pruned int8 (21 GB, recommended)"][0])
        clip = os.path.basename(hawk_colab.TEXT_ENCODERS["nvfp4 (16 GB, recommended on G4)"][0])
        self.assertEqual((unet, clip), (defaults.unet_name, defaults.clip_name))
        with open(os.path.join(ROOT, "deploy", "loras.example.json"), encoding="utf-8") as handle:
            required = json.load(handle)["defaults"][0]["name"]
        self.assertEqual(required, os.path.basename(hawk_colab.TURBO_LORA[0]))

    def test_notebook_choices_are_known(self):
        with open(os.path.join(ROOT, "deploy", "colab", "Hawk_H3_API_Colab.ipynb"), encoding="utf-8") as handle:
            source = "".join("".join(cell["source"]) for cell in json.load(handle)["cells"])
        for label in list(hawk_colab.DIFFUSION_MODELS) + list(hawk_colab.TEXT_ENCODERS):
            self.assertIn(f'"{label}"', source)

    def test_extra_loras(self):
        items = hawk_colab.manifest(
            "ref2va pruned fp8 (21 GB)",
            "int8 (27 GB)",
            "fal/MiniMax-H3-Realism-People-LoRA/h3-realism-people-t2v-i2v-r2v.safetensors,\n"
            "https://huggingface.co/TenStrip/Minimax-h3_Singularity-Lora/resolve/main/Minimax-h3_Singularity_64-fro95_lora.safetensors?download=true, "
            "https://example.com/files/style%20lora.safetensors",
            turbo_lora=False,
        )
        extras = items[4:]
        self.assertEqual((extras[0].repo, extras[0].path), ("fal/MiniMax-H3-Realism-People-LoRA", "h3-realism-people-t2v-i2v-r2v.safetensors"))
        self.assertEqual((extras[1].repo, extras[1].filename, extras[1].revision),
                         ("TenStrip/Minimax-h3_Singularity-Lora", "Minimax-h3_Singularity_64-fro95_lora.safetensors", "main"))
        self.assertEqual((extras[2].url, extras[2].filename), ("https://example.com/files/style%20lora.safetensors", "style lora.safetensors"))
        self.assertTrue(all(i.folder == "loras" for i in extras))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            hawk_colab.manifest("big model", "nvfp4 (16 GB, recommended on G4)")
        for spec in ("just-a-name", "https://example.com/page", "owner/repo"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                hawk_colab.parse_lora_source(spec)


class Detection(unittest.TestCase):
    FILES = {
        "diffusion_models": [
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "h3/minimax_h3_ref2va_bf16.safetensors",
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        ],
        "text_encoders": ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "umt5_xxl.safetensors"],
        "vae": ["minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_video_vae_fp16.safetensors", "wan_vae.safetensors"],
        "loras": [
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "drbaph/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            "MysticXXX_MMH3-V4-ref2va.safetensors",
        ],
    }

    def test_picks_ref2va_and_preferred_files(self):
        chosen = hawk_colab.pick_models(self.FILES)
        self.assertEqual(chosen, {
            "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
            "turbo_lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        })

    def test_fl2va_only_is_not_picked(self):
        chosen = hawk_colab.pick_models({"diffusion_models": self.FILES["diffusion_models"][:1], "loras": self.FILES["loras"][:1]})
        self.assertIsNone(chosen["unet_name"])
        self.assertIsNone(chosen["turbo_lora"])

    def test_resolve_from_disk_with_overrides_and_errors(self):
        import tempfile

        with tempfile.TemporaryDirectory() as comfy:
            for folder, names in self.FILES.items():
                for name in names:
                    path = os.path.join(comfy, "models", folder, name)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "wb").close()
            self.assertEqual(hawk_colab.list_model_files(comfy, "loras")[0], "MysticXXX_MMH3-V4-ref2va.safetensors")
            chosen = hawk_colab.resolve_models(comfy, clip_name="custom_te.safetensors")
            self.assertEqual((chosen["unet_name"], chosen["clip_name"]),
                             ("minimax_h3_ref2va_pruned_int8_convrot.safetensors", "custom_te.safetensors"))
            labelled = hawk_colab.resolve_models(comfy, text_encoder="nvfp4 (16 GB, recommended on G4)")
            self.assertEqual(labelled["clip_name"], "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(RuntimeError, "unet_name, clip_name, video_vae, audio_vae"):
                hawk_colab.resolve_models(empty)

    def test_lora_config_uses_the_turbo_file_on_disk(self):
        with open(os.path.join(ROOT, "deploy", "loras.example.json"), encoding="utf-8") as handle:
            example = json.load(handle)
        config = hawk_colab.lora_config(example, "drbaph/turbo_rank21.safetensors")
        self.assertEqual(config["defaults"], [{"name": "drbaph/turbo_rank21.safetensors", "strength": 1.0, "required": True, "turbo": True}])
        self.assertEqual(config["presets"], example["presets"])
        self.assertEqual(hawk_colab.lora_config(example, None)["defaults"], [])
        sys.path.insert(0, ROOT)
        from hawk_api.loras import parse_config

        parse_config(config)  # the API accepts it


class Runtime(unittest.TestCase):
    def test_tunnel_url(self):
        log = (
            "2026-09-14T10:00:00Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
            "2026-09-14T10:00:02Z INF |  https://brave-otter-lane-hills.trycloudflare.com  |\n"
        )
        self.assertEqual(hawk_colab.find_tunnel_url(log), "https://brave-otter-lane-hills.trycloudflare.com")
        self.assertIsNone(hawk_colab.find_tunnel_url("INF Starting tunnel"))

    def test_api_tunnel_pattern_spares_other_tunnels(self):
        import re

        api_tunnel = f"/content/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:{hawk_colab.API_PORT}"
        ui_tunnel = "/usr/local/bin/cloudflared tunnel --url http://localhost:8188 --no-autoupdate"
        self.assertTrue(re.search(hawk_colab.API_TUNNEL_PATTERN, api_tunnel))
        self.assertIsNone(re.search(hawk_colab.API_TUNNEL_PATTERN, ui_tunnel))

    def test_port_free(self):
        import socket

        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            self.assertFalse(hawk_colab.port_free(port))
        self.assertTrue(hawk_colab.port_free(port))

    def test_blackwell_torch_check(self):
        blackwell = {"cap": [12, 0], "arch": ["sm_80", "sm_90", "sm_120"]}
        self.assertFalse(hawk_colab.needs_blackwell_torch(blackwell))
        self.assertTrue(hawk_colab.needs_blackwell_torch({"cap": [12, 0], "arch": ["sm_80", "sm_90"]}))
        self.assertFalse(hawk_colab.needs_blackwell_torch({"cap": [8, 0], "arch": ["sm_80"]}))
        self.assertFalse(hawk_colab.needs_blackwell_torch({}))


class Snapshots(unittest.TestCase):
    """Bringing a previous runtime's chats back, before the API starts."""

    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="hawk_data_")
        self.drive = tempfile.mkdtemp(prefix="hawk_drive_")
        self.addCleanup(shutil.rmtree, self.data, True)
        self.addCleanup(shutil.rmtree, self.drive, True)
        self.backups = os.path.join(self.drive, hawk_colab.SNAPSHOT_FOLDER)
        os.makedirs(self.backups, exist_ok=True)

    def write_db(self, path, title):
        db = sqlite3.connect(path)
        db.executescript("CREATE TABLE IF NOT EXISTS agent_sessions(id TEXT PRIMARY KEY, data TEXT);")
        db.execute("INSERT OR REPLACE INTO agent_sessions VALUES('s1', ?)", (title,))
        db.commit()
        db.close()

    def titles(self, path):
        db = sqlite3.connect(path)
        try:
            return [row[0] for row in db.execute("SELECT data FROM agent_sessions")]
        finally:
            db.close()

    def test_a_snapshot_comes_back_on_a_fresh_runtime(self):
        self.write_db(os.path.join(self.backups, hawk_colab.SNAPSHOT_NAME), "Diwali shoot")
        said = hawk_colab.restore_snapshot(self.data, self.drive)
        self.assertIn("Restored", said, said)
        self.assertEqual(self.titles(os.path.join(self.data, hawk_colab.SNAPSHOT_NAME)), ["Diwali shoot"])

    def test_a_database_already_here_is_never_replaced(self):
        self.write_db(os.path.join(self.backups, hawk_colab.SNAPSHOT_NAME), "older")
        self.write_db(os.path.join(self.data, hawk_colab.SNAPSHOT_NAME), "the live one")
        said = hawk_colab.restore_snapshot(self.data, self.drive)
        self.assertIn("Keeping", said, said)
        self.assertEqual(self.titles(os.path.join(self.data, hawk_colab.SNAPSHOT_NAME)), ["the live one"],
                         "re-running the start cell mid-session must not drop an older copy on live chats")

    def test_a_truncated_snapshot_falls_back_to_the_one_before_it(self):
        open(os.path.join(self.backups, hawk_colab.SNAPSHOT_NAME), "wb").close()  # caught mid-copy
        self.write_db(os.path.join(self.backups, hawk_colab.SNAPSHOT_NAME + ".prev"), "the one before")
        said = hawk_colab.restore_snapshot(self.data, self.drive)
        self.assertIn("Restored", said, said)
        self.assertEqual(self.titles(os.path.join(self.data, hawk_colab.SNAPSHOT_NAME)), ["the one before"])

    def test_no_drive_mounted_is_a_message_not_a_crash(self):
        said = hawk_colab.restore_snapshot(self.data, os.path.join(self.drive, "not-mounted"))
        self.assertIn("No Google Drive", said, said)
        self.assertFalse(os.path.exists(os.path.join(self.data, hawk_colab.SNAPSHOT_NAME)))

    def test_an_empty_drive_folder_just_starts_fresh(self):
        self.assertIn("starting fresh", hawk_colab.restore_snapshot(self.data, self.drive))


if __name__ == "__main__":
    unittest.main()
