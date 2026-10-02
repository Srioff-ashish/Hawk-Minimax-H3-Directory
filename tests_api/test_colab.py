"""Pure helpers of the Colab launcher. Standard library only."""

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

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
            # The dropdown says nvfp4 and the fixture has only int8 on disk, which is what happens whenever
            # the download cell is skipped or run with a different choice. Pinning the label there configured
            # the API with a file ComfyUI does not have, and every render died at the loader with
            # "Value not in list" long after the notebook had reported success.
            labelled = hawk_colab.resolve_models(comfy, text_encoder="nvfp4 (16 GB, recommended on G4)")
            self.assertEqual(labelled["clip_name"], "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                             "a label that was never downloaded must give way to the file that is there")
            # and the label still wins when its file really is on disk
            present = hawk_colab.resolve_models(comfy, text_encoder="int8 (27 GB)")
            self.assertEqual(present["clip_name"], "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(RuntimeError, "unet_name, clip_name, video_vae, audio_vae"):
                hawk_colab.resolve_models(empty)

    def test_an_explicit_model_name_that_is_not_on_disk_is_used_but_reported(self):
        """The cell that started this pod named two files it did not have.

        A saved render_models.json happened to override both, so nothing broke and nothing was said. With
        no such override the API starts on a name ComfyUI has never heard of and every render dies at the
        loader, a failure with no visible connection to the cell that set it.
        """
        import contextlib
        import tempfile

        with tempfile.TemporaryDirectory() as comfy:
            for folder, names in self.FILES.items():
                for name in names:
                    path = os.path.join(comfy, "models", folder, name)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "wb").close()
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                chosen = hawk_colab.resolve_models(
                    comfy, unet_name="minimax_h3_ref2va_pruned_fp8_scaled.safetensors")
            self.assertEqual(chosen["unet_name"], "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
                             "an explicit name must still win: ComfyUI can reach roots this listing cannot")
            self.assertIn("is not in models/diffusion_models", said.getvalue(),
                          "naming a file that is absent has to say so, or the render fails far from here")
            self.assertIn("minimax_h3_ref2va_pruned_int8_convrot.safetensors", said.getvalue(),
                          "the note should list what is on disk, so the right name is one glance away")

            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                hawk_colab.resolve_models(comfy, unet_name="minimax_h3_ref2va_pruned_int8_convrot.safetensors")
            self.assertEqual(quiet.getvalue(), "",
                             "a name that is on disk is ordinary and must print nothing at all")

    def test_encoders_kept_in_subfolders_are_named_as_comfyui_lists_them(self):
        """Encoders sorted into text_encoders/h3/, krea2/, qwen/... are "h3/qwen3vl_..." to ComfyUI's loader,
        which refuses the bare file name. A label or explicit name must come back as that subfolder path."""
        import tempfile

        with tempfile.TemporaryDirectory() as comfy:
            files = dict(self.FILES, text_encoders=["h3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                                                    "krea2/qwen3vl_4b_fp8_scaled.safetensors"])
            for folder, names in files.items():
                for name in names:
                    path = os.path.join(comfy, "models", folder, name)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "wb").close()
            want = "h3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
            self.assertEqual(hawk_colab.resolve_models(comfy)["clip_name"], want)
            self.assertEqual(hawk_colab.resolve_models(comfy, text_encoder="int8 (27 GB)")["clip_name"], want)
            self.assertEqual(hawk_colab.resolve_models(
                comfy, clip_name="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")["clip_name"], want)

    def test_sage_gives_way_to_an_attention_flag_the_caller_chose(self):
        """ComfyUI's --use-*-attention flags are one mutually exclusive group.

        Adding sage beside a caller's own choice is an argparse error, so ComfyUI exits at once and the
        pod is left with none -- an expensive way to find out that two flags cannot be combined.
        """
        session = hawk_colab.Session(
            comfy_dir=".", pack_dir=".", token="t", env={}, log_dir=".",
            comfy_args=["--use-ck-attention"])
        with unittest.mock.patch.object(hawk_colab, "sage_installed", lambda: True):
            chosen = hawk_colab.comfy_command(session)
            self.assertNotIn("--use-sage-attention", chosen,
                             "sage must stand down: both flags together stop ComfyUI starting at all")
            self.assertIn("--use-ck-attention", chosen, "the caller's own choice is the one that survives")

            plain = hawk_colab.Session(comfy_dir=".", pack_dir=".", token="t", env={}, log_dir=".")
            self.assertIn("--use-sage-attention", hawk_colab.comfy_command(plain),
                          "with no competing flag sage is still added, as it always was")
            self.assertNotIn("--use-sage-attention", hawk_colab.comfy_command(plain, ["--use-flash-attention"]),
                             "a flag passed for one restart collides just as surely as a stored one")

    def test_the_comfy_compiler_is_disabled_only_where_comfyui_knows_the_flag(self):
        """The aimdo compiler kills long H3 segments ("aimdo memory compile error"), so it is switched off --
        but an older ComfyUI without the flag would refuse to start at all, so it is only passed when known."""
        import tempfile

        with tempfile.TemporaryDirectory() as comfy:
            session = hawk_colab.Session(comfy_dir=comfy, pack_dir=".", token="t", env={}, log_dir=".")
            self.assertNotIn("--disable-comfy-compiler", hawk_colab.comfy_command(session))
            os.makedirs(os.path.join(comfy, "comfy"))
            with open(os.path.join(comfy, "comfy", "cli_args.py"), "w", encoding="utf-8") as handle:
                handle.write('parser.add_argument("--disable-comfy-compiler", action="store_true")\n')
            self.assertIn("--disable-comfy-compiler", hawk_colab.comfy_command(session))
            self.assertEqual(hawk_colab.comfy_command(session, ["--disable-comfy-compiler"]).count(
                "--disable-comfy-compiler"), 1, "a caller's own copy is not doubled")

    def test_a_restart_waits_for_a_render_comfyui_is_too_busy_to_report(self):
        """Rerunning the start cell mid-render killed it: a ComfyUI loading the encoder did not answer /queue
        in time and silence read as idle. Silence, and any job the API still has running, now mean busy."""
        session = hawk_colab.Session(comfy_dir=".", pack_dir=".", token="t", env={}, log_dir=".")
        patch = unittest.mock.patch.object
        with patch(hawk_colab, "port_free", lambda port: False), patch(hawk_colab, "_http_json", lambda *a, **k: None):
            self.assertIn("did not answer", hawk_colab.render_in_progress(session))
            with self.assertRaisesRegex(RuntimeError, "did not answer"):
                hawk_colab.restart_comfyui(session)

        class Reply(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        jobs = json.dumps({"jobs": [{"id": "ebc51ece-1", "status": "rendering"}, {"id": "x", "status": "done"}]})
        with patch(hawk_colab, "port_free", lambda port: False), \
                patch(hawk_colab, "_http_json", lambda *a, **k: {"queue_running": []}), \
                patch(hawk_colab.urllib.request, "urlopen", lambda *a, **k: Reply(jobs.encode())):
            self.assertIn("ebc51ece", hawk_colab.render_in_progress(session),
                          "a render the API holds is in progress even when ComfyUI's queue is empty")
        idle = json.dumps({"jobs": [{"id": "x", "status": "done"}]})
        with patch(hawk_colab, "port_free", lambda port: False), \
                patch(hawk_colab, "_http_json", lambda *a, **k: {"queue_running": []}), \
                patch(hawk_colab.urllib.request, "urlopen", lambda *a, **k: Reply(idle.encode())):
            self.assertEqual(hawk_colab.render_in_progress(session), "")

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

    def test_a_busy_comfyui_is_not_a_dead_one(self):
        """watch() force-restarts a ComfyUI it believes has died, and a killed render is unrecoverable.

        With a ComfyUI started by another notebook cell there is no process to poll, so the only question
        left is over HTTP -- and one deep in a render answers /queue late. A single slow probe used to be
        enough to restart it and lose a 40-minute render, so it takes several misses in a row now.
        """
        probes = iter([None, None, 200, None, None, None])  # two slow, recovers, then genuinely gone
        seen = []

        def status(url, headers=None, timeout=10.0):
            seen.append(timeout)
            return next(probes)

        strikes, restarted, comfy_strikes = 0, [], 3
        for _ in range(6):
            answered = status("/queue", timeout=hawk_colab.COMFY_PROBE_SECONDS) == 200
            strikes = 0 if answered else strikes + 1
            if strikes >= comfy_strikes:
                restarted.append(True)
                strikes = 0
        self.assertEqual(len(restarted), 1, "only the three misses in a row count as death, not the first two")
        self.assertTrue(all(t >= 20 for t in seen),
                        "a five second patience is what made a rendering ComfyUI look dead in the first place")

    def test_a_restart_brings_back_the_comfyui_that_was_started(self):
        """One builder for the argv, because a restart that drops a flag is invisible until a render OOMs.

        --reserve-vram is the one that matters: the VAE decode at the very end of an otherwise finished
        render is what usually runs out of memory, and losing it there loses the whole render.
        """
        session = hawk_colab.Session.__new__(hawk_colab.Session)
        session.comfy_args = ["--preview-method", "auto"]
        saved = hawk_colab.sage_installed
        hawk_colab.sage_installed = lambda: True
        try:
            started = hawk_colab.comfy_command(session)
            restarted = hawk_colab.comfy_command(session, ["--enable-triton-backend"])
        finally:
            hawk_colab.sage_installed = saved
        for flag in ("--reserve-vram", "--use-sage-attention", "--preview-method", "--disable-auto-launch"):
            self.assertIn(flag, started, f"{flag} belongs on every ComfyUI this module starts")
            self.assertIn(flag, restarted, f"{flag} must survive a restart, not just the first start")
        self.assertIn("--enable-triton-backend", restarted, "and a restart can still add its own")
        self.assertNotIn("--enable-triton-backend", started)
        self.assertEqual(started[started.index("--port") + 1], str(hawk_colab.COMFY_PORT))

    def test_sage_is_left_off_the_command_when_it_is_not_installed(self):
        session = hawk_colab.Session.__new__(hawk_colab.Session)
        session.comfy_args = []
        saved = hawk_colab.sage_installed
        hawk_colab.sage_installed = lambda: False
        try:
            self.assertNotIn("--use-sage-attention", hawk_colab.comfy_command(session),
                             "ComfyUI refuses to start on that flag without the package behind it")
        finally:
            hawk_colab.sage_installed = saved

    def _report(self, mode: str, *, sol: bool, sage: bool) -> str:
        """attention_report with both backends under the test's control, neither on this machine."""
        session = hawk_colab.Session.__new__(hawk_colab.Session)
        session.env = {"HAWK_ATTENTION": mode}
        saved_json, saved_sage = hawk_colab._http_json, hawk_colab.sage_installed
        hawk_colab._http_json = lambda url, **k: ({"SolAttnPatch": {}} if sol and "SolAttn" in url else None)
        hawk_colab.sage_installed = lambda: sage
        try:
            return hawk_colab.attention_report(session)
        finally:
            hawk_colab._http_json, hawk_colab.sage_installed = saved_json, saved_sage

    def test_the_attention_report_names_a_backend_that_is_not_there(self):
        # hawk_h3 skips a missing backend with a log line and renders anyway, several times slower, and
        # nothing about the result looks wrong -- so the launcher has to say it out loud.
        report = self._report("sol scheduled", sol=False, sage=True)
        self.assertIn("MISSING", report, "a mode asking for sol without sol loaded has to be visible")
        self.assertIn("dense attention", report, "and it has to say what that costs")

    def test_the_attention_report_is_quiet_when_both_backends_are_there(self):
        report = self._report("sol scheduled", sol=True, sage=True)
        self.assertIn("loaded", report)
        self.assertNotIn("MISSING", report, "nothing is wrong, so nothing should look wrong")

    def test_sage_is_reported_even_when_the_mode_does_not_ask_for_it(self):
        # --use-sage-attention is ComfyUI's own global backend, so it applies to every render whatever
        # HAWK_ATTENTION says. Reporting it only when the mode named it hid a real speedup being absent.
        report = self._report("sol scheduled", sol=True, sage=False)
        self.assertIn("sage", report)
        self.assertIn("MISSING", report, "sage missing is worth knowing even under a sol-only mode")
        self.assertIn("global ComfyUI flag", report, "and it should say why it is listed under this mode")

    def test_comfy_default_attention_still_reports_sage(self):
        report = self._report("comfy default", sol=False, sage=True)
        self.assertNotIn("sol", report, "this mode never asks for sol, so sol is not its problem")
        self.assertIn("installed", report)

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
