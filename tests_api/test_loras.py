"""LoRA resolution rules. Pure Python: python -m unittest discover -s tests_api -p 'test_loras.py'"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.loras import (  # noqa: E402
    LoraError,
    LoraSpec,
    choose_steps,
    compare_applied,
    default_status,
    parse_applied,
    parse_config,
    resolve_name,
    resolve_request,
)

AVAILABLE = [
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "styles/h3-realism-people-t2v-i2v-r2v.safetensors",
    "styles/Anime_Motion.safetensors",
    "chars/anime_motion.safetensors",
    "Minimax-h3_Singularity_64-fro95_lora.safetensors",
]

CONFIG = parse_config(
    {
        "defaults": [{"name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "required": True, "turbo": True}],
        "presets": {"realism": [{"name": "realism-people", "strength": 0.8}]},
    }
)


class ResolveName(unittest.TestCase):
    def test_exact_basename_substring(self):
        self.assertEqual(resolve_name(AVAILABLE[1], AVAILABLE), AVAILABLE[1])
        self.assertEqual(resolve_name("H3-Realism-People-T2V-I2V-R2V", AVAILABLE), AVAILABLE[1])
        self.assertEqual(resolve_name("singularity", AVAILABLE), AVAILABLE[4])
        self.assertEqual(resolve_name("styles\\h3-realism-people-t2v-i2v-r2v.safetensors", AVAILABLE), AVAILABLE[1])

    def test_ambiguous(self):
        with self.assertRaisesRegex(LoraError, "several files") as ctx:
            resolve_name("anime_motion", AVAILABLE)
        self.assertEqual(len(ctx.exception.details["matches"]), 2)

    def test_missing_suggests(self):
        with self.assertRaises(LoraError) as ctx:
            resolve_name("realism-peple", AVAILABLE)
        self.assertIn(AVAILABLE[1], ctx.exception.details["suggestions"])
        self.assertIn("LoRA 'realism-peple' is not in ComfyUI's models/loras", str(ctx.exception))

    def test_other_folders(self):
        models = ["minimax_h3_ref2va_pruned_int8_convrot.safetensors", "minimax_h3_ref2va_pruned_bf16.safetensors"]
        self.assertEqual(resolve_name("bf16", models, label="Base model", folder="diffusion_models"), models[1])
        with self.assertRaisesRegex(LoraError, "Base model 'fp8' is not in ComfyUI's models/diffusion_models"):
            resolve_name("fp8", models, label="Base model", folder="diffusion_models")


class ResolveRequest(unittest.TestCase):
    def test_defaults_preset_request_merge(self):
        resolved, warnings = resolve_request(
            CONFIG, AVAILABLE, preset="realism", loras=[LoraSpec("realism-people", 0.5), LoraSpec("singularity", 0.7)]
        )
        self.assertEqual([r.file for r in resolved], [AVAILABLE[0], AVAILABLE[1], AVAILABLE[4]])
        self.assertEqual(resolved[1].strength, 0.5)
        self.assertEqual(resolved[1].source, "request")
        self.assertTrue(resolved[0].turbo)
        self.assertEqual(warnings, [])

    def test_strength_zero_switches_off_a_default(self):
        resolved, _ = resolve_request(CONFIG, AVAILABLE, loras=[LoraSpec("turbo", 0.0)])
        self.assertEqual(resolved, [])

    def test_required_default_missing_refuses(self):
        with self.assertRaisesRegex(LoraError, "Required default LoRA"):
            resolve_request(CONFIG, AVAILABLE[1:])
        resolved, _ = resolve_request(CONFIG, AVAILABLE[1:], use_defaults=False)
        self.assertEqual(resolved, [])

    def test_optional_default_missing_warns(self):
        config = parse_config({"defaults": [{"name": "nope.safetensors"}]})
        resolved, warnings = resolve_request(config, AVAILABLE)
        self.assertEqual(resolved, [])
        self.assertEqual(len(warnings), 1)

    def test_unknown_preset(self):
        with self.assertRaisesRegex(LoraError, "Unknown lora_preset"):
            resolve_request(CONFIG, AVAILABLE, preset="cinema")

    def test_default_status(self):
        rows = default_status(CONFIG, AVAILABLE[1:])
        self.assertEqual((rows[0]["present"], rows[0]["required"]), (False, True))


class Steps(unittest.TestCase):
    def test_turbo_decides_default_steps(self):
        with_turbo, _ = resolve_request(CONFIG, AVAILABLE)
        self.assertEqual(choose_steps(with_turbo, None)[0], 8)
        without, _ = resolve_request(CONFIG, AVAILABLE, use_defaults=False, loras=[LoraSpec("singularity")])
        self.assertEqual(choose_steps(without, None)[0], 30)
        self.assertEqual(choose_steps(without, 12), (12, "set by the request"))
        # A request LoRA with "turbo" in its file name counts as turbo too.
        turbo_by_name, _ = resolve_request(parse_config({}), AVAILABLE, loras=[LoraSpec("turbo")])
        self.assertEqual(choose_steps(turbo_by_name, None)[0], 8)


class Applied(unittest.TestCase):
    def test_parse_and_compare(self):
        text = f"{AVAILABLE[0]} @ 1\n{AVAILABLE[1]} @ 0.8  (v=1 a=0 t=1)"
        applied = parse_applied(text)
        self.assertEqual(applied, [(AVAILABLE[0], 1.0), (AVAILABLE[1], 0.8)])
        self.assertEqual(parse_applied("no LoRAs selected"), [])
        resolved = [{"file": AVAILABLE[0], "strength": 1.0}, {"file": AVAILABLE[1], "strength": 0.8}]
        self.assertEqual(compare_applied(resolved, applied), [])
        self.assertEqual(len(compare_applied(resolved, applied[:1])), 1)


if __name__ == "__main__":
    unittest.main()

class ShippedCatalogue(unittest.TestCase):
    """The defaults that ship with the code, and how they reach a pod that already has a loras.json."""

    def setUp(self):
        import json
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="hawk_loras_")
        self.path = os.path.join(self.tmp, "loras.json")
        self.json = json

    def test_video_defaults_are_the_turbo_lora_motion_booster_and_aio(self):
        from hawk_api.loras import load_config
        config = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "deploy", "loras.example.json"))
        names = [(spec.name, spec.strength, spec.required) for spec in config.defaults]
        self.assertEqual(names, [("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1.0, True),
                                 ("H3_Motion_BoosterV2.safetensors", 1.0, False),
                                 ("HMNSFW_AIO_V25.safetensors", 0.8, False)])
        self.assertTrue(all(not spec.name.lower().startswith("krea2") for spec in config.defaults),
                        "image LoRAs never belong in the video defaults")

    def test_new_shipped_defaults_reach_a_pod_that_already_has_a_copy(self):
        from hawk_api.loras import load_config
        with open(self.path, "w", encoding="utf-8") as handle:  # what a pod from before the new defaults has
            self.json.dump({"defaults": [{"name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                                          "required": True, "turbo": True}], "presets": {}}, handle)
        names = [spec.name for spec in load_config(self.path).defaults]
        self.assertEqual(names, ["minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                                 "H3_Motion_BoosterV2.safetensors", "HMNSFW_AIO_V25.safetensors"])
        with open(self.path, "r", encoding="utf-8") as handle:
            self.assertEqual(len(self.json.load(handle)["defaults"]), 1, "the pod's own file is never rewritten")

    def test_the_pods_own_settings_win(self):
        from hawk_api.loras import load_config
        mine = {"defaults": [{"name": "H3_Motion_BoosterV2.safetensors", "strength": 0},
                             {"name": "HMNSFW_AIO_V25.safetensors", "strength": 0.4}], "presets": {}}
        with open(self.path, "w", encoding="utf-8") as handle:
            self.json.dump(mine, handle)
        applied = {spec.name: spec.strength for spec in load_config(self.path).defaults}
        self.assertEqual(applied["H3_Motion_BoosterV2.safetensors"], 0.0, "strength 0 switches a shipped default off")
        self.assertEqual(applied["HMNSFW_AIO_V25.safetensors"], 0.4, "a strength set on the pod is kept")
        self.assertIn("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", applied, "and the rest still arrive")
