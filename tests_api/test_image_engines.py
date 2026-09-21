"""The image engine registry: capabilities, aliases, LoRA families and ladder order."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api import image_engines as ie


class Descriptors(unittest.TestCase):
    def test_every_alias_points_at_a_real_engine(self):
        for name, engine_id in ie.ALIASES.items():
            self.assertIn(engine_id, ie.ENGINES, f"alias {name!r} points at an unknown engine")

    def test_ids_match_their_keys(self):
        for key, engine in ie.ENGINES.items():
            self.assertEqual(key, engine.id)

    def test_only_local_engines_run_the_content_checks(self):
        for engine in ie.ENGINES.values():
            self.assertEqual(engine.checks, engine.local, f"{engine.id}: checks must follow where it runs")

    def test_an_engine_that_cannot_edit_takes_no_references(self):
        for engine in ie.ENGINES.values():
            if not engine.edit:
                self.assertEqual(engine.max_refs, 0, engine.id)
            else:
                self.assertGreater(engine.max_refs, 0, engine.id)

    def test_atlas_engines_carry_a_model_and_a_price_and_local_ones_do_not(self):
        for engine in ie.ENGINES.values():
            if engine.local:
                self.assertEqual(engine.atlas_model, "", engine.id)
                self.assertEqual(engine.price_key, "", f"{engine.id} runs here, so it is free")
                self.assertTrue(engine.lora_family, f"{engine.id} needs a LoRA family")
            else:
                self.assertTrue(engine.atlas_model, engine.id)
                self.assertTrue(engine.price_key, engine.id)

    def test_an_atlas_engine_that_edits_names_an_edit_model(self):
        for engine in ie.ENGINES.values():
            if engine.edit and not engine.local:
                self.assertTrue(engine.atlas_edit_model, engine.id)

    def test_tag_for_uses_the_edit_tag_only_on_edits(self):
        krea = ie.get("krea2")
        self.assertEqual(krea.tag_for("generate"), "krea2")
        self.assertEqual(krea.tag_for("edit"), "krea2-edit")
        # an engine with no separate edit tag reuses its own
        self.assertEqual(ie.get("seedream").tag_for("edit"), "seedream")


class Resolve(unittest.TestCase):
    def test_legacy_spellings_still_work(self):
        for name in ("local", "krea", "krea2", "KREA-2"):
            self.assertEqual(ie.resolve(name), "krea2", name)
        for name in ("turbo", "fast", "cheap", "z-image/turbo"):
            self.assertEqual(ie.resolve(name), "turbo", name)
        for name in ("seedream", "quality", "best"):
            self.assertEqual(ie.resolve(name), "seedream", name)
        self.assertEqual(ie.resolve("lite"), "seedream-lite")

    def test_z_image_now_means_the_local_engine(self):
        # the one deliberate break: it used to resolve to the Atlas engine
        self.assertEqual(ie.resolve("z-image"), "zimage")
        self.assertEqual(ie.resolve("zimage"), "zimage")
        self.assertTrue(ie.get("zimage").local)
        # and the change is announced rather than silent
        self.assertIn("local", ie.MOVED["z-image"])

    def test_auto_and_nonsense_resolve_to_nothing(self):
        self.assertEqual(ie.resolve("auto"), "")
        self.assertEqual(ie.resolve(""), "")
        self.assertEqual(ie.resolve("midjourney"), "")


class Families(unittest.TestCase):
    def test_the_files_the_notebook_installs_land_in_the_right_family(self):
        cases = {
            "H3_Motion_BoosterV2.safetensors": "h3",
            "HMNSFW_AIO_V25.safetensors": "h3",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": "h3",
            "MysticXXX_MMH3-V4-ref2va.safetensors": "h3",
            "krea2_mystic_xxx_v3.safetensors": "krea2",
            "snofs_krea2.safetensors": "krea2",
            "krea2_identity_edit_v1_2.safetensors": "krea2",
            "snofs_photodetail_slider.safetensors": "krea2",
            "zit_mystic_xxx.safetensors": "zit",
            "klein_snofs.safetensors": "klein",
            "klein_nsfw_no_face_change.safetensors": "klein",
        }
        for filename, family in cases.items():
            self.assertEqual(ie.family_of(filename), family, filename)

    def test_a_declared_family_wins_over_the_filename(self):
        self.assertEqual(ie.family_of("oddly_named.safetensors", declared="klein"), "klein")
        self.assertEqual(ie.family_of("krea2_thing.safetensors", declared="zit"), "zit")
        # a declared family that is not real is ignored rather than trusted
        self.assertEqual(ie.family_of("klein_thing.safetensors", declared="nonsense"), "klein")

    def test_an_unknown_file_is_treated_as_video(self):
        # video is the side that refuses unknown names, so it is the safe default
        self.assertEqual(ie.family_of("something_new.safetensors"), "h3")

    def test_a_subfolder_path_is_classified_by_its_basename(self):
        self.assertEqual(ie.family_of("Krea2/klein_snofs.safetensors"), "klein")

    def test_every_image_family_has_a_label(self):
        for family in ie.IMAGE_FAMILIES:
            self.assertIn(family, ie.LORA_FAMILIES)
            self.assertTrue(ie.family_label(family))


class Order(unittest.TestCase):
    STORED = [
        {"engine": "klein", "enabled": True},
        {"engine": "krea2", "enabled": True},
        {"engine": "zimage", "enabled": True},
        {"engine": "seedream", "enabled": True},
        {"engine": "turbo", "enabled": False},
    ]

    def test_generate_keeps_the_stored_order(self):
        self.assertEqual(ie.order(self.STORED, "generate"), ["klein", "krea2", "zimage", "seedream"])

    def test_a_generate_only_engine_is_left_out_of_the_edit_ladder(self):
        # zimage cannot edit, and turbo is both disabled and generate-only
        self.assertEqual(ie.order(self.STORED, "edit"), ["klein", "krea2", "seedream"])

    def test_disabled_engines_are_skipped(self):
        self.assertNotIn("turbo", ie.order(self.STORED, "generate"))

    def test_unknown_and_duplicate_rows_are_ignored(self):
        stored = [{"engine": "krea2"}, {"engine": "midjourney"}, {"engine": "local"}]
        self.assertEqual(ie.order(stored, "generate"), ["krea2"])

    def test_legacy_ids_in_a_stored_file_still_resolve(self):
        self.assertEqual(ie.order([{"engine": "local"}, {"engine": "turbo"}], "generate"), ["krea2", "turbo"])

    def test_nothing_stored_means_nothing_enabled(self):
        self.assertEqual(ie.order(None, "generate"), [])

    def test_full_order_appends_missing_engines_disabled(self):
        rows = ie.full_order([{"engine": "krea2", "enabled": True}], "generate")
        self.assertEqual(rows[0], {"engine": "krea2", "enabled": True})
        appended = {row["engine"]: row["enabled"] for row in rows[1:]}
        self.assertTrue(all(enabled is False for enabled in appended.values()))
        # a new engine is offered in Studio, never silently switched on
        self.assertIn("klein", appended)

    def test_full_order_omits_engines_that_cannot_do_the_action(self):
        engines = {row["engine"] for row in ie.full_order(self.STORED, "edit")}
        self.assertNotIn("zimage", engines)
        self.assertNotIn("turbo", engines)


class LegacyTags(unittest.TestCase):
    def test_tags_written_before_engine_ids_still_map_back(self):
        self.assertEqual(ie.id_for_tag("krea2"), "krea2")
        self.assertEqual(ie.id_for_tag("krea2-edit"), "krea2")
        self.assertEqual(ie.id_for_tag("seedream"), "seedream")

    def test_the_old_z_image_tag_means_the_atlas_engine(self):
        # the tag was written when z-image only ever meant Atlas
        self.assertEqual(ie.id_for_tag("z-image"), "turbo")

    def test_an_unknown_tag_maps_to_nothing(self):
        self.assertEqual(ie.id_for_tag("dalle"), "")


if __name__ == "__main__":
    unittest.main()
