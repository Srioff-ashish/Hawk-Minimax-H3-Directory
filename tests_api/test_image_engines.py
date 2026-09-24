"""The image engine registry: capabilities, aliases, LoRA families and ladder order."""

import json
import os
import shutil
import sys
import tempfile
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
            "qwen-image-2.1-fix-1.0-comfy.safetensors": "qwen21",
            "PornMaster_QI2.1_Age_Slider_V1.safetensors": "qwen21",
            "elusarcas-qwen2-1-detailer-v1.safetensors": "qwen21",
            "lenovo_qwen21.safetensors": "qwen21",
        }
        for filename, family in cases.items():
            self.assertEqual(ie.family_of(filename), family, filename)

    def test_a_qwen_lora_whose_own_name_says_nothing_is_classified_by_its_folder(self):
        # Three of the shipped Qwen 2.1 LoRAs are published with spaces or CJK in the name and no usable
        # prefix. Downloading them into models/loras/qwen21/ is what classifies them, so they never have to
        # be renamed -- and a rename is exactly the step a user would skip.
        for filename in ("NSFW Qwen Lora.safetensors", "qwen2.1\u89d2\u8272\u5361-4.safetensors",
                         "Qwen Image2.1_Anime2Real.safetensors"):
            self.assertEqual(ie.family_of(f"qwen21/{filename}"), "qwen21",
                             f"{filename} in a qwen21 folder should be a Qwen Image 2.1 LoRA")
        self.assertEqual(ie.family_of("NSFW Qwen Lora.safetensors"), "h3",
                         "out of that folder its name alone says nothing, so it falls back to video")

    def test_a_declared_family_wins_over_the_filename(self):
        self.assertEqual(ie.family_of("oddly_named.safetensors", declared="qwen21"), "qwen21")
        self.assertEqual(ie.family_of("krea2_thing.safetensors", declared="zit"), "zit")
        # a declared family that is not real is ignored rather than trusted
        self.assertEqual(ie.family_of("lenovo_qwen21.safetensors", declared="nonsense"), "qwen21")

    def test_an_unknown_file_is_treated_as_video(self):
        # video is the side that refuses unknown names, so it is the safe default
        self.assertEqual(ie.family_of("something_new.safetensors"), "h3")

    def test_a_subfolder_path_is_classified_by_its_basename(self):
        self.assertEqual(ie.family_of("Krea2/lenovo_qwen21.safetensors"), "qwen21")

    def test_every_image_family_has_a_label(self):
        for family in ie.IMAGE_FAMILIES:
            self.assertIn(family, ie.LORA_FAMILIES)
            self.assertTrue(ie.family_label(family))


class Order(unittest.TestCase):
    STORED = [
        {"engine": "qwen21", "enabled": True},
        {"engine": "krea2", "enabled": True},
        {"engine": "zimage", "enabled": True},
        {"engine": "seedream", "enabled": True},
        {"engine": "turbo", "enabled": False},
    ]

    def test_generate_keeps_the_stored_order(self):
        self.assertEqual(ie.order(self.STORED, "generate"), ["qwen21", "krea2", "zimage", "seedream"])

    def test_a_generate_only_engine_is_left_out_of_the_edit_ladder(self):
        # zimage cannot edit, and turbo is both disabled and generate-only
        self.assertEqual(ie.order(self.STORED, "edit"), ["qwen21", "krea2", "seedream"])

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
        self.assertIn("qwen21", appended)

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


    def test_a_model_name_maps_back_when_an_asset_predates_engine_ids(self):
        self.assertEqual(ie.id_for_generator("krea2/turbo"), "krea2")
        self.assertEqual(ie.id_for_generator("krea2_edit"), "krea2")
        # z-image meant the Atlas engine when these assets were written
        self.assertEqual(ie.id_for_generator("z-image/turbo"), "turbo")
        self.assertEqual(ie.id_for_generator("bytedance/seedream-v5.0-pro/edit"), "seedream")

    def test_no_generator_means_no_engine(self):
        self.assertEqual(ie.id_for_generator(""), "")
        self.assertEqual(ie.id_for_generator("   "), "")


class Store(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.store = ie.ImageEngineStore(self.dir)

    def test_an_unconfigured_pod_puts_the_local_engines_first(self):
        self.assertEqual(self.store.order("generate"), ["qwen21", "krea2", "zimage", "turbo", "seedream"])
        self.assertEqual(self.store.order("edit"), ["qwen21", "krea2", "seedream"], "zimage and turbo cannot edit")
        order = self.store.order("generate")
        self.assertLess(order.index("turbo"), order.index("seedream"), "the cheaper paid engine first")
        self.assertEqual(self.store.wait_seconds(), 0.0, "falling through is still the default")

    def test_saving_an_order_survives_a_reload(self):
        self.store.save(generate=[{"engine": "qwen21"}, {"engine": "krea2"}, {"engine": "zimage"}, {"engine": "seedream"}])
        self.assertEqual(ie.ImageEngineStore(self.dir).order("generate"), ["qwen21", "krea2", "zimage", "seedream"])

    def test_an_engine_left_out_of_a_saved_order_comes_back_switched_off(self):
        view = self.store.save(generate=[{"engine": "krea2"}, {"engine": "seedream"}])
        rows = {row["engine"]: row["enabled"] for row in view["generate"]}
        self.assertEqual(rows["turbo"], False)
        self.assertNotIn("turbo", self.store.order("generate"))

    def test_a_disabled_engine_keeps_its_place(self):
        self.store.save(generate=[{"engine": "qwen21"}, {"engine": "krea2", "enabled": False}, {"engine": "seedream"}])
        order = [row["engine"] for row in self.store.settings()["generate"]]
        self.assertEqual(order[:3], ["qwen21", "krea2", "seedream"], "off, but still second")
        self.assertEqual(self.store.order("generate"), ["qwen21", "seedream"])

    def test_only_the_parts_given_are_changed(self):
        self.store.save(generate=[{"engine": "qwen21"}, {"engine": "seedream"}])
        self.store.save(busy_mode="wait")
        self.assertEqual(self.store.order("generate"), ["qwen21", "seedream"], "the order was not touched")
        self.assertEqual(self.store.wait_seconds(), 120.0)

    def test_waiting_zero_seconds_is_falling_through(self):
        view = self.store.save(busy_mode="wait", busy_max_wait_seconds=0)
        self.assertEqual(view["busy"]["mode"], "fall_through")

    def test_a_wait_longer_than_a_render_is_capped(self):
        view = self.store.save(busy_mode="wait", busy_max_wait_seconds=99999)
        self.assertEqual(view["busy"]["max_wait_seconds"], ie.MAX_WAIT_SECONDS)

    def test_an_unknown_engine_is_refused(self):
        with self.assertRaises(ie.SettingsError) as caught:
            self.store.save(generate=[{"engine": "midjourney"}])
        self.assertIn("midjourney", str(caught.exception))

    def test_listing_an_engine_twice_is_refused(self):
        with self.assertRaises(ie.SettingsError):
            self.store.save(generate=[{"engine": "krea2"}, {"engine": "local"}])

    def test_a_generate_only_engine_in_the_edit_order_is_named_not_dropped(self):
        with self.assertRaises(ie.SettingsError) as caught:
            self.store.save(edit=[{"engine": "krea2"}, {"engine": "zimage"}])
        self.assertIn("Z-Image Turbo", str(caught.exception), "say which one, do not silently drop it")

    def test_switching_everything_off_is_refused(self):
        with self.assertRaises(ie.SettingsError):
            self.store.save(generate=[{"engine": "krea2", "enabled": False}])

    def test_a_refused_save_changes_nothing_on_disk(self):
        self.store.save(generate=[{"engine": "qwen21"}, {"engine": "seedream"}])
        with self.assertRaises(ie.SettingsError):
            self.store.save(generate=[{"engine": "nope"}])
        self.assertEqual(self.store.order("generate"), ["qwen21", "seedream"])

    def test_all_paid_warns_and_all_local_warns(self):
        view = self.store.save(generate=[{"engine": "seedream"}, {"engine": "turbo"}])
        self.assertTrue(any("billed to Atlas" in w for w in view["warnings"]))
        view = self.store.save(generate=[{"engine": "krea2"}, {"engine": "qwen21"}])
        self.assertTrue(any("ComfyUI is busy" in w for w in view["warnings"]))

    def test_a_mixed_ladder_warns_about_nothing(self):
        view = self.store.save(generate=[{"engine": "krea2"}, {"engine": "seedream"}],
                               edit=[{"engine": "krea2"}, {"engine": "seedream"}])
        self.assertEqual(view["warnings"], [])

    def test_a_corrupt_file_falls_back_to_the_defaults(self):
        with open(os.path.join(self.dir, "image_engines.json"), "w") as handle:
            handle.write("{ not json")
        self.assertEqual(self.store.order("generate"), ["qwen21", "krea2", "zimage", "turbo", "seedream"])

    def test_a_family_with_no_stored_defaults_falls_back_to_the_shipped_pair(self):
        self.assertIsNone(self.store.defaults("qwen21"), "never set is not the same as empty")
        self.assertEqual([r["name"] for r in self.store.view()["defaults"]["qwen21"]],
                         list(ie.DEFAULT_ADULT_LORAS["qwen21"]))

    def test_stored_defaults_replace_the_shipped_ones(self):
        view = self.store.save(defaults={"zit": [{"name": "zit_betternudes.safetensors", "strength": 0.6}]})
        self.assertEqual(view["defaults"]["zit"], [{"name": "zit_betternudes.safetensors", "strength": 0.6}])
        self.assertEqual(self.store.defaults("zit"), [{"name": "zit_betternudes.safetensors", "strength": 0.6}])

    def test_an_empty_list_switches_a_family_off(self):
        self.store.save(defaults={"krea2": []})
        self.assertEqual(self.store.defaults("krea2"), [], "off, not 'use the shipped pair'")

    def test_a_lora_from_another_family_is_refused(self):
        with self.assertRaises(ie.SettingsError) as caught:
            self.store.save(defaults={"qwen21": [{"name": "zit_mystic_xxx.safetensors"}]})
        self.assertIn("Z-Image Turbo", str(caught.exception), "it should name the family the file really belongs to")

    def test_an_unknown_family_is_refused(self):
        with self.assertRaises(ie.SettingsError):
            self.store.save(defaults={"h3": [{"name": "H3_Motion_BoosterV2.safetensors"}]})

    def test_strength_is_bounded(self):
        with self.assertRaises(ie.SettingsError):
            self.store.save(defaults={"zit": [{"name": "zit_mystic_xxx.safetensors", "strength": 9}]})

    def test_saving_defaults_leaves_the_ladders_alone(self):
        self.store.save(generate=[{"engine": "qwen21"}, {"engine": "seedream"}])
        self.store.save(defaults={"qwen21": [{"name": "NSFW Qwen Lora.safetensors"}]})
        self.assertEqual(self.store.order("generate"), ["qwen21", "seedream"])


    def test_the_file_is_written_atomically(self):
        self.store.save(busy_mode="wait")
        with open(os.path.join(self.dir, "image_engines.json")) as handle:
            self.assertIn("busy", json.load(handle))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "image_engines.json.tmp")))


if __name__ == "__main__":
    unittest.main()


class FamilyFolders(unittest.TestCase):
    """A folder named after a family classifies the files in it, so downloads keep their published names."""

    def test_a_named_folder_classifies_an_unprefixed_file(self):
        self.assertEqual(ie.family_of("qwen21/UltraReal_QI21_V4.safetensors"), "qwen21")
        self.assertEqual(ie.family_of("zit/Hands_v2.1.safetensors"), "zit")
        self.assertEqual(ie.family_of("Krea2/Identity_Edit.safetensors"), "krea2")
        self.assertEqual(ie.family_of("h3/Bouncing_REF2VA.safetensors"), "h3")

    def test_an_alias_folder_counts_too(self):
        self.assertEqual(ie.family_of("z-image/whatever.safetensors"), "zit")
        self.assertEqual(ie.family_of("qwen-image-2.1/whatever.safetensors"), "qwen21")

    def test_the_file_name_still_beats_the_folder(self):
        # a misfiled Qwen LoRA is still a Qwen LoRA; loading it into Krea 2 would just make a worse image
        self.assertEqual(ie.family_of("Krea2/lenovo_qwen21.safetensors"), "qwen21")

    def test_an_unknown_folder_and_name_is_still_video(self):
        self.assertEqual(ie.family_of("misc/Unknown_Thing.safetensors"), "h3")


class FolderDiscovery(unittest.TestCase):
    """The catalogue must classify a discovered file by where it sits, not only by its name."""

    def test_a_file_in_a_family_folder_is_that_family(self):
        # the case folders exist for: a Civitai download that kept its published name
        self.assertEqual(ie.family_of("qwen21/UltraReal_QI21_V4.safetensors"), "qwen21")
        self.assertNotEqual(ie.family_of("qwen21/UltraReal_QI21_V4.safetensors"), "h3")

    def test_the_same_file_loose_is_still_video(self):
        self.assertEqual(ie.family_of("UltraReal_QI21_V4.safetensors"), "h3")


class DefaultNameForms(unittest.TestCase):
    """A stored default may be written with or without .safetensors, or with its folder."""

    def test_the_stem_is_what_matches(self):
        from hawk_api.local_images import _same_lora
        for stored in ("lenovo_qwen21", "lenovo_qwen21.safetensors", "qwen21/lenovo_qwen21.safetensors"):
            self.assertTrue(_same_lora("qwen21/lenovo_qwen21.safetensors", stored), stored)
            self.assertTrue(_same_lora("lenovo_qwen21.safetensors", stored), stored)

    def test_a_different_lora_still_does_not_match(self):
        from hawk_api.local_images import _same_lora
        self.assertFalse(_same_lora("lenovo_qwen21.safetensors", "lenovo_krea2"))
        self.assertFalse(_same_lora("lenovo_qwen21.safetensors", ""))
