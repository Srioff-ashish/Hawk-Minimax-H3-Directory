"""The local engines' ComfyUI graphs, and the LoRAs attached before one is built.

Pure Python plus a stub service: python -m unittest discover -s tests_api -p 'test_local_images.py'
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api import image_engines as ie  # noqa: E402
from hawk_api import local_images as li  # noqa: E402

FILES = {"unet": "qwen_image_2.1_int8_convrot.safetensors",
         "clip": "qwen3vl_8b_bf16.safetensors",
         "vae": "qwen_image_2.1_vae_bf16.safetensors"}


def nodes_of(graph: dict, class_type: str) -> list[dict]:
    return [node for node in graph.values() if node["class_type"] == class_type]


class Qwen21Graph(unittest.TestCase):
    """Text to image."""

    def build(self, **over) -> dict:
        args = dict(width=1024, height=1536, n=2, seed=11, loras=[], steps=30, cfg=2.0,
                    sampler="euler", scheduler="simple", prefix="p", **FILES)
        return li.qwen21_graph("a lamp", **{**args, **over})

    def test_the_text_encoder_loads_as_qwen_image(self):
        # The one field that fails quietly: Klein's "flux2" or Z-Image's "lumina2" here would load and
        # produce nonsense rather than raise, so it is worth pinning.
        loader = nodes_of(self.build(), "CLIPLoader")[0]
        self.assertEqual(loader["inputs"]["type"], "qwen_image",
                         "Qwen Image 2.1 needs the qwen_image encoder type, not flux2 or lumina2")

    def test_the_sampler_reads_the_end_of_the_lora_chain(self):
        graph = self.build(loras=[("a.safetensors", 1.0), ("b.safetensors", 0.5)])
        chain = nodes_of(graph, "LoraLoaderModelOnly")
        self.assertEqual(len(chain), 2, "both LoRAs should be in the chain")
        sampler = nodes_of(graph, "KSampler")[0]
        self.assertEqual(sampler["inputs"]["model"], ["l1", 0],
                         "sampling must run through every LoRA, not off the bare UNET")

    def test_without_loras_the_sampler_reads_the_unet_directly(self):
        self.assertEqual(nodes_of(self.build(), "KSampler")[0]["inputs"]["model"], ["1", 0],
                         "with nothing to chain, the model comes straight off the loader")

    def test_the_cache_node_is_only_added_when_comfyui_has_it(self):
        self.assertEqual(nodes_of(self.build(cache=False), "QwenImage21Cache"), [],
                         "an older ComfyUI must still get a graph it can run")
        cached = self.build(cache=True, loras=[("a.safetensors", 1.0)])
        self.assertEqual(nodes_of(cached, "QwenImage21Cache")[0]["inputs"]["model"], ["1", 0])
        self.assertEqual(cached["l0"]["inputs"]["model"], ["c", 0],
                         "the LoRA chain should hang off the cache, not bypass it")

    def test_the_sampler_settings_are_the_ones_passed(self):
        sampler = nodes_of(self.build(steps=25, cfg=1.0, sampler="er_sde", scheduler="beta"), "KSampler")[0]
        self.assertEqual((sampler["inputs"]["steps"], sampler["inputs"]["cfg"]), (25, 1.0))
        self.assertEqual((sampler["inputs"]["sampler_name"], sampler["inputs"]["scheduler"]), ("er_sde", "beta"),
                         "a LoRA that only behaves on er_sde/beta has to be able to say so")

    def test_the_batch_and_size_reach_the_latent(self):
        latent = nodes_of(self.build(width=896, height=1600, n=3), "EmptyLatentImage")[0]
        self.assertEqual((latent["inputs"]["width"], latent["inputs"]["height"]), (896, 1600))
        self.assertEqual(latent["inputs"]["batch_size"], 3)


class Qwen21EditGraph(unittest.TestCase):
    """Edit, where every reference goes through one TextEncodeQwenImage21 node."""

    def build(self, images, **over) -> dict:
        args = dict(images=images, width=None, height=None, seed=5, loras=[], steps=30, cfg=2.0,
                    sampler="euler", scheduler="simple", prefix="p", **FILES)
        return li.qwen21_edit_graph("put her at the table", **{**args, **over})

    def test_every_reference_gets_a_loader_wired_to_its_own_slot(self):
        graph = self.build([f"ref{i}.png" for i in range(li.QWEN21_MAX_REFS)])
        self.assertEqual(len(nodes_of(graph, "LoadImage")), 16, "all sixteen references should be loaded")
        encode = nodes_of(graph, "TextEncodeQwenImage21")[0]["inputs"]
        for slot in range(1, 17):
            # namespaced by the Autogrow input's own id; a bare "image_1" reaches execute() as a
            # stray keyword instead of being gathered into the node's images argument
            self.assertEqual(encode[f"images.image_{slot}"], [f"i{slot}", 0],
                             f"images.image_{slot} should read the {slot}th reference loader")
            self.assertNotIn(f"image_{slot}", encode,
                             f"image_{slot} without the images. prefix matches no slot on the node")
        self.assertNotIn("images.image_17", encode, "the node has no seventeenth slot")

    def test_more_references_than_the_node_has_slots_is_refused(self):
        with self.assertRaises(li.LocalImageError) as caught:
            self.build([f"ref{i}.png" for i in range(17)])
        self.assertTrue(caught.exception.fatal, "asking for a 17th is a wrong request, not a reason to fall back")
        self.assertIn("16", str(caught.exception), "say what the limit actually is")

    def test_an_edit_with_no_reference_is_refused(self):
        with self.assertRaises(li.LocalImageError):
            self.build([])

    def test_with_no_size_the_latent_comes_from_the_references(self):
        graph = self.build(["a.png"])
        self.assertEqual(nodes_of(graph, "EmptyLatentImage"), [],
                         "the encode node already sized a latent, so nothing should override it")
        self.assertEqual(nodes_of(graph, "KSampler")[0]["inputs"]["latent_image"], ["4", 2],
                         "the sampler should take the encode node's third output")

    def test_an_asked_for_size_replaces_that_latent(self):
        graph = self.build(["a.png"], width=1024, height=1024)
        self.assertEqual(nodes_of(graph, "EmptyLatentImage")[0]["inputs"]["width"], 1024)
        self.assertEqual(nodes_of(graph, "KSampler")[0]["inputs"]["latent_image"], ["6", 0])

    def test_both_conditionings_come_from_the_one_encode_node(self):
        sampler = nodes_of(self.build(["a.png"]), "KSampler")[0]["inputs"]
        self.assertEqual((sampler["positive"], sampler["negative"]), (["4", 0], ["4", 1]),
                         "the node returns both, so no separate negative encode is built")

    def test_the_negative_prompt_rides_on_the_encode_node(self):
        encode = nodes_of(self.build(["a.png"], negative="blurry"), "TextEncodeQwenImage21")[0]["inputs"]
        self.assertEqual(encode["negative_prompt"], "blurry")
        self.assertEqual(encode["resolution"], li.QWEN21_EDIT_RESOLUTION)


CHROMA_FILES = {"unet": "chroma1_hd_fp8_scaled.safetensors",
                "clip": "t5xxl_fp8_e4m3fn.safetensors",
                "vae": "chroma_vae.safetensors"}


class ChromaGraph(unittest.TestCase):
    """Text to image. Chroma is the odd engine out: real guidance, its own sigmas, a live negative."""

    def build(self, **over) -> dict:
        args = dict(width=1152, height=1152, n=1, seed=7, loras=[], steps=li.CHROMA_STEPS,
                    cfg=li.CHROMA_CFG, sampler="euler", prefix="p", **CHROMA_FILES)
        return li.chroma_graph("a lamp", **{**args, **over})

    def test_the_text_encoder_loads_as_chroma(self):
        # The quiet failure the other engines have too: a Qwen or lumina2 type here loads and draws
        # nonsense rather than raising.
        self.assertEqual(nodes_of(self.build(), "CLIPLoader")[0]["inputs"]["type"], "chroma",
                         "Chroma needs the chroma encoder type over its T5, not a Qwen one")

    def test_both_prompts_are_encoded_through_the_tokenizer_options(self):
        graph = self.build()
        encodes = [n for n in graph.values() if n["class_type"] == "CLIPTextEncode"]
        self.assertEqual(len(encodes), 2, "Chroma encodes a positive and a real negative")
        for node in encodes:
            self.assertEqual(node["inputs"]["clip"], ["tok", 0],
                             "the padding options must be in the path, not bypassed by reading the loader")

    def test_a_blank_negative_falls_back_to_chromas_own(self):
        # Every other engine zeroes an empty negative out. At cfg 3.8 that is a visibly worse image,
        # so the blank case has to land on real text instead.
        graph = self.build(negative="   ")
        self.assertEqual(nodes_of(graph, "ConditioningZeroOut"), [],
                         "Chroma never zeroes its negative out")
        self.assertEqual(graph["6"]["inputs"]["text"], li.CHROMA_NEGATIVE,
                         "a blank negative should become the tuned default, not an empty string")

    def test_a_written_negative_replaces_it(self):
        self.assertEqual(self.build(negative="extra fingers")["6"]["inputs"]["text"], "extra fingers",
                         "what the caller wrote wins over the default")

    def test_the_sigmas_come_from_the_beta_scheduler_and_not_from_a_named_schedule(self):
        graph = self.build(steps=26)
        self.assertEqual(nodes_of(graph, "KSampler"), [],
                         "Chroma samples through SamplerCustomAdvanced; a KSampler would lose its alpha/beta")
        beta = nodes_of(graph, "BetaSamplingScheduler")[0]["inputs"]
        self.assertEqual((beta["steps"], beta["alpha"], beta["beta"]),
                         (26, li.CHROMA_BETA_ALPHA, li.CHROMA_BETA_BETA))
        self.assertEqual(nodes_of(graph, "SamplerCustomAdvanced")[0]["inputs"]["sigmas"], ["8", 0],
                         "the sampler must read those sigmas")

    def test_guidance_reaches_the_guider(self):
        graph = self.build(cfg=3.8)
        guider = nodes_of(graph, "CFGGuider")[0]["inputs"]
        self.assertEqual(guider["cfg"], 3.8)
        self.assertEqual((guider["positive"], guider["negative"]), (["5", 0], ["6", 0]),
                         "the negative must reach the guider, or guidance has nothing to push away from")

    def test_the_whole_model_path_runs_through_every_lora(self):
        graph = self.build(loras=[("a.safetensors", 1.0), ("b.safetensors", 0.5)])
        self.assertEqual(len(nodes_of(graph, "LoraLoaderModelOnly")), 2)
        self.assertEqual(graph["4"]["inputs"]["model"], ["l1", 0],
                         "shift must be applied on top of the LoRAs, not on the bare UNET")
        # Both the guider and the scheduler read the shifted model: a scheduler reading the unshifted
        # one would hand out sigmas for a different model than the one being sampled.
        self.assertEqual(nodes_of(graph, "CFGGuider")[0]["inputs"]["model"], ["4", 0])
        self.assertEqual(nodes_of(graph, "BetaSamplingScheduler")[0]["inputs"]["model"], ["4", 0])

    def test_the_seed_batch_and_size_reach_their_nodes(self):
        graph = self.build(seed=99, n=3, width=1024, height=1536)
        self.assertEqual(nodes_of(graph, "RandomNoise")[0]["inputs"]["noise_seed"], 99)
        latent = nodes_of(graph, "EmptySD3LatentImage")[0]["inputs"]
        self.assertEqual((latent["width"], latent["height"], latent["batch_size"]), (1024, 1536, 3))


class TheLocalModelPatterns(unittest.TestCase):
    """One ComfyUI folder holds every engine's files, so each engine's pattern has to pick its own."""

    #: What a pod running all four engines has on disk, plus the video encoder that shares text_encoders.
    LISTING = {
        "diffusion_models": ["krea2_turbo_nvfp4.safetensors", "qwen_image_2.1_int8_convrot.safetensors",
                             "z_image_turbo_nvfp4.safetensors", "chroma1_hd_fp8_scaled.safetensors",
                             "minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "text_encoders": ["qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_8b_bf16.safetensors",
                          "qwen_3_4b_fp4_mixed.safetensors", "umt5_xxl.safetensors",
                          "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "t5xxl_fp8_e4m3fn.safetensors"],
        "vae": ["qwen_image_vae.safetensors", "qwen_image_2.1_vae_bf16.safetensors",
                "z_image_ae.safetensors", "chroma_vae.safetensors",
                "minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
    }
    EXPECTED = {
        "krea2": ("krea2_turbo_nvfp4.safetensors", "qwen3vl_4b_fp8_scaled.safetensors", "qwen_image_vae.safetensors"),
        "qwen21": ("qwen_image_2.1_int8_convrot.safetensors", "qwen3vl_8b_bf16.safetensors",
                   "qwen_image_2.1_vae_bf16.safetensors"),
        "zimage": ("z_image_turbo_nvfp4.safetensors", "qwen_3_4b_fp4_mixed.safetensors", "z_image_ae.safetensors"),
        "chroma": ("chroma1_hd_fp8_scaled.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "chroma_vae.safetensors"),
    }

    def test_each_engine_finds_its_own_three_files_with_nothing_configured(self):
        for engine, expected in self.EXPECTED.items():
            with self.subTest(engine=engine):
                spec = li.LOCAL_MODELS[engine]
                found = tuple(li.pick_model("", self.LISTING[folder], pattern)
                              for folder, pattern in (spec["files"][key] for key in ("unet", "clip", "vae")))
                self.assertEqual(found, expected,
                                 f"{engine} must pick its own files out of the shared folders")

    def test_chroma_does_not_mistake_the_video_encoder_for_its_own(self):
        # umt5_xxl is Wan's encoder and lives in the same folder; a bare "t5xxl" pattern would take it,
        # and a wrong encoder loads and draws nonsense rather than raising.
        pattern = li.LOCAL_MODELS["chroma"]["files"]["clip"][1]
        self.assertIsNone(pattern.search("umt5_xxl.safetensors"),
                          "umt5_xxl belongs to video, not to Chroma")
        self.assertTrue(pattern.search("t5xxl_fp16.safetensors"), "a plain t5xxl is Chroma's")

    def test_the_names_chroma_is_published_under_are_all_recognised(self):
        # Chroma ships from several repackagers, and the VAE is Flux's, so it arrives under the bare
        # "ae.safetensors" as often as under a chroma name. None of these should need a setting.
        spec = li.LOCAL_MODELS["chroma"]["files"]
        for key, name in (("unet", "Chroma1-HD-fp8_scaled_rev2.safetensors"), ("unet", "Chroma1-HD.safetensors"),
                          ("clip", "flan-t5-xxl-fp16.safetensors"),
                          ("clip", "t5xxl_flan_latest_float8_e4m3fn_scaled_stochastic.safetensors"),
                          ("vae", "ae.safetensors"), ("vae", "chroma_vae.safetensors")):
            with self.subTest(name=name):
                self.assertTrue(spec[key][1].search(name), f"{name} should be recognised as Chroma's {key}")

    def test_every_wired_engine_has_a_model_entry_and_a_descriptor(self):
        for engine in li.WIRED_ENGINES:
            with self.subTest(engine=engine):
                self.assertIn(engine, li.LOCAL_MODELS, "a wired engine needs files to look for")
                self.assertTrue(ie.get(engine).local, "a wired engine must be a local one")


class StubSettings:
    krea_adult_default = True

    def __init__(self, data_dir):
        self.data_dir = data_dir


class StubService:
    """Only the parts resolve_loras reaches: a LoRA listing, the settings store and a data directory."""

    def __init__(self, data_dir, files):
        self.settings = StubSettings(data_dir)
        self.image_engines = ie.ImageEngineStore(data_dir)
        self._files = files

    async def available_models(self, folder, refresh=False):
        return list(self._files) if folder == "loras" else []


class BaseLoras(unittest.IsolatedAsyncioTestCase):
    """The always-on repair LoRA, which the adult defaults must not be able to displace."""

    FIX = "qwen-image-2.1-fix-1.0-comfy.safetensors"
    NSFW = "NSFW Qwen Lora.safetensors"

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        shutil.copyfile(li.EXAMPLE_LORAS, os.path.join(self.dir, "image_loras.json"))

    def engine(self, files):
        # the catalogue matches on the bare name, the graph is handed the path it was found at
        return li.LocalImageEngine(StubService(self.dir, [f"qwen21/{name}" for name in files]))

    async def resolve(self, files, requested=None, adult_default=True):
        chosen, used, warnings = await self.engine(files).resolve_loras(
            requested, adult_default=adult_default, family="qwen21")
        return [name.rsplit("/", 1)[-1] for name, _ in chosen], used, warnings

    async def test_it_attaches_with_nothing_asked_for(self):
        names, _, warnings = await self.resolve([self.FIX])
        self.assertEqual(names, [self.FIX], "the repair LoRA should be on every image of this family")
        self.assertEqual(warnings, [])

    async def test_it_survives_a_request_naming_its_own_adult_lora(self):
        # the case the old whole-list gate broke: naming an adult LoRA stood the defaults down entirely,
        # and the repair LoRA went with them
        names, _, _ = await self.resolve([self.FIX, self.NSFW], requested=[{"name": self.NSFW}])
        self.assertIn(self.FIX, names, "an adult LoRA in the request must not switch off the repair LoRA")
        self.assertIn(self.NSFW, names)

    async def test_it_is_not_applied_twice_when_the_request_names_it(self):
        names, _, _ = await self.resolve([self.FIX], requested=[{"name": self.FIX, "strength": 0.6}])
        self.assertEqual(names, [self.FIX], "naming it should set its strength, not stack a second copy")

    async def test_it_attaches_even_with_the_adult_defaults_switched_off(self):
        names, _, _ = await self.resolve([self.FIX, self.NSFW], adult_default=False)
        self.assertEqual(names, [self.FIX], "adult_default governs the adult pair, not the repair LoRA")

    async def test_it_does_not_count_as_one_of_the_engine_loras_the_user_chose(self):
        _, used, _ = await self.resolve([self.FIX, self.NSFW], requested=[{"name": self.NSFW}])
        automatic = {item.file.rsplit("/", 1)[-1] for item in used if item.automatic}
        self.assertEqual(automatic, {self.FIX}, "it arrived on its own, so its sampler hints must not take over")

    async def test_a_missing_one_warns_instead_of_vanishing(self):
        # a name that matches nothing attaches nothing and raises nothing, so this warning is the only
        # place a typo in an always-on LoRA can ever surface
        names, _, warnings = await self.resolve([self.NSFW])
        self.assertNotIn(self.FIX, names)
        self.assertEqual(len(warnings), 1, "a configured but missing always-on LoRA should say so")
        self.assertIn(self.FIX, warnings[0], "name the file, so the user can see which one to fix")

    async def test_it_warns_even_when_nothing_else_is_attached(self):
        _, _, warnings = await self.resolve([], adult_default=False)
        self.assertEqual(len(warnings), 1, "the early return for an empty request must not swallow it")

    async def test_a_shortened_name_does_not_stack_a_second_copy(self):
        # The duplicate guard compared exact stems while resolution accepted a substring, so a name the
        # agent shortened looked new to the guard, collected the automatic copy beside it, and then
        # resolved to that same file: one LoRA twice in the chain, at about double its allowed strength.
        names, _, _ = await self.resolve([self.FIX], requested=[{"name": "qwen-image-2.1-fix"}])
        self.assertEqual(names, [self.FIX],
                         "a shortened name means the same file, so it must not be attached again beside it")

    async def test_separators_do_not_decide_whether_a_lora_is_found(self):
        # The catalogue mixes all three -- "NSFW Qwen Lora.safetensors" beside "lenovo_qwen21" and
        # "qwen-image-2.1-fix-1.0-comfy" -- so an agent writing a name from memory picks whichever it saw.
        # Refusing the underscored spelling reports an installed LoRA as missing.
        for spelling in ("NSFW_Qwen_Lora", "nsfw-qwen-lora", "nsfw qwen lora"):
            with self.subTest(spelling=spelling):
                names, _, _ = await self.resolve([self.FIX, self.NSFW], requested=[{"name": spelling}])
                self.assertIn(self.NSFW, names, f"{spelling!r} names an installed LoRA and must reach it")

    async def test_a_name_missing_a_middle_word_still_reaches_one_file(self):
        # "nsfw lora" is neither the file nor a substring of it, but every word of it is in the name, and
        # exactly one installed LoRA answers to that. A name reaching several is still refused upstream.
        names, _, _ = await self.resolve([self.FIX, self.NSFW], requested=[{"name": "nsfw lora"}])
        self.assertIn(self.NSFW, names, "every word matched one installed file, so it should resolve")

    async def test_a_folder_qualified_name_resolves_to_its_file(self):
        # Results write LoRA names back with the family folder on the front, so the form the agent is most
        # likely to copy out of one job has to be usable in the next instead of matching nothing.
        names, _, _ = await self.resolve([self.FIX], requested=[{"name": f"qwen21/{self.FIX}", "strength": 0.9}])
        self.assertEqual(names, [self.FIX], "a folder-qualified name should resolve, and only once")


class StubComfy:
    """ComfyUI far enough for an edit to be set up. Submitting is an error: these tests are about what
    happens *before* anything renders."""

    def __init__(self, testcase):
        self.testcase = testcase

    async def queue_state(self):
        return [], []

    async def object_info(self, name):
        return {name: {}} if name == li.QWEN21_EDIT_NODE else None

    async def submit(self, graph, prompt_id):
        self.testcase.fail("a refused edit must never reach ComfyUI")


class EditService(StubService):
    def __init__(self, data_dir, loras, assets, testcase):
        super().__init__(data_dir, loras)
        self.comfy = StubComfy(testcase)
        self.store = self
        self._assets = assets
        self._models = {
            "diffusion_models": ["qwen_image_2.1_int8_convrot.safetensors"],
            "text_encoders": ["qwen3vl_8b_bf16.safetensors"],
            "vae": ["qwen_image_2.1_vae_bf16.safetensors"],
            "loras": [f"qwen21/{name}" for name in loras],
        }

    async def available_models(self, folder, refresh=False):
        return list(self._models.get(folder, []))

    def get_asset(self, asset_id):
        return self._assets.get(asset_id)


class EditGuardrails(unittest.IsolatedAsyncioTestCase):
    """check_edit on the Qwen 2.1 edit path. It moved from edit_klein to edit_qwen21 with the engine swap,
    and it is the rule that keeps adult LoRAs and sexual prompts off uploaded photos of possibly real people."""

    FIX = "qwen-image-2.1-fix-1.0-comfy.safetensors"
    NSFW = "NSFW Qwen Lora.safetensors"
    PHOTO = {"id": "up1", "filename": "her.jpg", "path": "her.jpg", "source": {"type": "upload"}}
    MADE = {"id": "gen1", "filename": "made.png", "path": "made.png",
            "source": {"type": "generated", "references": []}}

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        shutil.copyfile(li.EXAMPLE_LORAS, os.path.join(self.dir, "image_loras.json"))
        assets = {a["id"]: a for a in (self.PHOTO, self.MADE)}
        service = EditService(self.dir, [self.FIX, self.NSFW], assets, self)
        self.engine = li.LocalImageEngine(service)

    async def test_an_adult_lora_on_an_uploaded_photo_is_refused(self):
        with self.assertRaises(li.LocalImageError) as caught:
            await self.engine.edit_qwen21("put her on a balcony", [self.PHOTO], loras=[{"name": self.NSFW}])
        self.assertTrue(caught.exception.fatal, "a refusal must not fall through to another engine")
        self.assertIn("uploaded photos", str(caught.exception))

    async def test_a_sexual_edit_of_an_uploaded_photo_is_refused(self):
        with self.assertRaises(li.LocalImageError) as caught:
            await self.engine.edit_qwen21("remove her clothes", [self.PHOTO])
        self.assertTrue(caught.exception.fatal)
        self.assertIn("real people", str(caught.exception))

    async def test_an_image_made_from_an_uploaded_photo_is_still_a_photo(self):
        # the anti-laundering rule: generating from an upload and editing that does not clear it
        derived = {"id": "gen2", "filename": "d.png", "path": "d.png",
                   "source": {"type": "generated", "references": ["up1"]}}
        self.engine.service._assets["gen2"] = derived
        with self.assertRaises(li.LocalImageError):
            await self.engine.edit_qwen21("nude portrait", [derived])

    async def test_a_prompt_naming_a_minor_is_refused_before_anything_else(self):
        for prompt in ("a schoolgirl on a balcony", "make her look 15 years old", "a teen at the window"):
            with self.subTest(prompt=prompt), self.assertRaises(li.LocalImageError) as caught:
                await self.engine.edit_qwen21(prompt, [self.MADE])
            self.assertIn("under 18", str(caught.exception))

    def test_the_minor_check_reads_words_and_stated_ages_only(self):
        # Pinning the boundary rather than asserting a capability it does not have: _MINOR matches keywords
        # and an age written with its unit, so "15 years old" is caught and a bare "look 15" is not. It reads
        # the prompt and cannot see the output, so it is a filter on what is asked for, not a guarantee.
        li.check_prompt("a woman who looks 15")  # passes today
        with self.assertRaises(li.LocalImageError):
            li.check_prompt("a woman who looks 15 years old")

    async def test_the_adult_defaults_are_not_attached_to_an_uploaded_photo(self):
        chosen, used, _ = await self.engine.resolve_loras(
            None, adult_default=self.engine.adult_default and not li.from_upload(self.PHOTO), family="qwen21")
        self.assertNotIn(self.NSFW, [n.rsplit("/", 1)[-1] for n in dict(chosen)],
                         "an upload must not pick up the family's adult LoRA on its own")
        self.assertEqual([i.kind for i in used], ["detail"], "only the repair LoRA, which is not adult")


class WhichEngineCanLoadTheseLoras(unittest.IsolatedAsyncioTestCase):
    """Naming a LoRA names the engine, so the ladder has to be able to ask which engines qualify."""

    FILES = ["qwen21/NSFW Qwen Lora.safetensors", "qwen21/qwen-image-2.1-fix-1.0-comfy.safetensors",
             "krea2/snofs_krea2.safetensors", "krea2/krea2_mystic_xxx_v3.safetensors"]

    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        shutil.copyfile(li.EXAMPLE_LORAS, os.path.join(self.dir, "image_loras.json"))
        self.images = li.LocalImageEngine(StubService(self.dir, self.FILES))

    async def test_a_name_narrows_to_the_family_that_holds_it(self):
        self.assertEqual(await self.images.families_for_loras(["NSFW Qwen Lora"]), {"qwen21"})
        self.assertEqual(await self.images.families_for_loras(["snofs"]), {"krea2"})

    async def test_an_engine_with_nothing_installed_never_qualifies(self):
        # Chroma leads the generate ladder and has no catalogue on this pod. Without this the walk would
        # hand it every LoRA request and every one of them would be refused outright.
        self.assertNotIn("chroma", await self.images.families_for_loras(["snofs"]))

    async def test_names_from_two_families_leave_no_engine_able(self):
        # Nothing can load both, so the ladder is left alone and the request fails with its own reason.
        self.assertEqual(await self.images.families_for_loras(["snofs", "NSFW Qwen Lora"]), set())

    async def test_a_name_nothing_installed_matches_narrows_to_nothing(self):
        self.assertEqual(await self.images.families_for_loras(["no such lora"]), set())

    async def test_naming_none_leaves_every_family_in_play(self):
        self.assertEqual(await self.images.families_for_loras([]), set(ie.IMAGE_FAMILIES))
        self.assertEqual(await self.images.families_for_loras(["", "  ", None]), set(ie.IMAGE_FAMILIES))


class CorrectingWhereAFileCameFrom(unittest.IsolatedAsyncioTestCase):
    """Re-pointing a re-uploaded copy at the image it was copied from.

    A client that could not reach a generated image on disk downloaded it and uploaded it back. The copy
    arrived with no history, so the upload rules -- there because an uploaded photo may show a real person
    -- applied to a picture this pod drew from a prompt, and to everything later made from it.
    """

    def setUp(self):
        from hawk_api.jobs import HawkService

        self.assets = {
            "root": {"id": "root", "kind": "image", "source": {"type": "generated", "engine": "qwen21"}},
            "made": {"id": "made", "kind": "image",
                     "source": {"type": "generated", "engine": "krea2", "references": ["root"]}},
            "copy": {"id": "copy", "kind": "image", "filename": "lib_made_re.png", "source": {"type": "upload"}},
            "photo": {"id": "photo", "kind": "image", "source": {"type": "upload"}},
            "of_photo": {"id": "of_photo", "kind": "image",
                         "source": {"type": "generated", "engine": "krea2", "references": ["photo"]}},
        }
        service = type("Service", (), {
            "update_asset": HawkService.update_asset,
            "_correct_provenance": HawkService._correct_provenance,
            "_descends_from": HawkService._descends_from,
        })()
        service.store = type("Store", (), {
            "get_asset": staticmethod(lambda i: self.assets.get(i)),
            "add_asset": staticmethod(lambda a: self.assets.__setitem__(a["id"], a)),
        })()
        self.service = service

    def lookup(self, asset_id):
        return self.assets.get(asset_id)

    def test_a_corrected_copy_stops_counting_as_an_upload(self):
        self.assertTrue(li.from_upload(self.assets["copy"], self.lookup), "it arrives as an upload")
        self.service.update_asset("copy", generated_from="made")
        self.assertFalse(li.from_upload(self.assets["copy"], self.lookup),
                         "once it points at the image it was copied from, the chain is clean")

    def test_the_old_source_is_kept_and_the_change_can_be_undone(self):
        # The override of a content guardrail has to stay visible on the asset, not quietly rewrite history.
        self.service.update_asset("copy", generated_from="made")
        self.assertEqual(self.assets["copy"]["source"]["corrected_from"], {"type": "upload"})
        self.assertIn("corrected_at", self.assets["copy"]["source"])
        self.service.update_asset("copy", generated_from="")
        self.assertEqual(self.assets["copy"]["source"], {"type": "upload"}, "undone, exactly as it was")
        self.assertTrue(li.from_upload(self.assets["copy"], self.lookup))

    def test_an_origin_that_is_itself_an_upload_is_refused(self):
        from hawk_api.jobs import RequestError

        # Otherwise the correction launders what it exists to undo: naming an upload, or anything made from
        # one, as the origin only puts the photograph one more step away from the check.
        for origin in ("photo", "of_photo"):
            with self.subTest(origin=origin), self.assertRaises(RequestError):
                self.service.update_asset("copy", generated_from=origin)
        self.assertTrue(li.from_upload(self.assets["copy"], self.lookup), "and the copy is left alone")

    def test_an_origin_made_from_the_asset_itself_is_refused(self):
        from hawk_api.jobs import RequestError

        # Corrected first, so the chain through "later" is clean and the upload check has nothing to say:
        # what has to refuse this is the cycle guard, which is the only thing standing between a typo and
        # an asset that is its own ancestor.
        self.service.update_asset("copy", generated_from="made")
        self.assets["later"] = {"id": "later", "kind": "image",
                                "source": {"type": "generated", "references": ["copy"]}}
        self.assertFalse(li.from_upload(self.assets["later"], self.lookup))
        with self.assertRaises(RequestError) as caught:
            self.service.update_asset("copy", generated_from="later")
        self.assertIn("cannot also be its origin", str(caught.exception))
        self.assertEqual(self.assets["copy"]["source"]["references"], ["made"], "and the copy is untouched")

    def test_an_unknown_origin_and_an_asset_pointing_at_itself_are_refused(self):
        from hawk_api.jobs import RequestError

        for origin in ("nope", "copy"):
            with self.subTest(origin=origin), self.assertRaises(RequestError):
                self.service.update_asset("copy", generated_from=origin)

    def test_undoing_a_correction_that_was_never_made_is_refused(self):
        from hawk_api.jobs import RequestError

        with self.assertRaises(RequestError):
            self.service.update_asset("made", generated_from="")


class MirroredConstants(unittest.TestCase):
    def test_the_base_loras_table_is_the_same_on_both_sides(self):
        # local_images mirrors image_engines because image_engines cannot import it back
        self.assertEqual(li.BASE_LORAS, ie.BASE_LORAS)

    def test_every_default_lora_belongs_to_the_family_that_lists_it(self):
        for table in (ie.BASE_LORAS, ie.DEFAULT_ADULT_LORAS):
            for family, entries in table.items():
                self.assertIn(family, ie.IMAGE_FAMILIES, f"{family} is not an image family")
                for entry in entries:
                    name = entry[0] if isinstance(entry, tuple) else entry
                    self.assertTrue(name.endswith(".safetensors"), name)

    def test_the_shipped_catalogue_covers_every_named_default(self):
        with open(li.EXAMPLE_LORAS, encoding="utf-8") as handle:
            catalogued = {e["file"] for e in json.load(handle)["loras"]}
        for table in (ie.BASE_LORAS, ie.DEFAULT_ADULT_LORAS):
            for family, entries in table.items():
                for entry in entries:
                    name = entry[0] if isinstance(entry, tuple) else entry
                    self.assertIn(name, catalogued,
                                  f"{family} attaches {name} on its own, so it must be in the catalogue "
                                  "with a kind and a strength")


try:
    from hawk_api.schemas import ImageRequest
except ImportError as exc:  # pragma: no cover
    ImageRequest = None
    _SCHEMA_WHY = str(exc)


@unittest.skipIf(ImageRequest is None, "schema test dependencies missing")
class TheRequestSchemaAndTheEngines(unittest.TestCase):
    """The request schema is checked before any engine is chosen, so a cap lower than an engine's
    own reach refuses references the engine would have accepted, and no engine code ever runs."""

    def test_the_reference_cap_reaches_the_widest_engine(self):
        widest = max(e.max_refs for e in ie.ENGINES.values())
        cap = next(m.max_length for m in ImageRequest.model_fields["reference_asset_ids"].metadata
                   if getattr(m, "max_length", None) is not None)
        self.assertGreaterEqual(cap, widest,
                                f"the schema stops at {cap} references but {widest} are usable, so the widest "
                                "engine can never be given a full set")

    def test_a_full_set_of_references_is_accepted(self):
        widest = max(e.max_refs for e in ie.ENGINES.values())
        request = ImageRequest(prompt="put her at the table", reference_asset_ids=[f"a{i}" for i in range(widest)])
        self.assertEqual(len(request.reference_asset_ids), widest,
                         "every reference the widest engine takes should survive validation")


class ThePromptKeptOnAnAsset(unittest.TestCase):
    """What the library shows for a picture, and the only thing that makes one reproducible by hand."""

    def test_a_full_length_qwen_prompt_survives_being_recorded(self):
        from hawk_api.jobs import PROMPT_RECORD_LIMIT

        # The agent is told to write Qwen a paragraph of four to five hundred words. At an average of
        # six characters a word with its spaces, that is the length the record has to hold.
        longest_asked_for = 500 * 6
        self.assertGreaterEqual(
            PROMPT_RECORD_LIMIT, longest_asked_for,
            f"prompts are written up to about {longest_asked_for} characters but only "
            f"{PROMPT_RECORD_LIMIT} are kept, so the library shows a sentence or two and the rest is lost")


class AReferenceWhoseFileIsGone(unittest.IsolatedAsyncioTestCase):
    """What happens to an edit when a restarted runtime kept the database and lost the pixels.

    The methods are exercised straight off HawkService rather than through a built service: all three
    reach only settings, the input folder and the Drive exporter, and standing those up is the whole
    point of the test.
    """

    def setUp(self):
        from hawk_api.jobs import HawkService

        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.input_dir = os.path.join(self.root, "comfy_input")
        self.drive = os.path.join(self.root, "drive")
        os.makedirs(self.input_dir)

        service = type("Service", (), {
            "ensure_asset_on_disk": HawkService.ensure_asset_on_disk,
            "local_asset_path": HawkService.local_asset_path,
            "drive_asset_path": HawkService.drive_asset_path,
        })()
        service.settings = type("S", (), {"comfy_input_dir": self.input_dir})()
        browser = type("B", (), {"available": True, "root": self.drive})()
        service.drive_exporter = type("E", (), {
            "browser": browser, "settings": staticmethod(lambda: {"image_folder": "Images"}),
        })()
        self.service = service
        self.asset = {"id": "64d37a92a0e1", "kind": "image", "filename": "shot.png",
                      "path": "hawk_api/64d37a92a0e1/shot.png", "created_at": time.time()}

    def export(self, name="gen_a_shot_64d37a92.png"):
        """One exported copy, named the way export_assets names them: <slug>_<asset_id[:8]>.<ext>."""
        day = time.strftime("%Y-%m-%d", time.localtime(self.asset["created_at"]))
        folder = os.path.join(self.drive, "Images", day)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, name), "wb") as handle:
            handle.write(b"exported pixels")

    def on_disk(self):
        return os.path.join(self.input_dir, self.asset["path"])

    async def test_the_export_is_copied_back_under_the_path_the_asset_records(self):
        # ComfyUI loads a reference by that stored path, so restoring it anywhere else would not help.
        self.export()
        self.assertTrue(await self.service.ensure_asset_on_disk(self.asset),
                        "an export exists, so the file should have been restored")
        self.assertEqual(open(self.on_disk(), "rb").read(), b"exported pixels",
                         "the reference should be readable at exactly the path the asset names")

    async def test_a_file_already_there_is_left_alone(self):
        os.makedirs(os.path.dirname(self.on_disk()))
        with open(self.on_disk(), "wb") as handle:
            handle.write(b"original pixels")
        self.export()
        self.assertTrue(await self.service.ensure_asset_on_disk(self.asset))
        self.assertEqual(open(self.on_disk(), "rb").read(), b"original pixels",
                         "the file on disk is the original; an export must never overwrite it")

    async def test_no_export_to_restore_from_is_reported_rather_than_guessed_at(self):
        # The honest answer when the database outlived the pixels: the edit cannot run, and the caller
        # says so by name instead of letting ComfyUI fail inside its loader.
        self.assertFalse(await self.service.ensure_asset_on_disk(self.asset),
                         "nothing was exported, so this must report the file as unrecoverable")
        self.assertFalse(os.path.exists(self.on_disk()))

    async def test_a_render_reference_is_restored_too_and_not_only_an_edit(self):
        # A video render loads its references through the same input folder and lost them the same way,
        # but reached the graph by a different route -- ComfyUI answered "Invalid image file", which says
        # nothing about why. _refs restores first and names the file when it cannot.
        from hawk_api.jobs import HawkService, RequestError

        self.service.store = type("Store", (), {"get_asset": staticmethod(lambda _id: self.asset)})()
        self.service._refs = HawkService._refs.__get__(self.service)
        self.service._lost_files_message = HawkService._lost_files_message
        reference = type("Ref", (), {"asset_id": self.asset["id"], "role": "picture",
                                     "label": "her", "for_video": True})()

        with self.assertRaises(RequestError) as caught:
            await self.service._refs([reference])
        self.assertIn("shot.png", str(caught.exception), "say which file, not just that something is wrong")
        self.assertIn("the pixels did not", str(caught.exception))

        self.export()
        refs = await self.service._refs([reference])
        self.assertEqual(open(self.on_disk(), "rb").read(), b"exported pixels",
                         "with an export to hand, the render should just work")
        self.assertEqual(refs[0].path, self.asset["path"], "and still load by the recorded path")

    async def test_an_export_from_the_day_either_side_still_counts(self):
        # A restore can land either side of midnight from the export that made the file.
        self.asset["created_at"] = time.time() - 86400
        self.export()
        self.asset["created_at"] = time.time()
        self.assertTrue(await self.service.ensure_asset_on_disk(self.asset),
                        "the dated folder is a day out, which a restore near midnight makes ordinary")


if __name__ == "__main__":
    unittest.main()
