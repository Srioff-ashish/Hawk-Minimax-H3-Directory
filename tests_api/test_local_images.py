"""Qwen Image 2.1's ComfyUI graphs, and the LoRAs attached before one is built.

Pure Python plus a stub service: python -m unittest discover -s tests_api -p 'test_local_images.py'
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
