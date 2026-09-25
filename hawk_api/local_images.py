"""Local image generation and editing on the pod's own ComfyUI.

Three engines, each with its own graph and its own LoRA family, because a file or an encoder
type from one produces nonsense in another rather than an error:

* **Qwen Image 2.1** -- the lead engine. 30 steps at cfg 2.0 on euler / simple, encoder type
  "qwen_image". Edits take up to sixteen references through one TextEncodeQwenImage21 node.
* **Krea 2 Turbo** -- 8 steps at cfg 1, encoder type "krea2". Edits through Krea 2 Identity
  Edit, which needs the comfyui-krea2edit node pack and takes at most two references.
* **Z-Image Turbo** -- 8 steps, encoder type "lumina2", through ModelSamplingAuraFlow. No edits.

All three share the GPU and the queue with video renders, so ``busy()`` tells callers to use a
cloud engine instead of waiting behind a long render.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field

from . import image_engines
from .comfy_client import ComfyError

EXAMPLE_LORAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "image_loras.example.json")
DEFAULT_STEPS = 8
WAIT_SECONDS = 300.0  # first use loads ~18 GB of weights
POLL_SECONDS = 1.0
LORA_KINDS = ("realism", "detail", "style", "adult", "other")
#: Local engines whose ComfyUI graphs exist.
WIRED_ENGINES = ("krea2", "qwen21", "zimage")
# Any precision of the three Krea 2 files works (fp8_scaled, bf16, fp16…): the configured name wins,
# else the first match, higher precision first.
MODEL_FAMILIES = {
    "diffusion_models": re.compile(r"krea[-_ ]?2.*turbo", re.IGNORECASE),
    "text_encoders": re.compile(r"qwen[-_ ]?3[-_ ]?vl[-_ ]?4b", re.IGNORECASE),
    "vae": re.compile(r"qwen[-_ ]?image[-_ ]?vae", re.IGNORECASE),
}
#: Per local engine: which file to look for in each ComfyUI folder, and which setting names it.
#: The Qwen text encoders are deliberately distinct -- Krea 2 wants qwen3vl_4b, Z-Image qwen_3_4b and
#: Qwen Image 2.1 qwen3vl_8b -- so the patterns must not match each other. Klein's old encoder pattern was
#: a bare qwen_3_8b, which also matched qwen3vl_8b; requiring "vl" and "8b" together is what keeps
#: Qwen 2.1's unambiguous now that it owns the 8b slot.
LOCAL_MODELS: dict[str, dict] = {
    "krea2": {
        "files": {"unet": ("diffusion_models", MODEL_FAMILIES["diffusion_models"]),
                  "clip": ("text_encoders", MODEL_FAMILIES["text_encoders"]),
                  "vae": ("vae", MODEL_FAMILIES["vae"])},
        "configured": {"unet": "krea_unet", "clip": "krea_clip", "vae": "krea_vae"},
    },
    "qwen21": {
        "files": {"unet": ("diffusion_models", re.compile(r"qwen[-_ ]?image[-_ ]?2\.?1", re.IGNORECASE)),
                  "clip": ("text_encoders", re.compile(r"qwen[-_ ]?3[-_ ]?vl[-_ ]?8b", re.IGNORECASE)),
                  "vae": ("vae", re.compile(r"qwen[-_ ]?image[-_ ]?2\.?1[-_ ]?vae", re.IGNORECASE))},
        "configured": {"unet": "qwen21_unet", "clip": "qwen21_clip", "vae": "qwen21_vae"},
    },
    "zimage": {
        "files": {"unet": ("diffusion_models", re.compile(r"z[-_ ]?image(?!.*ae\.safetensors$)", re.IGNORECASE)),
                  "clip": ("text_encoders", re.compile(r"qwen[-_ ]?3[-_ ]?4b", re.IGNORECASE)),
                  "vae": ("vae", re.compile(r"z[-_ ]?image.*ae|^ae\.safetensors$", re.IGNORECASE))},
        "configured": {"unet": "zimage_unet", "clip": "zimage_clip", "vae": "zimage_vae"},
    },
}
#: Qwen Image 2.1's own defaults, from the reference ComfyUI workflows. A LoRA in the catalogue may override
#: any of them -- the NSFW one wants 25 steps at cfg 1 on er_sde/beta, which is why ImageLora carries a cfg.
QWEN21_STEPS, QWEN21_CFG = 30, 2.0
QWEN21_SAMPLER, QWEN21_SCHEDULER = "euler", "simple"
QWEN21_EDIT_RESOLUTION = 1024
QWEN21_MAX_REFS = 16  # TextEncodeQwenImage21 has image_1 .. image_16 and no more
QWEN21_EDIT_NODE = "TextEncodeQwenImage21"  # core, ComfyUI 0.36.0+; edits cannot be built without it
QWEN21_CACHE_NODE = "QwenImage21Cache"  # core, 0.36.0+; a speed-up, so its absence is not an error
ZIMAGE_STEPS = 8
_PRECISION = ("bf16", "fp16", "fp8", "")

# Krea 2 Identity Edit (conradlocke/krea2-identity-edit): a LoRA plus the comfyui-krea2edit node pack,
# which feeds the reference image in as context tokens and grounds the text encoder on it.
EDIT_NODES = ("Krea2EditModelPatch", "Krea2EditGroundedEncode")
EDIT_LORA = re.compile(r"krea[-_ ]?2[-_ ]?identity[-_ ]?edit", re.IGNORECASE)
EDIT_STEPS = 10  # 8 favours the instruction, 12 face detail
EDIT_REF_BOOST = 4.0  # likeness dial; >10 breaks removals, <1 frees the model
EDIT_GROUNDING_PX = 768
EDIT_MEGAPIXELS = 1.0  # the LoRA's sweet spot

# Local Krea 2 edits of uploaded photos may show real people: no nudity or sexual edits of them. Only local generation
# is checked here; Atlas engines (z-image/turbo, Seedream) are left to Atlas's own moderation.
_SEXUAL = re.compile(
    r"\b(nude|nudes|nudity|naked|topless|bottomless|undress\w*|nsfw|sex|sexy|sexual\w*|explicit|porn\w*|erotic\w*|"
    r"genitals?|nipples?|remove (?:her|his|their|the) (?:clothes|clothing|top|shirt|dress|bra))\b",
    re.IGNORECASE,
)


def _sampler_hint(items: list) -> "ImageLora | None":
    """The first LoRA with an opinion about how to sample it, or None.

    ``cfg`` counts like the rest: a LoRA that only names a guidance -- and several do, because guidance is
    the one setting that breaks them -- would otherwise be skipped and quietly sampled at the engine default.
    """
    return next((item for item in items
                 if item.steps or item.scheduler or item.sampler or item.cfg is not None), None)


def _stem(value) -> str:
    """A LoRA name reduced to what identifies it: no folder, no suffix, no case."""
    return str(value or "").strip().rsplit("/", 1)[-1].lower().removesuffix(".safetensors")


def _same_lora(file: str, name: str) -> bool:
    """Whether a stored or requested name points at this file.

    Compared on the bare stem, because a default may be written with or without the ``.safetensors``
    suffix and with or without its folder. An exact compare here fails silently -- the default saves,
    then simply never attaches -- so it has to be the forgiving kind.
    """
    return bool(_stem(name)) and _stem(file) == _stem(name)


def _match_loras(items: list, name) -> list:
    """The installed catalogue entries a requested name could mean, exact matches winning outright.

    One matcher, used both to resolve a name into a file and to decide whether an always-on LoRA is
    already named by the request. Those were two different comparisons: the duplicate guard tested exact
    stems while resolution accepted a substring, so a shortened name like "qwen-image-2.1-fix" looked new
    to the guard, got the automatic copy appended beside it, and then resolved to that same file -- one
    LoRA loaded twice in a chain, at roughly double the strength its catalogue entry allows.

    Matching the stem exactly also accepts the folder-qualified form ("qwen21/<file>.safetensors"), which
    is how results write LoRA names back, so a name copied out of one job is usable in the next.
    """
    key = str(name or "").strip().lower()
    if not key:
        return []
    stem = _stem(key)
    exact = [i for i in items if i.file.lower() == key or i.file.lower().rsplit(".", 1)[0] == key
             or _stem(i.file) == stem]
    return exact or [i for i in items if key in i.file.lower() or key == i.label.lower()
                     or (stem and stem in _stem(i.file))]


def pick_model(configured: str, files: list[str], family: re.Pattern) -> str | None:
    """The configured file if present, else the best file of the same model family."""
    for name in files:
        if name == configured or name.rsplit("/", 1)[-1] == configured:
            return name
    matches = [name for name in files if family.search(name.rsplit("/", 1)[-1])]
    rank = lambda name: next(i for i, tag in enumerate(_PRECISION) if tag in name.lower())
    return min(matches, key=lambda name: (rank(name), name)) if matches else None
MAX_ADULT_LORAS = 3  # adult LoRAs one image may stack (SNOFS + Mystic XXX is the go-to pair)
ADULT_STRENGTH_WARN = 2.0  # combined adult LoRA strength above this tends to over-cook Turbo
# Local Krea 2 text-to-image attaches the go-to adult pair by default: base Krea 2 is coy without them. A request
# that names its own adult LoRAs keeps those instead, and edits never get them (check_edit refuses them on uploads).
#: Per family, attached to local generation unless the request names its own adult LoRA. These are the
#: fallback: Studio's defaults panel overrides them per family, and an empty list there switches them off.
DEFAULT_ADULT_LORAS: dict[str, tuple[str, ...]] = {
    "krea2": ("snofs_krea2.safetensors", "krea2_mystic_xxx_v3.safetensors"),
    "qwen21": ("NSFW Qwen Lora.safetensors",),
    "zit": ("zit_mystic_xxx.safetensors",),
}
#: Per family, attached to every image whatever else was asked for. Unlike the adult defaults above, a
#: request naming its own adult LoRA does not displace these: Qwen Image 2.1 ships with broken layers that
#: its repair LoRA fixes, so an image made without it is simply a worse image, not a different choice.
#: Mirrored in image_engines.BASE_LORAS, which cannot import this module.
BASE_LORAS: dict[str, tuple[tuple[str, float], ...]] = dict(image_engines.BASE_LORAS)

# Local generation has no provider-side moderation, so refuse prompts that point at minors.
_MINOR = re.compile(
    r"\b(child|children|kid|kids|minor|minors|underage|under-age|teen|teens|teenage|teenager|preteen|pre-teen|"
    r"schoolgirl|schoolboy|school ?girl|school ?boy|loli|lolita|shota|toddler|infant|baby-faced|"
    r"(?:1[0-7]|[1-9])[ -]?(?:yo|y/o|year[ -]old|years[ -]old))\b",
    re.IGNORECASE,
)


class LocalImageError(RuntimeError):
    """The local engine can't make this image (not installed, busy, failed or refused).

    fatal: the request itself is wrong (a LoRA name, too many adult LoRAs), so falling back to
    another engine would silently drop what was asked for."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class ImageLora:
    file: str
    kind: str = "other"
    label: str = ""
    strength: float = 0.8
    range: tuple[float, float] = (0.0, 2.0)
    trigger: str = ""
    notes: str = ""
    steps: int | None = None
    scheduler: str | None = None
    sampler: str | None = None
    cfg: float | None = None  # some LoRAs only behave at a particular guidance (the Qwen 2.1 NSFW one wants 1.0)
    installed: bool = False
    family: str = "krea2"  # which model this LoRA is for; a file from another family produces garbage, not an error
    automatic: bool = False  # attached by default (the adult pair), not asked for: its sampler hints don't take over

    def view(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if v not in (None, "")}
        data["range"] = list(self.range)
        return data


@dataclass
class LocalResult:
    images: list[bytes]
    loras: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _names_adult(requested: list[dict], items: list) -> bool:
    """True when the request already picks an adult LoRA, so the default pair is left out."""
    for spec in requested:
        key = str(spec.get("name") or "").strip().lower()
        if not key:
            continue
        for item in items:
            if getattr(item, "kind", "") != "adult":
                continue
            file = item.file.lower()
            if key in (file, file.rsplit(".", 1)[0], item.label.lower()) or key in file:
                return True
    return False


def check_prompt(prompt: str) -> None:
    match = _MINOR.search(prompt or "")
    if match:
        raise LocalImageError(
            f"Refused: the prompt mentions {match.group(0)!r}. The image engines on this GPU don't make images of "
            "anyone under 18 (a fixed rule).",
            fatal=True,
        )


def from_upload(asset: dict, lookup=None, depth: int = 0) -> bool:
    """True when an image is an upload, or was made from one (an edit of an edit of a photo still counts).
    A reference that no longer exists counts as an upload, to be safe."""
    source = asset.get("source") or {}
    if source.get("type") != "generated":
        return True
    if lookup is None or depth > 20:
        return depth > 20
    for ref_id in source.get("references") or []:
        ref = lookup(ref_id)
        if ref is None or from_upload(ref, lookup, depth + 1):
            return True
    return False


def check_edit(prompt: str, sources: list[dict], loras: list, lookup=None) -> None:
    """Uploaded photos can be real people: edits of them, and images made from them, stay non-sexual and
    use no adult LoRA. Pictures made here from a prompt are fictional characters and follow the normal rules."""
    real = [a for a in sources if from_upload(a, lookup)]
    if not real:
        return
    names = ", ".join(a.get("filename") or a.get("id", "") for a in real)
    if any(getattr(item, "kind", "") == "adult" for item in loras):
        raise LocalImageError(f"Refused: adult LoRAs can't be used to edit uploaded photos ({names}); they may show real people. "
                              "Generate a fictional character first and edit that.", fatal=True)
    match = _SEXUAL.search(prompt or "")
    if match:
        raise LocalImageError(f"Refused: {match.group(0)!r} edits of uploaded photos ({names}) aren't allowed; they may show "
                              "real people.", fatal=True)




def edit_size(width: int, height: int, megapixels: float = EDIT_MEGAPIXELS) -> tuple[int, int]:
    """The source's aspect ratio at about 1 MP, in multiples of 16."""
    scale = (megapixels * 1_000_000 / max(1, width * height)) ** 0.5
    return tuple(max(512, min(2048, round(v * scale / 16) * 16)) for v in (width, height))


def load_catalogue(path: str) -> list[ImageLora]:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        shutil.copyfile(EXAMPLE_LORAS, path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if os.path.abspath(path) != os.path.abspath(EXAMPLE_LORAS):
        with open(EXAMPLE_LORAS, "r", encoding="utf-8") as handle:
            example = json.load(handle)
        if int(data.get("version") or 1) < int(example.get("version") or 1):  # a newer catalogue ships with the code
            shutil.copyfile(path, path + ".bak")
            shutil.copyfile(EXAMPLE_LORAS, path)
            data = example
    entries = [e for e in data.get("loras", []) if isinstance(e, dict)]
    if os.path.abspath(path) != os.path.abspath(EXAMPLE_LORAS):  # LoRAs added to the example later still show up
        with open(EXAMPLE_LORAS, "r", encoding="utf-8") as handle:
            known = {e.get("file") for e in entries}
            entries += [e for e in json.load(handle).get("loras", []) if isinstance(e, dict) and e.get("file") not in known]
    items = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        low, high = (entry.get("range") or [0.0, 2.0])[:2]
        items.append(ImageLora(
            file=str(entry["file"]), kind=str(entry.get("kind") or "other").lower(), label=str(entry.get("label") or ""),
            strength=float(entry.get("strength", 0.8)), range=(float(low), float(high)), trigger=str(entry.get("trigger") or ""),
            notes=str(entry.get("notes") or ""), steps=entry.get("steps"), scheduler=entry.get("scheduler"), sampler=entry.get("sampler"),
            cfg=None if entry.get("cfg") is None else float(entry["cfg"]),
            family=str(entry.get("family") or "krea2"),
        ))
    return items


def parse_size(size: str | None, default: str = "1024x1536") -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[x*×]\s*(\d+)\s*", size or default)
    if not match:
        raise LocalImageError(f"size {size!r} should look like 1024x1536.")
    width, height = (max(512, min(2048, int(v) // 16 * 16)) for v in match.groups())
    return width, height


def _negative_node(text: str, positive: list, clip_node: str = "2") -> dict:
    """The negative conditioning slot.

    Empty means ConditioningZeroOut, which is what every graph did before there was a field for this. Note that
    at cfg 1.0 the guider collapses to the positive term, so a negative prompt is inert on Krea 2 Turbo and
    Z-Image, and only does anything on an engine that samples above cfg 1 -- Qwen Image 2.1, at cfg 2.0.
    """
    if not text.strip():
        return {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": positive}}
    return {"class_type": "CLIPTextEncode", "inputs": {"clip": [clip_node, 0], "text": text}}


def krea_graph(prompt: str, *, width: int, height: int, n: int, seed: int, loras: list[tuple[str, float]],
               unet: str, clip: str, vae: str, steps: int, sampler: str, scheduler: str, prefix: str,
               negative: str = "", cfg: float = 1.0) -> dict:
    """ComfyUI API-format graph for Krea 2 Turbo text to image."""
    graph: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model = ["1", 0]
    for index, (name, strength) in enumerate(loras):
        node = f"l{index}"
        graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [node, 0]
    graph.update({
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": _negative_node(negative, ["4", 0]),
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": n}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": model, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed,
            "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    })
    return graph


def krea_edit_graph(prompt: str, *, images: list[str], width: int, height: int, seed: int, loras: list[tuple[str, float]],
                    unet: str, clip: str, vae: str, steps: int, ref_boost: float, prefix: str,
                    grounding_px: int = EDIT_GROUNDING_PX, cfg: float = 1.0) -> dict:
    """Krea 2 Identity Edit, wired like the node pack's krea2_identity_edit.json workflow.
    images: 1 or 2 ComfyUI input paths; with 2, the first is the scene and the second the subject."""
    graph: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "lat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    model = ["1", 0]
    for index, (name, strength) in enumerate(loras):  # the identity-edit LoRA comes first
        node = f"l{index}"
        graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [node, 0]
    for index, path in enumerate(images[:2], 1):
        graph[f"img{index}"] = {"class_type": "LoadImage", "inputs": {"image": path}}
        graph[f"enc{index}"] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"img{index}", 0], "vae": ["3", 0]}}
    patch = {"model": model, "source_latent": ["enc1", 0], "vae": ["3", 0], "source_image": ["img1", 0],
             "target_latent": ["lat", 0], "ref_boost": ref_boost, "ref_boost_a": 1.0, "fit_mode": "fit"}
    grounded = {"clip": ["2", 0], "image": ["img1", 0], "grounding_px": grounding_px, "system_prompt": ""}
    if len(images) > 1:
        patch.update(source_latent_b=["enc2", 0], source_image_b=["img2", 0])
        grounded["image_b"] = ["img2", 0]
    graph.update({
        "patch": {"class_type": "Krea2EditModelPatch", "inputs": patch},
        "pos": {"class_type": "Krea2EditGroundedEncode", "inputs": {**grounded, "prompt": prompt}},
        "neg": {"class_type": "Krea2EditGroundedEncode", "inputs": {**grounded, "prompt": ""}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["patch", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0], "seed": seed,
            "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    })
    return graph


def zimage_graph(prompt: str, *, width: int, height: int, n: int, seed: int, loras: list[tuple[str, float]],
                 unet: str, clip: str, vae: str, steps: int, sampler: str, scheduler: str, prefix: str,
                 negative: str = "", cfg: float = 1.0) -> dict:
    """Z-Image Turbo text to image, from the ComfyUI template.

    Its text encoder loads as "lumina2", not as a Qwen type, and the model goes through ModelSamplingAuraFlow
    before sampling. Both are easy to get wrong and neither fails loudly.
    """
    graph: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model = ["1", 0]
    for index, (name, strength) in enumerate(loras):
        node = f"l{index}"
        graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [node, 0]
    graph.update({
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": _negative_node(negative, ["5", 0]),
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": n}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "seed": seed,
            "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    })
    return graph


def _qwen21_base(unet: str, clip: str, vae: str, loras: list[tuple[str, float]], cache: bool = False) -> tuple[dict, list]:
    """Loaders and the LoRA chain shared by both Qwen Image 2.1 graphs.

    The CLIP type is the field to be careful with: "qwen_image", not Klein's "flux2" or Z-Image's "lumina2".
    A wrong type loads and produces nonsense rather than failing.

    ``cache`` adds QwenImage21Cache, a speed-up that needs ComfyUI 0.36.0. The caller decides from
    object_info whether the node exists; without it the chain is identical, only slower.
    """
    graph: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model = ["1", 0]
    if cache:
        graph["c"] = {"class_type": "QwenImage21Cache", "inputs": {"model": model, "device": "auto", "dtype": "default"}}
        model = ["c", 0]
    for index, (name, strength) in enumerate(loras):
        node = f"l{index}"
        graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [node, 0]
    return graph, model


def qwen21_graph(prompt: str, *, width: int, height: int, n: int, seed: int, loras: list[tuple[str, float]],
                 unet: str, clip: str, vae: str, steps: int, cfg: float, sampler: str, scheduler: str,
                 prefix: str, negative: str = "", cache: bool = False) -> dict:
    """Qwen Image 2.1 text to image. A plain KSampler, unlike Klein's SamplerCustomAdvanced path."""
    graph, model = _qwen21_base(unet, clip, vae, loras, cache)
    graph.update({
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": _negative_node(negative, ["4", 0]),
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": n}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": model, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed,
            "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    })
    return graph


def qwen21_edit_graph(prompt: str, *, images: list[str], width: int | None, height: int | None, seed: int,
                      loras: list[tuple[str, float]], unet: str, clip: str, vae: str, steps: int, cfg: float,
                      sampler: str, scheduler: str, prefix: str, negative: str = "",
                      resolution: int = QWEN21_EDIT_RESOLUTION, cache: bool = False) -> dict:
    """Qwen Image 2.1 edit, with up to sixteen reference images.

    One TextEncodeQwenImage21 node does the whole job: it takes the references on images.image_1 ..
    images.image_16, the
    prompt and the negative prompt, and returns conditioning for both plus a latent sized from the
    references. That is why this takes sixteen where Klein's chained ReferenceLatents took six.

    Output size follows the references unless one was asked for, in which case an EmptyLatentImage replaces
    the node's own latent.
    """
    if not images:
        raise LocalImageError("Qwen Image 2.1 edit needs at least one reference image.", fatal=True)
    if len(images) > QWEN21_MAX_REFS:
        raise LocalImageError(f"Qwen Image 2.1 edit takes at most {QWEN21_MAX_REFS} reference images; "
                              f"{len(images)} were given.", fatal=True)
    graph, model = _qwen21_base(unet, clip, vae, loras, cache)
    encode = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "negative_prompt": negative,
              "resolution": resolution}
    for index, path in enumerate(images, start=1):
        graph[f"i{index}"] = {"class_type": "LoadImage", "inputs": {"image": path}}
        # "images.image_1", not "image_1": the node takes its references through an Autogrow input
        # named "images", and ComfyUI matches the slots by their namespaced id, gathering them into one
        # dict argument. A flat "image_1" matches nothing, survives as a stray keyword and lands as
        # "execute() got an unexpected keyword argument 'image_1'" at sampling time.
        encode[f"images.image_{index}"] = [f"i{index}", 0]
    graph["4"] = {"class_type": "TextEncodeQwenImage21", "inputs": encode}
    if width and height:
        graph["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["6", 0]
    else:  # the encode node already sized a latent from the references
        latent = ["4", 2]
    graph.update({
        "7": {"class_type": "KSampler", "inputs": {
            "model": model, "positive": ["4", 0], "negative": ["4", 1], "latent_image": latent, "seed": seed,
            "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    })
    return graph


class LocalImageEngine:
    def __init__(self, service):
        self.service = service
        self.settings = service.settings
        self.path = os.path.join(self.settings.data_dir, "image_loras.json")
        self.boost_broken = False  # the likeness boost hit the Blackwell cuDNN error: edits run at 1.0 until restart
        self.adult_default = getattr(self.settings, "krea_adult_default", True)  # the go-to pair on every generation

    # ------------------------------------------------------------ status

    async def status(self, engine: str = "krea2") -> dict:
        """Installed model files for one local engine, whether ComfyUI is busy, and its LoRA catalogue."""
        spec = LOCAL_MODELS[engine]
        missing, files = [], {}
        try:
            folders = {folder for folder, _ in spec["files"].values()}
            listing = {folder: await self.service.available_models(folder, refresh=True) for folder in folders}
            for key, (folder, pattern) in spec["files"].items():
                name = getattr(self.settings, spec["configured"][key], "")
                found = pick_model(name, listing[folder], pattern)
                if found:
                    files[key] = found
                else:
                    missing.append(f"{folder}/{name}")
            busy, queue = await self.busy()
            reachable = True
        except Exception as exc:  # ComfyUI down
            return {"installed": False, "busy": False, "reachable": False, "missing": [], "error": str(exc), "loras": []}
        family = image_engines.get(engine).lora_family
        return {"engine": engine, "installed": not missing, "busy": busy, "queue": queue, "reachable": reachable,
                "missing": missing, "model": files.get("unet", ""), "files": files,
                "edit": await self.edit_status(engine),
                # each engine lists its own family's LoRAs; loading another family's makes a worse image, not an error
                "loras": [lora.view() for lora in await self.catalogue(family)]}

    async def statuses(self) -> dict:
        """Every local engine's status, keyed by engine id."""
        return {engine: await self.status(engine) for engine in LOCAL_MODELS}

    async def _has_cache_node(self) -> bool:
        """Whether this ComfyUI has QwenImage21Cache (0.36.0+). Without it the graph is the same, only slower,
        so a missing node is not worth failing over -- unlike TextEncodeQwenImage21, which edits need."""
        try:
            return bool(await self.service.comfy.object_info(QWEN21_CACHE_NODE))
        except Exception:  # ComfyUI down: the caller is about to fail on something louder than this
            return False

    async def edit_status(self, engine: str = "krea2") -> dict:
        """What an engine needs to edit, beyond its base files.

        Krea 2 needs the comfyui-krea2edit nodes and the identity-edit LoRA. Qwen Image 2.1 needs one core
        node that only exists from ComfyUI 0.36.0, so an older pod is told which node rather than having
        ComfyUI reject the graph at submit time. Everything else edits with its own weights alone.
        """
        if engine != "krea2":
            spec = image_engines.get(engine)
            if not spec or not spec.edit:
                return {"installed": False, "missing": [f"{spec.label if spec else engine} does not edit images."], "lora": None}
            if engine == "qwen21" and not await self.service.comfy.object_info(QWEN21_EDIT_NODE):
                return {"installed": False, "lora": None,
                        "missing": [f"core node {QWEN21_EDIT_NODE} (needs ComfyUI 0.36.0 or newer)"]}
            return {"installed": True, "missing": [], "lora": None}
        missing = [f"custom node {node} (comfyui-krea2edit)" for node in EDIT_NODES if not await self.service.comfy.object_info(node)]
        lora = self._edit_lora(await self.service.available_models("loras"))
        if not lora:
            missing.append("loras/krea2_identity_edit_v1_2.safetensors")
        return {"installed": not missing, "missing": missing, "lora": lora, "boost": not self.boost_broken}

    @staticmethod
    def _edit_lora(files: list[str]) -> str | None:
        matches = [name for name in files if EDIT_LORA.search(name.rsplit("/", 1)[-1])]
        # the full-rank file first, then the r128 / r64 low-VRAM variants; newest version first
        return min(matches, key=lambda n: ("_r64" in n, "_r128" in n, [-int(x) for x in re.findall(r"\d+", n)])) if matches else None

    async def wait_until_idle(self, seconds: float) -> bool:
        """Poll until ComfyUI's queue drains, or the budget runs out. True when it drained.

        Polls busy() and never status(): status re-lists every model folder, so polling it for two minutes
        would be two minutes of model listings.
        """
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            busy, _ = await self.busy()
            if not busy:
                return True
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            await asyncio.sleep(min(POLL_SECONDS, left))

    async def busy(self) -> tuple[bool, int]:
        running, pending = await self.service.comfy.queue_state()
        return bool(running or pending), len(running) + len(pending)

    async def catalogue(self, family: str = "") -> list[ImageLora]:
        """The curated LoRAs plus any other image LoRA on the pod, marked installed or not.

        One models/loras folder holds every family, so each file is classified by name (or by the catalogue's
        own ``family`` key). ``family=""`` returns every image family; pass one to get just that engine's.
        """
        items = load_catalogue(self.path)
        files = await self.service.available_models("loras")
        known = {item.file for item in items}
        for item in items:
            at = next((f for f in files if f == item.file or f.endswith("/" + item.file)), item.file)
            item.installed = at != item.file or item.file in files
            item.family = image_engines.family_of(at, item.family)
        for name in files:
            base = name.rsplit("/", 1)[-1]
            # The whole path, not the base: a file in models/loras/qwen21 is a Qwen 2.1 LoRA even when its own
            # name says nothing, which is the point of downloading into a folder named after the family.
            found = image_engines.family_of(name)
            if base not in known and found in image_engines.IMAGE_FAMILIES and not EDIT_LORA.search(base):
                items.append(ImageLora(file=name, kind="other", label=base, installed=True, family=found))
        return [item for item in items if not family or item.family == family]

    async def lora_basenames(self, family: str = "") -> set[str]:
        """File names in models/loras that belong to images rather than to video renders.

        Includes the Krea 2 Identity Edit LoRA, which the catalogue leaves out because edit() applies it itself.
        """
        names = {item.file.rsplit("/", 1)[-1].lower() for item in await self.catalogue(family) if item.installed}
        if family and family != "krea2":
            return names
        files = await self.service.available_models("loras")
        return names | {f.rsplit("/", 1)[-1].lower() for f in files if EDIT_LORA.search(f.rsplit("/", 1)[-1])}

    async def resolve_loras(self, requested: list[dict] | None, max_adult: int = MAX_ADULT_LORAS,
                            adult_default: bool = False, family: str = "krea2",
                            ) -> tuple[list[tuple[str, float]], list[ImageLora], list[str]]:
        """Match the requested LoRAs against the ones this engine can actually load.

        Only ``family``'s files are candidates: a Qwen 2.1 LoRA in a Krea 2 graph does not fail, it just makes
        a worse image, so the wrong family must never be reachable by name.
        """
        requested = list(requested or [])
        items = [item for item in await self.catalogue(family) if item.installed]
        automatic: set[str] = set()
        # Read before anything is attached: whether the *request* names an adult LoRA is what decides if the
        # family's adult defaults step aside, and the base LoRAs appended below must not change that answer.
        names_adult = _names_adult(requested, items)

        def resolved_file(name) -> str | None:
            """The one installed file a name means, or None when it means none or several."""
            matches = _match_loras(items, name)
            return matches[0].file if len(matches) == 1 else None

        def attach(wanted: list[dict]) -> None:
            """Append defaults the request has not already named, remembering which arrived on their own."""
            # Compared by the file each name resolves to, not by the spelling: "qwen-image-2.1-fix" and
            # "qwen-image-2.1-fix-1.0-comfy.safetensors" are the same LoRA, and attaching it beside itself
            # applies it twice at double strength. A name resolving to nothing installed attaches nothing,
            # so a pod missing one still generates.
            taken = {file for file in (resolved_file(spec.get("name")) for spec in requested) if file}
            for entry in wanted:
                target = resolved_file(entry.get("name"))
                if target is None or target in taken:
                    continue
                automatic.add(_stem(target))
                requested.append(dict(entry))
                taken.add(target)

        # Always-on repair LoRAs, attached first and *not* displaced by a request naming its own adult LoRA.
        base = BASE_LORAS.get(family, ())
        attach([{"name": name, "strength": strength} for name, strength in base])
        # A name that does not match any installed file attaches nothing and says nothing, so the only place
        # a typo in an always-on LoRA can surface is here.
        warnings = [f"{image_engines.family_label(family)} always attaches {name!r}, which is not installed on "
                    "this pod, so this image was made without it."
                    for name, _ in base if not any(_same_lora(i.file, name) for i in items)]
        if adult_default and not names_adult:
            stored = self.service.image_engines.defaults(family)
            wanted = stored if stored is not None else [{"name": n} for n in DEFAULT_ADULT_LORAS.get(family, ())]
            attach(wanted)
        if not requested:
            return [], [], warnings
        files = await self.service.available_models("loras")
        chosen: list[tuple[str, float]] = []
        used: list[ImageLora] = []
        for spec in requested:
            name = str(spec.get("name") or "").strip()
            matches = _match_loras(items, name)
            if len(matches) != 1:
                options = ", ".join(i.file for i in items) or "none installed"
                raise LocalImageError(f"Image LoRA {name!r} matches {len(matches)} installed files. Installed: {options}.", fatal=True)
            item = matches[0]
            path = next((f for f in files if f == item.file or f.endswith("/" + item.file)), item.file)
            strength = float(spec["strength"]) if spec.get("strength") is not None else item.strength
            if strength == 0:
                continue
            item.automatic = item.file.rsplit("/", 1)[-1].lower().removesuffix(".safetensors") in automatic
            chosen.append((path, strength))
            used.append(item)
        adult = [(item, strength) for item, (_, strength) in zip(used, chosen) if item.kind == "adult"]
        if len(adult) > max_adult:
            raise LocalImageError("Use one adult LoRA at a time; they overlap and fight each other." if max_adult == 1
                                  else f"At most {max_adult} adult LoRAs in one image.", fatal=True)
        total = sum(abs(strength) for _, strength in adult)
        if len(adult) > 1 and total > ADULT_STRENGTH_WARN:
            warnings.append(f"{len(adult)} adult LoRAs at a combined strength of {total:.2f}: above about {ADULT_STRENGTH_WARN:.1f} "
                            "they tend to over-cook (melted anatomy, plastic skin). Lower them if the result looks off.")
        return chosen, used, warnings

    # ------------------------------------------------------------ generate

    async def _ready(self, status: dict, engine: str, action: str = "generate", wait_seconds: float = 0.0) -> None:
        """Raise unless this engine can run right now. Never fatal: the ladder may have somewhere else to go."""
        label = image_engines.get(engine).label
        if not status.get("reachable"):
            raise LocalImageError(f"ComfyUI is not reachable: {status.get('error')}")
        if not status["installed"]:
            raise LocalImageError(f"{label} is not installed on the pod (missing " + ", ".join(status["missing"]) + ").")
        if action == "edit" and not status["edit"]["installed"]:
            raise LocalImageError(f"{label} can't edit here (missing " + ", ".join(status["edit"]["missing"]) + ").")
        if status["busy"] and wait_seconds > 0 and await self.wait_until_idle(wait_seconds):
            return  # the render finished while we waited, so this engine can have the GPU
        if status["busy"]:
            raise LocalImageError(f"ComfyUI is busy ({status['queue']} job(s) running or queued, usually a video render).")

    async def _reference_paths(self, sources: list[dict]) -> list[str]:
        """Each source's path for the graph, restoring any file a restarted runtime lost.

        Not fatal: the ladder can still try a paid engine, which fetches the bytes over HTTP and does not
        need the file on this disk at all.
        """
        missing = []
        for asset in sources:
            if not await self.service.ensure_asset_on_disk(asset):
                missing.append(asset.get("filename") or asset.get("id", "?"))
        if missing:
            raise LocalImageError(
                "The file is gone from ComfyUI's input folder for " + ", ".join(missing) +
                ", and no Drive export was found to restore it from. The library still lists the image "
                "because the database survived the restart; the pixels did not.")
        return [asset["path"] for asset in sources]

    def _with_triggers(self, prompt: str, used: list) -> str:
        text = prompt.strip()
        for item in used:  # trigger words go in automatically
            if item.trigger and item.trigger.lower() not in text.lower():
                text = f"{text}, {item.trigger}"
        return text

    async def _submit(self, graph: dict, label: str) -> list[bytes]:
        prompt_id = str(uuid.uuid4())
        try:
            await self.service.comfy.submit(graph, prompt_id)
        except ComfyError as exc:
            raise LocalImageError(f"ComfyUI rejected the {label} graph: {exc}") from exc
        return await self._collect(prompt_id, label)

    async def generate(self, prompt: str, *, size: str | None = None, n: int = 1, seed: int | None = None,
                       loras: list[dict] | None = None, steps: int | None = None,
                       max_adult_loras: int = MAX_ADULT_LORAS, engine: str = "krea2",
                       wait_seconds: float = 0.0, negative: str = "", cfg: float | None = None) -> LocalResult:
        check_prompt(prompt)
        status = await self.status(engine)
        await self._ready(status, engine, wait_seconds=wait_seconds)
        spec = image_engines.get(engine)
        chosen, used, warnings = await self.resolve_loras(loras, max(1, min(MAX_ADULT_LORAS, max_adult_loras)),
                                                          adult_default=self.adult_default, family=spec.lora_family)
        width, height = parse_size(size)
        text = self._with_triggers(prompt, used)
        asked = [item for item in used if not item.automatic]
        hint = _sampler_hint(asked or used)
        files = status["files"]
        seed = seed if seed is not None else int.from_bytes(os.urandom(6), "big")
        batch = max(1, min(4, n))
        started = time.monotonic()
        if engine == "qwen21":
            graph = qwen21_graph(text, width=width, height=height, n=batch, seed=seed, loras=chosen,
                                 unet=files["unet"], clip=files["clip"], vae=files["vae"],
                                 steps=steps or (hint.steps if hint and hint.steps else QWEN21_STEPS),
                                 cfg=cfg if cfg is not None else (hint.cfg if hint and hint.cfg is not None else QWEN21_CFG),
                                 sampler=(hint.sampler if hint and hint.sampler else QWEN21_SAMPLER),
                                 scheduler=(hint.scheduler if hint and hint.scheduler else QWEN21_SCHEDULER),
                                 prefix="hawk_images/qwen21", negative=negative,
                                 cache=await self._has_cache_node())
        elif engine == "zimage":
            graph = zimage_graph(text, width=width, height=height, n=batch, seed=seed, loras=chosen,
                                 unet=files["unet"], clip=files["clip"], vae=files["vae"],
                                 steps=steps or (hint.steps if hint and hint.steps else ZIMAGE_STEPS),
                                 sampler=(hint.sampler if hint and hint.sampler else "res_multistep"),
                                 scheduler=(hint.scheduler if hint and hint.scheduler else "simple"),
                                 prefix="hawk_images/zimage", negative=negative,
                                 cfg=cfg if cfg is not None else 1.0)
        else:
            graph = krea_graph(text, width=width, height=height, n=batch, seed=seed, loras=chosen,
                               unet=files["unet"], clip=files["clip"], vae=files["vae"],
                               steps=steps or (hint.steps if hint and hint.steps else DEFAULT_STEPS),
                               sampler=(hint.sampler if hint and hint.sampler else "euler"),
                               scheduler=(hint.scheduler if hint and hint.scheduler else "simple"),
                               prefix="hawk_images/krea2", negative=negative,
                               cfg=cfg if cfg is not None else 1.0)
        images = await self._submit(graph, spec.label)
        return LocalResult(images, [{"file": f, "strength": v} for f, v in chosen], round(time.monotonic() - started, 1), warnings)

    async def edit_qwen21(self, prompt: str, sources: list[dict], *, size: str | None = None, n: int = 1,
                          seed: int | None = None, loras: list[dict] | None = None, steps: int | None = None,
                          max_adult_loras: int = MAX_ADULT_LORAS, wait_seconds: float = 0.0,
                          negative: str = "", cfg: float | None = None) -> LocalResult:
        """Qwen Image 2.1 edit: up to sixteen references, all through one TextEncodeQwenImage21 node.

        Unlike the Klein edit this replaced, a LoRA's sampler hints apply here too -- the go-to adult LoRA
        for this family only behaves at cfg 1 on er_sde/beta, which a fixed edit recipe could not express.
        """
        check_prompt(prompt)
        status = await self.status("qwen21")
        await self._ready(status, "qwen21", "edit", wait_seconds=wait_seconds)
        photo = any(from_upload(asset, self.service.store.get_asset) for asset in sources)
        chosen, used, warnings = await self.resolve_loras(loras, max(1, min(MAX_ADULT_LORAS, max_adult_loras)),
                                                          adult_default=self.adult_default and not photo, family="qwen21")
        check_edit(prompt, sources, used, self.service.store.get_asset)
        text = self._with_triggers(prompt, used)
        width, height = parse_size(size) if size else (None, None)
        files = status["files"]
        asked = [item for item in used if not item.automatic]
        hint = _sampler_hint(asked or used)
        cache = await self._has_cache_node()
        seed = seed if seed is not None else int.from_bytes(os.urandom(6), "big")
        paths = await self._reference_paths(sources)
        started, images = time.monotonic(), []
        for index in range(max(1, min(4, n))):
            graph = qwen21_edit_graph(text, images=paths, width=width, height=height,
                                      seed=seed + index, loras=chosen, unet=files["unet"], clip=files["clip"],
                                      vae=files["vae"],
                                      steps=steps or (hint.steps if hint and hint.steps else QWEN21_STEPS),
                                      cfg=cfg if cfg is not None else (hint.cfg if hint and hint.cfg is not None else QWEN21_CFG),
                                      sampler=(hint.sampler if hint and hint.sampler else QWEN21_SAMPLER),
                                      scheduler=(hint.scheduler if hint and hint.scheduler else QWEN21_SCHEDULER),
                                      prefix="hawk_images/qwen21_edit", negative=negative, cache=cache)
            images += await self._submit(graph, "Qwen Image 2.1 edit")
        return LocalResult(images, [{"file": f, "strength": v} for f, v in chosen], round(time.monotonic() - started, 1), warnings)

    async def edit(self, prompt: str, sources: list[dict], *, size: str | None = None, n: int = 1, seed: int | None = None,
                   loras: list[dict] | None = None, steps: int | None = None, ref_boost: float | None = None,
                   wait_seconds: float = 0.0, max_adult_loras: int = MAX_ADULT_LORAS,
                   cfg: float | None = None) -> LocalResult:
        """Edit one image (or put the person from a second image into the first) with Krea 2 Identity Edit."""
        check_prompt(prompt)
        if not 1 <= len(sources) <= 2:
            raise LocalImageError("Krea 2 edit takes 1 image, or 2 (scene first, then the person).")
        status = await self.status()
        if not status.get("reachable"):
            raise LocalImageError(f"ComfyUI is not reachable: {status.get('error')}")
        if not status["installed"]:
            raise LocalImageError("Krea 2 is not installed on the pod (missing " + ", ".join(status["missing"]) + ").")
        if not status["edit"]["installed"]:
            raise LocalImageError("Krea 2 edit is not installed (missing " + ", ".join(status["edit"]["missing"]) + ").")
        if status["busy"] and wait_seconds > 0 and await self.wait_until_idle(wait_seconds):
            pass  # the render finished while we waited
        elif status["busy"]:
            raise LocalImageError(f"ComfyUI is busy ({status['queue']} job(s) running or queued, usually a video render).")
        # Editing a picture made here means a fictional character, so the go-to adult pair applies as it does to
        # generation. Anything tracing back to an uploaded photo may be a real person and never gets them; check_edit
        # refuses them there anyway, and would fail every ordinary photo edit if they were attached blindly.
        photo = any(from_upload(asset, self.service.store.get_asset) for asset in sources)
        chosen, used, warnings = await self.resolve_loras(loras, max(1, min(MAX_ADULT_LORAS, max_adult_loras)),
                                                          adult_default=self.adult_default and not photo, family="krea2")
        check_edit(prompt, sources, used, self.service.store.get_asset)
        if size:
            width, height = parse_size(size)
        else:
            width, height = edit_size(*(await self.service.image_dimensions(sources[0]) or (1024, 1024)))
        text = prompt.strip()
        for item in used:
            if item.trigger and item.trigger.lower() not in text.lower():
                text = f"{text}, {item.trigger}"
        seed = seed if seed is not None else int.from_bytes(os.urandom(6), "big")
        boost = EDIT_REF_BOOST if ref_boost is None else max(0.0, min(20.0, float(ref_boost)))
        started = time.monotonic()
        wanted = boost
        if boost != 1.0 and self.boost_broken:
            boost = 1.0
        images = []
        # One prompt at a time: after a cuDNN failure the next queued masked prompt can segfault ComfyUI.
        for index in range(max(1, min(4, n))):
            try:
                images += await self._run_edit(text, sources, width, height, seed + index, status, chosen, steps, boost,
                                              cfg=cfg if cfg is not None else 1.0)
            except LocalImageError as exc:
                # Blackwell: cuDNN attention has no plan for the likeness boost's attention mask. Without the
                # boost there is no mask, so run at 1.0 rather than failing, and stay at 1.0 from now on.
                if boost == 1.0 or "cudnn" not in str(exc).lower():
                    raise
                self.boost_broken = True
                boost = 1.0
                images += await self._run_edit(text, sources, width, height, seed + index, status, chosen, steps, boost,
                                              cfg=cfg if cfg is not None else 1.0)
        if boost != wanted:
            warnings.append(f"The likeness boost ({wanted:g}) fails on this GPU (cuDNN), so this edit ran at 1.0. Update the "
                            "Hawk nodes and restart ComfyUI, then restart the API, to use it.")
        applied = [{"file": status["edit"]["lora"], "strength": 1.0}] + [{"file": f, "strength": v} for f, v in chosen]
        return LocalResult(images, applied, round(time.monotonic() - started, 1), warnings)

    async def _run_edit(self, text, sources, width, height, seed, status, chosen, steps, boost,
                        cfg: float = 1.0) -> list[bytes]:
        files = status["files"]
        graph = krea_edit_graph(
            text, images=await self._reference_paths(sources), width=width, height=height, seed=seed,
            loras=[(status["edit"]["lora"], 1.0)] + chosen, unet=files["unet"], clip=files["clip"], vae=files["vae"],
            steps=steps or EDIT_STEPS, ref_boost=boost, prefix="hawk_images/krea2_edit", cfg=cfg,
        )
        return await self._submit(graph, "Krea 2 edit")

    async def _collect(self, prompt_id: str, label: str) -> list[bytes]:
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            try:
                history = await self.service.comfy.history(prompt_id)
            except ComfyError as exc:
                raise LocalImageError(str(exc)) from exc
            if history:
                status = history.get("status") or {}
                if status.get("status_str") == "error":
                    error = next((data for name, data in status.get("messages", []) if name == "execution_error"), {})
                    message = str(error.get("exception_message") or "ComfyUI reported an error.").strip()
                    where = f" in {error['node_type']} (node {error.get('node_id')})" if error.get("node_type") else ""
                    raise LocalImageError(f"{label} failed{where}: {message}")
                files = [image for output in (history.get("outputs") or {}).values() for image in output.get("images") or []]
                if files:
                    return [await self._read(image) for image in files]
                if status.get("completed"):
                    raise LocalImageError("Krea 2 finished without images.")
            if time.monotonic() > deadline:
                try:
                    await self.service.comfy.cancel(prompt_id)
                except ComfyError:
                    pass
                raise LocalImageError(f"Krea 2 did not finish within {int(WAIT_SECONDS)} s.")
            await asyncio.sleep(POLL_SECONDS)

    async def _read(self, image: dict) -> bytes:
        filename, subfolder, type_ = image["filename"], image.get("subfolder", ""), image.get("type", "output")
        local = self.service.local_output(filename, subfolder, type_)
        if local:
            return await asyncio.to_thread(_read_file, local)
        _, _, body = await self.service._view(filename, subfolder, type_)
        return b"".join([chunk async for chunk in body])


def _read_file(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()
