"""Local image generation with Krea 2 Turbo on the pod's own ComfyUI (text to image, no edits).

The graph follows Comfy-Org's ``image_krea2_turbo_t2i`` template without its optional
prompt-expansion step: UNETLoader -> LoraLoaderModelOnly x N -> KSampler (8 steps, cfg 1,
euler / simple), CLIPLoader(type "krea2") -> CLIPTextEncode, ConditioningZeroOut as the
negative, EmptyLatentImage, VAEDecode -> SaveImage.

It shares the GPU and the queue with video renders, so ``busy()`` tells callers to use a
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
#: Local engines whose ComfyUI graphs exist. Klein and Z-Image join this as their graphs land.
WIRED_ENGINES = ("krea2", "klein", "zimage")
# Any precision of the three Krea 2 files works (fp8_scaled, bf16, fp16…): the configured name wins,
# else the first match, higher precision first.
MODEL_FAMILIES = {
    "diffusion_models": re.compile(r"krea[-_ ]?2.*turbo", re.IGNORECASE),
    "text_encoders": re.compile(r"qwen[-_ ]?3[-_ ]?vl[-_ ]?4b", re.IGNORECASE),
    "vae": re.compile(r"qwen[-_ ]?image[-_ ]?vae", re.IGNORECASE),
}
#: Per local engine: which file to look for in each ComfyUI folder, and which setting names it.
#: The Qwen text encoders are deliberately distinct -- Krea 2 wants qwen3vl_4b, Z-Image qwen_3_4b and Klein
#: qwen_3_8b -- so the patterns must not match each other.
LOCAL_MODELS: dict[str, dict] = {
    "krea2": {
        "files": {"unet": ("diffusion_models", MODEL_FAMILIES["diffusion_models"]),
                  "clip": ("text_encoders", MODEL_FAMILIES["text_encoders"]),
                  "vae": ("vae", MODEL_FAMILIES["vae"])},
        "configured": {"unet": "krea_unet", "clip": "krea_clip", "vae": "krea_vae"},
    },
    "klein": {
        "files": {"unet": ("diffusion_models", re.compile(r"flux.?2.?klein", re.IGNORECASE)),
                  "clip": ("text_encoders", re.compile(r"qwen[-_ ]?3[-_ ]?8b", re.IGNORECASE)),
                  "vae": ("vae", re.compile(r"flux2[-_ ]?vae|full_encoder_small_decoder", re.IGNORECASE))},
        "configured": {"unet": "klein_unet", "clip": "klein_clip", "vae": "klein_vae"},
    },
    "zimage": {
        "files": {"unet": ("diffusion_models", re.compile(r"z[-_ ]?image(?!.*ae\.safetensors$)", re.IGNORECASE)),
                  "clip": ("text_encoders", re.compile(r"qwen[-_ ]?3[-_ ]?4b", re.IGNORECASE)),
                  "vae": ("vae", re.compile(r"z[-_ ]?image.*ae|^ae\.safetensors$", re.IGNORECASE))},
        "configured": {"unet": "zimage_unet", "clip": "zimage_clip", "vae": "zimage_vae"},
    },
}
#: FLUX.2 Klein ships distilled and "base" builds that want very different settings. Guessing wrong does not
#: fail, it just makes slow over-cooked images, so read it off the file name.
#: The base build's guidance is the one number worth being careful with: 5.0 posterises it -- blown greens and
#: blues, banded surfaces -- on any prompt, with or without LoRAs. Pass ``cfg`` on the request to tune it.
KLEIN_BASE_STEPS, KLEIN_BASE_CFG = 20, 3.0
KLEIN_TURBO_STEPS, KLEIN_TURBO_CFG = 4, 1.0
ZIMAGE_STEPS = 8


def klein_settings(unet: str) -> tuple[int, float]:
    return (KLEIN_BASE_STEPS, KLEIN_BASE_CFG) if "base" in unet.lower() else (KLEIN_TURBO_STEPS, KLEIN_TURBO_CFG)
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


def _same_lora(file: str, name: str) -> bool:
    """Whether a stored or requested name points at this file.

    Compared on the bare stem, because a default may be written with or without the ``.safetensors``
    suffix and with or without its folder. An exact compare here fails silently -- the default saves,
    then simply never attaches -- so it has to be the forgiving kind.
    """
    stem = lambda value: str(value or "").rsplit("/", 1)[-1].lower().removesuffix(".safetensors")
    return bool(stem(name)) and stem(file) == stem(name)


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
    "klein": ("klein_snofs.safetensors", "klein_nsfw_no_face_change.safetensors"),
    "zit": ("zit_mystic_xxx.safetensors",),
}

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
    at cfg 1.0 the guider collapses to the positive term, so a negative prompt is inert on the distilled builds
    and only starts doing anything on a build that samples above cfg 1 (FLUX.2 Klein "base").
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


def _klein_base(unet: str, clip: str, vae: str, loras: list[tuple[str, float]]) -> tuple[dict, list]:
    """Loaders and the LoRA chain shared by both FLUX.2 Klein graphs."""
    graph: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model = ["1", 0]
    for index, (name, strength) in enumerate(loras):
        node = f"l{index}"
        graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [node, 0]
    return graph, model


def klein_graph(prompt: str, *, width: int, height: int, n: int, seed: int, loras: list[tuple[str, float]],
                unet: str, clip: str, vae: str, steps: int, cfg: float, prefix: str, negative: str = "") -> dict:
    """FLUX.2 Klein text to image. Uses the advanced sampler path, not KSampler."""
    graph, model = _klein_base(unet, clip, vae, loras)
    graph.update({
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": _negative_node(negative, ["4", 0]),
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": n}},
        "7": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["4", 0], "negative": ["5", 0], "cfg": cfg}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0], "guider": ["7", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    })
    return graph


def klein_edit_graph(prompt: str, *, images: list[str], width: int | None, height: int | None, seed: int,
                     loras: list[tuple[str, float]], unet: str, clip: str, vae: str, steps: int, cfg: float,
                     megapixels: float, prefix: str, negative: str = "") -> dict:
    """FLUX.2 Klein edit, with any number of reference images.

    Each reference is scaled to about one megapixel, encoded, and folded into both the positive and the negative
    conditioning by its own ReferenceLatent. They chain, which is why Klein takes more than two references where
    Krea 2 Identity Edit takes two. Output size follows the first reference unless one was asked for.
    """
    graph, model = _klein_base(unet, clip, vae, loras)
    graph["4"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}}
    graph["5"] = _negative_node(negative, ["4", 0])
    positive, negative = ["4", 0], ["5", 0]
    for index, path in enumerate(images):
        graph[f"i{index}"] = {"class_type": "LoadImage", "inputs": {"image": path}}
        graph[f"s{index}"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [f"i{index}", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 1}}
        graph[f"e{index}"] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"s{index}", 0], "vae": ["3", 0]}}
        graph[f"rp{index}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": [f"e{index}", 0]}}
        graph[f"rn{index}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative, "latent": [f"e{index}", 0]}}
        positive, negative = [f"rp{index}", 0], [f"rn{index}", 0]
    if width and height:
        size_w, size_h = width, height
    else:  # follow the first reference, which is already at a size the model likes
        graph["gs"] = {"class_type": "GetImageSize", "inputs": {"image": ["s0", 0]}}
        size_w, size_h = ["gs", 0], ["gs", 1]
    graph.update({
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size_w, "height": size_h, "batch_size": 1}},
        "7": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": positive, "negative": negative, "cfg": cfg}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": size_w, "height": size_h}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0], "guider": ["7", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
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

    async def edit_status(self, engine: str = "krea2") -> dict:
        """What an engine needs to edit, beyond its base files.

        Krea 2 needs the comfyui-krea2edit nodes and the identity-edit LoRA. Klein edits with core nodes and
        its own weights, so it is ready as soon as those are.
        """
        if engine != "krea2":
            spec = image_engines.get(engine)
            if not spec or not spec.edit:
                return {"installed": False, "missing": [f"{spec.label if spec else engine} does not edit images."], "lora": None}
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
            # The whole path, not the base: a file in models/loras/klein is a Klein LoRA even when its own
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

        Only ``family``'s files are candidates: a Klein LoRA in a Krea 2 graph does not fail, it just makes
        a worse image, so the wrong family must never be reachable by name.
        """
        requested = list(requested or [])
        items = [item for item in await self.catalogue(family) if item.installed]
        automatic = set()
        if adult_default and not _names_adult(requested, items):
            stored = self.service.image_engines.defaults(family)
            wanted = stored if stored is not None else [{"name": n} for n in DEFAULT_ADULT_LORAS.get(family, ())]
            # never add one the request already names, or it is applied twice at double strength
            asked = {str(spec.get("name") or "").strip().lower().removesuffix(".safetensors") for spec in requested}
            automatic = {str(entry.get("name") or "").rsplit("/", 1)[-1].lower().removesuffix(".safetensors")
                         for entry in wanted
                         if any(_same_lora(i.file, entry.get("name"))
                                for i in items)  # only what is installed, so a pod missing one still generates
                         and str(entry.get("name") or "").lower().removesuffix(".safetensors") not in asked}
            requested += [dict(entry) for entry in wanted
                          if str(entry.get("name") or "").rsplit("/", 1)[-1].lower().removesuffix(".safetensors")
                          in automatic]
        if not requested:
            return [], [], []
        files = await self.service.available_models("loras")
        chosen: list[tuple[str, float]] = []
        used: list[ImageLora] = []
        for spec in requested:
            name = str(spec.get("name") or "").strip()
            key = name.lower()
            matches = [i for i in items if i.file.lower() == key or i.file.lower().rsplit(".", 1)[0] == key] or \
                      [i for i in items if key and (key in i.file.lower() or key == i.label.lower())]
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
        warnings = []
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
        return await self._collect(prompt_id)

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
        hint = next((item for item in (asked or used) if item.steps or item.scheduler or item.sampler), None)
        files = status["files"]
        seed = seed if seed is not None else int.from_bytes(os.urandom(6), "big")
        batch = max(1, min(4, n))
        started = time.monotonic()
        if engine == "klein":
            klein_steps, klein_cfg = klein_settings(files["unet"])
            graph = klein_graph(text, width=width, height=height, n=batch, seed=seed, loras=chosen,
                                unet=files["unet"], clip=files["clip"], vae=files["vae"],
                                steps=steps or klein_steps, cfg=cfg if cfg is not None else klein_cfg,
                                prefix="hawk_images/klein", negative=negative)
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

    async def edit_klein(self, prompt: str, sources: list[dict], *, size: str | None = None, n: int = 1,
                         seed: int | None = None, loras: list[dict] | None = None, steps: int | None = None,
                         max_adult_loras: int = MAX_ADULT_LORAS, wait_seconds: float = 0.0,
                         negative: str = "", cfg: float | None = None) -> LocalResult:
        """FLUX.2 Klein edit: every reference folded in through its own ReferenceLatent."""
        check_prompt(prompt)
        status = await self.status("klein")
        await self._ready(status, "klein", "edit", wait_seconds=wait_seconds)
        photo = any(from_upload(asset, self.service.store.get_asset) for asset in sources)
        chosen, used, warnings = await self.resolve_loras(loras, max(1, min(MAX_ADULT_LORAS, max_adult_loras)),
                                                          adult_default=self.adult_default and not photo, family="klein")
        check_edit(prompt, sources, used, self.service.store.get_asset)
        text = self._with_triggers(prompt, used)
        width, height = parse_size(size) if size else (None, None)
        files = status["files"]
        klein_steps, klein_cfg = klein_settings(files["unet"])
        seed = seed if seed is not None else int.from_bytes(os.urandom(6), "big")
        started, images = time.monotonic(), []
        for index in range(max(1, min(4, n))):
            graph = klein_edit_graph(text, images=[a["path"] for a in sources], width=width, height=height,
                                     seed=seed + index, loras=chosen, unet=files["unet"], clip=files["clip"],
                                     vae=files["vae"], steps=steps or klein_steps,
                                     cfg=cfg if cfg is not None else klein_cfg,
                                     megapixels=EDIT_MEGAPIXELS, prefix="hawk_images/klein_edit",
                                     negative=negative)
            images += await self._submit(graph, "FLUX.2 Klein edit")
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
            text, images=[a["path"] for a in sources], width=width, height=height, seed=seed,
            loras=[(status["edit"]["lora"], 1.0)] + chosen, unet=files["unet"], clip=files["clip"], vae=files["vae"],
            steps=steps or EDIT_STEPS, ref_boost=boost, prefix="hawk_images/krea2_edit", cfg=cfg,
        )
        prompt_id = str(uuid.uuid4())
        try:
            await self.service.comfy.submit(graph, prompt_id)
        except ComfyError as exc:
            raise LocalImageError(f"ComfyUI rejected the Krea 2 edit graph: {exc}") from exc
        return await self._collect(prompt_id)

    async def _collect(self, prompt_id: str) -> list[bytes]:
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
                    raise LocalImageError(f"Krea 2 failed{where}: {message}")
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
