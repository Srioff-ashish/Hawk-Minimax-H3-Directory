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

from .comfy_client import ComfyError

EXAMPLE_LORAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "image_loras.example.json")
DEFAULT_STEPS = 8
WAIT_SECONDS = 300.0  # first use loads ~18 GB of weights
POLL_SECONDS = 1.0
LORA_KINDS = ("realism", "detail", "style", "adult", "other")
MAX_ADULT_LORAS = 3  # the most a manual (Studio) request may stack; the agent stays at 1
ADULT_STRENGTH_WARN = 1.0  # combined adult LoRA strength above this tends to over-cook Turbo

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


def check_prompt(prompt: str) -> None:
    match = _MINOR.search(prompt or "")
    if match:
        raise LocalImageError(
            f"Refused: the prompt mentions {match.group(0)!r}. Images of anyone under 18 are not allowed "
            "(this rule applies to every engine and can't be changed).",
            fatal=True,
        )


def load_catalogue(path: str) -> list[ImageLora]:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        shutil.copyfile(EXAMPLE_LORAS, path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
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
        ))
    return items


def parse_size(size: str | None, default: str = "1024x1536") -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[x*×]\s*(\d+)\s*", size or default)
    if not match:
        raise LocalImageError(f"size {size!r} should look like 1024x1536.")
    width, height = (max(512, min(2048, int(v) // 16 * 16)) for v in match.groups())
    return width, height


def krea_graph(prompt: str, *, width: int, height: int, n: int, seed: int, loras: list[tuple[str, float]],
               unet: str, clip: str, vae: str, steps: int, sampler: str, scheduler: str, prefix: str) -> dict:
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
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": n}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": model, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed,
            "steps": steps, "cfg": 1.0, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    })
    return graph


class LocalImageEngine:
    def __init__(self, service):
        self.service = service
        self.settings = service.settings
        self.path = os.path.join(self.settings.data_dir, "image_loras.json")

    # ------------------------------------------------------------ status

    async def status(self) -> dict:
        """Installed model files, whether ComfyUI is busy, and the LoRA catalogue."""
        s = self.settings
        missing = []
        try:
            for folder, name in (("diffusion_models", s.krea_unet), ("text_encoders", s.krea_clip), ("vae", s.krea_vae)):
                if name not in await self.service.available_models(folder, refresh=True):
                    missing.append(f"{folder}/{name}")
            busy, queue = await self.busy()
            reachable = True
        except Exception as exc:  # ComfyUI down
            return {"installed": False, "busy": False, "reachable": False, "missing": [], "error": str(exc), "loras": []}
        return {"installed": not missing, "busy": busy, "queue": queue, "reachable": reachable, "missing": missing,
                "model": s.krea_unet, "loras": [lora.view() for lora in await self.catalogue()]}

    async def busy(self) -> tuple[bool, int]:
        running, pending = await self.service.comfy.queue_state()
        return bool(running or pending), len(running) + len(pending)

    async def catalogue(self) -> list[ImageLora]:
        """The curated LoRAs plus any other Krea LoRA file on the pod, marked installed or not."""
        items = load_catalogue(self.path)
        files = await self.service.available_models("loras")
        known = {item.file for item in items}
        for item in items:
            item.installed = item.file in files or any(f.endswith("/" + item.file) for f in files)
        for name in files:
            base = name.rsplit("/", 1)[-1]
            if base not in known and "krea" in base.lower():
                items.append(ImageLora(file=name, kind="other", label=base, installed=True))
        return items

    async def resolve_loras(self, requested: list[dict] | None, max_adult: int = 1
                            ) -> tuple[list[tuple[str, float]], list[ImageLora], list[str]]:
        if not requested:
            return [], [], []
        items = [item for item in await self.catalogue() if item.installed]
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

    async def generate(self, prompt: str, *, size: str | None = None, n: int = 1, seed: int | None = None,
                       loras: list[dict] | None = None, steps: int | None = None, wait_if_busy: bool = False,
                       max_adult_loras: int = 1) -> LocalResult:
        check_prompt(prompt)
        status = await self.status()
        if not status.get("reachable"):
            raise LocalImageError(f"ComfyUI is not reachable: {status.get('error')}")
        if not status["installed"]:
            raise LocalImageError("Krea 2 is not installed on the pod (missing " + ", ".join(status["missing"]) + ").")
        if status["busy"] and not wait_if_busy:
            raise LocalImageError(f"ComfyUI is busy ({status['queue']} job(s) running or queued, usually a video render).")
        chosen, used, warnings = await self.resolve_loras(loras, max(1, min(MAX_ADULT_LORAS, max_adult_loras)))
        width, height = parse_size(size)
        text = prompt.strip()
        for item in used:  # trigger words go in automatically
            if item.trigger and item.trigger.lower() not in text.lower():
                text = f"{text}, {item.trigger}"
        hint = next((item for item in used if item.steps or item.scheduler or item.sampler), None)
        s = self.settings
        graph = krea_graph(
            text, width=width, height=height, n=max(1, min(4, n)), seed=seed if seed is not None else int.from_bytes(os.urandom(6), "big"),
            loras=chosen, unet=s.krea_unet, clip=s.krea_clip, vae=s.krea_vae,
            steps=steps or (hint.steps if hint and hint.steps else DEFAULT_STEPS),
            sampler=(hint.sampler if hint and hint.sampler else "euler"), scheduler=(hint.scheduler if hint and hint.scheduler else "simple"),
            prefix="hawk_images/krea2",
        )
        prompt_id = str(uuid.uuid4())
        started = time.monotonic()
        try:
            await self.service.comfy.submit(graph, prompt_id)
        except ComfyError as exc:
            raise LocalImageError(f"ComfyUI rejected the Krea 2 graph: {exc}") from exc
        images = await self._collect(prompt_id)
        return LocalResult(images, [{"file": f, "strength": v} for f, v in chosen], round(time.monotonic() - started, 1), warnings)

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
                    message = next((str(data.get("exception_message", "")).strip() for name, data in status.get("messages", [])
                                    if name == "execution_error"), "ComfyUI reported an error.")
                    raise LocalImageError(f"Krea 2 failed: {message}")
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
