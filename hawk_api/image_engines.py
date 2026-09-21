"""Every image engine the gateway can use, described once.

The order engines are tried in used to be written out in four unrelated places — the local rung and the
Atlas chain in ``jobs.py``, ``ENGINE_LADDER`` in ``agent.py``, and again in JavaScript in ``studio.html`` —
so adding an engine meant finding all four. A descriptor here replaces them: it says what an engine can do
(generate, edit, how many references), where it runs, which LoRA family it loads, what it costs, and whether
the local content checks apply to it.

Nothing in this module does I/O, so ``jobs.py``, ``agent.py``, ``mcp_server.py`` and the settings store can
all import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

# Atlas model ids. These live here rather than in jobs.py because a descriptor names them.
IMAGE_MODEL = "bytedance/seedream-v5.0-pro/text-to-image"
IMAGE_EDIT_MODEL = "bytedance/seedream-v5.0-pro/edit"
IMAGE_LITE_MODEL = "bytedance/seedream-v5.0-lite"  # 2K-4K only, a little cheaper than Pro 2K, a little below Pro in quality
IMAGE_LITE_EDIT_MODEL = "bytedance/seedream-v5.0-lite/edit"
IMAGE_FAST_MODEL = "z-image/turbo"  # ~$0.01 an image, text only (no edits)


@dataclass(frozen=True)
class ImageEngine:
    """One rung of a ladder.

    ``checks`` is the content-check boundary made explicit: engines that run on the user's own GPU have no
    provider-side moderation, so ``check_prompt`` / ``check_edit`` apply to them. Atlas engines are left to
    Atlas's own moderation, which is how they have always worked here.
    """

    id: str
    label: str
    where: str  # "local" (this GPU) or "atlas" (paid, over the network)
    generate: bool
    edit: bool
    max_refs: int  # reference images an edit may take; 0 when edit is False
    lora_family: str  # key into LORA_FAMILIES; "" for engines that take no LoRAs
    checks: bool  # run the local content checks before this engine draws anything
    atlas_model: str = ""
    atlas_edit_model: str = ""
    price_key: str = ""  # key into jobs.IMAGE_PRICES; "" means free
    tag: str = ""  # asset tag written by _image_result; user-visible in the Media library
    edit_tag: str = ""

    @property
    def local(self) -> bool:
        return self.where == "local"

    def supports(self, action: str) -> bool:
        return self.edit if action == "edit" else self.generate

    def tag_for(self, action: str) -> str:
        return (self.edit_tag or self.tag) if action == "edit" else self.tag


#: Canonical engines, in the order they are shown in Studio. The order a request actually tries them in
#: comes from the stored settings, not from here.
ENGINES: dict[str, ImageEngine] = {
    "klein": ImageEngine(
        id="klein", label="FLUX.2 Klein 9B", where="local", generate=True, edit=True, max_refs=6,
        lora_family="klein", checks=True, tag="klein", edit_tag="klein-edit",
    ),
    "krea2": ImageEngine(
        id="krea2", label="Krea 2", where="local", generate=True, edit=True, max_refs=2,
        lora_family="krea2", checks=True, tag="krea2", edit_tag="krea2-edit",
    ),
    "zimage": ImageEngine(
        id="zimage", label="Z-Image Turbo", where="local", generate=True, edit=False, max_refs=0,
        lora_family="zit", checks=True, tag="zimage",
    ),
    "turbo": ImageEngine(
        id="turbo", label="z-image/turbo (Atlas)", where="atlas", generate=True, edit=False, max_refs=0,
        lora_family="", checks=False, atlas_model=IMAGE_FAST_MODEL, price_key="z-image", tag="z-image",
    ),
    "seedream": ImageEngine(
        id="seedream", label="Seedream v5.0 Pro (Atlas)", where="atlas", generate=True, edit=True, max_refs=10,
        lora_family="", checks=False, atlas_model=IMAGE_MODEL, atlas_edit_model=IMAGE_EDIT_MODEL,
        price_key="pro-1.5k", tag="seedream",
    ),
    "seedream-lite": ImageEngine(
        id="seedream-lite", label="Seedream v5.0 Lite (Atlas)", where="atlas", generate=True, edit=True, max_refs=10,
        lora_family="", checks=False, atlas_model=IMAGE_LITE_MODEL, atlas_edit_model=IMAGE_LITE_EDIT_MODEL,
        price_key="lite", tag="seedream",
    ),
}

#: Every spelling a caller may send as ``engine``, mapped to a canonical id.
#:
#: ``z-image`` is the one deliberate break: it used to mean the Atlas engine, and now means the local one,
#: because Z-Image Turbo runs on this GPU for free. ``model="z-image"`` is a different field and still means
#: the Atlas model — see IMAGE_ALIASES in jobs.py.
ALIASES = {
    "local": "krea2", "krea": "krea2", "krea2": "krea2", "krea-2": "krea2",
    "klein": "klein", "flux": "klein", "flux2": "klein", "flux-2": "klein",
    "z-image": "zimage", "zimage": "zimage", "zit": "zimage", "z-image-local": "zimage",
    "turbo": "turbo", "fast": "turbo", "cheap": "turbo", "z-image-turbo": "turbo", "z-image/turbo": "turbo",
    "seedream": "seedream", "quality": "seedream", "best": "seedream",
    "seedream-lite": "seedream-lite", "lite": "seedream-lite",
}

#: Engine ids whose meaning changed, and the note to attach so the change is visible rather than silent.
MOVED = {
    "z-image": "engine 'z-image' now means the local Z-Image Turbo on this GPU; use 'turbo' for the Atlas one.",
    "zimage": "engine 'z-image' now means the local Z-Image Turbo on this GPU; use 'turbo' for the Atlas one.",
}

#: LoRA families: one ComfyUI models/loras folder holds all of them, and a file from the wrong family
#: produces garbage rather than an error, so every file is classified before it reaches a graph.
#: Each entry is (label, filename prefixes).
LORA_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "h3": ("MiniMax H3 video", ("minimax_h3", "h3_", "hmnsfw", "hmbreasts", "hmpenis", "mysticxxx_mmh3", "synthpussy_h3")),
    "krea2": ("Krea 2", ("krea2", "snofs_krea2", "snofs_photodetail")),
    "klein": ("FLUX.2 Klein", ("klein_",)),
    "zit": ("Z-Image Turbo", ("zit_",)),
}

#: The families that belong to images; everything else is a video LoRA.
IMAGE_FAMILIES = frozenset({"krea2", "klein", "zit"})

#: Tags written before engines had ids, so an asset made by an older build can still be traced back.
_TAG_IDS = {"krea2": "krea2", "krea2-edit": "krea2", "z-image": "turbo", "seedream": "seedream", "atlas": "seedream"}


def get(engine_id: str) -> ImageEngine | None:
    return ENGINES.get(engine_id)


def resolve(name: str) -> str:
    """A caller's ``engine`` string as a canonical id, or "" when it is not one of ours ("auto" included)."""
    key = (name or "").strip().lower()
    return ALIASES.get(key, key if key in ENGINES else "")


def family_of(filename: str, declared: str = "") -> str:
    """Which model family a LoRA file belongs to.

    A ``family`` declared in the catalogue wins; then the filename prefix; then anything unrecognised is a
    video LoRA, because that is the side that refuses unknown names instead of silently loading them.
    """
    if declared and declared in LORA_FAMILIES:
        return declared
    base = filename.rsplit("/", 1)[-1].lower()
    for family in IMAGE_FAMILIES:
        if any(base.startswith(prefix) for prefix in LORA_FAMILIES[family][1]):
            return family
    return "h3"


def family_label(family: str) -> str:
    entry = LORA_FAMILIES.get(family)
    return entry[0] if entry else family


def id_for_tag(tag: str) -> str:
    """The engine id behind an asset tag written before ``source["engine"]`` existed."""
    return _TAG_IDS.get((tag or "").strip().lower(), "")


def id_for_generator(generator: str) -> str:
    """The engine id behind an asset's model name, for assets made before ``source["engine"]`` existed.

    Only three engines could have written one, so the old guess still holds for them.
    """
    name = (generator or "").strip().lower()
    if not name:
        return ""
    if name.startswith("krea2"):
        return "krea2"
    if name.startswith("z-image"):  # at the time, z-image only ever meant the Atlas engine
        return "turbo"
    return "seedream"


def order(stored: list[dict] | None, action: str = "generate") -> list[str]:
    """Enabled engine ids for this action, best first.

    Engines missing from the stored list are appended disabled, so an engine shipped after a pod's settings
    were written is offered in Studio rather than silently active — the same idea as loras._with_new_defaults.
    """
    seen, result = set(), []
    for row in stored or []:
        engine_id = resolve(str((row or {}).get("engine") or ""))
        if not engine_id or engine_id in seen:
            continue
        seen.add(engine_id)
        if (row or {}).get("enabled", True) and ENGINES[engine_id].supports(action):
            result.append(engine_id)
    return result


def full_order(stored: list[dict] | None, action: str = "generate") -> list[dict]:
    """Every engine that can do this action, in stored order, each with its enabled flag."""
    rows, seen = [], set()
    for row in stored or []:
        engine_id = resolve(str((row or {}).get("engine") or ""))
        if not engine_id or engine_id in seen:
            continue
        seen.add(engine_id)
        if ENGINES[engine_id].supports(action):
            rows.append({"engine": engine_id, "enabled": bool((row or {}).get("enabled", True))})
    rows += [{"engine": e.id, "enabled": False} for e in ENGINES.values() if e.id not in seen and e.supports(action)]
    return rows
