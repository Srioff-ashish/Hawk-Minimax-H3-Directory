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

import json
import os
import threading
import time
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
    "qwen21": ImageEngine(
        id="qwen21", label="Qwen Image 2.1", where="local", generate=True, edit=True, max_refs=16,
        lora_family="qwen21", checks=True, tag="qwen21", edit_tag="qwen21-edit",
    ),
    "krea2": ImageEngine(
        id="krea2", label="Krea 2", where="local", generate=True, edit=True, max_refs=2,
        lora_family="krea2", checks=True, tag="krea2", edit_tag="krea2-edit",
    ),
    "zimage": ImageEngine(
        id="zimage", label="Z-Image Turbo", where="local", generate=True, edit=False, max_refs=0,
        lora_family="zit", checks=True, tag="zimage",
    ),
    "chroma": ImageEngine(
        id="chroma", label="Chroma1-HD", where="local", generate=True, edit=False, max_refs=0,
        lora_family="chroma", checks=True, tag="chroma",
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
    "qwen21": "qwen21", "qwen": "qwen21", "qwen2.1": "qwen21", "qwen-image-2.1": "qwen21",
    "qwen_image_2.1": "qwen21", "qwen-image": "qwen21",
    # Klein's spellings are kept pointing at its replacement so assets, pinned engines and retakes made
    # before the swap still resolve; MOVED says so out loud rather than letting it look like Klein ran.
    "klein": "qwen21", "flux": "qwen21", "flux2": "qwen21", "flux-2": "qwen21",
    "z-image": "zimage", "zimage": "zimage", "zit": "zimage", "z-image-local": "zimage",
    "chroma": "chroma", "chroma1": "chroma", "chroma1-hd": "chroma", "chroma-hd": "chroma",
    "chroma1hd": "chroma", "chroma1_hd": "chroma",
    "turbo": "turbo", "fast": "turbo", "cheap": "turbo", "z-image-turbo": "turbo", "z-image/turbo": "turbo",
    "seedream": "seedream", "quality": "seedream", "best": "seedream",
    "seedream-lite": "seedream-lite", "lite": "seedream-lite",
}

#: Engine ids whose meaning changed, and the note to attach so the change is visible rather than silent.
MOVED = {
    "z-image": "engine 'z-image' now means the local Z-Image Turbo on this GPU; use 'turbo' for the Atlas one.",
    "zimage": "engine 'z-image' now means the local Z-Image Turbo on this GPU; use 'turbo' for the Atlas one.",
    "klein": "FLUX.2 Klein has been replaced by Qwen Image 2.1; engine 'klein' now runs Qwen.",
    "flux": "FLUX.2 Klein has been replaced by Qwen Image 2.1; engine 'flux' now runs Qwen.",
    "flux2": "FLUX.2 Klein has been replaced by Qwen Image 2.1; engine 'flux2' now runs Qwen.",
    "flux-2": "FLUX.2 Klein has been replaced by Qwen Image 2.1; engine 'flux-2' now runs Qwen.",
}

#: LoRA families: one ComfyUI models/loras folder holds all of them, and a file from the wrong family
#: produces garbage rather than an error, so every file is classified before it reaches a graph.
#: Each entry is (label, filename prefixes).
LORA_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "h3": ("MiniMax H3 video", ("minimax_h3", "h3_", "hmnsfw", "hmbreasts", "hmpenis", "mysticxxx_mmh3", "synthpussy_h3")),
    "krea2": ("Krea 2", ("krea2", "snofs_krea2", "snofs_photodetail")),
    # Qwen 2.1 LoRAs are published under the author's own names rather than a common prefix, so these cover
    # the catalogued ones and FAMILY_FOLDERS covers the rest -- several ship with spaces or CJK in the name.
    "qwen21": ("Qwen Image 2.1", ("qwen21_", "qwen_image_2.1", "qwen-image-2.1", "qwen2.1", "qwen image2.1",
                                  "lenovo_qwen21", "pornmaster_qi2.1", "elusarcas-qwen2-1")),
    "zit": ("Z-Image Turbo", ("zit_",)),
    # Chroma LoRAs are published under a dozen different names; the "chroma" prefix covers the ones that
    # carry it and models/loras/chroma covers the rest, the same arrangement Qwen 2.1 needs.
    "chroma": ("Chroma1-HD", ("chroma",)),
}

#: Mirrors local_images.DEFAULT_ADULT_LORAS so view() can show what a family falls back to. Kept here as
#: plain names to avoid importing local_images, which imports this module.
DEFAULT_ADULT_LORAS: dict[str, tuple[str, ...]] = {
    "krea2": ("snofs_krea2.safetensors", "krea2_mystic_xxx_v3.safetensors"),
    "qwen21": ("NSFW Qwen Lora.safetensors",),
    "zit": ("zit_mystic_xxx.safetensors",),
}

#: Attached to every image from this family, whatever else the request asks for -- unlike the adult
#: defaults above, which step aside when the request names its own adult LoRA. A model that needs a repair
#: LoRA to produce its advertised quality should not depend on the user remembering to ask for it.
#: Mirrored in local_images.BASE_LORAS for the same reason as DEFAULT_ADULT_LORAS.
BASE_LORAS: dict[str, tuple[tuple[str, float], ...]] = {
    "qwen21": (("qwen-image-2.1-fix-1.0-comfy.safetensors", 1.0),),
}

#: The families that belong to images; everything else is a video LoRA.
IMAGE_FAMILIES = frozenset({"krea2", "qwen21", "zit", "chroma"})

#: Tags written before engines had ids, so an asset made by an older build can still be traced back.
_TAG_IDS = {"krea2": "krea2", "krea2-edit": "krea2", "z-image": "turbo", "seedream": "seedream", "atlas": "seedream"}


def get(engine_id: str) -> ImageEngine | None:
    return ENGINES.get(engine_id)


def resolve(name: str) -> str:
    """A caller's ``engine`` string as a canonical id, or "" when it is not one of ours ("auto" included)."""
    key = (name or "").strip().lower()
    return ALIASES.get(key, key if key in ENGINES else "")


#: Folder names under ComfyUI's ``models/loras`` that mean a family, besides the family id itself.
FAMILY_FOLDERS: dict[str, tuple[str, ...]] = {
    "h3": ("minimax_h3", "minimax-h3", "video"),
    "krea2": ("krea-2", "krea"),
    # The folder is how a Qwen 2.1 LoRA whose own name says nothing -- "NSFW Qwen Lora.safetensors",
    # "qwen2.1角色卡-4.safetensors" -- gets classified without being renamed first.
    "qwen21": ("qwen-image-2.1", "qwen_image_2.1", "qwenimage21", "qwen2.1", "qwen"),
    "zit": ("zimage", "z_image", "z-image"),
    "chroma": ("chroma1", "chroma1-hd", "chroma1_hd", "chroma-hd", "chroma1hd"),
}


def family_of(filename: str, declared: str = "") -> str:
    """Which model family a LoRA file belongs to.

    A ``family`` declared in the catalogue wins; then the filename prefix; then anything unrecognised is a
    video LoRA, because that is the side that refuses unknown names instead of silently loading them.
    """
    if declared and declared in LORA_FAMILIES:
        return declared
    head, _, base = filename.replace("\\", "/").rpartition("/")
    base = base.lower()
    # The file name wins: it is the publisher's own label, so "klein_snofs" in a Krea 2 folder is a misfiled
    # Klein LoRA, not a Krea 2 one. Only a name that says nothing falls through to the folder.
    for family in IMAGE_FAMILIES:
        if any(base.startswith(prefix) for prefix in LORA_FAMILIES[family][1]):
            return family
    # A folder named after a family classifies what it holds, so downloads keep the names they were published
    # under instead of having to be renamed to carry a prefix.
    folder = head.rsplit("/", 1)[-1].lower()
    for family in LORA_FAMILIES:
        if folder == family or folder in FAMILY_FOLDERS.get(family, ()):
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


#: How the ladders start out on a pod that has never been configured. Local engines first, cheapest of the
#: paid ones last. An engine that is enabled but not installed reports ready: false in image_options and is
#: skipped by the walk, so listing all three local engines here costs a pod that lacks one nothing.
DEFAULTS = {
    "generate": [
        {"engine": "qwen21", "enabled": True},
        {"engine": "krea2", "enabled": True},
        {"engine": "zimage", "enabled": True},
        # Off unless asked for by name. Chroma samples at cfg 3.8 with a real negative prompt and has an
        # aesthetic of its own, so having "auto" fall onto it would change images nobody asked to change.
        # Naming it as the engine pins it whatever this says, which is what makes it an option rather than a rung.
        {"engine": "chroma", "enabled": False},
        # below all three local engines, so it only ever fires when none of them can run -- and then it is
        # a third of Seedream's price for the same job
        {"engine": "turbo", "enabled": True},
        {"engine": "seedream", "enabled": True},
        {"engine": "seedream-lite", "enabled": False},
    ],
    "edit": [
        {"engine": "qwen21", "enabled": True},  # 16 references, so a group shot no longer falls onto paid Seedream
        {"engine": "krea2", "enabled": True},
        {"engine": "seedream", "enabled": True},
        {"engine": "seedream-lite", "enabled": False},
    ],
    "busy": {"mode": "fall_through", "max_wait_seconds": 120},
    #: A retake that has exhausted the free engines asks before spending on a paid one.
    "confirm_paid": True,
    #: Inspection rejecting a take stops and shows it, instead of the agent silently taking it again.
    "pick_takes": True,
}
BUSY_MODES = ("wait", "fall_through")
MAX_WAIT_SECONDS = 300  # matches local_images.WAIT_SECONDS: waiting longer than one render is pointless
#: Ceiling on a stored default's strength. 2.0 covers the content LoRAs; slider-style ones (the Qwen 2.1
#: age slider runs to 3) are meant to be pushed past it, so the cap is theirs rather than the common case's.
MAX_DEFAULT_STRENGTH = 3.0


class SettingsError(ValueError):
    """A ladder the gateway would not be able to use. Raised to the caller as a 422."""


class ImageEngineStore:
    """``DATA_DIR/image_engines.json``: which engines are tried, in what order, and what a busy GPU means.

    Read on every request, so an edit in Studio applies to the next image without a restart.
    """

    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "image_engines.json")
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp, self.path)

    def settings(self) -> dict:
        """Stored ladders, with any engine the file predates appended switched off."""
        stored = self._load()
        busy = dict(DEFAULTS["busy"])
        if isinstance(stored.get("busy"), dict):
            if stored["busy"].get("mode") in BUSY_MODES:
                busy["mode"] = stored["busy"]["mode"]
            try:
                busy["max_wait_seconds"] = max(0, min(MAX_WAIT_SECONDS, int(stored["busy"]["max_wait_seconds"])))
            except (KeyError, TypeError, ValueError):
                pass
        return {
            "generate": full_order(stored.get("generate") or DEFAULTS["generate"], "generate"),
            "edit": full_order(stored.get("edit") or DEFAULTS["edit"], "edit"),
            "busy": busy,
            "confirm_paid": self.confirm_paid(),
            "pick_takes": self.pick_takes(),
        }

    def confirm_paid(self) -> bool:
        """Whether a failed take must be approved before it retries on a paid engine."""
        stored = self._load().get("confirm_paid")
        return DEFAULTS["confirm_paid"] if not isinstance(stored, bool) else stored

    def pick_takes(self) -> bool:
        """Whether a rejected take is put to the user before anything is generated again."""
        stored = self._load().get("pick_takes")
        return DEFAULTS["pick_takes"] if not isinstance(stored, bool) else stored

    def defaults(self, family: str) -> list[dict] | None:
        """LoRAs this family attaches on its own, or None when the user has never set them.

        None means "use the shipped fallback"; an empty list means the user switched them off, which is a
        different thing and must not be confused with it.
        """
        stored = (self._load().get("defaults") or {}).get(family)
        if not isinstance(stored, list):
            return None
        return [{"name": str(e.get("name") or ""), "strength": float(e.get("strength", 0.8))}
                for e in stored if isinstance(e, dict) and e.get("name")]

    def order(self, action: str = "generate") -> list[str]:
        """The enabled engine ids for this action, best first."""
        return order(self.settings().get(action), action)

    def wait_seconds(self) -> float:
        busy = self.settings()["busy"]
        return float(busy["max_wait_seconds"]) if busy["mode"] == "wait" else 0.0

    def warnings(self, data: dict | None = None, installed: set[str] | None = None) -> list[str]:
        """What is wrong with this ladder. ``installed`` is the LoRA basenames on the pod, when the caller
        has them: a base LoRA that is configured but missing attaches to nothing and says nothing, so the
        only place a typo in its name can surface is here."""
        data = data or self.settings()
        found = []
        for action in ("generate", "edit"):
            live = [ENGINES[row["engine"]] for row in data[action] if row["enabled"]]
            if not live:
                continue  # save() refuses this; an older file could still hold it
            if not any(engine.local for engine in live):
                found.append(f"No local engine is on for {action}: every image will be billed to Atlas.")
            elif all(engine.local for engine in live):
                found.append(f"Only local engines are on for {action}: it fails when ComfyUI is busy or down.")
        found += self.missing_base_loras(installed)
        return found

    @staticmethod
    def missing_base_loras(installed: set[str] | None) -> list[str]:
        """One warning per always-on LoRA that is not on the pod. Empty when the caller has no listing."""
        if installed is None:
            return []
        stems = {str(name).rsplit("/", 1)[-1].lower().removesuffix(".safetensors") for name in installed}
        return [f"{family_label(family)} always attaches {name!r}, which is not installed: every image from it "
                "is missing that LoRA. Put the file in models/loras or correct the name."
                for family, entries in BASE_LORAS.items() for name, _ in entries
                if name.rsplit("/", 1)[-1].lower().removesuffix(".safetensors") not in stems]

    def view(self, installed: set[str] | None = None) -> dict:
        data = self.settings()
        shown = {family: (self.defaults(family) if self.defaults(family) is not None
                          else [{"name": n, "strength": 0.8} for n in DEFAULT_ADULT_LORAS.get(family, ())])
                 for family in sorted(IMAGE_FAMILIES)}
        # Base LoRAs are not editable here -- they are shown so the panel can say what is already attached,
        # rather than leaving the user to wonder why a LoRA they never chose is in every result.
        base = {family: [{"name": n, "strength": s} for n, s in BASE_LORAS.get(family, ())]
                for family in sorted(IMAGE_FAMILIES)}
        return {**data, "warnings": self.warnings(data, installed), "defaults": shown, "base": base,
                "families": {f: family_label(f) for f in sorted(IMAGE_FAMILIES)}, "engines": [
            {"id": e.id, "label": e.label, "where": e.where, "generate": e.generate, "edit": e.edit,
             "max_refs": e.max_refs, "lora_family": e.lora_family, "price_key": e.price_key}
            for e in ENGINES.values()]}

    @staticmethod
    def _clean(rows, action: str) -> list[dict]:
        cleaned, seen = [], set()
        for row in rows:
            name = str((row or {}).get("engine") or "")
            engine_id = resolve(name)
            if not engine_id:
                raise SettingsError(f"No image engine {name!r}; use one of: {', '.join(ENGINES)}.")
            if engine_id in seen:
                raise SettingsError(f"{ENGINES[engine_id].label} is listed twice in the {action} order.")
            seen.add(engine_id)
            if not ENGINES[engine_id].supports(action):
                # say which one, rather than dropping it and leaving the user wondering where it went
                raise SettingsError(f"{ENGINES[engine_id].label} can't {action} images, so it can't be in that order.")
            cleaned.append({"engine": engine_id, "enabled": bool((row or {}).get("enabled", True))})
        if not any(row["enabled"] for row in cleaned):
            raise SettingsError(f"Leave at least one engine on for {action}, or no image can be made.")
        return cleaned

    def _clean_defaults(self, family: str, entries) -> list[dict]:
        if family not in IMAGE_FAMILIES:
            raise SettingsError(f"No image model family {family!r}; use one of: {', '.join(sorted(IMAGE_FAMILIES))}.")
        cleaned = []
        for entry in entries:
            name = str((entry or {}).get("name") or "").strip()
            if not name:
                raise SettingsError(f"Give a LoRA file name for {family_label(family)}.")
            # Refuse a name that positively belongs to another image family; accept one whose name says
            # nothing. Several published Qwen 2.1 LoRAs carry no usable prefix ("NSFW Qwen Lora.safetensors"),
            # and the folder that classifies them on disk is not part of the name the panel round-trips. What
            # actually protects a render is resolve_loras, which only ever loads installed files of the
            # engine's own family -- this check is here to catch an obvious mix-up, not to be the gate.
            found = family_of(name)
            if found != family and found in IMAGE_FAMILIES:
                raise SettingsError(f"{name!r} is a {family_label(found)} LoRA, so it can't be one of "
                                    f"{family_label(family)}'s defaults.")
            try:
                strength = float((entry or {}).get("strength", 0.8))
            except (TypeError, ValueError):
                raise SettingsError(f"{name!r} needs a number for strength.") from None
            if not 0.0 <= strength <= MAX_DEFAULT_STRENGTH:
                raise SettingsError(f"{name!r}: strength should be between 0 and {MAX_DEFAULT_STRENGTH:g}.")
            cleaned.append({"name": name, "strength": strength})
        return cleaned

    def save(self, *, generate=None, edit=None, busy_mode=None, busy_max_wait_seconds=None, defaults=None,
             confirm_paid=None, pick_takes=None) -> dict:
        """Update the parts that were given. Anything left as None keeps its current value."""
        with self._lock:
            current = self.settings()
            if generate is not None:
                current["generate"] = self._clean(generate, "generate")
            if edit is not None:
                current["edit"] = self._clean(edit, "edit")
            if busy_mode is not None:
                if busy_mode not in BUSY_MODES:
                    raise SettingsError(f"busy mode should be {' or '.join(BUSY_MODES)}.")
                current["busy"]["mode"] = busy_mode
            if busy_max_wait_seconds is not None:
                current["busy"]["max_wait_seconds"] = max(0, min(MAX_WAIT_SECONDS, int(busy_max_wait_seconds)))
            if current["busy"]["mode"] == "wait" and not current["busy"]["max_wait_seconds"]:
                current["busy"]["mode"] = "fall_through"  # waiting zero seconds is falling through
            if confirm_paid is not None:
                current["confirm_paid"] = bool(confirm_paid)
            if pick_takes is not None:
                current["pick_takes"] = bool(pick_takes)
            stored_defaults = dict(self._load().get("defaults") or {})
            for family, entries in (defaults or {}).items():
                stored_defaults[family] = self._clean_defaults(family, entries or [])
            self._write({**current, "defaults": stored_defaults, "updated_at": time.time()})
        return self.view()
