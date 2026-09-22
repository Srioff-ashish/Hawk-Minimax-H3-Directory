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

#: Mirrors local_images.DEFAULT_ADULT_LORAS so view() can show what a family falls back to. Kept here as
#: plain names to avoid importing local_images, which imports this module.
DEFAULT_ADULT_LORAS: dict[str, tuple[str, ...]] = {
    "krea2": ("snofs_krea2.safetensors", "krea2_mystic_xxx_v3.safetensors"),
    "klein": ("klein_snofs.safetensors", "klein_nsfw_no_face_change.safetensors"),
    "zit": ("zit_mystic_xxx.safetensors",),
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


#: Folder names under ComfyUI's ``models/loras`` that mean a family, besides the family id itself.
FAMILY_FOLDERS: dict[str, tuple[str, ...]] = {
    "h3": ("minimax_h3", "minimax-h3", "video"),
    "krea2": ("krea-2", "krea"),
    "klein": ("flux2", "flux-2", "flux2_klein"),
    "zit": ("zimage", "z_image", "z-image"),
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
        {"engine": "klein", "enabled": True},
        {"engine": "krea2", "enabled": True},
        {"engine": "zimage", "enabled": True},
        # below all three local engines, so it only ever fires when none of them can run -- and then it is
        # a third of Seedream's price for the same job
        {"engine": "turbo", "enabled": True},
        {"engine": "seedream", "enabled": True},
        {"engine": "seedream-lite", "enabled": False},
    ],
    "edit": [
        {"engine": "klein", "enabled": True},  # the only local engine that takes more than two references
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

    def warnings(self, data: dict | None = None) -> list[str]:
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
        return found

    def view(self) -> dict:
        data = self.settings()
        shown = {family: (self.defaults(family) if self.defaults(family) is not None
                          else [{"name": n, "strength": 0.8} for n in DEFAULT_ADULT_LORAS.get(family, ())])
                 for family in sorted(IMAGE_FAMILIES)}
        return {**data, "warnings": self.warnings(data), "defaults": shown,
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
            if family_of(name) != family:
                raise SettingsError(f"{name!r} is not a {family_label(family)} LoRA, so it can't be one of its defaults.")
            try:
                strength = float((entry or {}).get("strength", 0.8))
            except (TypeError, ValueError):
                raise SettingsError(f"{name!r} needs a number for strength.") from None
            if not 0.0 <= strength <= 2.0:
                raise SettingsError(f"{name!r}: strength should be between 0 and 2.")
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
