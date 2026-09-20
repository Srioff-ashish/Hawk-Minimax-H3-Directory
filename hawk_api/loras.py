"""Which LoRAs a render gets, checked against what is really in ComfyUI's models/loras.

Pure Python: the live file list is passed in (the service fetches it from ComfyUI's
``GET /models/loras``), so every rule here is unit-tested without a pod.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
from dataclasses import dataclass

EXAMPLE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "loras.example.json")


class LoraError(ValueError):
    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass
class LoraSpec:
    name: str
    strength: float = 1.0
    required: bool = False
    turbo: bool = False

    @classmethod
    def from_dict(cls, data, where: str) -> "LoraSpec":
        if isinstance(data, str):
            data = {"name": data}
        if not isinstance(data, dict) or not str(data.get("name", "")).strip():
            raise LoraError(f"{where}: every LoRA entry needs a name.")
        return cls(
            str(data["name"]).strip(),
            float(data.get("strength", 1.0)),
            bool(data.get("required", False)),
            bool(data.get("turbo", False)),
        )


@dataclass
class LoraConfig:
    defaults: list[LoraSpec]
    presets: dict[str, list[LoraSpec]]


@dataclass
class ResolvedLora:
    requested: str
    file: str
    strength: float
    turbo: bool
    source: str


def parse_config(data, where: str = "loras.json") -> LoraConfig:
    if not isinstance(data, dict):
        raise LoraError(f"{where} must be a JSON object with 'defaults' and 'presets'.")
    defaults = [LoraSpec.from_dict(item, f"{where} defaults") for item in data.get("defaults") or []]
    presets = {
        str(name): [LoraSpec.from_dict(item, f"{where} preset {name!r}") for item in items or []]
        for name, items in (data.get("presets") or {}).items()
    }
    return LoraConfig(defaults, presets)


def load_config(path: str) -> LoraConfig:
    """Read loras.json, creating it from deploy/loras.example.json on first use.
    Read on every request so edits apply without restarting the gateway. A newer catalogue shipped with the code
    (a higher "version") replaces the copy, keeping the old one as loras.json.bak, so new defaults reach pods that
    already have a loras.json. This mirrors the image catalogue in local_images.load_catalogue."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        shutil.copyfile(EXAMPLE_CONFIG, path)
    elif os.path.abspath(path) != os.path.abspath(EXAMPLE_CONFIG):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            with open(EXAMPLE_CONFIG, "r", encoding="utf-8") as handle:
                example = json.load(handle)
            if int(example.get("version") or 1) > int(stored.get("version") or 1):
                shutil.copyfile(path, path + ".bak")
                shutil.copyfile(EXAMPLE_CONFIG, path)
        except (OSError, ValueError, TypeError):
            pass  # a broken or hand-written file is reported by the read below
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise LoraError(f"{path} is not valid JSON: {exc}") from None
    return parse_config(data, path)


def _normalise(path: str) -> str:
    return path.replace("\\", "/")


def _stem(path: str) -> str:
    base = _normalise(path).rsplit("/", 1)[-1]
    return re.sub(r"\.(safetensors|pt|pth|ckpt|bin)$", "", base, flags=re.IGNORECASE).lower()


def resolve_name(name: str, available: list[str], *, label: str = "LoRA", folder: str = "loras") -> str:
    """Match a requested file (a LoRA by default) to a real one, in order: exact path,
    file name (any folder, any case, extension optional), then a unique substring."""
    wanted = _normalise(name.strip())
    originals = {_normalise(f): f for f in available}
    if wanted in originals:
        return originals[wanted]

    by_name = sorted(f for n, f in originals.items() if _stem(n) == _stem(wanted))
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise LoraError(
            f"{label} {name!r} matches several files: {', '.join(by_name)}. Use the full path.",
            details={"requested": name, "matches": by_name},
        )

    needle = wanted.lower()
    contains = sorted(f for n, f in originals.items() if needle in n.lower())
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        raise LoraError(
            f"{label} {name!r} matches several files: {', '.join(contains)}. Use a longer name or the full path.",
            details={"requested": name, "matches": contains},
        )

    stems = {_stem(n): f for n, f in originals.items()}
    close = [stems[s] for s in difflib.get_close_matches(_stem(wanted), list(stems), n=5, cutoff=0.4)]
    hint = f" Closest files: {', '.join(close)}." if close else " No similar files are in the folder."
    raise LoraError(
        f"{label} {name!r} is not in ComfyUI's models/{folder}.{hint}",
        details={"requested": name, "suggestions": close, "available_count": len(available)},
    )


def resolve_request(
    config: LoraConfig,
    available: list[str],
    *,
    loras: list[LoraSpec] = (),
    preset: str | None = None,
    use_defaults: bool = True,
) -> tuple[list[ResolvedLora], list[str]]:
    """Defaults, then the preset, then the request's own LoRAs. The same file listed
    twice keeps its first position and takes the later strength; strength 0 drops it
    (which is how a request switches off a default)."""
    warnings: list[str] = []
    chosen: dict[str, ResolvedLora] = {}

    def add(spec: LoraSpec, source: str) -> None:
        file = resolve_name(spec.name, available)
        turbo = spec.turbo or "turbo" in _stem(file)
        if file in chosen:
            entry = chosen[file]
            entry.strength, entry.source, entry.requested = spec.strength, source, spec.name
            entry.turbo = entry.turbo or turbo
        else:
            chosen[file] = ResolvedLora(spec.name, file, spec.strength, turbo, source)

    if use_defaults:
        for spec in config.defaults:
            try:
                add(spec, "default")
            except LoraError as exc:
                if spec.required:
                    raise LoraError(
                        f"Required default LoRA {spec.name!r} is missing on the pod, so the render was refused. "
                        f"{exc} Put the file in ComfyUI/models/loras, fix loras.json, or send use_default_loras=false.",
                        details={**exc.details, "required": True},
                    ) from None
                warnings.append(f"Default LoRA {spec.name!r} skipped: {exc}")

    if preset:
        if preset not in config.presets:
            names = ", ".join(sorted(config.presets)) or "none configured"
            raise LoraError(f"Unknown lora_preset {preset!r}. Presets: {names}.", details={"presets": sorted(config.presets)})
        for spec in config.presets[preset]:
            add(spec, f"preset:{preset}")

    for spec in loras:
        add(spec, "request")

    return [entry for entry in chosen.values() if entry.strength != 0.0], warnings


def default_status(config: LoraConfig, available: list[str]) -> list[dict]:
    rows = []
    for spec in config.defaults:
        row = {"name": spec.name, "strength": spec.strength, "required": spec.required, "turbo": spec.turbo}
        try:
            row.update(file=resolve_name(spec.name, available), present=True)
        except LoraError as exc:
            row.update(file=None, present=False, problem=str(exc))
        rows.append(row)
    return rows


def choose_steps(resolved: list[ResolvedLora], requested: int | None) -> tuple[int, str]:
    if requested:
        return int(requested), "set by the request"
    turbo = next((entry for entry in resolved if entry.turbo), None)
    if turbo:
        return 8, f"8 steps because turbo LoRA {turbo.file} is applied"
    return 30, "30 steps because no turbo LoRA is applied"


_APPLIED = re.compile(r"^(?P<name>.+?) @ (?P<strength>-?\d+(?:\.\d+)?(?:e-?\d+)?)")


def parse_applied(text: str) -> list[tuple[str, float]]:
    """Read Hawk H3 LoRA Stack's on-node summary ("file @ 0.8" lines)."""
    applied = []
    for line in (text or "").splitlines():
        match = _APPLIED.match(line.strip())
        if match:
            applied.append((match.group("name"), float(match.group("strength"))))
    return applied


def compare_applied(resolved: list[dict], applied: list[tuple[str, float]]) -> list[str]:
    expected = {entry["file"]: entry["strength"] for entry in resolved}
    actual = dict(applied)
    warnings = []
    for file, strength in expected.items():
        if file not in actual:
            warnings.append(f"LoRA {file} was requested but the LoRA Stack did not report applying it.")
        elif abs(actual[file] - strength) > 1e-3:
            warnings.append(f"LoRA {file} applied at {actual[file]:g} instead of {strength:g}.")
    for file in actual.keys() - expected.keys():
        warnings.append(f"LoRA {file} was applied but not requested.")
    return warnings
