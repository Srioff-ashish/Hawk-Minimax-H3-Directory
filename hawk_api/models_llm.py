"""Which model id answers for a role, on whichever provider is switched on right now.

Every role used to hold one bare model id, and an id belongs to a provider. ``VISION_FALLBACK_MODELS`` was
the visible casualty -- Atlas spells Grok ``xai/grok-4.6`` and OpenRouter spells it ``x-ai/grok-4.6``, so on
OpenRouter the vision ladder 404ed and image inspection was left with only the chat's own model, which for a
prose finetune cannot see at all. The director, summary and planner ids have exactly the same defect; they
only looked healthy because Atlas was the default. Flipping one setting made five settings wrong.

The fix is not a table of equivalent ids per provider: that has to be maintained by hand for every model on
every service, and it goes stale the same way the hardcoded tuple did. Instead a role carries an ordered
**chain** of candidates and they are resolved against the catalogue the provider actually serves, matching on
the part of the id after the vendor prefix. A chain is also how a role gets a cross-provider fallback with no
new mechanism: ``anthracite-org/magnum-v4-72b, xai/grok-4.6`` is Magnum on OpenRouter and Grok on Atlas,
because on Atlas the first candidate resolves to nothing.

Nothing here does I/O -- a catalogue is handed in, in the shape ``AtlasClient.list_models`` returns -- so
``agent.py``, ``jobs.py`` and the settings store can all import it without a cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: How many candidates a vision ladder may try. A model that refuses to look at an image still bills for
#: being asked, so an unbounded ladder turns one refusal into a charge per installed model.
MAX_VISION_CANDIDATES = 3


@dataclass(frozen=True)
class Role:
    """One job a model is asked to do, and what it takes to do it.

    ``chain`` is the built-in fallback used when the setting for this role is blank. It is not "the best
    models" -- it is the shortest list that keeps the role working on either provider.
    """

    chain: tuple[str, ...]
    vision: bool = False
    #: Shown next to the field in Studio, and in the report a settings save returns.
    label: str = ""


ROLES: dict[str, Role] = {
    # The tail every chat falls back on, whatever model the chat itself names: a chat set to deepseek-v4-pro
    # (which can spend its whole budget reasoning and return nothing) tries the same family's flash model
    # before paying grok's price.
    "director": Role(chain=("deepseek-ai/deepseek-v4.1-flash", "xai/grok-4.6", "xai/grok-4.3"),
                     label="Director (tools and orchestration)"),
    # Deliberately the same chain as the director: turning the split on must not, by itself, change any
    # existing chat's behaviour. The uncensored chain is something the user opts into.
    "prose": Role(chain=("deepseek-ai/deepseek-v4.1-flash", "xai/grok-4.6", "xai/grok-4.3"),
                  label="Prose (the characters' voices)"),
    "summary": Role(chain=("deepseek-ai/deepseek-v4.1-flash", "xai/grok-4.3"), label="Summary and compaction"),
    "vision": Role(chain=("xai/grok-4.6", "google/gemini-pro-1.5"), vision=True, label="Image inspection"),
    "planner": Role(chain=("xai/grok-4.6", "xai/grok-4.3"), label="Film planner"),
}


def chain(value) -> list[str]:
    """A model setting read as an ordered list of candidates.

    Settings are comma-separated so that a fallback needs no new field, and so that every id already stored
    keeps working: a single id is simply a one-element chain.
    """
    if isinstance(value, (list, tuple)):
        items = [str(item or "").strip() for item in value]
    else:
        items = [part.strip() for part in str(value or "").split(",")]
    seen: dict[str, None] = {}
    for item in items:
        if item:
            seen.setdefault(item, None)
    return list(seen)


def slug(model_id: str) -> str:
    """An id without its vendor prefix, for comparing the same model across two providers.

    ``xai/grok-4.6`` and ``x-ai/grok-4.6`` are one model under two spellings, and so are ``deepseek-ai/...``
    and ``deepseek/...``. Only the last path segment is kept, because that is the part the providers agree on;
    any ``:free`` or ``:nitro`` variant suffix is dropped so a chain written without one still matches.
    """
    text = str(model_id or "").strip().lower()
    text = text.split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "", text.split(":")[0])


def _usable(entry: dict, want_vision: bool) -> bool:
    return bool(entry.get("id")) and (entry.get("vision") is True or not want_vision)


def resolve(role: str, setting, catalogue: list[dict] | None, *, vision: bool = False) -> str:
    """The id to send for ``role``: the first candidate this provider actually serves.

    ``setting`` is the configured chain (blank falls back to the role's own), ``catalogue`` is what
    ``AtlasClient.list_models`` returned. Returns "" only when there is no candidate anywhere, which callers
    treat as "this role is unavailable" rather than sending a guess.

    ``vision`` adds the image requirement to a role that does not always carry it. The planner is the case:
    it is a text role until the plan has reference photos, and then the very same call is multimodal.
    """
    return (resolve_many(role, setting, catalogue, vision=vision) or [""])[0]


def resolve_many(role: str, setting, catalogue: list[dict] | None, *, vision: bool = False) -> list[str]:
    """Every id worth trying for ``role``, best first.

    Only the vision role has more than one caller-visible rung today -- ``_inspect_images`` walks the list
    until a model agrees to look -- but returning a list keeps that from being a special case.
    """
    spec = ROLES.get(role) or Role(chain=())
    # A role can need vision for one call and not the next, so the requirement is the role's OR the
    # caller's. Resolving a multimodal call against a text-only model is not a quality problem: the
    # provider rejects the request outright ("No endpoints found that support image input").
    want_vision = bool(vision or spec.vision)
    wanted = chain(setting) or list(spec.chain)
    # No catalogue is not the same as an empty catalogue. A provider outage, a missing key or a /models call
    # that raised must never be the reason a call does not happen, so the configured id goes out unchanged and
    # behaviour is exactly what it was before this module existed.
    if not catalogue:
        return wanted or list(spec.chain)

    listed = [entry for entry in catalogue if _usable(entry, want_vision)]
    by_id = {str(entry["id"]): entry for entry in listed}
    by_slug: dict[str, list[str]] = {}
    for model_id in by_id:
        by_slug.setdefault(slug(model_id), []).append(model_id)

    found: list[str] = []
    for candidate in wanted + [c for c in spec.chain if c not in wanted]:
        if candidate in by_id:
            found.append(candidate)
            continue
        matches = by_slug.get(slug(candidate)) or []
        # Two listed models sharing a slug name neither: the same rule cast_talk.mentioned_all uses for a
        # first name two characters share. A wrong model is worse than falling through to the next candidate.
        if len(matches) == 1:
            found.append(matches[0])

    if not found:
        # Nothing configured is served here. Rather than fail the feature, take the cheapest model that can do
        # the job -- which for the vision role is the old hardcoded fallback, generalised.
        cheapest = sorted(listed, key=lambda entry: (float(entry.get("price_in") or 0.0), str(entry["id"])))
        found = [str(entry["id"]) for entry in cheapest[:1]]

    ordered = list(dict.fromkeys(found))
    return ordered[:MAX_VISION_CANDIDATES] if want_vision else ordered


def report(settings: dict, catalogue: list[dict] | None) -> dict:
    """Per role: what is configured, what it resolved to here, and whether that was a real match.

    Returned when the LLM settings are saved. Switching provider silently re-points five roles, and without
    this the first sign of a bad switch is a render failing much later; ``fallback`` is the flag worth
    reading, because it means nothing configured for that role is served by this provider.
    """
    out = {}
    for role, spec in ROLES.items():
        configured = chain(settings.get(role, ""))
        resolved = resolve_many(role, configured, catalogue)
        asked = configured or list(spec.chain)
        first = resolved[0] if resolved else ""
        out[role] = {
            "label": spec.label,
            "configured": asked,
            "resolved": resolved,
            "listed": bool(catalogue),
            # Whether what this role will actually send is a model the user asked for. False means their own
            # setting named nothing this provider serves -- worth showing even when the built-in chain
            # rescued the role, because otherwise a typo or a provider-only model looks like it took effect.
            "honoured": bool(first) and any(slug(first) == slug(a) for a in asked),
            # True only when neither the configured chain nor the role's own matched and the cheapest listed
            # model was taken instead: the role still works, but on a model nobody chose.
            "fallback": bool(catalogue) and bool(first)
            and not any(slug(first) == slug(a) for a in asked + list(spec.chain)),
        }
    return out
