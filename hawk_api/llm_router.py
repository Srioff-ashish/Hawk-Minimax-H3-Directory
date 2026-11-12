"""Chat calls across Atlas and OpenRouter as one service.

Every feature calls ``service.atlas.chat(model, ...)`` with a model id it was configured with. Three things went
wrong with that on a pod that has both keys:

* An id is a provider's spelling. ``deepseek-ai/deepseek-v4.1-flash`` is Atlas's; OpenRouter calls the same model
  ``deepseek/deepseek-v4.1-flash``. Story, manhwa and the agent's own chains pass ids straight through, so a chain
  written for one provider misfired on the other.
* One provider's rate limit failed the whole chain, because every candidate in it went to the same provider. A live
  plan failed with "429 for deepseek-ai/deepseek-v4.1-flash" while the other provider was idle.
* There was no way to say "this role on OpenRouter, that one on Atlas".

``ChatRouter`` keeps the client interface (``chat``, ``list_models``, ``configured``, ``routing``...) and adds:

* ``atlas:<id>`` / ``openrouter:<id>`` picks the provider for that one call, so a role's chain can mix providers;
* an id is corrected to the spelling the chosen provider serves, matched on the part after the vendor prefix
  (``models_llm.slug``), only when exactly one served model matches;
* a call that fails for a reason the other provider could not share (rate limit, server error, network) is tried
  once on the other provider, when it has a key and serves the model.

A refusal, a truncated reply or a bad request is never moved: the same request would fail the same way there.
"""

from __future__ import annotations

import logging
import re

from . import models_llm
from .atlas import AtlasClient, AtlasError

log = logging.getLogger("hawk_api.llm")

PREFIX = re.compile(r"^\s*(atlas|openrouter)\s*:\s*", re.IGNORECASE)
#: Failures that belong to one provider at one moment, not to the request.
_TRANSIENT = re.compile(r"\b(429|500|502|503|504|408|425)\b|could not reach|timed? ?out|rate.?limit|overloaded|"
                        r"request failed", re.IGNORECASE)
_NEVER_MOVE = re.compile(r"cut off at the token cap|refused|content.?policy|no .*api key", re.IGNORECASE)


def split_provider(model: str) -> tuple[str, str]:
    """``"openrouter:x-ai/grok-4.6"`` -> ``("openrouter", "x-ai/grok-4.6")``; no prefix -> ``("", model)``."""
    text = str(model or "")
    found = PREFIX.match(text)
    if not found:
        return "", text.strip()
    return found.group(1).lower(), text[found.end():].strip()


class ChatRouter:
    """The chat client the service hands out: a primary provider, and the other one when it has a key."""

    def __init__(self, primary: AtlasClient, secondary: AtlasClient | None = None):
        self.primary = primary
        self.secondary = secondary if secondary is not None and secondary.configured else None

    # Everything that is not routing reads from the primary, so existing code (and tests) that look at
    # ``service.atlas.base_url`` or ``.routing`` see what they always saw.
    def __getattr__(self, name):
        return getattr(self.primary, name)

    @property
    def on_call(self):
        return self.primary.on_call

    @on_call.setter
    def on_call(self, value) -> None:
        self.primary.on_call = value
        if self.secondary is not None:
            self.secondary.on_call = value

    def _client(self, provider: str) -> AtlasClient:
        for client in (self.primary, self.secondary):
            if client is not None and client.service == provider:
                return client
        return self.primary

    def _other(self, client: AtlasClient) -> AtlasClient | None:
        other = self.secondary if client is self.primary else self.primary
        return other if other is not None and other.configured else None

    @staticmethod
    async def _spelling(client: AtlasClient, model: str) -> str | None:
        """The id ``client`` serves for ``model``: itself, a unique same-slug match, or None when not served.

        No catalogue (no key, /models down) means "send it as written", the rule models_llm already follows."""
        try:
            catalogue = await client.list_models()
        except AtlasError:
            return model
        if not catalogue:
            return model
        ids = [str(entry.get("id") or "") for entry in catalogue]
        if model in ids:
            return model
        want = models_llm.slug(model)
        matches = [model_id for model_id in ids if models_llm.slug(model_id) == want]
        plain = [model_id for model_id in matches if ":" not in model_id]
        if len(plain) == 1:
            return plain[0]
        if len(matches) == 1:
            return matches[0]
        return None

    async def chat(self, model: str, messages: list[dict], **kwargs) -> tuple[str, dict]:
        provider, bare = split_provider(model)
        client = self._client(provider) if provider else self.primary
        sent = await self._spelling(client, bare) or bare
        try:
            return await client.chat(sent, messages, **kwargs)
        except AtlasError as exc:
            text = str(exc)
            other = self._other(client)
            if other is None or _NEVER_MOVE.search(text) or not _TRANSIENT.search(text):
                raise
            moved = await self._spelling(other, bare)
            if not moved:
                raise
            log.warning("hawk_api: %s failed on %s (%s); trying %s on %s", sent, client.service, text[:120],
                        moved, other.service)
            try:
                return await other.chat(moved, messages, **kwargs)
            except AtlasError as second:
                raise AtlasError(f"{text} -- and on {other.label}: {second}") from None
