"""Async Atlas Cloud client for the gateway: model catalogue and chat completions.

Torch-free (httpx only), so the API process never imports the ComfyUI-side
``hawk_h3/atlas.py``. The retry policy matches it: 408/409/425/429/5xx are retried
with jittered backoff, other 4xx fail at once.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import time

import httpx

DEFAULT_URL = "https://api.atlascloud.ai/v1"
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
USER_AGENT = "HawkH3Director/0.1 (hawk_api)"
#: Shown first, in this order, when present.
FAVOURITES = [
    "xai/grok-4.6", "xai/grok-4.5", "xai/grok-4.3",
    "deepseek/deepseek-chat", "deepseek/deepseek-reasoner", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.6",
    "anthracite-org/magnum-v4-72b", "qwen/qwen-2.5-72b-instruct",
    "nvidia/nemotron-4-340b-instruct", "nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning",
]
#: Not chat models a planner or agent can use.
_EXCLUDE = re.compile(r"(image|ocr|embed|whisper|tts|-coding$|-ccmax$|codex|grok-build)", re.IGNORECASE)
MODEL_CACHE_SECONDS = 600.0

#: How OpenRouter chooses between the several services that each serve one model id.
#:
#: Naming a model is only half the choice there. The same id is served by a dozen providers at different
#: prices, and -- the part that actually shows in the output -- at different weights: an int4 copy of a
#: model is a visibly worse model than its bf16 copy for the same id. Left alone, a request goes wherever
#: OpenRouter's own balance of price and uptime sends it, so two identical calls can come back at two
#: different qualities and nothing in the reply says which was used.
#:
#: Each preset is sent as the request's "provider" block. "quantizations" is the quality floor and "sort"
#: decides among whatever clears it, which is the order that gets a cheap call without a cheap model.
#: "require_parameters" keeps out providers that would silently drop the JSON-mode and temperature settings
#: we send; a provider that ignores response_format answers in prose, which reads here as a model that
#: cannot follow the format rather than a provider that was never asked to.
ROUTING: dict[str, dict] = {
    # A prompt cache is the provider's, so it only helps while requests keep landing on the same one. "balanced"
    # sorts by price and lets OpenRouter fall back, which moves a chat between services and throws the warm
    # prefix away every time -- paying full price for ~6k tokens to save a fraction of a cent on the rest.
    # "sticky" keeps the same quality floor and stops the moving.
    "sticky": {"quantizations": ["bf16", "fp16", "fp8"], "require_parameters": True,
               "data_collection": "deny", "allow_fallbacks": False},
    "balanced": {"sort": "price", "quantizations": ["bf16", "fp16", "fp8"],
                 "require_parameters": True, "data_collection": "deny", "allow_fallbacks": True},
    "quality": {"quantizations": ["bf16", "fp16"],
                "require_parameters": True, "data_collection": "deny", "allow_fallbacks": True},
    "cheapest": {"sort": "price", "allow_fallbacks": True},
    "fastest": {"sort": "throughput", "quantizations": ["bf16", "fp16", "fp8"], "allow_fallbacks": True},
    "default": {},  # whatever OpenRouter would do on its own
}
#: A refusal that means the filters matched no provider, rather than anything being wrong with the request.
_NO_PROVIDER = re.compile(r"no (?:allowed |eligible )?provider|provider.*(?:not found|no match)", re.IGNORECASE)
IMAGE_OK = frozenset({"completed", "succeeded", "success"})
IMAGE_FAILED = frozenset({"failed", "error", "canceled", "cancelled"})


def routing_block(name: str) -> dict | None:
    """The provider block for a preset name, or None when there is nothing to send.

    An unknown name falls back to the sticky preset rather than to no filtering: a typo in a setting should not
    quietly re-open the cheap heavily-quantised services this exists to keep out, nor start moving a warm
    prompt cache between providers.
    """
    block = ROUTING.get(str(name or "").strip().lower(), ROUTING["sticky"])
    return dict(block) if block else None


class AtlasError(RuntimeError):
    """Any Atlas failure, worded for a chat or an API caller."""


class AtlasTruncated(AtlasError):
    """The model ran into the token cap mid-reply (finish_reason="length").

    Only raised for a JSON-mode call, where it is never recoverable: an object cut off mid-string cannot be
    parsed, and every caller on that path asked for JSON because it intends to parse it. It used to pass
    through as ordinary content, because the only check was that the reply was not empty -- a half-written
    film plan was stored as a finished one and was found out at render time, several minutes later.
    """


def _price(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class AtlasClient:
    def __init__(self, base_url: str = DEFAULT_URL, api_key: str = "", *, timeout: float = 240.0,
                 routing: dict | None = None):
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout
        #: An OpenRouter "provider" block, or None for a service that has no such thing. Atlas serves each
        #: model itself, so sending it one would be an unknown field on a request it cannot act on.
        self.routing = routing or None
        self._models: tuple[float, list[dict]] = (0.0, [])

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }

    async def list_models(self, refresh: bool = False) -> list[dict]:
        """Chat models on the account: ``[{id, name, vision, context, price_in, price_out, price_cache}]``."""
        if not self.configured:
            return []
        stamp, cached = self._models
        if cached and not refresh and time.monotonic() - stamp < MODEL_CACHE_SECONDS:
            return cached
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                response = await http.get(f"{self.base_url}/models", headers=self._headers())
        except httpx.HTTPError as exc:
            raise AtlasError(f"Could not reach Atlas: {exc}") from exc
        if response.status_code != 200:
            raise AtlasError(f"Atlas /models failed ({response.status_code}): {response.text[:300]}")
        data = response.json()
        items = data.get("data", data) if isinstance(data, dict) else data
        models = []
        for item in items or []:
            model_id = str(item.get("id") or "")
            # Atlas puts the modalities at the top level; OpenRouter nests them under "architecture". Read
            # both, because falling back to "text" for the outputs and nothing for the inputs marked every
            # OpenRouter model as blind, and inspect_image only ever offers a model it believes can see.
            shape = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
            outputs = item.get("output_modalities") or shape.get("output_modalities") or ["text"]
            inputs_ = item.get("input_modalities") or shape.get("input_modalities") or []
            if not model_id or "text" not in outputs or _EXCLUDE.search(model_id):
                continue
            pricing = item.get("pricing") or {}
            models.append({
                "id": model_id,
                "name": item.get("name") or model_id,
                "vision": "image" in inputs_,
                "context": item.get("context_length"),
                "price_in": _price(pricing.get("prompt")),
                "price_out": _price(pricing.get("completion")),
                # cached prompt tokens (a repeated prefix) bill at this lower rate; models without one bill them in full
                "price_cache": _price(pricing.get("input_cache_read")) or _price(pricing.get("prompt")),
            })
        rank = {model_id: index for index, model_id in enumerate(FAVOURITES)}
        models.sort(key=lambda m: (rank.get(m["id"], len(rank)), m["name"].lower()))
        self._models = (time.monotonic(), models)
        return models

    async def model_info(self, model_id: str) -> dict | None:
        try:
            return next((m for m in await self.list_models() if m["id"] == model_id), None)
        except AtlasError:
            return None

    @property
    def image_base(self) -> str:
        """``https://api.atlascloud.ai`` -- image generation lives under /api/v1, not /v1."""
        base = self.base_url
        for suffix in ("/api/v1", "/v1"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    async def generate_image(self, payload: dict, *, timeout: float = 420.0, poll_seconds: float = 2.0, max_retries: int = 3) -> list[bytes]:
        """Submit to ``/api/v1/model/generateImage``, poll the prediction, return image bytes.
        Edit models take ``images``: a list of data URIs."""
        if not self.configured:
            raise AtlasError("No Atlas API key on the server. Set ATLAS_API_KEY and restart the API.")
        base = self.image_base
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0), follow_redirects=True) as http:
            submitted = await self._json(http, "POST", f"{base}/api/v1/model/generateImage", payload, max_retries)
            data = submitted.get("data") if isinstance(submitted, dict) else None
            prediction = data.get("id") if isinstance(data, dict) else None
            if not prediction:
                raise AtlasError(f"Atlas did not return a prediction id: {json.dumps(submitted)[:400]}")
            while True:
                result = await self._json(http, "GET", f"{base}/api/v1/model/prediction/{prediction}", None, 1)
                data = result.get("data") if isinstance(result, dict) else {}
                status = str((data or {}).get("status", "")).lower()
                if status in IMAGE_OK:
                    outputs = data.get("outputs") or []
                    if not outputs:
                        raise AtlasError(f"Image generation {prediction} finished without images.")
                    break
                if status in IMAGE_FAILED:
                    raise AtlasError(f"Image generation failed: {data.get('error') or 'no reason given'}")
                if time.monotonic() > deadline:
                    raise AtlasError(f"Image generation {prediction} did not finish within {int(timeout)} s.")
                await asyncio.sleep(poll_seconds)
            images = []
            for item in outputs:
                item = str(item)
                if item.startswith(("http://", "https://")):
                    response = await http.get(item)
                    if response.status_code != 200:
                        raise AtlasError(f"Could not download the generated image ({response.status_code}).")
                    images.append(response.content)
                else:
                    images.append(base64.b64decode(item.split(",", 1)[-1]))
            return images

    async def _json(self, http: httpx.AsyncClient, method: str, url: str, payload: dict | None, max_retries: int):
        attempt, last_error = 0, ""
        while True:
            try:
                response = await http.request(method, url, headers=self._headers(), json=payload)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                if attempt >= max_retries:
                    raise AtlasError(f"Could not reach Atlas: {exc}") from exc
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        raise AtlasError(f"Atlas returned non-JSON from {url}: {response.text[:300]}") from None
                if response.status_code not in RETRY_STATUSES or attempt >= max_retries:
                    raise AtlasError(f"Atlas error {response.status_code}: {response.text[:600] or '(empty body)'}")
                last_error = f"{response.status_code}: {response.text[:200]}"
            attempt += 1
            await asyncio.sleep(min(2**attempt, 30) * (0.5 + random.random() / 2))

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        json_mode: bool = True,
        max_tokens: int = 8192,
        temperature: float = 0.4,
        max_retries: int = 3,
    ) -> tuple[str, dict]:
        """One chat completion. Returns ``(text, usage)``."""
        if not self.configured:
            raise AtlasError("No Atlas API key on the server. Set ATLAS_API_KEY and restart the API.")
        payload: dict = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.routing:
            payload["provider"] = self.routing
        url = f"{self.base_url}/chat/completions"
        last_error = ""
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=30.0)) as http:
            attempt = 0
            while True:
                try:
                    response = await http.post(url, headers=self._headers(), json=payload)
                except httpx.HTTPError as exc:
                    if attempt >= max_retries:
                        raise AtlasError(f"Could not reach Atlas: {exc}") from exc
                    last_error = str(exc)
                else:
                    if response.status_code == 200:
                        try:
                            return self._parse(response, json_mode=json_mode)
                        except AtlasTruncated as exc:
                            # Nothing but asking again can help, so this takes the same backoff as a 429.
                            if attempt >= max_retries:
                                raise AtlasError(str(exc)) from None
                            last_error = str(exc)
                            attempt += 1
                            await asyncio.sleep(min(2**attempt, 30) * (0.5 + random.random() / 2))
                            continue
                    body = response.text
                    if response.status_code == 400 and "response_format" in payload and "response_format" in body:
                        payload.pop("response_format")  # this model rejects JSON mode; the prompt still asks for JSON
                        continue
                    if "provider" in payload and (response.status_code == 404 or _NO_PROVIDER.search(body)):
                        # No service serving this model clears the quality filters. Running the call
                        # unfiltered beats stopping the chat, and it is dropped for this request only, so
                        # the next model is filtered again. A genuine unknown model simply 404s twice.
                        payload.pop("provider")
                        continue
                    if response.status_code not in RETRY_STATUSES or attempt >= max_retries:
                        raise AtlasError(f"Atlas error {response.status_code} for {model}: {body[:600] or '(empty body)'}")
                    last_error = f"{response.status_code}: {body[:200]}"
                attempt += 1
                await asyncio.sleep(min(2**attempt, 30) * (0.5 + random.random() / 2))
                if attempt > max_retries:
                    raise AtlasError(f"Atlas request failed: {last_error}")

    @staticmethod
    def _parse(response: httpx.Response, json_mode: bool = True) -> tuple[str, dict]:
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise AtlasError(f"Atlas returned non-JSON: {response.text[:300]}") from None
        if data.get("error"):
            raise AtlasError(f"Atlas error: {json.dumps(data['error'])[:400]}")
        choices = data.get("choices") or []
        if not choices:
            raise AtlasError(f"Atlas returned no choices: {json.dumps(data)[:300]}")
        message = choices[0].get("message") or {}
        if message.get("refusal"):
            raise AtlasError(f"The model refused: {message['refusal']}")
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not content:
            raise AtlasError(f"The model returned an empty message (finish_reason={choices[0].get('finish_reason')}).")
        if json_mode and choices[0].get("finish_reason") == "length":
            raise AtlasTruncated(f"The model's reply was cut off at the token cap after {len(content)} characters, "
                                 "so it is not the complete JSON object that was asked for.")
        return content, data.get("usage") or {}
