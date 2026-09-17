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
FAVOURITES = ["xai/grok-4.6", "xai/grok-4.5", "xai/grok-4.3"]
#: Not chat models a planner or agent can use.
_EXCLUDE = re.compile(r"(image|ocr|embed|whisper|tts|-coding$|-ccmax$|codex|grok-build)", re.IGNORECASE)
MODEL_CACHE_SECONDS = 600.0
IMAGE_OK = frozenset({"completed", "succeeded", "success"})
IMAGE_FAILED = frozenset({"failed", "error", "canceled", "cancelled"})


class AtlasError(RuntimeError):
    """Any Atlas failure, worded for a chat or an API caller."""


def _price(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class AtlasClient:
    def __init__(self, base_url: str = DEFAULT_URL, api_key: str = "", *, timeout: float = 240.0):
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout
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
        """Chat models on the account: ``[{id, name, vision, context, price_in, price_out}]``."""
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
            outputs = item.get("output_modalities") or ["text"]
            if not model_id or "text" not in outputs or _EXCLUDE.search(model_id):
                continue
            pricing = item.get("pricing") or {}
            models.append({
                "id": model_id,
                "name": item.get("name") or model_id,
                "vision": "image" in (item.get("input_modalities") or []),
                "context": item.get("context_length"),
                "price_in": _price(pricing.get("prompt")),
                "price_out": _price(pricing.get("completion")),
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
                        return self._parse(response)
                    body = response.text
                    if response.status_code == 400 and "response_format" in payload and "response_format" in body:
                        payload.pop("response_format")  # this model rejects JSON mode; the prompt still asks for JSON
                        continue
                    if response.status_code not in RETRY_STATUSES or attempt >= max_retries:
                        raise AtlasError(f"Atlas error {response.status_code} for {model}: {body[:600] or '(empty body)'}")
                    last_error = f"{response.status_code}: {body[:200]}"
                attempt += 1
                await asyncio.sleep(min(2**attempt, 30) * (0.5 + random.random() / 2))
                if attempt > max_retries:
                    raise AtlasError(f"Atlas request failed: {last_error}")

    @staticmethod
    def _parse(response: httpx.Response) -> tuple[str, dict]:
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
        return content, data.get("usage") or {}
