"""Minimal async Atlas Cloud chat client.

Vendored from HawkNodes (github.com/Srioff-ashish/HawkNodes) so this pack does
not require HawkNodes to be installed. OpenAI-compatible
``POST {base}/v1/chat/completions``; async so a slow call never blocks ComfyUI.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import random

import aiohttp
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("HawkH3")

DEFAULT_CHAT_URL = "https://api.atlascloud.ai/v1"
#: Atlas takes an int32 seed; anything larger comes back as a bare 400.
MAX_SEED = 2147483647
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: Default library agent strings draw a 403 from the edge proxy.
USER_AGENT = "HawkH3Director/0.1 (ComfyUI custom node)"


class AtlasError(RuntimeError):
    """Any Atlas API failure; the message is shown on the node."""


def resolve_api_key(api_key: str | None) -> str:
    """Widget value wins; a blank widget falls back to ATLAS_API_KEY.

    Blank + environment variable is the safer setup: ComfyUI saves widget values
    into workflow JSON and PNG metadata, so a typed key travels with shared files.
    """
    key = (api_key or "").strip() or os.environ.get("ATLAS_API_KEY", "").strip()
    if not key:
        raise AtlasError(
            "No Atlas API key. Type one into the node's api_key widget, or set the "
            "ATLAS_API_KEY environment variable before starting ComfyUI."
        )
    return key


def normalize_chat_url(url: str | None) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_CHAT_URL
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


async def chat_completion(
    api_url: str,
    api_key: str,
    payload: dict,
    *,
    timeout: int = 180,
    max_retries: int = 3,
) -> dict:
    url = f"{normalize_chat_url(api_url)}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    last_error = ""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        for attempt in range(max_retries + 1):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    body = await response.text()
                    if response.status == 200:
                        try:
                            result = json.loads(body)
                        except json.JSONDecodeError:
                            raise AtlasError(f"Atlas returned non-JSON from {url}: {body[:500]}") from None
                        if not isinstance(result, dict):
                            raise AtlasError(f"Unexpected chat response shape: {type(result).__name__}")
                        return result
                    # 4xx auth/validation errors fail identically every time: never retried.
                    if response.status not in RETRY_STATUSES or attempt == max_retries:
                        raise AtlasError(f"Atlas API error {response.status} at {url}: {body[:1000] or '(empty body)'}")
                    last_error = f"{response.status}: {body[:200]}"
            except aiohttp.ClientError as exc:
                if attempt == max_retries:
                    raise AtlasError(f"Could not reach Atlas at {url}: {exc}") from exc
                last_error = str(exc)
            except asyncio.TimeoutError:
                if attempt == max_retries:
                    raise AtlasError(
                        f"Atlas request timed out after {timeout}s. Raise the node's timeout widget."
                    ) from None
                last_error = "timeout"

            # Full jitter, so parallel nodes do not retry in lockstep.
            delay = min(2**attempt, 30) * (0.5 + random.random() / 2)
            logger.warning("HawkH3: retrying Atlas in %.1fs (%s)", delay, last_error)
            await asyncio.sleep(delay)

    raise AtlasError(f"Atlas request to {url} failed: {last_error}")


def extract_message_text(response: dict) -> str:
    error = response.get("error")
    if error:
        if isinstance(error, dict):
            raise AtlasError(f"Atlas error ({error.get('code', 'unknown')}): {error.get('message') or json.dumps(error)}")
        raise AtlasError(f"Atlas error: {error}")

    choices = response.get("choices") or []
    if not choices:
        raise AtlasError(f"Atlas returned no choices: {json.dumps(response)[:500]}")
    message = choices[0].get("message") or {}
    if message.get("refusal"):
        raise AtlasError(f"The model refused to respond: {message['refusal']}")

    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content:
        finish = choices[0].get("finish_reason", "unknown")
        raise AtlasError(
            f"Atlas returned an empty message (finish_reason={finish}). If this is `length`, raise max_tokens."
        )
    return content


def frame_to_data_uri(frame: torch.Tensor, max_side: int = 1024, quality: int = 90) -> str:
    """One IMAGE frame (``[H, W, C]`` or ``[1, H, W, C]``) to a JPEG data URI."""
    if frame.dim() == 4:
        frame = frame[0]
    array = (frame[..., :3].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
