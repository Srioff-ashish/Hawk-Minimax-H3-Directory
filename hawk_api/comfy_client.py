"""Async client for the private ComfyUI server: uploads, prompts, history, files, events."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, Awaitable, Callable

import httpx
import websockets

log = logging.getLogger("hawk_api")


class ComfyError(RuntimeError):
    """ComfyUI unreachable or answered unexpectedly."""


class ComfyValidationError(ComfyError):
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


class ComfyNotFound(ComfyError):
    pass


def summarise_errors(data: dict) -> str:
    parts = []
    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        parts.append(error["message"])
    for node_id, node in (data.get("node_errors") or {}).items():
        for item in node.get("errors", []):
            detail = f" ({item['details']})" if item.get("details") else ""
            parts.append(f"node {node_id} {node.get('class_type', '')}: {item.get('message', '')}{detail}")
    return "; ".join(parts) or "ComfyUI rejected the prompt."


class ComfyClient:
    def __init__(self, base_url: str, client_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex
        self.http = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(60.0, read=600.0))

    async def close(self) -> None:
        await self.http.aclose()

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            return await self.http.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise ComfyError(f"ComfyUI is unreachable at {self.base_url}: {exc}") from exc

    async def ping(self) -> bool:
        try:
            return (await self._request("GET", "/queue")).status_code == 200
        except ComfyError:
            return False

    async def upload(self, fileobj, filename: str, subfolder: str, content_type: str = "application/octet-stream") -> dict:
        response = await self._request(
            "POST",
            "/upload/image",
            files={"image": (filename, fileobj, content_type)},
            data={"subfolder": subfolder, "type": "input", "overwrite": "true"},
            timeout=httpx.Timeout(60.0, write=None, read=600.0),
        )
        if response.status_code != 200:
            raise ComfyError(f"ComfyUI upload failed ({response.status_code}): {response.text[:300]}")
        return response.json()

    async def submit(self, prompt: dict, prompt_id: str) -> dict:
        response = await self._request(
            "POST", "/prompt", json={"prompt": prompt, "client_id": self.client_id, "prompt_id": prompt_id}
        )
        if response.status_code == 400:
            data = response.json()
            raise ComfyValidationError(summarise_errors(data), data)
        if response.status_code != 200:
            raise ComfyError(f"ComfyUI /prompt failed ({response.status_code}): {response.text[:300]}")
        data = response.json()
        if data.get("node_errors"):
            raise ComfyValidationError(summarise_errors(data), data)
        return data

    async def history(self, prompt_id: str) -> dict | None:
        response = await self._request("GET", f"/history/{prompt_id}")
        if response.status_code != 200:
            raise ComfyError(f"ComfyUI /history failed ({response.status_code})")
        return response.json().get(prompt_id)

    async def queue_ids(self) -> set[str]:
        response = await self._request("GET", "/queue")
        data = response.json()
        return {
            str(item[1])
            for key in ("queue_running", "queue_pending")
            for item in data.get(key, [])
            if isinstance(item, list) and len(item) > 1
        }

    async def cancel(self, prompt_id: str) -> bool:
        response = await self._request("POST", f"/api/jobs/{prompt_id}/cancel")
        if response.status_code == 404:
            return False
        return bool(response.json().get("cancelled")) if response.status_code == 200 else False

    async def list_models(self, folder: str) -> list[str]:
        response = await self._request("GET", f"/models/{folder}")
        if response.status_code != 200:
            raise ComfyError(f"ComfyUI /models/{folder} failed ({response.status_code})")
        return list(response.json())

    async def object_info(self, node_class: str) -> dict:
        response = await self._request("GET", f"/object_info/{node_class}")
        return response.json().get(node_class, {}) if response.status_code == 200 else {}

    async def view(self, filename: str, subfolder: str, type_: str = "output") -> tuple[str, str | None, AsyncIterator[bytes]]:
        request = self.http.build_request(
            "GET", "/view", params={"filename": filename, "subfolder": subfolder, "type": type_}
        )
        try:
            response = await self.http.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise ComfyError(f"ComfyUI is unreachable at {self.base_url}: {exc}") from exc
        if response.status_code != 200:
            await response.aclose()
            raise ComfyNotFound(f"{subfolder}/{filename} is not available ({response.status_code}).")

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes(1 << 20):
                    yield chunk
            finally:
                await response.aclose()

        return response.headers.get("content-type", "application/octet-stream"), response.headers.get("content-length"), body()

    async def listen(self, handler: Callable[[dict], Awaitable[None]]) -> None:
        """Forward ComfyUI websocket events forever, reconnecting with backoff.
        Progress and "executed" events only reach the client_id that queued the prompt."""
        ws_url = self.base_url.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
        delay = 1.0
        while True:
            try:
                async with websockets.connect(ws_url, max_size=None, ping_interval=20) as socket:
                    delay = 1.0
                    async for message in socket:
                        if isinstance(message, bytes):
                            continue  # binary previews
                        try:
                            event = json.loads(message)
                        except ValueError:
                            continue
                        try:
                            await handler(event)
                        except Exception:
                            log.exception("hawk_api: handling ComfyUI event %s failed", event.get("type"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("hawk_api: ComfyUI websocket closed (%s); reconnecting in %.0fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
