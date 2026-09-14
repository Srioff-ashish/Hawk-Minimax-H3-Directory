"""FastAPI app: REST endpoints, the upload page and the MCP endpoint behind one token check.

Run with ``uvicorn --factory hawk_api.app:create_app`` (deploy/start_pod.sh does).
"""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from urllib.parse import parse_qs

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from mcp.server.transport_security import TransportSecuritySettings

from .auth import bearer, signature_valid, split_path_token, token_matches
from .config import Settings
from .jobs import Conflict, HawkService, NotFound, RequestError, Unavailable
from .mcp_server import build_mcp
from .schemas import PlanRequest, UrlAssetRequest, VideoRequest

#: No token needed: health, and the API schema/docs page (they contain no data).
PUBLIC_PATHS = {"/healthz", "/docs", "/openapi.json", "/docs/oauth2-redirect"}
#: Paths a signed link (?exp=&sig=) may open without the token.
SIGNABLE = re.compile(r"^/(upload|v1/jobs/[^/]+/video|v1/jobs/[^/]+/segments/\d+)$")
UPLOAD_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "upload.html")


class AuthMiddleware:
    """Bearer header, /t/<token>/ path prefix, or a signed link -- otherwise 401."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        path_token, rest = split_path_token(path)
        if path_token is not None:
            if not token_matches(self.token, path_token):
                return await self._deny(send)
            scope = dict(scope, path=rest, raw_path=rest.encode("utf-8"))
            return await self.app(scope, receive, send)

        if path in PUBLIC_PATHS:
            return await self.app(scope, receive, send)

        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        if token_matches(self.token, bearer(headers.get("authorization"))):
            return await self.app(scope, receive, send)

        if SIGNABLE.match(path):
            query = parse_qs(scope.get("query_string", b"").decode("utf-8"))
            if signature_valid(self.token, path, (query.get("exp") or [None])[0], (query.get("sig") or [None])[0]):
                return await self.app(scope, receive, send)

        return await self._deny(send)

    @staticmethod
    async def _deny(send):
        body = json.dumps({"error": "Missing or invalid token. Send 'Authorization: Bearer <token>', use a /t/<token>/ URL, or a fresh signed link."}).encode()
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})


def create_app(settings: Settings | None = None, service: HawkService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or HawkService(settings)
    mcp = build_mcp(service)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        # The token middleware authenticates every request; the pod proxy's host name varies.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await service.start()
        async with mcp.session_manager.run():
            yield
        await service.stop()

    app = FastAPI(
        title="Hawk MiniMax H3 Director API",
        version="0.1.0",
        description="Plan and render long MiniMax H3 videos with native audio. Auth: Bearer token, /t/<token>/ prefix, or signed links.",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.exception_handler(RequestError)
    async def _request_error(_request, exc: RequestError):
        return JSONResponse({"error": str(exc), "details": exc.details}, status_code=422)

    @app.exception_handler(NotFound)
    async def _not_found(_request, exc: NotFound):
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(Conflict)
    async def _conflict(_request, exc: Conflict):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(Unavailable)
    async def _unavailable(_request, exc: Unavailable):
        return JSONResponse({"error": str(exc)}, status_code=503)

    # ------------------------------------------------------------- info

    @app.get("/healthz", tags=["info"])
    async def healthz():
        health = await service.health()
        return JSONResponse(health, status_code=200 if health["ok"] else 503)

    @app.get("/v1/options", tags=["info"])
    async def options():
        return await service.options()

    # ----------------------------------------------------------- assets

    async def _store_uploads(files: list[UploadFile]) -> dict:
        limit = settings.max_upload_mb * 1024 * 1024
        assets = []
        for upload in files:
            if upload.size is not None and upload.size > limit:
                raise RequestError(f"{upload.filename} is larger than {settings.max_upload_mb} MB.")
            assets.append(await service.add_asset(upload.filename or "upload", upload.file, upload.content_type, upload.size))
        return {"assets": assets}

    @app.post("/v1/assets", tags=["assets"], status_code=201)
    async def upload_assets(files: list[UploadFile] = File(..., description="One or more image, audio or video files.")):
        return await _store_uploads(files)

    @app.post("/v1/assets/from-url", tags=["assets"], status_code=201)
    async def asset_from_url(body: UrlAssetRequest):
        return await service.add_asset_from_url(body.url, body.filename)

    @app.get("/v1/assets", tags=["assets"])
    async def list_assets(limit: int = 100):
        return {"assets": service.list_assets(limit)}

    @app.get("/upload", include_in_schema=False)
    async def upload_page():
        with open(UPLOAD_PAGE, "r", encoding="utf-8") as handle:
            return HTMLResponse(handle.read())

    @app.post("/upload", include_in_schema=False, status_code=201)
    async def upload_page_post(files: list[UploadFile] = File(...)):
        return await _store_uploads(files)

    # ------------------------------------------------------------- jobs

    @app.post("/v1/plans", tags=["jobs"], status_code=202)
    async def create_plan(body: PlanRequest):
        return service.job_view(await service.create_plan(body))

    @app.post("/v1/videos", tags=["jobs"], status_code=202)
    async def create_video(body: VideoRequest):
        return service.job_view(await service.create_video(body))

    @app.get("/v1/jobs", tags=["jobs"])
    async def list_jobs(limit: int = 50):
        return {"jobs": [service.job_view(job) for job in service.list_jobs(limit)]}

    @app.get("/v1/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str):
        return service.job_view(service.get_job(job_id))

    @app.post("/v1/jobs/{job_id}/cancel", tags=["jobs"])
    async def cancel_job(job_id: str):
        return service.job_view(await service.cancel(job_id))

    @app.post("/v1/jobs/{job_id}/retry", tags=["jobs"], status_code=202)
    async def retry_job(job_id: str):
        return service.job_view(await service.retry(job_id))

    def _stream(opened, filename: str) -> StreamingResponse:
        content_type, length, body = opened
        headers = {"content-disposition": f'attachment; filename="{filename}"'}
        if length:
            headers["content-length"] = length
        return StreamingResponse(body, media_type=content_type, headers=headers)

    @app.get("/v1/jobs/{job_id}/video", tags=["downloads"])
    async def download_video(job_id: str):
        return _stream(await service.open_video(job_id), f"hawk_h3_{job_id[:8]}.mp4")

    @app.get("/v1/jobs/{job_id}/segments/{number}", tags=["downloads"])
    async def download_segment(job_id: str, number: int):
        return _stream(await service.open_segment(job_id, number), f"hawk_h3_{job_id[:8]}_segment_{number:03d}.mp4")

    app.mount("/", mcp_app)  # serves /mcp; mounted last so the routes above win
    app.add_middleware(AuthMiddleware, token=settings.token)
    return app
