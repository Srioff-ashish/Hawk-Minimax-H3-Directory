"""FastAPI app: REST endpoints, the upload page and the MCP endpoint behind one token check.

Run with ``uvicorn --factory hawk_api.app:create_app`` (deploy/start_pod.sh does).
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import asynccontextmanager
from urllib.parse import parse_qs

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.middleware.gzip import GZipMiddleware
from mcp.server.transport_security import TransportSecuritySettings

from .agent import AgentService
from .library import DriveBrowser, DriveExporter, ImportManager
from .auth import bearer, signature_valid, split_path_token, token_matches
from .comfy_client import ComfyError
from .config import Settings
from .jobs import Conflict, HawkService, NotFound, RequestError, Unavailable
from .mcp_server import build_mcp
from .prompts import PROMPT_NAMES
from .schemas import (
    AgentMessageIn, AgentSessionIn, AgentTalkIn, AssetBulk, AssetUpdate, DriveExportSettings, DriveImportIn, ImageRequest, PlanRequest, PromptIn,
    UrlAssetRequest, VideoRequest,
)

#: No token needed: health, the API schema/docs page and the Studio page itself
#: (they contain no data; every API call the Studio makes still needs the token).
PUBLIC_PATHS = {"/healthz", "/docs", "/openapi.json", "/docs/oauth2-redirect", "/studio"}
#: Paths a signed link (?exp=&sig=) may open without the token.
SIGNABLE = re.compile(r"^/(upload|v1/jobs/[^/]+/(video|thumb)|v1/jobs/[^/]+/segments/\d+|v1/assets/[^/]+/file)$")
#: Media is never gzipped (it is already compressed, and ranges must stay intact).
NO_GZIP = ("video/mp4", "video/quicktime", "video/webm", "audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/ogg",
           "image/jpeg", "image/png", "image/webp", "application/octet-stream", "text/event-stream")
CACHE = {"cache-control": "private, max-age=86400"}
UPLOAD_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "upload.html")
STUDIO_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "studio.html")


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

        if path in PUBLIC_PATHS or path.startswith("/.well-known/"):
            # Connector apps probe /.well-known/oauth-* first; a 401 there makes them
            # demand OAuth credentials. Let it through so the app answers 404: no OAuth.
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
        # No WWW-Authenticate header on purpose: MCP clients treat it as the start of an
        # OAuth flow, and this API authenticates with the token in the header or URL instead.
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


def create_app(settings: Settings | None = None, service: HawkService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or HawkService(settings)
    drive = DriveBrowser(settings.drive_root)
    imports = ImportManager(service, drive)
    exporter = DriveExporter(service, drive)
    service.render_done_hooks.append(exporter.schedule)
    mcp = build_mcp(service, drive=drive, imports=imports)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        # The token middleware authenticates every request; the pod proxy's host name varies.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    agent = AgentService(service, mcp)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await service.start()
        await agent.start()
        async with mcp.session_manager.run():
            yield
        await imports.stop()
        await exporter.stop()
        await agent.stop()
        await service.stop()

    app = FastAPI(
        title="Hawk MiniMax H3 Director API",
        version="0.1.0",
        description="Plan and render long MiniMax H3 videos with native audio. Auth: Bearer token, /t/<token>/ prefix, or signed links.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.agent = agent

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

    async def _store_uploads(files: list[UploadFile], collection: str | None = None, tags: str | None = None) -> dict:
        limit = settings.max_upload_mb * 1024 * 1024
        assets = []
        for upload in files:
            if upload.size is not None and upload.size > limit:
                raise RequestError(f"{upload.filename} is larger than {settings.max_upload_mb} MB.")
            asset = await service.add_asset(upload.filename or "upload", upload.file, upload.content_type, upload.size,
                                            collection=collection, tags=tags)
            assets.append(service.asset_view(asset))
        return {"assets": assets}

    @app.post("/v1/assets", tags=["assets"], status_code=201)
    async def upload_assets(
        files: list[UploadFile] = File(..., description="One or more image, audio or video files."),
        collection: str | None = Form(None, description="Collection for these files (default Uploads)."),
        tags: str | None = Form(None, description="Comma-separated tags."),
    ):
        return await _store_uploads(files, collection, tags)

    @app.post("/v1/assets/from-url", tags=["assets"], status_code=201)
    async def asset_from_url(body: UrlAssetRequest):
        return service.asset_view(await service.add_asset_from_url(body.url, body.filename))

    @app.get("/v1/assets", tags=["assets"])
    async def list_assets(limit: int = 100):
        return {"assets": [service.asset_view(asset) for asset in service.list_assets(limit)]}

    @app.get("/v1/library", tags=["assets"])
    async def library(kind: str | None = None, collection: str | None = None, tag: str | None = None, q: str | None = None,
                      limit: int = 200, offset: int = 0):
        items, total = service.search_assets(kind=kind, collection=collection, tag=tag, query=q, limit=min(limit, 1000), offset=offset)
        return {"assets": [service.asset_view(a) for a in items], "total": total,
                "collections": service.collections(), "tags": service.tags()}

    @app.patch("/v1/assets/{asset_id}", tags=["assets"])
    async def update_asset(asset_id: str, body: AssetUpdate):
        return service.asset_view(service.update_asset(asset_id, collection=body.collection, tags=body.tags,
                                                       add_tags=body.add_tags, remove_tags=body.remove_tags, filename=body.filename))

    @app.delete("/v1/assets/{asset_id}", tags=["assets"])
    async def delete_asset(asset_id: str):
        service.delete_asset(asset_id)
        return {"deleted": [asset_id]}

    @app.post("/v1/assets/bulk", tags=["assets"])
    async def bulk_assets(body: AssetBulk):
        done, errors = [], []
        for asset_id in body.ids:
            try:
                if body.action == "delete":
                    service.delete_asset(asset_id)
                elif body.action == "move":
                    if not (body.collection or "").strip():
                        raise RequestError("Give a collection to move to.")
                    service.update_asset(asset_id, collection=body.collection)
                elif body.action == "tag":
                    service.update_asset(asset_id, add_tags=body.tags)
                else:
                    service.update_asset(asset_id, remove_tags=body.tags)
                done.append(asset_id)
            except (NotFound, RequestError) as exc:
                errors.append(f"{asset_id}: {exc}")
        return {"action": body.action, "done": done, "errors": errors}

    @app.get("/v1/drive", tags=["assets"])
    async def drive_browse(path: str = ""):
        if not drive.available:
            return {"available": False, "root": settings.drive_root,
                    "hint": "Mount Google Drive in the Colab notebook: from google.colab import drive; drive.mount('/content/drive')"}
        return drive.browse(path)

    @app.post("/v1/drive/import", tags=["assets"], status_code=202)
    async def drive_import(body: DriveImportIn):
        return await imports.start(body.paths, recursive=body.recursive, collection=body.collection, tags=body.tags)

    @app.get("/v1/imports", tags=["assets"])
    async def imports_recent():
        return {"imports": imports.recent()}

    @app.get("/v1/imports/{import_id}", tags=["assets"])
    async def import_status(import_id: str):
        return imports.get(import_id)

    @app.get("/v1/assets/{asset_id}/file", tags=["assets"])
    async def asset_file(asset_id: str, w: int = 0):
        """The asset itself (byte ranges when stored locally), or a JPEG thumbnail at most ``w`` pixels wide."""
        asset = service.store.get_asset(asset_id)
        local = service.local_asset_path(asset) if asset and w <= 0 else None
        if local:
            import mimetypes

            return FileResponse(local, media_type=mimetypes.guess_type(asset["filename"])[0] or "application/octet-stream",
                                filename=asset["filename"], content_disposition_type="inline", headers=CACHE)
        content, media_type = await service.asset_file(asset_id, w)
        return Response(content, media_type=media_type, headers={"cache-control": "private, max-age=86400"})

    @app.post("/v1/images", tags=["assets"], status_code=201)
    async def generate_images(body: ImageRequest):
        return await service.generate_images(body.prompt, model=body.model, reference_asset_ids=body.reference_asset_ids,
                                             size=body.size, n=body.n, seed=body.seed, engine=body.engine,
                                             loras=[l.model_dump() for l in body.loras], steps=body.steps,
                                             max_adult_loras=body.max_adult_loras, ref_boost=body.ref_boost)

    @app.get("/v1/debug/comfy-logs", tags=["system"])
    async def comfy_logs(lines: int = 200, grep: str = ""):
        """The tail of ComfyUI's log (optionally only lines containing `grep`), to diagnose node errors."""
        try:
            text = await service.comfy.logs()
        except ComfyError as exc:
            raise Unavailable(str(exc)) from exc
        rows = text.splitlines()
        if grep:
            rows = [row for row in rows if grep.lower() in row.lower()]
        return {"lines": rows[-max(1, min(2000, lines)):]}

    @app.get("/v1/images/options", tags=["assets"])
    async def image_options():
        """Image engines: local Krea 2 (installed? busy?) with its LoRA catalogue, and the Atlas models."""
        return await service.image_options()

    @app.get("/studio", include_in_schema=False)
    async def studio_page():
        with open(STUDIO_PAGE, "r", encoding="utf-8") as handle:
            return HTMLResponse(handle.read(), headers={"cache-control": "no-store"})

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
    async def list_jobs(limit: int = 50, view: str = "full", since: float | None = None):
        """view=summary drops scripts, prompts and segment links; since returns only jobs changed after that time."""
        now = time.time()
        jobs = service.list_jobs(min(limit, 500))
        if since is not None:
            jobs = [job for job in jobs if job["updated_at"] > since]
        render = service.job_summary if view == "summary" else service.job_view
        return {"jobs": [render(job) for job in jobs], "server_time": now}

    @app.get("/v1/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str):
        return service.job_view(service.get_job(job_id))

    @app.post("/v1/jobs/{job_id}/cancel", tags=["jobs"])
    async def cancel_job(job_id: str):
        return service.job_view(await service.cancel(job_id))

    @app.post("/v1/jobs/{job_id}/retry", tags=["jobs"], status_code=202)
    async def retry_job(job_id: str):
        return service.job_view(await service.retry(job_id))

    async def _media(request: Request, location: tuple[str, str, str], name: str, download: bool):
        """Serve a video: from disk with byte ranges when ComfyUI's output folder is local,
        otherwise proxied from ComfyUI with the Range header forwarded."""
        disposition = "attachment" if download else "inline"
        local = service.local_output(*location)
        if local:
            return FileResponse(local, media_type="video/mp4", filename=name, content_disposition_type=disposition, headers=CACHE)
        opened = await service.open_remote(*location, request.headers.get("range"))
        headers = {**opened["headers"], **CACHE, "content-disposition": f'{disposition}; filename="{name}"'}
        return StreamingResponse(opened["body"], status_code=opened["status"], headers=headers,
                                 media_type=headers.get("content-type", "video/mp4"))

    @app.get("/v1/jobs/{job_id}/video", tags=["downloads"])
    async def download_video(request: Request, job_id: str, download: int = 0):
        return await _media(request, service.video_location(job_id), f"hawk_h3_{job_id[:8]}.mp4", bool(download))

    @app.get("/v1/jobs/{job_id}/segments/{number}", tags=["downloads"])
    async def download_segment(request: Request, job_id: str, number: int, download: int = 0):
        return await _media(request, service.segment_location(job_id, number), f"hawk_h3_{job_id[:8]}_segment_{number:03d}.mp4", bool(download))

    @app.get("/v1/jobs/{job_id}/thumb", tags=["downloads"])
    async def job_thumb(job_id: str, w: int = 640):
        return Response(await service.job_thumb(job_id, w), media_type="image/jpeg", headers=CACHE)

    @app.post("/v1/jobs/{job_id}/drive", tags=["downloads"], status_code=202)
    async def export_to_drive(job_id: str):
        service.video_location(job_id)
        exporter.schedule(job_id, force=True)
        return service.job_view(service.get_job(job_id))

    @app.get("/v1/drive/export", tags=["downloads"])
    async def drive_export_settings():
        return exporter.settings()

    @app.put("/v1/drive/export", tags=["downloads"])
    async def drive_export_update(body: DriveExportSettings):
        return exporter.save_settings(enabled=body.enabled, folder=body.folder, segments=body.segments)

    # ------------------------------------------------------------ prompts

    def _prompt_name(name: str) -> str:
        if name not in PROMPT_NAMES:
            raise NotFound(f"No prompt {name!r}; use one of {', '.join(PROMPT_NAMES)}.")
        return name

    @app.get("/v1/prompts", tags=["prompts"])
    async def prompts_list():
        return {"prompts": [service.prompts.view(name) for name in PROMPT_NAMES]}

    @app.get("/v1/prompts/{name}", tags=["prompts"])
    async def prompt_get(name: str):
        return service.prompts.view(_prompt_name(name))

    @app.put("/v1/prompts/{name}", tags=["prompts"])
    async def prompt_save(name: str, body: PromptIn):
        return service.prompts.save(_prompt_name(name), body.text)

    @app.post("/v1/prompts/{name}/reset", tags=["prompts"])
    async def prompt_reset(name: str):
        return service.prompts.save(_prompt_name(name), None)

    # ------------------------------------------------------------ agent

    @app.get("/v1/agent/models", tags=["agent"])
    async def agent_models():
        try:
            models = await service.atlas.list_models()
        except Exception as exc:  # AtlasError or network
            raise Unavailable(str(exc)) from None
        return {"models": models, "default_agent_model": settings.agent_model, "default_planner_model": settings.planner_model,
                "configured": service.atlas.configured}

    @app.post("/v1/agent/sessions", tags=["agent"], status_code=201)
    async def agent_create(body: AgentSessionIn):
        session = agent.create_session(body.title, body.persona, body.model)
        if body.name is not None or body.avatar_asset_id is not None or body.cast is not None or body.adaptive is not None:
            session = agent.update_session(session["id"], name=body.name, avatar_asset_id=body.avatar_asset_id,
                                           cast=[m.model_dump() for m in body.cast] if body.cast is not None else None,
                                           adaptive=body.adaptive)
        return agent.public(session)

    @app.get("/v1/agent/sessions", tags=["agent"])
    async def agent_list():
        return {"sessions": [agent.public(session) for session in agent.list_sessions()]}

    @app.get("/v1/agent/sessions/{session_id}", tags=["agent"])
    async def agent_get(session_id: str, after: int = 0):
        return agent.view(session_id, after)

    @app.patch("/v1/agent/sessions/{session_id}", tags=["agent"])
    async def agent_update(session_id: str, body: AgentSessionIn):
        return agent.public(agent.update_session(session_id, title=body.title, persona=body.persona, model=body.model,
                                                 name=body.name, avatar_asset_id=body.avatar_asset_id,
                                                 cast=[m.model_dump() for m in body.cast] if body.cast is not None else None,
                                                 adaptive=body.adaptive))

    @app.post("/v1/agent/sessions/{session_id}/messages", tags=["agent"], status_code=202)
    async def agent_send(session_id: str, body: AgentMessageIn):
        sent = await agent.send(session_id, body.text, body.attachments)
        return {**sent, "session": agent.public(sent["session"])}

    @app.post("/v1/agent/sessions/{session_id}/talk", tags=["agent"], status_code=202)
    async def agent_talk(session_id: str, body: AgentTalkIn):
        """Group chats: let the characters talk to each other for a few rounds (stop any time)."""
        return agent.public(agent.talk(session_id, body.rounds))

    @app.post("/v1/agent/sessions/{session_id}/compact", tags=["agent"])
    async def agent_compact(session_id: str):
        """Summarise all but the last few messages now, so each model call sends less. The full history is kept."""
        result = await agent.compact(session_id)
        return {**result, "session": agent.public(result["session"])}

    @app.post("/v1/agent/sessions/{session_id}/stop", tags=["agent"])
    async def agent_stop(session_id: str):
        return agent.public(agent.request_stop(session_id))

    @app.delete("/v1/agent/sessions/{session_id}", tags=["agent"])
    async def agent_delete(session_id: str):
        await agent.delete_session(session_id)
        return {"deleted": session_id}

    app.mount("/", mcp_app)  # serves /mcp; mounted last so the routes above win
    app.add_middleware(GZipMiddleware, minimum_size=1024, exclude_content_types=NO_GZIP)
    app.add_middleware(AuthMiddleware, token=settings.token)
    return app
