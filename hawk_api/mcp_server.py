"""MCP tools over the same HawkService the REST API uses.

Connect a chat assistant to ``<PUBLIC_BASE_URL>/t/<HAWK_API_TOKEN>/mcp`` (claude.ai
custom connector) or to ``<PUBLIC_BASE_URL>/mcp`` with an ``Authorization: Bearer``
header (Grok connectors, xAI API remote MCP, Claude Desktop/Code).
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from .jobs import Conflict, HawkService, NotFound, RequestError, Unavailable
from .schemas import LoraIn, PlannerOptions, PlanRequest, ReferenceIn, RenderSettings, VideoRequest

INSTRUCTIONS = """\
Hawk H3 Director renders MiniMax H3 videos with native audio on the user's GPU pod, one clip or a long film of joined segments.

Typical flow:
1. References: call upload_page_link and give the user the link to upload images / audio / video from their device, or call add_reference_from_url for a public file URL. Each file gets an asset_id. list_references shows what exists.
2. Plan (optional but recommended for films): plan_film with a brief and the references. Poll get_job until status is done, show the user the script, and let them edit it.
3. Render: render_film with the approved script (or plan_job_id, or story to plan and render in one job). Start with settings.megapixels=0.4 for a cheap preview.
4. Progress: poll get_job every 30-60 seconds; progress.segments_done / segments_total. When done, give the user video_url (a signed download link).
5. If a render failed with resumable=true, retry_job resumes: finished segments are reused.

Reference roles: picture (identity, outfit, place -> <Picture N>), pose (body pose only -> <Pose N>), video (motion or camera -> <Video N>), audio (voice, music -> <Audio N>), video_soundtrack (audio of video for_video).
Script rules: segments separated by a line '---'; optional headers title:, duration: (1-15 s), pictures: 1,2, poses:, videos:, audios:, continuity: (off, last_frame, tail_5, tail_22, tail_39), seed:; a 'style:' block is prepended to every segment. Number tags by the order references were listed. Write each prompt as a director's brief: reference roles, action in order, one camera move, quoted dialogue, sounds, 'Music N/A' if no score.
LoRAs: the server adds its default LoRAs (usually the turbo LoRA) automatically; list_options shows the files available on the pod and the presets.
Renders take many minutes. Never wait inside a tool call; poll get_job instead.
"""


def _error(exc: Exception) -> ValueError:
    details = getattr(exc, "details", None)
    message = str(exc)
    if details:
        message += "\nDetails: " + json.dumps(details, ensure_ascii=False)[:1500]
    return ValueError(message)


def build_mcp(service: HawkService) -> MCPServer:
    mcp = MCPServer(
        name="hawk-h3-director",
        title="Hawk MiniMax H3 Director",
        instructions=INSTRUCTIONS,
        log_level="WARNING",  # the SDK configures global logging; INFO logs every HTTP call
    )

    async def run(coro):
        try:
            return await coro
        except (RequestError, NotFound, Conflict, Unavailable) as exc:
            raise _error(exc) from None

    @mcp.tool(description="Get a signed link to a web page where the user uploads reference files (images, audio, video) from their device. The page shows each file's asset_id.")
    async def upload_page_link() -> dict:
        return {"upload_url": service.upload_page_link(), "note": "Open in a browser, upload files, then paste the asset ids into the chat."}

    @mcp.tool(description="Download a public image, audio or video URL onto the pod as a reference asset. Returns its asset_id.")
    async def add_reference_from_url(url: str, filename: str | None = None) -> dict:
        return await run(service.add_asset_from_url(url, filename))

    @mcp.tool(description="List uploaded reference assets (newest first) with asset_id, kind and file name.")
    async def list_references(limit: int = 30) -> dict:
        return {"assets": service.list_assets(limit)}

    @mcp.tool(description="Show what the pod can use: LoRA files in models/loras, the default LoRAs (and whether they are present), LoRA presets, samplers, schedulers, aspect ratios and continuity modes.")
    async def list_options() -> dict:
        return await run(service.options())

    @mcp.tool(description="Start an LLM plan: turns a brief plus references into a segment script. Returns a job; poll get_job until done, then read its script.")
    async def plan_film(
        story: str,
        references: list[ReferenceIn] | None = None,
        segment_count: int = 3,
        segment_seconds: float = 10.0,
        aspect_ratio: str = "16:9",
        model: str | None = None,
        seed: int | None = None,
    ) -> dict:
        try:
            request = PlanRequest(
                story=story, references=references or [], segment_count=segment_count,
                segment_seconds=segment_seconds, aspect_ratio=aspect_ratio, model=model, seed=seed,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        return service.job_view(await run(service.create_plan(request)))

    @mcp.tool(description=(
        "Start a video render. Give exactly one of: script (an approved Hawk H3 script), plan_job_id (a finished plan), "
        "or story (plan and render in one job). references lists assets with roles. settings controls size, length, "
        "seed, continuity and extra LoRAs. Returns a job; poll get_job."
    ))
    async def render_film(
        references: list[ReferenceIn] | None = None,
        script: str | None = None,
        plan_job_id: str | None = None,
        story: str | None = None,
        segment_count: int = 3,
        segment_seconds: float = 10.0,
        settings: RenderSettings | None = None,
        loras: list[LoraIn] | None = None,
        lora_preset: str | None = None,
    ) -> dict:
        settings = settings or RenderSettings()
        if loras is not None:
            settings = settings.model_copy(update={"loras": loras})
        if lora_preset is not None:
            settings = settings.model_copy(update={"lora_preset": lora_preset})
        try:
            request = VideoRequest(
                references=references or [],
                script=script,
                plan_job_id=plan_job_id,
                story=PlannerOptions(
                    story=story, segment_count=segment_count, segment_seconds=segment_seconds,
                    aspect_ratio=settings.aspect_ratio,
                ) if story else None,
                settings=settings,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        return service.job_view(await run(service.create_video(request)))

    @mcp.tool(description="Status of a plan or render job: progress (segments_done / segments_total), script, LoRAs applied, errors, and the signed video_url when done.")
    async def get_job(job_id: str) -> dict:
        try:
            return service.job_view(service.get_job(job_id))
        except NotFound as exc:
            raise ValueError(str(exc)) from None

    @mcp.tool(description="Recent jobs, newest first.")
    async def list_jobs(limit: int = 10) -> dict:
        return {"jobs": [service.job_view(job) for job in service.list_jobs(limit)]}

    @mcp.tool(description="Cancel a queued or running job. Finished segments stay on disk; retry_job resumes.")
    async def cancel_job(job_id: str) -> dict:
        return service.job_view(await run(service.cancel(job_id)))

    @mcp.tool(description="Resume a failed or cancelled job with the same seed and run folder; finished segments are reused.")
    async def retry_job(job_id: str) -> dict:
        return service.job_view(await run(service.retry(job_id)))

    return mcp
