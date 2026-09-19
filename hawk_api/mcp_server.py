"""MCP tools over the same HawkService the REST API uses.

Connect a chat assistant to ``<PUBLIC_BASE_URL>/t/<HAWK_API_TOKEN>/mcp`` (claude.ai
custom connector) or to ``<PUBLIC_BASE_URL>/mcp`` with an ``Authorization: Bearer``
header (Grok connectors, xAI API remote MCP, Claude Desktop/Code).
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .jobs import Conflict, HawkService, NotFound, RequestError, Unavailable
from .schemas import ImageLoraIn, LoraIn, PlannerOptions, PlanRequest, ReferenceIn, RenderSettings, VideoRequest

INSTRUCTIONS = """\
Hawk H3 Director renders MiniMax H3 videos with native audio on the user's GPU pod, one clip or a long film of joined segments.

Typical flow:
1. References: call upload_page_link and give the user the link to upload images / audio / video from their device, or call add_reference_from_url for a public file URL. Each file gets an asset_id. list_references shows what exists.
2. Plan (optional but recommended for films): plan_film with a brief and the references. Poll get_job until status is done, show the user the script, and let them edit it.
3. Render: render_film with the approved script (or plan_job_id, or story to plan and render in one job). Start with settings.megapixels=0.4 for a cheap preview.
4. Progress: poll get_job every 30-60 seconds; progress.segments_done / segments_total. One job runs at a time: status queued with queue_position N means N-th in line behind the current render. When done, give the user video_url (a signed download link); if drive.view_url is present, also mention the Google Drive copy (faster, and it doesn't go through the tunnel).
5. If a render failed with resumable=true, retry_job resumes: finished segments are reused.

Reference roles: picture (identity, outfit, place -> <Picture N>), pose (body pose only -> <Pose N>), video (motion or camera -> <Video N>), audio (voice, music -> <Audio N>), video_soundtrack (audio of video for_video).
Script rules: segments separated by a line '---'; optional headers title:, duration: (1-15 s), pictures: 1,2, poses:, videos:, audios:, continuity: (off, last_frame, tail_5, tail_22, tail_39), seed:; 'style:' (one paragraph, ended by a blank line or a header) is prepended to every segment; count segments in the job's progress.segments_total and fix the script if it differs from the scenes you wrote. Number tags by the order references were listed. Write each prompt as a director's brief: reference roles, action in order, one camera move, quoted dialogue, sounds, 'Music N/A' if no score.
LoRAs: the server adds its default LoRAs (usually the turbo LoRA) automatically. list_loras shows every LoRA file on the pod, the defaults and the presets. To use one, pass settings.loras=[{"name": "<file name or a unique part of it>", "strength": 0.6}] in render_film (or lora_preset); get_job's loras_applied confirms what was applied. strength 0 switches a default off.
Music: H3 composes new music in every segment, so a multi-segment film jumps at each cut. For one continuous track, upload it (upload page or add_reference_from_url) and pass settings.music_asset_id; the Director mixes it under the whole film (settings.music_volume_db, scene_volume_db, music_fade_seconds) and turns off each segment's own music. Do not also list the track in references.
Planner: plan_film takes model, any id from list_options.planner_models (default xai/grok-4.3); pick a vision model when references are attached.
Models: list_options shows diffusion_models (ref2va base models), text_encoders and the defaults. settings.unet_name / settings.clip_name pick others for one render, by file name or a unique part such as "bf16". bf16 gives the best quality but is slowest; int8 / fp8 / nvfp4 are faster. Omit them to use the defaults.
Renders take many minutes. Never wait inside a tool call; poll get_job instead.
""" + """
Image prompts (generate_image): write for the engine that will run it. engine auto: text only -> local Krea 2 when idle, else z-image/turbo; with reference images -> Krea 2 Identity Edit, else Seedream edit. When a retake moves to another engine (inspect_image's next_engine), rewrite the prompt for that engine.
- Every engine: natural descriptive language, not keyword lists; concrete over vague ("soft window light from the left", not "beautiful lighting"); no filler such as "masterpiece, best quality, 8K"; none of these engines reads negative prompts, so say what you want ("a plain empty wall" rather than "no clutter"); state positions and relationships ("the woman on the left in a red leather jacket"); 3-5 key visual elements; no contradictory styles.
- Krea 2 (local, text to image), 40-100 words: lead with the subject in specific visual language (textures, materials, skin, fabric), then camera and composition terms (close-up, low-angle, 85mm, shallow depth of field, tightly cropped), lighting, and a rendering word that sets the look (editorial photography, wildlife photography, candid phone snapshot, film grain, risograph, grainy VHS still). Example: "resting cheetah in close-up profile facing right, golden fur with dense black spots, sharp focus on amber eye and textured fur, paws on flat reddish-brown stone, shallow depth of field, soft dark green blurred background, warm natural daylight, wildlife photography".
- Krea 2 Identity Edit (local, reference images), 1-2 sentences: one plain instruction with a concrete verb (put, place, restage, replace, remove, change, turn, recolor, add) naming only what changes. "Put a retro red trenchcoat on her." "Turn her head to the left, she looks to her left." "Restage the photo in psychedelic rainbow light." Two images: the scene first, the person second: "Create a photo of this man next to the tractor." Replace is reliable; removals and outfit swaps less so. ref_boost about 4 keeps the likeness; lower it for bigger changes; above 10 hurts removals.
- z-image/turbo, 80-250 words, the most detailed: subject (age, features, hair, clothing, expression, action) -> environment (place, time of day, weather, background) -> style (camera body and lens, film stock such as Kodak Portra 400, or an art style) -> composition (framing, rule of thirds, depth of field). Text in the image: exact words in quotes (English or Chinese).
- Seedream 5 Pro and Lite, 30-100 words (Lite: 2-4 sentences), conversational: subject and action first, then setting, light with direction and quality ("hard midday sun", "large softbox from camera left"), materials and surfaces ("brushed aluminium, frosted glass, matte skin"), lens and framing, mood; realism comes from light and materials more than style words. Text: exact words in double quotes with placement ("a headline across the top reading \"MONSOON SALE\""); for another language write the words in it and name it; large display text is reliable, dense small text is not. Posters and infographics: give the layout in reading order. Edits: name the elements to change and say what stays ("Make the mug matte black and remove the plant in the top-right corner. Keep the lighting, the table and the composition exactly the same."); refer to several references as image 1, image 2 in reference_asset_ids order ("Put the dress from image 2 on the woman in image 1"). Improving an existing image with an edit is often better than regenerating.
"""


def _error(exc: Exception) -> ToolError:
    """An expected failure the chat model should read and act on. The SDK only passes a
    ToolError's message to the client; any other exception becomes 'Error executing tool'."""
    details = getattr(exc, "details", None)
    message = str(exc)
    if details:
        message += "\nDetails: " + json.dumps(details, ensure_ascii=False)[:1500]
    return ToolError(message)


def build_mcp(service: HawkService, drive=None, imports=None) -> MCPServer:
    mcp = MCPServer(
        name="hawk-h3-director",
        title="Hawk MiniMax H3 Director",
        instructions=INSTRUCTIONS,
        log_level="WARNING",  # the SDK configures global logging; INFO logs every HTTP call
    )

    async def _sync(fn, *args):
        return fn(*args)

    async def _viewed(coro):
        return service.asset_view(await coro)

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
        return await run(_viewed(service.add_asset_from_url(url, filename)))

    @mcp.tool(description=(
        "Search the asset library (newest first): images, videos and audio uploaded, imported from Google Drive or generated. "
        "Filter by kind (image/video/audio), collection, tag, or query (matches file name, id, tags, collection). "
        "Each asset has id, kind, filename, collection, tags and a thumb_url for images and videos."
    ))
    async def list_references(limit: int = 30, kind: str | None = None, collection: str | None = None,
                              tag: str | None = None, query: str | None = None) -> dict:
        items, total = service.search_assets(kind=kind, collection=collection, tag=tag, query=query, limit=max(1, min(limit, 200)))
        return {"assets": [service.asset_view(asset) for asset in items], "total": total}

    @mcp.tool(description="List the library's collections (with file counts per kind) and tags.")
    async def list_collections() -> dict:
        return {"collections": service.collections(), "tags": service.tags()}

    @mcp.tool(description="Move assets to a collection and/or add or remove tags, to keep the library organised.")
    async def organize_assets(asset_ids: list[str], collection: str | None = None, add_tags: list[str] | None = None,
                              remove_tags: list[str] | None = None) -> dict:
        updated = []
        for asset_id in asset_ids:
            try:
                updated.append(service.asset_view(service.update_asset(asset_id, collection=collection, add_tags=add_tags, remove_tags=remove_tags)))
            except NotFound as exc:
                raise ToolError(str(exc)) from None
        return {"assets": updated}

    @mcp.tool(description=(
        "Generate or edit images. engine 'auto' (default, text only) tries local Krea 2 on this GPU (free; used when installed "
        "and ComfyUI is idle), then Atlas z-image/turbo (about $0.01), then Seedream v5.0 Pro. engine 'local', 'turbo', "
        "'seedream' or 'seedream-lite' picks one; Seedream Pro costs about $0.036 an image up to 2.36 MP and $0.072 above "
        "(e.g. 2048x2048), so stay at 1536x1536 or smaller unless the user wants high resolution; Lite gives 2K+ for about "
        "$0.032, a little below Pro in quality. Results carry cost_usd for Atlas images. the result says which engine made it and what was skipped (tried). loras (local Krea 2 only): "
        "[{name, strength}] from image_options, e.g. a realism or detail LoRA for photo portraits, a style LoRA for a look; follow "
        "the GO-TO / AVOID notes; up to 3 adult LoRAs, only for fictional adults the user explicitly asked for. "
        "Edits: with reference_asset_ids, 'auto' / 'local' use Krea 2 Identity Edit on this GPU when installed and idle (free; "
        "1 image, or 2: the scene first, then the person to place in it; plain-English instructions like 'Change her outfit "
        "to a red raincoat', 'Place this person at the cafe table'; ref_boost is the likeness dial: 4 default, 1 looser, "
        "keep it under 10), else Seedream edit (up to 10 images; also 'seedream'). z-image can't edit. Keep a face, change "
        "outfit, pose, scene, light or style, make variations. Krea 2 edits of uploaded photos (not images made here) are "
        "refused if sexual or with adult LoRAs. Returns new image assets (asset_id, thumb_url) that work as picture "
        "references in plan_film and render_film, e.g. to lock a character's identity across segments. size like 1024x1536 "
        "or 1536x1536 (z-image: 512-2048 a side; Seedream snaps to its nearest preset) is optional; n is 1-4. Write rich, "
        "specific prompts: subject, face, hair, expression, outfit, setting, light, camera and lens, mood, style. Never create sexual content involving anyone who appears under 18, or sexual or nude images of real, identifiable people."
    ))
    async def generate_image(
        prompt: str,
        reference_asset_ids: list[str] | None = None,
        model: str | None = None,
        size: str | None = None,
        n: int = 1,
        seed: int | None = None,
        engine: str | None = None,
        loras: list[ImageLoraIn] | None = None,
        steps: int | None = None,
        ref_boost: float | None = None,
    ) -> dict:
        return await run(service.generate_images(prompt, model=model, reference_asset_ids=reference_asset_ids or [],
                                                 size=size, n=max(1, min(4, n)), seed=seed, engine=engine,
                                                 loras=[l.model_dump() for l in loras or []], steps=steps,
                                                 ref_boost=None if ref_boost is None else max(0.0, min(20.0, ref_boost))))

    @mcp.tool(description=(
        "Image engines and Krea 2 LoRAs: whether local Krea 2 is installed and busy (a video render is using ComfyUI), and "
        "each LoRA's file, kind (realism, detail, style, adult), label, default strength and range, trigger and notes. Call it "
        "before choosing loras for generate_image."
    ))
    async def image_options() -> dict:
        return await run(service.image_options())

    @mcp.tool(description="Show what the pod can use: LoRA files in models/loras, the default LoRAs (and whether they are present), LoRA presets, ref2va base models and text encoders (with the defaults), planner model ids, samplers, schedulers, aspect ratios and continuity modes.")
    async def list_options() -> dict:
        options = await run(service.options())
        # Compact for chat models: LoRAs first, and planner models as ids (the full catalogue
        # with pricing is tens of KB and pushed the LoRA list out of view).
        first = ("available_loras", "default_loras", "lora_presets")
        compact = {key: options[key] for key in first}
        compact.update({key: value for key, value in options.items() if key not in first})
        compact["planner_models"] = [model["id"] for model in options.get("planner_models") or []]
        return compact

    @mcp.tool(description="LoRA files on the pod (models/loras), the default LoRAs the server adds (and whether they are present) and the presets from loras.json. Use a file name, or a unique part of it, in render_film settings.loras.")
    async def list_loras() -> dict:
        options = await run(service.options())
        return {key: options[key] for key in ("available_loras", "default_loras", "lora_presets")}

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
            raise ToolError(str(exc)) from None
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
            raise ToolError(str(exc)) from None
        return service.job_view(await run(service.create_video(request)))

    @mcp.tool(description="Status of a plan or render job: status (queued with queue_position, planning, rendering, done, failed, cancelled), progress (segments_done / segments_total), script, models and LoRAs applied, warnings, errors, and the signed video_url when done.")
    async def get_job(job_id: str) -> dict:
        try:
            return service.job_view(service.get_job(job_id))
        except NotFound as exc:
            raise ToolError(str(exc)) from None

    @mcp.tool(description="Recent jobs, newest first.")
    async def list_jobs(limit: int = 10) -> dict:
        return {"jobs": [service.job_view(job) for job in service.list_jobs(limit)]}

    @mcp.tool(description="Cancel a queued or running job. Finished segments stay on disk; retry_job resumes.")
    async def cancel_job(job_id: str) -> dict:
        return service.job_view(await run(service.cancel(job_id)))

    @mcp.tool(description="Resume a failed or cancelled job with the same seed and run folder; finished segments are reused.")
    async def retry_job(job_id: str) -> dict:
        return service.job_view(await run(service.retry(job_id)))

    if drive is not None and imports is not None:
        @mcp.tool(description=(
            "Browse the user's Google Drive mounted on the server: folders and media files (image/audio/video) in a folder. "
            "path is relative to My Drive ('' for the top). Returns available=false when Drive is not mounted."
        ))
        async def browse_drive(path: str = "") -> dict:
            if not drive.available:
                return {"available": False, "hint": "Ask the user to mount Google Drive in the Colab notebook."}
            return await run(_sync(drive.browse, path))

        @mcp.tool(description=(
            "Import files or whole folders from the mounted Google Drive into the asset library (copied on the server, no size limit; "
            "duplicates are skipped). collection defaults to the folder name. Waits up to wait_seconds and returns progress with "
            "the imported asset ids; call get_import for a longer import."
        ))
        async def import_from_drive(paths: list[str], collection: str | None = None, tags: list[str] | None = None,
                                    recursive: bool = True, wait_seconds: int = 60) -> dict:
            state = await run(imports.start(paths, recursive=recursive, collection=collection, tags=tags or []))
            return await imports.wait(state["id"], max(0, min(wait_seconds, 300)))

        @mcp.tool(description="Progress of a Google Drive import: total, done, imported, duplicates, failed and asset ids.")
        async def get_import(import_id: str) -> dict:
            try:
                return imports.get(import_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

    return mcp
