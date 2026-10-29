"""MCP tools over the same HawkService the REST API uses.

Connect a chat assistant to ``<PUBLIC_BASE_URL>/t/<HAWK_API_TOKEN>/mcp`` (claude.ai
custom connector) or to ``<PUBLIC_BASE_URL>/mcp`` with an ``Authorization: Bearer``
header (Grok connectors, xAI API remote MCP, Claude Desktop/Code).
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from hawk_h3.script import max_segment_seconds

from . import pose_guide as pose_guides
from . import storyboard as storyboards
from .jobs import Conflict, HawkService, NotFound, RequestError, Unavailable
from .schemas import FilmIn, ImageLoraIn, LoraIn, ManhwaContextIn, ManhwaConvertIn, ManhwaPipelineIn, ManhwaSettingsIn, PlannerOptions, PlanRequest, ReferenceIn, RenderSettings, VideoRequest

INSTRUCTIONS = """\
Hawk H3 Director renders MiniMax H3 videos with native audio on the user's GPU pod, one clip or a long film of joined segments.

Typical flow:
1. References: call upload_page_link and give the user the link to upload images / audio / video from their device, or call add_reference_from_url for a public file URL. Each file gets an asset_id. list_references shows what exists.
   Images the user attaches in this chat do NOT reach the server: you can see them, but you cannot pass them to a tool (upload_image needs the file's bytes, which a chat attachment does not give you). So when the user attaches a picture to edit or to use as a reference, never call generate_image or render_film without it and never claim you used it: call upload_page_link, ask them to upload the same picture there, then call list_references(kind="image", limit=5) -- the newest is theirs -- confirm it by its filename or thumb_url, and pass its asset_id.
2. Plan: plan_film with a brief and the references. Poll get_job until status is done, show the user the script, and let them edit it. Whenever the video uses reference pictures (any character or face to keep), always plan -- with plan_film, or render_film with story -- even for a dance or music video with no dialogue: the planner writes H3's six-section reference format (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music), which is what keeps each face tied to its picture. Put the people, their looks and outfits, the action, camera and music in the brief; the planner does the rest.
   Several shots of one recurring character (a reel, a photoshoot, a dance, an adult scene)? make_film is the one call: name the characters from the library (list_characters; save_character once), give an idea, the platform and the quality -- the shots are written, drawn as $0.018 Seedream Flash sheets, checked, and filmed after the user approves (approve_film). The storyboard_* tools below are its parts, for step-by-step control. Up to 40 shots; get_storyboard shows the panels and the film, redo_storyboard_panel fixes one shot ($0.018). For a recurring character, make_identity_pack once gives clean front/back/side/face references to use everywhere. The user's own images need no drawing: storyboard_film with panel_asset_ids, or compose_sheet then storyboard_film with sheet_asset_id ($0 for the images).
3. Render: render_film with plan_job_id (or the approved, possibly edited script, or story to plan and render in one job). A hand-written script is for clips with no reference pictures, or one already in the six-section format; get_job warns when a script with pictures skips it. Start with settings.megapixels=0.4 for a cheap preview.
4. Progress: poll get_job every 30-60 seconds; progress.segments_done / segments_total. One job runs at a time: status queued with queue_position N means N-th in line behind the current render. When done, give the user video_url (a signed download link); if drive.view_url is present, also mention the Google Drive copy (faster, and it doesn't go through the tunnel).
5. If a render failed with resumable=true, retry_job resumes: finished segments are reused.

Reference roles: picture (identity, outfit, place -> <Picture N>), pose (body pose only -> <Pose N>), video (motion or camera -> <Video N>), audio (voice, music -> <Audio N>), video_soundtrack (audio of video for_video).
Script rules: segments separated by a line '---'; optional headers title:, duration: (1-{cap} s; longer is clamped, so split a longer beat into two segments), pictures: 1,2, poses:, videos:, audios:, continuity: (off, last_frame, tail_5, tail_22, tail_39), seed:; 'style:' (one paragraph, ended by a blank line or a header) is prepended to every segment; count segments in the job's progress.segments_total and fix the script if it differs from the scenes you wrote. Number tags by the order references were listed. Write each prompt as a director's brief: reference roles, action in order, one camera move, dialogue, sounds. plan_film writes the full H3 format and is the path for anything with reference pictures or dialogue; when you write a script yourself, keep these rules, because whatever audio a prompt leaves unscripted H3 fills with invented speech and noise:
- Dialogue: every line names who says it, the person's short description plus a stable speaker ID in order of first speaking, then the words in a <d> tag with a language: the woman in the red saree (S1) says, <d>[Hindi] यार, आज का weather एकदम perfect है।</d> Describe each speaker's voice once, on their first line (age, gender, pitch, pace, accent). After each line say what the speaker does next and that their lips close; one speaker per sentence. About 2 spoken words per second of the segment.
- Hinglish (the default): Hindi words in Devanagari, English words in Latin letters as spelled in English, tag [Hindi]. Never Hindi in Roman letters ("hai", "nahi"): H3 reads those as English and mispronounces them. Keep any words the user wrote for a line exactly as written.
- Sound: end each segment with one main ambient sound, then an exclusion sentence naming the language: "Only their Hindi lines; no other voices, no background chatter, no other language at any point." A segment without dialogue says "No speech, no voices; all lips stay closed." Fill silent gaps with visible actions. Music N/A unless asked; with dialogue, music very low or N/A.
Sex scenes (adults only): before writing any explicit prompt -- a video segment or an image -- call pose_guide with the position (no argument lists them all) and write the bodies the way it does. The models know no position names: spell out where each body is, what holds it up, where all four limbs of each and both pairs of hands go, where the faces point, the camera angle that shows the contact, and the motion. One position per segment; vaginal or anal stated plainly. plan_film already gets the guide on its own.
LoRAs: the server adds its default LoRAs (usually the turbo LoRA) automatically. list_loras shows every LoRA file on the pod, the defaults and the presets. To use one, pass settings.loras=[{"name": "<file name or a unique part of it>", "strength": 0.6}] in render_film (or lora_preset); get_job's loras_applied confirms what was applied. strength 0 switches a default off.
Music: H3 composes new music in every segment, so a multi-segment film jumps at each cut. For one continuous track, upload it (upload page or add_reference_from_url) and pass settings.music_asset_id; the Director mixes it under the whole film (settings.music_volume_db, scene_volume_db, music_fade_seconds) and turns off each segment's own music. Do not also list the track in references.
Planner: plan_film takes model, any id from list_options.planner_models (default: the server's planner chain, list_options.model_chains.planner); pick a vision model when references are attached. The call is made by the gateway on whichever provider its settings name, so an id from another provider is matched by name where it can be.
Models: list_options shows diffusion_models (H3 base models that read references: ref2va, hybrids such as 10Eros-Max beta5, and turbo bases such as 10Eros-Max beta6), text_encoders and the defaults. settings.unet_name / settings.clip_name pick others for one render, by file name or a unique part such as "bf16". bf16 gives the best quality but is slowest; int8 / fp8 / nvfp4 are faster. Omit them to use the defaults.
Renders take many minutes. Never wait inside a tool call; poll get_job instead.
""" + """
Image prompts (generate_image): write for the engine that will run it. engine auto: text only -> local Qwen Image 2.1 when idle, then Krea 2, else z-image/turbo; with reference images -> Qwen Image 2.1, then Krea 2 Identity Edit, else Seedream edit. When a retake moves to another engine (inspect_image's next_engine), rewrite the prompt for that engine.
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


def build_mcp(service: HawkService, drive=None, imports=None, manhwa=None, stories=None, boards=None,
              characters=None) -> MCPServer:
    mcp = MCPServer(
        name="hawk-h3-director",
        title="Hawk MiniMax H3 Director",
        instructions=INSTRUCTIONS.replace("{cap}", f"{max_segment_seconds():g}"),
        log_level="WARNING",  # the SDK configures global logging; INFO logs every HTTP call
    )

    async def _sync(fn, *args, **kwargs):
        return fn(*args, **kwargs)

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

    @mcp.tool(description=(
        "Upload an image whose file bytes you actually hold (a file in your own workspace) to the library, as base64 "
        "or a data: URI. Returns its asset_id. It is stored as an upload and copied to the user's Google Drive image folder. "
        "An image attached in the chat is NOT such a file -- you cannot produce its bytes, so do not try: use "
        "upload_page_link and have the user upload it there."))
    async def upload_image(data_base64: str, filename: str = "upload.png", collection: str | None = None,
                           tags: list[str] | None = None) -> dict:
        return await run(_viewed(service.add_asset_from_base64(data_base64, filename, collection, tags)))

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
        "Generate or edit images. engine 'auto' (default) tries local Qwen Image 2.1 on this GPU (free; used when "
        "installed and ComfyUI is idle), then local Krea 2, then Atlas z-image/turbo (about $0.01), then Seedream "
        "v5.0 Pro; with reference images it edits with Qwen Image 2.1, then Krea 2, then Seedream. engine 'qwen21', "
        "'krea2', 'zimage', 'local', 'turbo', 'seedream' or 'seedream-lite' picks one ('klein' still "
        "resolve, to qwen21). Seedream Pro costs about $0.036 an image up to 2.36 MP and $0.072 above (e.g. "
        "2048x2048), so stay at 1536x1536 or smaller unless the user wants high resolution; Lite gives 2K+ for about "
        "$0.032, a little below Pro in quality. 'nano-banana' (Google Nano Banana 2, about $0.067) and "
        "'nano-banana-pro' (about $0.134) and 'nano-banana-lite' (about $0.034, 1K only) generate and edit with up to 14 references, SFW only: an SFW check runs "
        "first and anything it does not pass goes to the next engine (they are off until switched on in Studio). "
        "Results carry cost_usd for paid images. The result says which engine "
        "made it and what was skipped (tried). loras (local engines only): [{name, strength}] from image_options, from "
        "the family of the engine that will run -- a Qwen 2.1 LoRA cannot go on Krea 2 or the reverse; follow the GO-TO "
        "/ AVOID notes. Qwen Image 2.1 always adds its own repair LoRA, and each family's adult default is attached to "
        "local generations automatically; naming your own adult LoRA replaces that default, up to 3. "
        "Prompts: Qwen 2.1 wants a long paragraph describing the finished frame as if you were looking at it -- medium "
        "and style first, then the background, then the frame walked with positional phrases, light its own sentence, "
        "and text to be rendered in straight double quotes. Give ages as a life stage ('in her thirties'), never a "
        "number of years. No 'masterpiece, 8K'. "
        "An edit needs the picture on the server as an asset_id: an image attached in the chat is not on the server -- "
        "get it there with upload_page_link (see the instructions), never edit without it. "
        "Edits: with reference_asset_ids, 'auto' / 'local' use Qwen Image 2.1 on this GPU when idle (free; up to 16 "
        "images). Refer to them in the prompt as <image1>, <image2> in the order given -- nothing labels them for you, "
        "so 'the woman in the second photo' points at nothing -- but that tagging is for TWO OR MORE references "
        "only. With a SINGLE reference prefer no tag: Qwen's rewriter refers to one image naturally ('the image', "
        "'the woman in the image'), which is the phrasing the model saw most; a tag there is not an error and still "
        "points at the only image, so never strip one the user wrote. Krea 2 takes references as latents and never "
        "reads a tag either way, so <image1> in a Krea prompt is text it will try to draw. "
        "Two kinds of edit want opposite amounts of writing. "
        "CHANGING ONE PICTURE (swap an object, replace a background, recolour, restyle): the canvas is the picture the "
        "change happens to, which is not always the one passed first -- for a face swap it is the body, for a change of "
        "clothes the person not the garment, for a restyle the picture being restyled not the style reference, for a "
        "new background the picture holding the subject. Its framing and untargeted content survive and the output "
        "takes its size, so pass it as <image1> where you can and say which one it is; lead with the operation, name only what "
        "changes, and cover the rest with one short clause ('Keep the room and the lighting of <image1> exactly as they "
        "are'). Do not describe a face or garment you want kept -- a preservation description reads as a paint "
        "instruction and the thing drifts; point at its image instead. BUILDING A NEW PICTURE from references (a "
        "character or model sheet, a shoot, a poster, a cosplay, the same person moved to another era, a person from "
        "one image wearing clothes from others): one reference is enough for this -- what decides the job is whether "
        "the user wants that picture altered or a new picture of that person, not how many images there are. Nothing is "
        "preserved from a canvas, so build it fully in this order -- (1) a role clause per reference, each image named "
        "separately ('Using the woman in <image1> as the sole identity reference', 'the dress from <image2>'); (2) an "
        "identity lock enumerating face shape, eye colour, skin tone, hair, and small fixed details like a bindi or "
        "particular earrings, ending 'completely unchanged'; (3) the artifact named in one phrase ('a two-panel "
        "full-body model character sheet'); (4) one block per panel opening on its position, with the pose and which "
        "reference's item it uses; (5) each transferred item described as a pattern cutter would -- neckline, closure, "
        "hardware, seams, cutout shapes and placement -- ending 'reproduced with exact <what usually goes wrong>'; (6) "
        "backdrop, lighting and framing once for all regions; (7) closing invariants across references and between "
        "regions ('both panels show identical facial identity and hairstyle'). When nothing is being altered there is no "
        "canvas, so no reference decides the shape: state it in size. Never write a size, a ratio or a resolution into "
        "the prompt text ('2K', '16:9', '1024x1024') -- those go in size, and this model renders text well enough to "
        "print a stray ratio across the picture. Write the prompt in English whatever language the request came in; only "
        "text meant to be legible in the image keeps another language, in one language per quoted string, taken from "
        "what the user quoted, else what already appears in the reference, else the language they asked in. "
        "Krea 2 (engine 'krea2') is the "
        "1-or-2 image alternative -- scene first, then the person -- and is the only engine using ref_boost (likeness "
        "dial: 4 default, 1 looser, under 10). Seedream edit takes up to 10 ('seedream'). z-image can't edit. "
        "Local edits of uploaded photos (not images made here) are refused if sexual or with adult LoRAs. Returns new "
        "image assets (asset_id, thumb_url) that work as picture references in plan_film and render_film, e.g. to lock "
        "a character's identity across segments. size like 1024x1536 or 1536x1536 (z-image: 512-2048 a side; Seedream "
        "snaps to its nearest preset) is optional; n is 1-4. Never create sexual content involving anyone who appears "
        "under 18, or sexual or nude images of real, identifiable people."
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
        "Image engines and their LoRAs: the generate and edit ladders with each engine's max_refs and whether it is "
        "ready, which local engines are installed and whether ComfyUI is busy (a video render), and per family each "
        "LoRA's file, kind (realism, detail, style, adult), label, default strength and range, trigger and notes. Call "
        "it before choosing loras for generate_image, and read lora_family off the engine that will run: a LoRA only "
        "works on its own family."
    ))
    async def image_options() -> dict:
        return await run(service.image_options())

    @mcp.tool(description="Show what the pod can use: LoRA files in models/loras, the default LoRAs (and whether they are present), LoRA presets, H3 base models (ref2va, hybrid and turbo bases) and text encoders (with the defaults), planner model ids, samplers, schedulers, aspect ratios and continuity modes.")
    async def list_options() -> dict:
        options = await run(service.options())
        # Compact for chat models: LoRAs first, and planner models as ids (the full catalogue
        # with pricing is tens of KB and pushed the LoRA list out of view).
        first = ("available_loras", "default_loras", "lora_presets")
        compact = {key: options[key] for key in first}
        compact.update({key: value for key, value in options.items() if key not in first})
        compact["planner_models"] = [model["id"] for model in options.get("planner_models") or []]
        return compact

    @mcp.tool(description="Video LoRA files on the pod (the MiniMax H3 ones in models/loras), the default LoRAs the server adds to every render (and whether they are present) and the presets from loras.json. Use a file name, or a unique part of it, in render_film settings.loras. Krea 2 image LoRAs are not listed here and cannot be used in a render: image_options lists those, and they belong to generate_image.")
    async def list_loras() -> dict:
        options = await run(service.options())
        return {key: options[key] for key in ("available_loras", "default_loras", "lora_presets")}

    @mcp.tool(description=(
        "Sex-position guide for explicit scenes between adults: for a position (e.g. 'standing full nelson', "
        "'cowgirl', 'doggy'), how to write it so a video or image model draws it -- each body's placement and "
        "support, hands, faces, the camera angle that shows it, the motion, what models usually get wrong, and a "
        "ready prompt sentence -- plus the rules every position shares. No argument lists every position. Call it "
        "before writing any explicit prompt."))
    async def pose_guide(pose: str | None = None) -> dict:
        return pose_guides.lookup(pose)

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

    async def boarded(coro):
        if boards is None:
            coro.close()
            raise ToolError("Storyboards aren't available on this server.")
        try:
            return await run(coro)
        except storyboards.StoryboardError as exc:
            raise ToolError(str(exc)) from None

    @mcp.tool(description=(
        "Storyboard step 1: draw up to 8 shots of one character as a single Seedream Flash sheet ($0.018 for the "
        "whole sheet instead of one image per shot; the same face, body and light in every shot). beats: one short "
        "shot description per panel, in story order. character: who they are and how they look. "
        "reference_asset_ids: the character's identity picture(s), ideally full body on a plain background. "
        "aspect_ratio: the video's, so the panels come out that shape. Returns the sheet and its grid for step 2."))
    async def make_storyboard_sheet(beats: list[str], character: str, reference_asset_ids: list[str],
                                    aspect_ratio: str = "9:16", style: str = "") -> dict:
        return await boarded(storyboards.make_sheet(service, beats, character, reference_asset_ids, aspect_ratio, style))

    @mcp.tool(description=(
        "Storyboard step 2: cut a sheet into its panels (columns x rows from step 1's grid; 0 and 0 find the grid from "
        "a merged sheet's record or the sheet's borders, for any sheet). Each panel is stored as "
        "an image asset in reading order. Look at the panels before step 3 and redo any bad one with generate_image."))
    async def split_storyboard_sheet(sheet_asset_id: str, columns: int = 0, rows: int = 0, panels: int | None = None) -> dict:
        return await boarded(storyboards.split_sheet(service, sheet_asset_id, columns, rows, panels))

    @mcp.tool(description=(
        "Storyboard step 3: render a film of the panels in order -- each segment opens on its panel (keyframe "
        "completion) with the identity picture beside it, planned in the H3 reference format. beats: one motion per "
        "panel (what happens from that frame). Returns a render job; poll get_job."))
    async def render_storyboard(panel_asset_ids: list[str], identity_asset_id: str, beats: list[str], character: str,
                                segment_seconds: float = 6.0, settings: RenderSettings | None = None,
                                mood: str = "", music: str = "") -> dict:
        job = await boarded(storyboards.render_film(service, panel_asset_ids, identity_asset_id, beats, character,
                                                    settings, segment_seconds, mood, music))
        return service.job_view(job)

    @mcp.tool(description=(
        "THE way to make a multi-shot video (a reel, a photoshoot, a dance, an adult scene, anything with a recurring "
        "character): who (characters, names from the character library -- or identity_asset_ids), what (an idea, and "
        "the shots, motions and a mode per shot are written for you; or your own shots, panel_asset_ids or a "
        "sheet_asset_id), where it goes (platform: instagram keeps everyone clothed and checks it; fanvue allows "
        "adult nudity), quality (draft fast, final to post). The engine draws Seedream Flash sheets ($0.018 per 8 "
        "shots), checks every panel with a vision model and redraws rejects, then films: frame shots on the "
        "face-faithful base model, look shots (dancing, positions -- the video model composes them) on the "
        "action one, joined into one video with one music track (music_prompt composes one, about $0.15). "
        "review ask (default): it stops at status review -- show the user the panels (get_storyboard: panel_views, "
        "checks) and call approve_film; review auto films straight away. Returns the board; poll get_storyboard."))
    async def make_film(film: FilmIn) -> dict:
        if boards is None:
            raise ToolError("Films aren't available on this server.")
        return await boarded(storyboards.make_film(boards, settings=film.settings, **film.model_dump(exclude={"settings"})))

    @mcp.tool(description="Approve a film stopped at status review (after showing the user its panels): it renders now. Poll get_storyboard.")
    async def approve_film(board_id: str, segment_seconds: float | None = None, mood: str | None = None) -> dict:
        return await boarded(_sync(boards.approve, board_id, None, segment_seconds, mood))

    @mcp.tool(description="The character library: every saved person with their facts, identity pictures, pack views and default platform. Use their names in make_film.")
    async def list_characters() -> dict:
        if characters is None:
            raise ToolError("The character library isn't available on this server.")
        return {"characters": characters.list()}

    @mcp.tool(description=(
        "Save a person to the character library once, so every film can just name them: facts (what every prompt "
        "keeps -- age, ethnicity, figure, hair; adults only), identity_asset_ids (up to 3, full body on a plain "
        "background is best), notes (wording that fixes a model's habit), platform (their usual destination). An "
        "existing name is updated with whatever is given."))
    async def save_character(name: str, facts: str = "", identity_asset_ids: list[str] | None = None, notes: str | None = None,
                             platform: str | None = None) -> dict:
        if characters is None:
            raise ToolError("The character library isn't available on this server.")
        try:
            characters.find(name)
        except NotFound:
            return await run(_sync(characters.create, name, facts, identity_asset_ids or [], notes or "", platform or "general"))
        return await run(_sync(characters.update, name, facts=facts or None, identity_asset_ids=identity_asset_ids,
                               notes=notes, platform=platform))

    @mcp.tool(description="Draw a library character's identity pack ($0.018): front, back, side and face views, kept on the character. anatomy=true adds nude views (adult platforms only).")
    async def make_character_pack(name: str, anatomy: bool = False) -> dict:
        if characters is None:
            raise ToolError("The character library isn't available on this server.")
        return await run(characters.make_pack(name, anatomy=anatomy))

    @mcp.tool(description=(
        "Storyboard film in one call -- the cheapest way to a multi-shot video of one recurring character: the shots "
        "are drawn as Seedream Flash sheets ($0.018 per 8 shots), cut into panels, and as soon as that is done the "
        "film renders with each segment opening on its panel; up to 40 shots (rendered in parts and joined). "
        "shots: one per line, in order (pose, place, clothing). identity_asset_ids: the character's identity "
        "picture(s), ideally full body on a plain background. Returns the board: poll get_storyboard "
        "(drawing -> rendering -> joining -> done, with film). panel_asset_ids: the user's own images as the panels "
        "instead (nothing drawn, no image cost; shots may be empty, motions say what moves). sheet_asset_id: a "
        "ready-made sheet instead -- one from compose_sheet (its images become the panels), a sheet drawn here, or "
        "one the user made elsewhere (cut into panels; grid from sheet_columns/sheet_rows or found from its borders). "
        "No shots at all: give an idea (and shot_count) and the shots, motions, character and look are written for "
        "you first from the idea and the identity pictures (write_storyboard_shots does only that step)."))
    async def storyboard_film(identity_asset_ids: list[str], shots: list[str] | None = None, character: str = "",
                              title: str = "Storyboard", idea: str = "", shot_count: int = 8,
                              aspect_ratio: str = "9:16", style: str = "", motions: list[str] | None = None,
                              segment_seconds: float = 6.0, settings: RenderSettings | None = None,
                              mood: str = "", music: str = "", panel_asset_ids: list[str] | None = None,
                              sheet_asset_id: str | None = None, sheet_columns: int = 0, sheet_rows: int = 0) -> dict:
        follow = {"settings": settings.model_dump() if settings else None, "segment_seconds": segment_seconds,
                  "mood": mood, "music": music}
        if not shots and not panel_asset_ids and not sheet_asset_id:
            if not idea:
                raise ToolError("Give shots, your own panel_asset_ids or a sheet_asset_id -- or an idea for the shots to be written.")
            written = await boarded(storyboards.suggest_shots(service, idea, identity_asset_ids, shot_count, aspect_ratio,
                                                              character=character))
            shots, motions = written["shots"], motions or written["motions"]
            character, style = character or written["character"], style or written["style"]
        return await boarded(_sync(boards.create, title=title, character=character or title, identity_asset_ids=identity_asset_ids,
                                   shots=shots or [], aspect_ratio=aspect_ratio, style=style, motions=motions,
                                   then_render=follow, panel_asset_ids=panel_asset_ids, sheet_asset_id=sheet_asset_id,
                                   sheet_columns=sheet_columns, sheet_rows=sheet_rows))

    @mcp.tool(description=(
        "Write a storyboard for the user to check: from a one-line idea and the character's reference image(s), an "
        "LLM (Gemma 4 31B, then DeepSeek, then Kimi K2.6; the editable 'storyboard' system prompt) writes the "
        "character as the images show them, the look, and count shot lines and motion lines. Nothing is drawn; pass "
        "them to create_storyboard or storyboard_film. character: what is already known about them (age, ethnicity, "
        "figure) -- kept as fixed facts, so the writer does not guess them from the picture."))
    async def write_storyboard_shots(idea: str, reference_asset_ids: list[str] | None = None, count: int = 8,
                                     aspect_ratio: str = "9:16", character: str = "") -> dict:
        return await boarded(storyboards.suggest_shots(service, idea, reference_asset_ids or [], count, aspect_ratio,
                                                       character=character))

    @mcp.tool(description=(
        "Storyboard board without rendering: draws the shots (Seedream Flash, $0.018 per 8) and cuts the panels, then "
        "waits so they can be checked; fix any with redo_storyboard_panel, then render_storyboard_board. The board "
        "also shows on the Studio Storyboard page. panel_asset_ids: the user's own images as the panels (nothing drawn); "
        "sheet_asset_id: a ready-made sheet, cut into its panels."))
    async def create_storyboard(shots: list[str], character: str, identity_asset_ids: list[str], title: str = "Storyboard",
                                aspect_ratio: str = "9:16", style: str = "", motions: list[str] | None = None,
                                panel_asset_ids: list[str] | None = None, sheet_asset_id: str | None = None,
                                sheet_columns: int = 0, sheet_rows: int = 0) -> dict:
        return await boarded(_sync(boards.create, title=title, character=character, identity_asset_ids=identity_asset_ids,
                                   shots=shots, aspect_ratio=aspect_ratio, style=style, motions=motions,
                                   panel_asset_ids=panel_asset_ids, sheet_asset_id=sheet_asset_id,
                                   sheet_columns=sheet_columns, sheet_rows=sheet_rows))

    @mcp.tool(description="A storyboard board: status, shots, sheets, panel images (panel_views), render jobs and the finished film.")
    async def get_storyboard(board_id: str) -> dict:
        return await boarded(_sync(boards.view, board_id))

    @mcp.tool(description=(
        "Draw one shot of a board again on its own ($0.018), optionally with new shot text; keep_setting gives the old "
        "panel as a reference so only what was wrong changes. Or pass asset_id to use an image from the library."))
    async def redo_storyboard_panel(board_id: str, number: int, shot: str | None = None, keep_setting: bool = True,
                                    asset_id: str | None = None) -> dict:
        if asset_id:
            return await boarded(_sync(boards.set_panel, board_id, number, asset_id))
        return await boarded(boards.redo_panel(board_id, number, shot=shot, keep_setting=keep_setting))

    @mcp.tool(description="Render a drawn board as one film (each shot opens on its panel; parts of up to 9 pictures, joined). Poll get_storyboard.")
    async def render_storyboard_board(board_id: str, segment_seconds: float = 6.0, settings: RenderSettings | None = None,
                                      mood: str = "", music: str = "") -> dict:
        return await boarded(_sync(boards.start_render, board_id, settings, segment_seconds, mood, music))

    @mcp.tool(description=(
        "Merge existing images into one sheet image -- no model call, $0. For a contact sheet to share, or to give a "
        "render several views of a character as ONE reference picture (one slot of its nine). fit: contain keeps each "
        "image whole, cover fills each cell and crops. columns 0 picks the most square grid."))
    async def compose_sheet(asset_ids: list[str], columns: int = 0, cell_aspect: str = "", fit: str = "contain",
                            border: int = 24, background: str = "black", title: str = "") -> dict:
        return await boarded(storyboards.compose_sheet(service, asset_ids, columns, cell_aspect, fit, border, background, title))

    @mcp.tool(description=(
        "A character's identity pack: front, back, side and face views as one $0.018 Seedream Flash sheet, cut into "
        "panels tagged identity-pack -- clean references for every later image and video of them. anatomy=true adds a "
        "nude front view and a rear close-up (adult platforms only)."))
    async def make_identity_pack(name: str, identity_asset_ids: list[str], description: str = "", style: str = "",
                                 anatomy: bool = False) -> dict:
        return await boarded(storyboards.identity_pack(service, name, description, identity_asset_ids, style, anatomy))

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

    if manhwa is not None:
        @mcp.tool(description=(
            "A manhwa chapter as a live-action film through the film engine: its panels (blank, repeated and "
            "under-18 ones left out) turned into live-action stills eight at a time on Seedream Flash ($0.018 per 8) "
            "with the cast's portraits for their faces, checked, then filmed with each panel's lines word for word. "
            "review ask (default) stops for approval: show the user the panels (get_storyboard) and call "
            "approve_film. Needs the cast's portraits (make_manhwa_cast). tile_ids: only these tiles, to leave scenes "
            "out (ids from get_manhwa_chapter)."))
        async def make_manhwa_film(chapter_id: str, platform: str = "general", quality: str = "final", review: str = "ask",
                                   style: str = "", aspect_ratio: str = "9:16", max_panels: int = 40,
                                   music_prompt: str = "", segment_seconds: float = 5.0,
                                   tile_ids: list[str] | None = None) -> dict:
            return await run(manhwa.make_film(chapter_id, platform=platform, quality=quality, review=review, style=style,
                                              aspect_ratio=aspect_ratio, max_panels=max_panels,
                                              music_prompt=music_prompt, segment_seconds=segment_seconds,
                                              tile_ids=tile_ids))

        @mcp.tool(description=(
            "Catalogue a manhwa chapter from its reader page URL: downloads the panel images in order into the library "
            "(collection manhwa/<series>/<chapter>), and a vision model writes per-tile scene, characters, dialogue "
            "(original + English) and tags, plus a chapter summary and a running character roster. follow_next=true "
            "keeps going through the next chapters (max_chapters, up to 50). cookies: an optional Cookie header for a "
            "site the user is logged in to. Returns a run; poll get_manhwa_run. Chapters already catalogued are skipped."))
        async def catalog_chapter(url: str, series: str, chapter: str = "", follow_next: bool = False,
                                  max_chapters: int = 5, cookies: str = "", wait_seconds: int = 20) -> dict:
            state = await run(manhwa.start(url, series, chapter, cookies=cookies, follow_next=follow_next,
                                           max_chapters=max_chapters))
            return await manhwa.wait(state["id"], max(0, min(wait_seconds, 120)))

        @mcp.tool(description="Progress of a manhwa catalogue run: status, chapters_done, the chapter and tile being worked on, chapter_ids, stop_reason, errors.")
        async def get_manhwa_run(run_id: str) -> dict:
            try:
                return manhwa.get(run_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description="Stop a manhwa catalogue run now. Tiles already described are kept; the chapter is marked cancelled.")
        async def cancel_manhwa_run(run_id: str) -> dict:
            try:
                return manhwa.cancel(run_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description="Catalogued manhwa: with no series, every series and its chapter count; with series, its chapters (id, number, status, summary).")
        async def list_manhwa(series: str = "") -> dict:
            if not series:
                return {"series": manhwa.store.series()}
            return {"chapters": [{k: c.get(k) for k in ("id", "chapter", "status", "summary", "tiles", "url")}
                                 for c in manhwa.store.list_chapters(series)]}

        @mcp.tool(description="One catalogued chapter in full: summary and every tile's scene, characters, dialogue and thumb_url.")
        async def get_manhwa_chapter(chapter_id: str) -> dict:
            try:
                return manhwa.chapter_view(chapter_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description="Full-text search over catalogued manhwa tiles (scenes, dialogue in both languages, character names, tags). Every word must match. Returns tiles with chapter, scene, dialogue and thumb_url.")
        async def search_manhwa(query: str, series: str = "", limit: int = 20) -> dict:
            return {"results": manhwa.search(query, series, max(1, min(limit, 100)))}

        @mcp.tool(description=(
            "Delete a catalogued manhwa series, or one chapter of it (chapter_id from list_manhwa), with its "
            "downloaded pages, cut panels and the photographs made from them (images=false keeps the images). "
            "Rendered videos are always kept. Cancels runs on that series first. "
            "Only when the user asks for it."))
        async def delete_manhwa(series: str, chapter_id: str = "", images: bool = True) -> dict:
            return await run(manhwa.delete(series, chapter_id, images=images))

        @mcp.tool(description=(
            "Turn a catalogued manhwa chapter's panels into realistic photographs with Qwen Image 2.1 on this GPU. "
            "With analyze (default) Grok 4.3 first looks at each panel and writes its prompt (people, poses, "
            "clothing, light, camera); a panel where someone reads as a minor is skipped. With the quality check on "
            "(qa, default from the settings) a vision model scores each photograph against its panel and below the "
            "bar rewrites the prompt and tries again, up to qa_retries times, keeping the best. prompt_model picks another "
            "model for that, e.g. deepseek-ai/deepseek-v4.1-flash (reads images, cheaper). "
            "Each page is cut into its panels at the gutters; each panel is edited with the newest anime-to-real LoRA "
            "in models/loras/qwen21, the lenovo realism LoRA (realism, 0 = off) and the age slider (age_slider, "
            "0 = off), using the tile's catalogued scene and character looks in the prompt. loras=[{name, strength}] "
            "replaces those defaults. tile_ids limits it to some tiles. Results go to manhwa/<series>/<chapter>/real. "
            "Runs in the background: poll get_manhwa_conversion. The usual image rules apply to every panel. "
            "With plan_after (default) it then starts the chapter's video plans, as plan_manhwa_video does; their "
            "job ids are in the conversion's plans."))
        async def convert_manhwa_to_real(chapter_id: str, tile_ids: list[str] | None = None,
                                         loras: list[dict] | None = None, prompt: str | None = None,
                                         realism: float | None = None, age_slider: float | None = None,
                                         steps: int | None = None, cfg: float | None = None,
                                         negative: str | None = None, max_panels: int = 300,
                                         plan_after: bool | None = None, analyze: bool | None = None,
                                         prompt_model: str | None = None, qa: bool | None = None,
                                         qa_retries: int | None = None, wait_seconds: int = 20) -> dict:
            try:
                ManhwaConvertIn(tile_ids=tile_ids or [], loras=loras, prompt=prompt, realism=realism,
                                age_slider=age_slider, steps=steps, cfg=cfg, negative=negative,
                                max_panels=max_panels)
            except ValueError as exc:
                raise ToolError(str(exc).splitlines()[0]) from None
            state = await run(manhwa.convert(chapter_id, tile_ids=tile_ids, loras=loras, prompt=prompt,
                                             realism=realism, age_slider=age_slider, max_panels=max_panels,
                                             plan_after=plan_after, analyze=analyze, prompt_model=prompt_model,
                                             steps=steps, cfg=cfg, negative=negative, qa=qa,
                                             qa_retries=qa_retries))
            return await manhwa.wait_conversion(state["id"], max(0, min(wait_seconds, 120)))

        @mcp.tool(description=(
            "Autopilot for the whole manhwa pipeline, no further input needed: catalogue the chapter at url (and the "
            "next ones with follow_next, up to max_chapters), make every panel's photograph, plan each chapter's "
            "video and render it, using the Manhwa page's saved Photograph and Video settings. context "
            "{characters, setting, action, one_scene} is given to chapters that have none. render=false stops after "
            "the plans. One autopilot at a time. Returns its state; poll get_manhwa_pipeline."))
        async def run_manhwa_pipeline(url: str, series: str, chapter: str = "", follow_next: bool = False,
                                      max_chapters: int = 1, cookies: str = "", context: dict | None = None,
                                      render: bool = True) -> dict:
            try:
                body = ManhwaPipelineIn(url=url, series=series, chapter=chapter, follow_next=follow_next,
                                        max_chapters=max_chapters, cookies=cookies, context=context, render=render)
            except ValueError as exc:
                raise ToolError(str(exc).splitlines()[0]) from None
            return await run(manhwa.start_pipeline(body.url, body.series, body.chapter, cookies=body.cookies,
                                                   follow_next=body.follow_next, max_chapters=body.max_chapters,
                                                   context=body.context.model_dump() if body.context else None,
                                                   render=body.render))

        @mcp.tool(description="Progress of a manhwa autopilot: stage, and per chapter its stage (photographs, planning, rendering, done, failed), conversion id, plan and video job ids, and any error.")
        async def get_manhwa_pipeline(pipeline_id: str) -> dict:
            try:
                return manhwa.pipeline(pipeline_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description="Stop a manhwa autopilot and the step it is on. Everything made so far is kept; videos already queued keep rendering.")
        async def cancel_manhwa_pipeline(pipeline_id: str) -> dict:
            try:
                return await manhwa.cancel_pipeline(pipeline_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description=(
            "Join a manhwa chapter's rendered parts, in part order, into one video (the chapter's film) in Media under "
            "manhwa/<series>/<chapter>, copied to Drive like other renders. Fails while a part is still rendering. "
            "The autopilot does this on its own; use it after render_manhwa_chapter. force=true joins again."))
        async def merge_manhwa_chapter(chapter_id: str, force: bool = False) -> dict:
            return await run(manhwa.merge_chapter(chapter_id, force=force))

        @mcp.tool(description=(
            "Render a manhwa chapter's video: one render job per finished video plan (from plan_manhwa_video or a "
            "conversion's plan_after). Plans still being written come back in not_ready -- call again when they are "
            "done. Parts already rendering or rendered are skipped unless force. settings: RenderSettings fields "
            "over the Manhwa defaults, e.g. {\"unet_name\": \"beta6\"}. Poll each job with get_job."))
        async def render_manhwa_chapter(chapter_id: str, parts: list[int] | None = None, force: bool = False,
                                        settings: dict | None = None) -> dict:
            return await run(manhwa.render_chapter(chapter_id, parts=parts, force=force, settings=settings))

        @mcp.tool(description=(
            "Draft a manhwa chapter's scene context (characters, setting, action, one_scene) from its catalogue: the "
            "chapter summary, the cast and the scenes. Not saved; show it to the user, then set_manhwa_chapter_context."))
        async def draft_manhwa_chapter_context(chapter_id: str) -> dict:
            return {"chapter_id": chapter_id, "context": await run(manhwa.draft_context(chapter_id))}

        @mcp.tool(description=(
            "Give a catalogued manhwa chapter its scene context: characters (who, and how they look), setting (where "
            "and when), action (what happens), and one_scene=true when the chapter is one continuous scene. Every "
            "panel prompt in convert_manhwa_to_real and the chapter's video plans follow it. Empty strings clear it."))
        async def set_manhwa_chapter_context(chapter_id: str, characters: str = "", setting: str = "",
                                             action: str = "", one_scene: bool = False) -> dict:
            try:
                body = ManhwaContextIn(characters=characters, setting=setting, action=action, one_scene=one_scene)
            except ValueError as exc:
                raise ToolError(str(exc).splitlines()[0]) from None
            try:
                return manhwa.set_context(chapter_id, body.model_dump())
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description=(
            "The Manhwa page's own Qwen Image 2.1 settings for converting panels (separate from image_options): "
            "loras (null = automatic: newest Anime2Real + lenovo realism + age slider), realism, age_slider, steps, "
            "cfg, negative, prompt, analyze, prompt_model, plan_after, aspect_ratio, segment_seconds; plus the LoRAs "
            "automatic would pick now and every installed Qwen 2.1 LoRA."))
        async def get_manhwa_settings() -> dict:
            return await manhwa.settings_view()

        @mcp.tool(description=(
            "Change the Manhwa page's Qwen Image 2.1 settings: pass only the fields to change (see "
            "get_manhwa_settings). loras=[] goes back to automatic. Only when the user asks."))
        async def set_manhwa_settings(changes: dict) -> dict:
            try:
                valid = ManhwaSettingsIn(**changes).model_dump(exclude_unset=True)
            except (ValueError, TypeError) as exc:
                raise ToolError(str(exc).splitlines()[0]) from None
            manhwa.save_settings(valid)
            return await manhwa.settings_view()

        @mcp.tool(description=(
            "Plan a film from a manhwa chapter's converted photographs: story-planner jobs, one per nine photographs "
            "(a render's picture limit), one segment per photograph in reading order, with each tile's catalogued "
            "dialogue spoken word for word (narration as voice-over). Returns the plan job ids; poll get_job, then "
            "render each with create_video plan_job_id. aspect_ratio defaults to 9:16 for tall webtoon panels."))
        async def plan_manhwa_video(chapter_id: str, aspect_ratio: str = "9:16", segment_seconds: float = 10.0,
                                    model: str = "") -> dict:
            return await run(manhwa.plan_chapter(chapter_id, aspect_ratio=aspect_ratio,
                                                 segment_seconds=max(1.0, min(segment_seconds, 15.0)),
                                                 model=model or None))

        @mcp.tool(description="Progress of a manhwa-to-real conversion: panels_total, panels_done, converted, refused, failed, and per panel the image asset ids or the reason.")
        async def get_manhwa_conversion(conversion_id: str) -> dict:
            try:
                return manhwa.conversion(conversion_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description="Stop a manhwa-to-real conversion. Images already made are kept.")
        async def cancel_manhwa_conversion(conversion_id: str) -> dict:
            try:
                return manhwa.cancel_conversion(conversion_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description=(
            "Make portraits for a catalogued manhwa series' cast: one photoreal head-and-shoulders portrait per character "
            "whose role is cast (or the names given), made from source_asset_id when given (one name), else the panel "
            "that shows them most plainly, else the user's description. Never for the narrator, a hidden character, "
            "or anyone who reads as under 18. "
            "Every later conversion gives Qwen the portraits of the people in a panel as extra references, so faces "
            "stay the same across panels and chapters. Characters with a portrait are skipped unless force. "
            "Runs in the background; poll get_manhwa_cast_run."))
        async def make_manhwa_cast(series: str, names: list[str] | None = None, force: bool = False,
                                   source_asset_id: str = "") -> dict:
            return await run(manhwa.make_cast(series, names=names, force=force, source_asset_id=source_asset_id or None))

        @mcp.tool(description=(
            "Edit a manhwa character as the user says: new_name (the old one becomes an alias), appearance (their "
            "description), role -- cast (portrait + face reference), extra (appears, no portrait), narrator (voice "
            "only, no image), hidden (left out) -- and aliases. A character who reads as under 18 can't be cast. "
            "Only when the user asks."))
        async def edit_manhwa_character(series: str, name: str, new_name: str = "", appearance: str | None = None,
                                        role: str = "", aliases: list[str] | None = None) -> dict:
            return await run(_sync(lambda: manhwa.edit_character(series, name, new_name=new_name or None,
                                                                 appearance=appearance, role=role or None,
                                                                 aliases=aliases)))

        @mcp.tool(description="Add a character the catalogue missed (role cast by default). Only when the user asks.")
        async def add_manhwa_character(series: str, name: str, appearance: str = "", role: str = "cast") -> dict:
            return await run(_sync(lambda: manhwa.add_character(series, name, appearance, role)))

        @mcp.tool(description="Fold duplicate manhwa characters into one (keep); the others' names become its aliases. Only when the user asks.")
        async def merge_manhwa_characters(series: str, keep: str, merge: list[str]) -> dict:
            return await run(_sync(lambda: manhwa.merge_characters(series, keep, merge)))

        @mcp.tool(description="Progress of a cast run: done (name, asset_id), skipped, errors.")
        async def get_manhwa_cast_run(run_id: str) -> dict:
            try:
                return manhwa.cast_run(run_id)
            except NotFound as exc:
                raise ToolError(str(exc)) from None

        @mcp.tool(description="Use an image as a manhwa character's cast portrait (asset_id), or remove it (asset_id empty). Only when the user asks.")
        async def set_manhwa_cast_portrait(series: str, name: str, asset_id: str = "") -> dict:
            return await run(manhwa.assign_portrait(series, name, asset_id or None))

        @mcp.tool(description="The character roster of a catalogued series: name, aliases, appearance, chapters they appear in.")
        async def manhwa_characters(series: str) -> dict:
            return {"characters": manhwa.characters(series)}

    if stories is not None:
        @mcp.tool(description="The user's stories (Story page): id, title, premise, chapters and how many have video.")
        async def list_stories() -> dict:
            return {"stories": stories.list_stories()}

        @mcp.tool(description=(
            "One story in full: premise, style, settings, cast (roles, descriptions, pictures), chapters (title, "
            "summary, scenes with exact dialogue, supporting images, video status) and the main film."))
        async def get_story(story_id: str) -> dict:
            return await run(_sync(lambda: stories.view(story_id)))

        @mcp.tool(description="Start a new story on the Story page. Only when the user asks.")
        async def create_story(title: str, premise: str = "", style: str = "") -> dict:
            return await run(_sync(lambda: stories.create_story(title, premise, style)))

        @mcp.tool(description=(
            "Send the user's idea to the story's co-writer (Gemma 4 31B, then Kimi K2.6, then DeepSeek V4.1 Flash) "
            "and return its answer. Both are kept in the story's chat. attachments: image asset ids to show it."))
        async def story_chat(story_id: str, text: str, attachments: list[str] | None = None) -> dict:
            return await run(stories.chat(story_id, text, attachments))

        @mcp.tool(description=(
            "Turn the story chat since the last chapter into the next chapter: title, summary and scenes (what is "
            "seen, setting, who is there, exact lines). Saved as an editable draft. When the user says the chapter "
            "is ready."))
        async def draft_story_chapter(story_id: str) -> dict:
            return await run(stories.draft_chapter(story_id))

        @mcp.tool(description=(
            "Edit a story chapter's draft: title, summary, scenes ([{description, setting, characters, dialogue: "
            "[{speaker, text}]}]) or its supporting images ([{asset_id, label}]). Only what the user asks to change."))
        async def edit_story_chapter(story_id: str, chapter_id: str, title: str | None = None, summary: str | None = None,
                                     scenes: list[dict] | None = None, images: list[dict] | None = None) -> dict:
            return await run(_sync(lambda: stories.edit_chapter(story_id, chapter_id, title=title, summary=summary,
                                                                scenes=scenes, images=images)))

        @mcp.tool(description=(
            "Make a story chapter's video with no stops: plan it (cast pictures and the chapter's images as "
            "references), render it, and append it to the story's main film. redo=true replaces an existing video "
            "in place. Poll get_story for the chapter's status (planning, rendering, appending, done, failed)."))
        async def make_story_chapter_video(story_id: str, chapter_id: str, redo: bool = False) -> dict:
            return await run(_sync(lambda: stories.make_video(story_id, chapter_id, redo=redo)))

        @mcp.tool(description=(
            "Draw a story chapter's scenes as a storyboard (one $0.018 Seedream Flash sheet per 8 scenes) with the "
            "cast's pictures keeping their faces. Once drawn, make_story_chapter_video opens each scene on its panel. "
            "Fix panels with redo_storyboard_panel on the returned storyboard id."))
        async def storyboard_story_chapter(story_id: str, chapter_id: str) -> dict:
            return await run(_sync(lambda: stories.make_storyboard(story_id, chapter_id)))

        @mcp.tool(description=(
            "A story character's identity pack: front, back, side and face views drawn as one $0.018 sheet from their "
            "picture; the front view and face become their first pictures. anatomy=true adds nude views (adults, "
            "adult platforms only)."))
        async def story_identity_pack(story_id: str, name: str, anatomy: bool = False) -> dict:
            return await run(stories.identity_pack(story_id, name, anatomy=anatomy))

        @mcp.tool(description=(
            "Story autopilot, from the premise alone: outline the next `chapters` chapters and the cast, make a picture "
            "for every cast character without one (portraits), then draft each chapter and (videos) make its video, "
            "appended to the main film. Runs in the background and survives a restart; poll get_story (autopilot)."))
        async def run_story_autopilot(story_id: str, chapters: int = 5, portraits: bool = True, videos: bool = True) -> dict:
            return await run(_sync(lambda: stories.start_autopilot(story_id, chapters=max(1, min(chapters, 30)),
                                                                   portraits=portraits, videos=videos)))

        @mcp.tool(description="Stop a story's autopilot after its current step. Everything made so far is kept.")
        async def stop_story_autopilot(story_id: str) -> dict:
            return await run(stories.stop_autopilot(story_id))

        @mcp.tool(description=(
            "Give a story character a picture from Media (asset_id; empty removes them). A picture of, or made from "
            "a photo of, a real identifiable person is refused; minors and the narrator get none."))
        async def set_story_character_image(story_id: str, name: str, asset_id: str = "", add: bool = False) -> dict:
            return await run(stories.set_character_image(story_id, name, asset_id or None, add=add))

    return mcp
