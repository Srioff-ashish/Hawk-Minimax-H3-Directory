"""Request models shared by REST (OpenAPI docs) and MCP (tool schemas)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "match first picture"]
Role = Literal["picture", "pose", "video", "audio", "video_soundtrack"]
Continuity = Literal["off", "last_frame", "tail_5", "tail_22", "tail_39"]
Attention = Literal["comfy default", "sage", "sol scheduled", "sol scheduled + sage"]


class ReferenceIn(BaseModel):
    asset_id: str = Field(description="Id returned when the file was uploaded.")
    role: Role = Field(
        description=(
            "picture: identity, outfit, product or place, mentioned as <Picture N>. "
            "pose: body pose only, <Pose N>. video: motion or camera clip, <Video N>. "
            "audio: voice, music or sound, <Audio N> (a video asset here uses its soundtrack). "
            "video_soundtrack: the audio belonging to video number for_video."
        )
    )
    label: str = Field("", description="What this reference is for. Shown to the story planner.")
    for_video: int | None = Field(None, ge=1, le=3, description="Only for video_soundtrack: which video (1-3) it belongs to.")


class LoraIn(BaseModel):
    name: str = Field(description="A file in ComfyUI models/loras, its file name, or a unique part of it such as 'turbo'.")
    strength: float = Field(1.0, ge=-10, le=10, description="0 removes the LoRA; use it to switch off a default.")


class ImageLoraIn(BaseModel):
    name: str = Field(description="A LoRA file from image_options, for the family of the local engine that will run "
                                  "(Qwen Image 2.1 or Krea 2), or a unique part of its name such as 'realism'.")
    strength: float | None = Field(None, ge=-4, le=4, description="Omit for the LoRA's recommended strength; 0 leaves it out.")


class PlannerOptions(BaseModel):
    story: str = Field(min_length=1, description="The brief: plot, characters, mood, locations, exact dialogue lines.")
    segment_count: int = Field(3, ge=0, le=40, description="Segments to write; 0 lets the planner decide.")
    segment_seconds: float = Field(10.0, ge=1, le=15, description="Target length of each segment.")
    aspect_ratio: AspectRatio = "16:9"
    model: str | None = Field(None, description="Atlas chat model id. Default comes from the server.")
    temperature: float = Field(0.7, ge=0, le=2)
    seed: int | None = Field(None, ge=0, description="Change it for a different plan.")


class PlanRequest(PlannerOptions):
    references: list[ReferenceIn] = Field(default_factory=list)


class RenderSettings(BaseModel):
    aspect_ratio: AspectRatio = "16:9"
    megapixels: float = Field(0.98, ge=0.1, le=2.2, description="0.98 at 16:9 is 1344x768. Use 0.4 for cheap previews.")
    default_seconds: float = Field(10.0, ge=0.5, le=15, description="Length of segments that do not set a duration.")
    steps: int | None = Field(None, ge=1, le=100, description="Default: 8 when a turbo LoRA is applied, otherwise 30.")
    sampler_name: str = "res_multistep"
    scheduler: str = "simple"
    seed: int | None = Field(None, ge=0, description="Kept fixed for the job so retries resume. Random when omitted.")
    continuity: Continuity = "tail_22"
    carry_audio: bool = True
    ref_image_size: Literal["match", "max"] = "match"
    interpolation: Literal["off", "48 fps (RIFE)", "60 fps (RIFE)"] = "off"
    audio_crossfade_ms: int = Field(60, ge=0, le=1000)
    loras: list[LoraIn] = Field(default_factory=list, description="Added after the server's default LoRAs and preset.")
    lora_preset: str | None = Field(None, description="A preset name from the server's loras.json.")
    use_default_loras: bool = Field(True, description="false skips the server's default LoRAs (including the turbo LoRA).")
    attention: Attention | None = Field(None, description="Override the server's attention backend.")
    unet_name: str | None = Field(
        None,
        description="ref2va base model from list_options.diffusion_models: a file name or a unique part such as 'bf16'. "
        "Default: the server's model. bf16 is best quality but slowest; int8 / fp8 are faster.",
    )
    music_asset_id: str | None = Field(
        None,
        description="Music bed: an uploaded audio asset mixed under the whole film (looped or trimmed, faded out). "
        "Use it for one continuous track across segments; do not also list it in references.",
    )
    music_volume_db: float = Field(-3.0, ge=-40, le=12, description="Music bed level in dB.")
    scene_volume_db: float = Field(0.0, ge=-60, le=12, description="Level of the rendered voices / ambience / effects under the music bed.")
    music_fade_seconds: float = Field(2.0, ge=0, le=15, description="Music fade-out at the end of the film.")
    mute_generated_music: bool = Field(True, description="With a music bed, stop H3 from composing its own music in each segment.")
    clip_name: str | None = Field(
        None,
        description="Qwen3-VL MiniMax text encoder from list_options.text_encoders (file name or unique part). Default: the server's.",
    )


class VideoRequest(BaseModel):
    references: list[ReferenceIn] = Field(
        default_factory=list, description="Empty with plan_job_id reuses the plan's references."
    )
    script: str | dict | None = Field(
        None, description="A finished Hawk H3 script: plain text with --- between segments, or the planner's JSON."
    )
    plan_job_id: str | None = Field(None, description="Render the script of a finished plan job.")
    story: PlannerOptions | None = Field(None, description="Plan and render in one job.")
    settings: RenderSettings = Field(default_factory=RenderSettings)

    @model_validator(mode="after")
    def _one_source(self):
        given = [name for name in ("script", "plan_job_id", "story") if getattr(self, name) not in (None, "")]
        if len(given) != 1:
            raise ValueError("Give exactly one of script, plan_job_id or story.")
        return self


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, description="What the image should show, or how to change the reference images.")
    reference_asset_ids: list[str] = Field(default_factory=list, max_length=16, description=
        "Image assets to edit or combine. Qwen Image 2.1 (auto/local) takes up to 16: the first is the canvas whose "
        "framing and untargeted content survive and whose size the output follows, the rest supply material. Refer to "
        "them in the prompt as <image1>, <image2> in this order -- nothing labels them otherwise. Krea 2 takes 1, or 2 "
        "with the scene first and the person second; Seedream edit takes up to 10.")
    model: str | None = Field(None, description="Atlas image model or alias: 'turbo' (z-image/turbo, the text-to-image default: fast, ~$0.01, no edits) "
                              "'seedream' (Seedream v5.0 Pro: best quality, ~$0.036 up to 2.36 MP, ~$0.072 above) or 'seedream-lite' "
                              "(Seedream v5.0 Lite: 2K+ output for ~$0.032). Reference images use the matching Seedream edit.")
    size: str | None = Field(None, description="e.g. 1024x1536 or 1536x1536 (z-image: 512-2048 a side). Seedream snaps to its nearest preset: up to 2.36 MP "
                      "stays in Pro's cheaper 1.5K tier (e.g. 1328x1776); larger, like 2048x2048, bills the 2K tier at twice the price. Omit for the default.")
    n: int = Field(1, ge=1, le=4, description="How many images.")
    seed: int | None = Field(None, ge=0)
    engine: str | None = Field(None, description="auto (default: local Qwen Image 2.1 when idle, then "
                               "Krea 2, then Z-Image, then z-image/turbo, then Seedream; with references it starts at Qwen "
                               "Image 2.1), or one of qwen21, krea2, zimage, local, turbo, seedream, seedream-lite, "
                               "nano-banana, nano-banana-pro, nano-banana-lite. 'klein' still resolves, to qwen21. The Nano Banana engines "
                               "(Google) take SFW requests only: an SFW check runs first, and anything it does not pass "
                               "goes to the next engine in the ladder, or fails when one is pinned.")
    loras: list[ImageLoraIn] = Field(default_factory=list, description="LoRAs for local generation, from the running "
                                     "engine's family (file name or a unique part, optional strength). Qwen Image 2.1 "
                                     "always adds its repair LoRA on top, whatever is named here.")
    steps: int | None = Field(None, ge=1, le=50, description="Steps for the local engine. Omit for its default: "
                              "Qwen Image 2.1 25, Krea 2 8, Z-Image 8, or a LoRA's own recommendation.")
    ref_boost: float | None = Field(None, ge=0, le=20, description="Krea 2 edit only, ignored by every other engine: "
                                   "likeness dial, 4 (default) strong likeness, 1 looser and more creative, above 10 "
                                   "breaks removals. On Qwen Image 2.1 say what stays in the prompt instead.")
    negative: str = Field("", max_length=2000, description=
        "What to keep out, for the local engines only (Atlas ignores it). It has no effect while the engine samples "
        "at cfg 1.0 -- Krea 2 Turbo and Z-Image -- because guidance at cfg 1 collapses to the positive prompt. On "
        "Qwen Image 2.1 it needs cfg above 1 (its default is 1.0, so pass cfg 1.5-2 with it); there it is the place "
        "for \"fused bodies, missing limbs, extra arms\".")
    cfg: float | None = Field(None, ge=0.0, le=15.0, description=
        "Guidance for the local engines. Omit for the engine's own default. Above 1.0 the negative prompt starts "
        "working; too high posterises (blown greens and blues, banded surfaces). Qwen Image 2.1 defaults to 1.0 (its "
        "official setting; 2 follows dense prompts more closely but over-sharpens edges); Krea 2 Turbo and Z-Image "
        "are trained for 1.0 and should be left there.")
    max_adult_loras: int = Field(3, ge=1, le=3, description="How many adult LoRAs one image may stack (up to 3; a note "
                                 "warns above a combined strength of 2.0). Lower it to be stricter.")


class AICredentialCheck(BaseModel):
    made_from: list[str] = Field(default_factory=list, max_length=16, description=
        "Library images this one was made from, when its content credential lists input images. Each must itself "
        "count as generated (verify its credential first), or an AI edit of a real photo would pass.")


class AssetUpdate(BaseModel):
    collection: str | None = Field(None, description="Move to this collection (created if new).")
    tags: list[str] | None = Field(None, description="Replace all tags.")
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)
    filename: str | None = Field(None, description="Rename the display name.")
    generated_from: str | None = Field(None, description=
        "Correct the provenance of a file that is a copy of an image made here: the asset id it was copied "
        "from. Use it when a client downloaded a generated image and uploaded it back, which loses the "
        "history and puts a picture this pod drew from a prompt under the rules meant for photographs of "
        "real people. The origin must itself be generated and not descend from an upload. The previous "
        "source is kept on the asset as corrected_from and the change is logged; \"\" undoes it. This "
        "overrides a content guardrail, so use it only on a file you have compared against its origin.")


class AssetBulk(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=2000)
    action: Literal["move", "tag", "untag", "delete"]
    collection: str | None = None
    tags: list[str] = Field(default_factory=list)


class DriveImportIn(BaseModel):
    paths: list[str] = Field(min_length=1, description="Drive files or folders, relative to My Drive.")
    recursive: bool = Field(True, description="Include subfolders.")
    collection: str | None = Field(None, description="Target collection; default: the folder name.")
    tags: list[str] = Field(default_factory=list)


class DriveExportSettings(BaseModel):
    enabled: bool | None = Field(None, description="Copy every finished render into Google Drive.")
    folder: str | None = Field(None, description="Folder inside My Drive, e.g. Hawk H3/Videos (a dated subfolder is added).")
    segments: bool | None = Field(None, description="Also copy each segment file.")
    images: bool | None = Field(None, description="Copy every generated image into Google Drive.")
    uploads: bool | None = Field(None, description="Copy every uploaded image (Studio, the upload page, MCP, URL "
                                 "imports) into the same Drive image folder, so a runtime restart cannot lose it.")
    image_folder: str | None = Field(None, description="Folder inside My Drive for generated images, e.g. Hawk H3/Images (a dated subfolder is added).")
    snapshots: bool | None = Field(None, description="Copy the chats-and-jobs database into Drive, so a Colab runtime that ends does not take them with it.")
    snapshot_folder: str | None = Field(None, description="Folder inside My Drive for the database copy, e.g. Hawk H3/Backups. Two files are kept there.")
    snapshot_minutes: float | None = Field(None, ge=1, le=120, description="Minutes between copies. A copy is only written when something actually changed.")


class ImageEngineRung(BaseModel):
    engine: str = Field(description="Engine id from /v1/images/engines, e.g. qwen21, krea2, zimage, turbo, seedream.")
    enabled: bool = Field(True, description="Off keeps its place in the order but is never tried.")


class ImageLoraDefault(BaseModel):
    name: str = Field(description="LoRA file name, from image_options. Must belong to that family.")
    # 2.0 covers the content LoRAs; slider-style ones run to 3. Kept in step with image_engines.MAX_DEFAULT_STRENGTH.
    strength: float = Field(0.8, ge=0.0, le=3.0)


class ImageEngineSettings(BaseModel):
    generate: list[ImageEngineRung] | None = Field(None, description="Order tried for text-only images, best first.")
    edit: list[ImageEngineRung] | None = Field(None, description="Order tried when reference_asset_ids are given, best first.")
    busy_mode: str | None = Field(None, description='What an image does while ComfyUI renders a video: "wait" or "fall_through" to a paid engine.')
    busy_max_wait_seconds: int | None = Field(None, ge=0, le=300, description="How long to wait before falling through.")
    confirm_paid: bool | None = Field(
        None, description="Ask before a failed take retries on a paid engine, instead of spending automatically.")
    pick_takes: bool | None = Field(
        None, description="Show a rejected take and wait for your choice, instead of letting the agent take it again.")
    qwen21_pe: bool | None = Field(
        None, description="Run Qwen Image 2.1's prompt enhancer (a local Qwen3.5-9B) over the prompt first. Slower; "
                          "needs the qwen3.5_9b_qwen_image_2.1_pe_t2i / _pe_i2i files in models/text_encoders.")
    defaults: dict[str, list[ImageLoraDefault]] | None = Field(
        None, description='LoRAs a family attaches on its own, keyed by family (krea2, qwen21, zit). An empty list switches them off.')


class RenderModelSettings(BaseModel):
    unet_name: str | None = Field(None, description="ref2va base model from ComfyUI's diffusion_models. \"\" restores the pod's own.")
    clip_name: str | None = Field(None, description="Qwen3-VL text encoder from text_encoders. It also encodes the reference images.")
    video_vae: str | None = Field(None, description="Video VAE from the vae folder.")
    audio_vae: str | None = Field(None, description="Audio VAE from the vae folder.")


class VideoLoraDefault(BaseModel):
    name: str = Field(description="MiniMax H3 LoRA file from list_loras.")
    strength: float = Field(1.0, ge=0.0, le=2.0, description="0 switches it off without removing it.")


class VideoLoraDefaults(BaseModel):
    defaults: list[VideoLoraDefault] = Field(description="The full list, in order. Required entries must stay.")


class PromptIn(BaseModel):
    text: str = Field(description="The full prompt text. Saving the default text (or an empty text) resets to the built-in prompt.")


class CastMemberIn(BaseModel):
    id: str | None = Field(None, description="Keep a character's id when editing; new characters get one.")
    name: str | None = Field(None, description="Display name, e.g. Riya. Required when the chat has several characters (or taken from the persona).")
    persona: str = Field("", description="Who this character is.")
    avatar_asset_id: str | None = Field(None, description="An image asset shown as this character's face.")
    growth: list[str] | None = Field(None, description="How the character has grown in this chat (adaptive chats). Omit to keep; [] resets.")
    feelings: dict[str, list[str]] | None = Field(None, description="Private feelings about the user (key 'user') or another "
                                                  "character (their id), group chats. Omit to keep; {} resets.")


class AgentTalkIn(BaseModel):
    rounds: int = Field(5, ge=1, le=10, description="How many rounds the characters talk to each other (each speaks about once a round).")
    makes: int = Field(2, ge=0, le=5, description="How many images or videos EACH character may have made during this talk (the whole cast is capped at 8).")


class AgentSessionIn(BaseModel):
    title: str | None = Field(None, description="Chat title; the agent renames it once the task is clear.")
    persona: str | None = Field(None, description="Who the agent should be, e.g. 'Bollywood ad-film director'.")
    model: str | None = Field(None, description="Chat model id for the director (tools and orchestration); default HAWK_AGENT_MODEL.")
    prose_model: str | None = Field(None, description="Chat model id for the characters' spoken turns in this "
                                    "chat. Empty uses the pod's prose model, then the chat's own model.")
    name: str | None = Field(None, description="The persona's display name, e.g. Maya. Empty: taken from the persona text.")
    avatar_asset_id: str | None = Field(None, description="An image asset shown as the agent's avatar in this chat; empty removes it.")
    cast: list[CastMemberIn] | None = Field(None, description="Several characters in one chat (up to 4); the first is the lead.")
    adaptive: bool | None = Field(None, description="Characters adapt: they grow from the conversation with you and with each other.")
    whispers: bool | None = Field(None, description="Group chats: a message starting with @Name is private to that character.")


class AgentMessageIn(BaseModel):
    text: str = Field("", description="Your message.")
    attachments: list[str] = Field(default_factory=list, description="Asset ids uploaded with this message.")
    narration: bool = Field(False, description="This is the scene, not speech: something that has happened and is "
                                               "true from now on. Wrapping the message in *asterisks* does the same.")


class UrlAssetRequest(BaseModel):
    url: str = Field(description="A direct http(s) link to an image, audio or video file.")
    filename: str | None = Field(None, description="Override the file name (its extension decides the kind).")


class LLMSettingsUpdate(BaseModel):
    """Which service answers planning and chat calls. Only the fields sent are changed; "" clears one back
    to the pod's own setting. A key is write-only: it is never returned, so it can never be sent back."""

    # Refused rather than dropped: agent_summary_model_override was missing from this model while Studio
    # sent it, so the panel reported a save that pydantic had discarded on the way in.
    model_config = ConfigDict(extra="forbid")

    llm_provider: Literal["atlas", "openrouter"] | None = Field(None, description="Which service to call.")
    atlas_api_key_override: str | None = Field(None, description="Replaces ATLAS_API_KEY. Write-only.")
    google_api_key: str | None = Field(
        None, description="Replaces GOOGLE_API_KEY: the Google AI Studio key the Nano Banana image engines use. "
                          "Write-only.")
    openrouter_url: str | None = Field(None, description="OpenRouter's OpenAI-compatible base URL.")
    openrouter_routing: Literal["sticky", "balanced", "quality", "cheapest", "fastest", "default"] | None = Field(
        None, description="Which service OpenRouter picks among the many serving one model id. One id is "
        "served at different prices and different weights, so this is a quality setting as much as a price "
        "one: 'sticky' keeps every call on one service so a cached prompt prefix stays warm, 'balanced' takes "
        "the cheapest of the bf16/fp16/fp8 services, 'quality' stays on bf16/fp16, "
        "'cheapest' takes the cheapest at any weight, 'fastest' sorts by throughput, 'default' lets "
        "OpenRouter decide. Ignored on Atlas.")
    openrouter_api_key: str | None = Field(None, description="OpenRouter key. Write-only.")
    openrouter_management_key: str | None = Field(
        None, description="OpenRouter management key, used only by the billing page to read spend per model. "
                          "Never used for chat. Write-only.")
    #: Each of these takes a comma-separated chain, best first, because a model id belongs to a provider: the
    #: later ids are what run when the provider in force does not serve the earlier ones. A single id still
    #: works and is read as a one-element chain.
    planner_model_override: str | None = Field(None, description="Model chain for plan_film, best first.")
    agent_model_override: str | None = Field(None, description="Model chain for new agent chats (the director).")
    agent_summary_model_override: str | None = Field(None, description="Model chain for chat summaries.")
    agent_prose_model_override: str | None = Field(
        None, description="Model chain for the characters' spoken turns, kept apart from the director's tool "
        "calling. Must be able to reply with a JSON object; a model that will not is retried on the "
        "director's. Blank means the chat's own model, which is how chats behaved before the split.")
    agent_vision_model_override: str | None = Field(
        None, description="Model chain for looking at images (inspect_image), used when the chat's own model "
        "cannot see. Only models the provider lists as accepting image input are tried.")


class ManhwaLoraIn(BaseModel):
    name: str = Field(description="A Qwen Image 2.1 LoRA file from models/loras, or a unique part of its name.")
    strength: float = Field(1.0, ge=-4, le=4)


class ManhwaSettingsIn(BaseModel):
    """The Manhwa page's own Qwen Image 2.1 settings. Only the fields sent are changed."""
    loras: list[ManhwaLoraIn] | None = Field(None, description="LoRAs every panel is converted with. [] goes back to "
                                             "automatic: the newest Anime2Real + lenovo realism + age slider from the folder.")
    realism: float | None = Field(None, ge=0, le=1, description="Automatic LoRAs only: lenovo realism strength; 0 leaves it out.")
    age_slider: float | None = Field(None, ge=0, le=3, description="Automatic LoRAs only: age slider strength; 0 leaves it out.")
    steps: int | None = Field(None, ge=1, le=50, description="Sampling steps. Null: the engine's (or a LoRA's) own.")
    cfg: float | None = Field(None, ge=0, le=15, description="Guidance. Null: the engine's own (1.0 with the adult LoRA).")
    negative: str | None = Field(None, max_length=2000, description="Added to every panel's negative prompt.")
    prompt: str | None = Field(None, max_length=2000, description="Added to every panel's prompt, e.g. 'soft evening light'.")
    analyze: bool | None = Field(None, description="A vision model writes each panel's prompt; false uses the template.")
    prompt_model: str | None = Field(None, description="The model that writes panel prompts, e.g. xai/grok-4.3 or deepseek-ai/deepseek-v4.1-flash.")
    plan_after: bool | None = Field(None, description="Plan the chapter's video when its photographs are made.")
    aspect_ratio: AspectRatio | None = Field(None, description="Video plans' aspect ratio (9:16 suits webtoon panels).")
    segment_seconds: float | None = Field(None, ge=1, le=15, description="Video plans' segment length.")
    planner_model: str | None = Field(None, description="Model that writes the chapter's video plans. \"\": the server's planner.")
    qa: bool | None = Field(None, description="Quality check: a vision model scores each photograph against its panel "
                            "(pose, camera, anatomy, photographic look) and below qa_min_score rewrites the prompt for another try.")
    qa_model: str | None = Field(None, description="The checking model, e.g. deepseek-ai/deepseek-v4.1-flash (default) or xai/grok-4.3.")
    qa_retries: int | None = Field(None, ge=0, le=5, description="Extra tries per panel after a low score.")
    qa_min_score: float | None = Field(None, ge=0, le=10, description="Score (0-10) a photograph needs to be kept without a retry.")
    auto_context: bool | None = Field(None, description="The autopilot drafts a chapter's scene context from its catalogue when none was given.")
    cast_refs: bool | None = Field(None, description="Give Qwen each character's cast portrait as an extra reference on the panels they are in.")
    auto_cast: bool | None = Field(None, description="The autopilot makes missing cast portraits before converting.")
    cast_size: int | None = Field(None, ge=1, le=20, description="How many main characters get a portrait.")
    triage: bool | None = Field(None, description="Skip text-only and blank panels, and near-duplicates of the panel before.")
    subtitles: Literal["file", "burn", "off"] | None = Field(None, description="Chapter films: an .srt beside the film "
                                                         "(file), drawn into the picture (burn), or none (off).")
    group_shots: bool | None = Field(None, description="Video plans join neighbouring panels of one moment into a shot "
                                     "and time each shot by its dialogue; false is one segment of segment_seconds per panel.")
    render: dict | None = Field(None, description="Render chapter's video settings, as RenderSettings fields: unet_name, "
                                "steps, megapixels, sampler_name, scheduler, continuity, attention, interpolation, "
                                "carry_audio, loras [{name, strength}], lora_preset, use_default_loras. {} = server defaults.")

    @model_validator(mode="after")
    def _render_is_valid(self):
        if self.render:
            RenderSettings(**{k: v for k, v in self.render.items() if v not in (None, "")})
        return self


class ManhwaConvertIn(BaseModel):
    """One conversion. Fields left out come from the Manhwa page's settings (GET /v1/manhwa/settings)."""
    tile_ids: list[str] = Field(default_factory=list, description="Only the panels in these tiles (ids from the chapter). Empty: the whole chapter.")
    loras: list[ManhwaLoraIn] | None = Field(None, description="[{name, strength}] for this run only.")
    prompt: str | None = Field(None, description="Extra instruction added to every panel's prompt.")
    realism: float | None = Field(None, ge=0, le=1)
    age_slider: float | None = Field(None, ge=0, le=3)
    steps: int | None = Field(None, ge=1, le=50)
    cfg: float | None = Field(None, ge=0, le=15)
    negative: str | None = Field(None, max_length=2000)
    max_panels: int = Field(300, ge=1, le=300)
    plan_after: bool | None = Field(None, description="When the photographs are made, start the video plans for the chapter.")
    analyze: bool | None = Field(None, description="A vision model looks at each panel and writes its own photograph prompt; "
                                 "false uses the fixed template with the catalogue's description.")
    prompt_model: str | None = Field(None, description="The model that analyses each panel, e.g. xai/grok-4.3 or "
                                     "deepseek-ai/deepseek-v4.1-flash (reads images, about a quarter of Grok's price).")
    qa: bool | None = Field(None, description="Quality check with retries (default from the settings).")
    qa_retries: int | None = Field(None, ge=0, le=5)


class ManhwaContextIn(BaseModel):
    characters: str = Field("", max_length=2000, description="Who is in the chapter and how they look, e.g. 'Mrs. Cha: "
                            "late 30s, auburn hair in a low bun; Dae-ho: early 20s, short dark hair'.")
    setting: str = Field("", max_length=2000, description="Where and when, e.g. 'a small tiled bathroom at night, warm light'.")
    action: str = Field("", max_length=2000, description="What happens across the chapter, in a sentence or two.")
    one_scene: bool = Field(False, description="The chapter is one continuous scene: one place, the same people, no cuts.")


class ManhwaRenderIn(BaseModel):
    parts: list[int] = Field(default_factory=list, description="Only these plan parts. Empty: every finished plan.")
    force: bool = Field(False, description="Render again a part that is already rendering or rendered.")
    settings: dict = Field(default_factory=dict, description="RenderSettings fields over the Manhwa defaults "
                           "(aspect ratio and segment length from the Photograph settings), e.g. {\"unet_name\": \"beta6\"}.")


class ManhwaPipelineIn(BaseModel):
    url: str = Field(description="The first chapter's reader page URL.")
    series: str = Field(description="Series name.")
    chapter: str = Field("", description="Chapter number for the first chapter. Blank: from the URL or page title.")
    cookies: str = Field("", description="Optional Cookie header for a site you are logged in to.")
    follow_next: bool = Field(False, description="Also do the next chapters, following the page's 'next' link.")
    max_chapters: int = Field(1, ge=1, le=50, description="With follow_next: how many chapters in all.")
    context: ManhwaContextIn | None = Field(None, description="Scene context for every chapter that has none yet.")
    render: bool = Field(True, description="Render the videos; false stops after the plans are written.")


class ManhwaCastIn(BaseModel):
    series: str
    names: list[str] = Field(default_factory=list, description="Only these characters. Empty: every character whose role is cast.")
    force: bool = Field(False, description="Make a new portrait even for characters that have one.")
    source_asset_id: str | None = Field(None, description="With one name: the picture to make the portrait from "
                                        "(a panel, art, an image made here). Default: the catalogue's best panel, or "
                                        "the description alone for a character you added or described.")


CharacterRole = Literal["cast", "extra", "narrator", "hidden"]


class ManhwaCharacterIn(BaseModel):
    series: str
    name: str
    new_name: str | None = Field(None, description="Rename; the old name becomes an alias.")
    appearance: str | None = Field(None, max_length=2000, description="Your description: adult age, build, face, hair, clothes.")
    role: CharacterRole | None = Field(None, description="cast: portrait + face reference; extra: appears, no portrait; "
                                       "narrator: voice only, no image; hidden: left out.")
    aliases: list[str] | None = None


class ManhwaNewCharacterIn(BaseModel):
    series: str
    name: str
    appearance: str = Field("", max_length=2000)
    role: CharacterRole = "cast"


class ManhwaMergeCharactersIn(BaseModel):
    series: str
    keep: str = Field(description="The character to keep.")
    merge: list[str] = Field(min_length=1, description="Duplicates folded into it (their names become aliases).")


class ManhwaPortraitIn(BaseModel):
    series: str
    name: str
    asset_id: str | None = Field(None, description="The image to use as this character's portrait; null removes it.")


class ManhwaMergeIn(BaseModel):
    force: bool = Field(False, description="Join again even when a film of these same parts exists.")


class ManhwaPlanIn(BaseModel):
    aspect_ratio: AspectRatio | None = Field(None, description="Default from the Manhwa settings (9:16).")
    segment_seconds: float | None = Field(None, ge=1, le=15)
    model: str | None = Field(None, description="Planner model id. Default comes from the server.")


class ManhwaCatalogIn(BaseModel):
    url: str = Field(description="The chapter page's http(s) URL (the reader page whose <img> tags are the panels).")
    series: str = Field(description="Series name. Groups chapters and their character roster.")
    chapter: str = Field("", description="Chapter number for the first chapter. Blank: read from the URL or page title.")
    cookies: str = Field("", description="Optional Cookie header for a site you are logged in to, e.g. 'session=abc; age_ok=1'.")
    follow_next: bool = Field(False, description="Keep going to the next chapters by following the page's 'next' link.")
    max_chapters: int = Field(5, ge=1, le=50, description="With follow_next: stop after this many chapters.")


# ------------------------------------------------------------------ story


StoryRole = Literal["cast", "extra", "narrator"]


class StoryIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    premise: str = Field("", max_length=2000, description="What the story is about, in a few sentences.")
    style: str = Field("", max_length=600, description="Genre, tone, look: e.g. 'slow-burn romance, warm film grain'.")


class StoryPatch(BaseModel):
    title: str | None = Field(None, max_length=120)
    premise: str | None = Field(None, max_length=2000)
    style: str | None = Field(None, max_length=600)
    settings: dict | None = Field(None, description="models (a comma-separated chain), aspect_ratio, segment_seconds, "
                                  "render (RenderSettings fields), subtitles (file | off).")


class StoryCharacterIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field("", max_length=2000, description="Adult age, build, face, hair, usual clothes, voice.")
    role: StoryRole = "cast"


class StoryCharacterPatch(BaseModel):
    name: str
    new_name: str | None = Field(None, max_length=80)
    description: str | None = Field(None, max_length=2000)
    role: StoryRole | None = None


class StoryImageIn(BaseModel):
    name: str = Field(description="The character.")
    asset_id: str | None = Field(None, description="An image from Media; null removes the character's pictures.")
    add: bool = Field(False, description="Keep the pictures they have and add this one.")


class StoryChatIn(BaseModel):
    text: str = Field("", max_length=6000)
    attachments: list[str] = Field(default_factory=list, max_length=4, description="Image asset ids to show the co-writer.")


class StoryChapterPatch(BaseModel):
    title: str | None = Field(None, max_length=160)
    summary: str | None = Field(None, max_length=1200)
    scenes: list[dict] | None = Field(None, description="[{description, setting, characters, dialogue:[{speaker, text}]}]")
    images: list[dict] | None = Field(None, description="[{asset_id, label}]: the chapter's supporting pictures.")


class StoryAutopilotIn(BaseModel):
    chapters: int = Field(5, ge=1, le=30, description="How many chapters to plan and make.")
    portraits: bool = Field(True, description="Make a picture for every cast character that has none, from the description.")
    videos: bool = Field(True, description="Make each chapter's video and append it to the main film; false stops at drafts.")


class StoryChapterImageIn(BaseModel):
    asset_id: str
    label: str = Field("", max_length=200, description="What it is for, e.g. 'the rooftop bar at night'.")


# --------------------------------------------------------------------------- storyboards


class StoryboardIn(BaseModel):
    title: str = Field("Storyboard", max_length=120)
    character: str = Field(min_length=1, max_length=2000, description="Who is in every shot and how they look.")
    identity_asset_ids: list[str] = Field(min_length=1, max_length=3,
                                          description="The character's identity picture(s), ideally full body on a plain background.")
    shots: list[str] = Field(default_factory=list, max_length=40, description="One shot per line, in story order (pose, place, clothing).")
    motions: list[str] | None = Field(None, description="Optional: what moves in each shot's video; default the shot text.")
    aspect_ratio: AspectRatio = "9:16"
    style: str = Field("", max_length=1000, description="Look of the whole sheet, e.g. 'luxury boudoir photography, warm light'.")
    draw: bool = Field(True, description="Draw the sheet(s) straight away ($0.018 per 8 shots on Seedream Flash).")
    panel_asset_ids: list[str] | None = Field(None, max_length=40, description=(
        "Your own images as the panels, in order: nothing is drawn (no model call) and the board is ready at once. "
        "shots may then be left empty."))
    sheet_asset_id: str | None = Field(None, description=(
        "A ready-made sheet instead: one merged from your images gives them back as the panels; any other is cut "
        "into its panels (grid from sheet_columns/sheet_rows, or found from its borders)."))
    sheet_columns: int = Field(0, ge=0, le=40)
    sheet_rows: int = Field(0, ge=0, le=40)


class StoryboardPatch(BaseModel):
    title: str | None = None
    character: str | None = None
    style: str | None = None
    aspect_ratio: AspectRatio | None = None
    shots: list[str] | None = Field(None, max_length=40)
    motions: list[str] | None = None


class StoryboardPanelIn(BaseModel):
    shot: str | None = Field(None, max_length=1000, description="New text for the shot before it is drawn again.")
    keep_setting: bool = Field(True, description="Give the old panel as a reference, so the setting and framing stay.")
    asset_id: str | None = Field(None, description="Use this image from Media as the panel instead of drawing one.")


class StoryboardRenderIn(BaseModel):
    settings: RenderSettings | None = None
    segment_seconds: float = Field(6.0, ge=1, le=15)
    mood: str = Field("", max_length=1000)
    music: str = Field("", max_length=500, description="The music to compose, when no music bed is set.")


class IdentityPackIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field("", max_length=2000)
    identity_asset_ids: list[str] = Field(min_length=1, max_length=3)
    style: str = Field("", max_length=1000)
    anatomy: bool = Field(False, description="Add a nude front view and a rear close-up (adult platforms only).")


class StoryPackIn(BaseModel):
    name: str = Field(description="The character.")
    anatomy: bool = Field(False, description="Add a nude front view and a rear close-up (adult platforms only).")


class ComposeSheetIn(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=16, description="The images, in reading order.")
    columns: int = Field(0, ge=0, le=16, description="0 picks the most square grid.")
    cell_aspect: str = Field("", description="Cell shape, e.g. 9:16; blank uses the first image's.")
    fit: Literal["contain", "cover"] = Field("contain", description="contain keeps each image whole; cover fills and crops.")
    border: int = Field(24, ge=0, le=200)
    background: Literal["black", "white", "grey"] = "black"
    title: str = Field("", max_length=80)


class StoryboardSuggestIn(BaseModel):
    idea: str = Field(min_length=1, max_length=2000, description="What the storyboard is about, in a line or two.")
    reference_asset_ids: list[str] = Field(default_factory=list, max_length=4, description="The character's picture(s), looked at.")
    count: int = Field(8, ge=1, le=40, description="How many shots.")
    aspect_ratio: AspectRatio = "9:16"
    models: str | None = Field(None, description="A comma-separated model chain; default Gemma 4 31B, DeepSeek, Kimi K2.6.")
