"""Request models shared by REST (OpenAPI docs) and MCP (tool schemas)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

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
    reference_asset_ids: list[str] = Field(default_factory=list, max_length=10, description="Image assets to edit or combine; switches to the edit model.")
    model: str | None = Field(None, description="Atlas image model; default bytedance/seedream-v5.0-pro/text-to-image (or /edit with references).")
    size: str | None = Field(None, description="e.g. 2048x2048, 1536x2048 or 2048x1152. Omit for the model default.")
    n: int = Field(1, ge=1, le=4, description="How many images.")
    seed: int | None = Field(None, ge=0)


class AssetUpdate(BaseModel):
    collection: str | None = Field(None, description="Move to this collection (created if new).")
    tags: list[str] | None = Field(None, description="Replace all tags.")
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)
    filename: str | None = Field(None, description="Rename the display name.")


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


class PromptIn(BaseModel):
    text: str = Field(description="The full prompt text. Saving the default text (or an empty text) resets to the built-in prompt.")


class AgentSessionIn(BaseModel):
    title: str | None = Field(None, description="Chat title; the agent renames it once the task is clear.")
    persona: str | None = Field(None, description="Who the agent should be, e.g. 'Bollywood ad-film director'.")
    model: str | None = Field(None, description="Atlas chat model id; default HAWK_AGENT_MODEL (xai/grok-4.6).")


class AgentMessageIn(BaseModel):
    text: str = Field("", description="Your message.")
    attachments: list[str] = Field(default_factory=list, description="Asset ids uploaded with this message.")


class UrlAssetRequest(BaseModel):
    url: str = Field(description="A direct http(s) link to an image, audio or video file.")
    filename: str | None = Field(None, description="Override the file name (its extension decides the kind).")
