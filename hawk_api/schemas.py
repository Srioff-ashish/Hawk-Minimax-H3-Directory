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
    engine: str | None = Field(None, description="auto (default: local Chroma1-HD when idle, then Qwen Image 2.1, then "
                               "Krea 2, then z-image/turbo, then Seedream; with references it starts at Qwen Image 2.1), "
                               "or one of qwen21, krea2, zimage, chroma, local, turbo, seedream, seedream-lite. "
                               "'klein' still resolves, to qwen21.")
    loras: list[ImageLoraIn] = Field(default_factory=list, description="LoRAs for local generation, from the running "
                                     "engine's family (file name or a unique part, optional strength). Qwen Image 2.1 "
                                     "always adds its repair LoRA on top, whatever is named here.")
    steps: int | None = Field(None, ge=1, le=50, description="Steps for the local engine. Omit for its default: "
                              "Qwen Image 2.1 30, Krea 2 8, Chroma1-HD 26, or a LoRA's own recommendation.")
    ref_boost: float | None = Field(None, ge=0, le=20, description="Krea 2 edit only, ignored by every other engine: "
                                   "likeness dial, 4 (default) strong likeness, 1 looser and more creative, above 10 "
                                   "breaks removals. On Qwen Image 2.1 say what stays in the prompt instead.")
    negative: str = Field("", max_length=2000, description=
        "What to keep out, for the local engines only (Atlas ignores it). It has no effect while the engine samples "
        "at cfg 1.0 -- Krea 2 Turbo and Z-Image -- because guidance at cfg 1 collapses to the positive prompt. On "
        "Qwen Image 2.1, which samples at cfg 2.0, it is the place for \"fused bodies, missing limbs, extra arms\". "
        "Chroma1-HD samples at cfg 3.8 and relies on it most; leaving it out there falls back to Chroma's own "
        "default negative rather than to nothing.")
    cfg: float | None = Field(None, ge=0.0, le=15.0, description=
        "Guidance for the local engines. Omit for the engine's own default. Above 1.0 the negative prompt starts "
        "working; too high posterises (blown greens and blues, banded surfaces). Qwen Image 2.1 defaults to 2.0, "
        "and drops to 1.0 when its adult LoRA is attached; Chroma1-HD defaults to 3.8; Krea 2 Turbo and Z-Image "
        "are trained for 1.0 and should be left there.")
    max_adult_loras: int = Field(3, ge=1, le=3, description="How many adult LoRAs one image may stack (up to 3; a note "
                                 "warns above a combined strength of 2.0). Lower it to be stricter.")


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
    image_folder: str | None = Field(None, description="Folder inside My Drive for generated images, e.g. Hawk H3/Images (a dated subfolder is added).")
    snapshots: bool | None = Field(None, description="Copy the chats-and-jobs database into Drive, so a Colab runtime that ends does not take them with it.")
    snapshot_folder: str | None = Field(None, description="Folder inside My Drive for the database copy, e.g. Hawk H3/Backups. Two files are kept there.")
    snapshot_minutes: float | None = Field(None, ge=1, le=120, description="Minutes between copies. A copy is only written when something actually changed.")


class ImageEngineRung(BaseModel):
    engine: str = Field(description="Engine id from /v1/images/engines, e.g. qwen21, krea2, zimage, chroma, turbo, seedream.")
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
    model: str | None = Field(None, description="Atlas chat model id; default HAWK_AGENT_MODEL (xai/grok-4.6).")
    name: str | None = Field(None, description="The persona's display name, e.g. Maya. Empty: taken from the persona text.")
    avatar_asset_id: str | None = Field(None, description="An image asset shown as the agent's avatar in this chat; empty removes it.")
    cast: list[CastMemberIn] | None = Field(None, description="Several characters in one chat (up to 4); the first is the lead.")
    adaptive: bool | None = Field(None, description="Characters adapt: they grow from the conversation with you and with each other.")
    whispers: bool | None = Field(None, description="Group chats: a message starting with @Name is private to that character.")


class AgentMessageIn(BaseModel):
    text: str = Field("", description="Your message.")
    attachments: list[str] = Field(default_factory=list, description="Asset ids uploaded with this message.")


class UrlAssetRequest(BaseModel):
    url: str = Field(description="A direct http(s) link to an image, audio or video file.")
    filename: str | None = Field(None, description="Override the file name (its extension decides the kind).")
