"""Gateway settings, read from environment variables."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class ModelSettings:
    """What Hawk H3 Model Loader loads for every render."""

    unet_name: str = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    clip_name: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    video_vae: str = "minimax_h3_video_vae_fp16.safetensors"
    audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors"
    shift_video: float = 12.0
    shift_audio: float = 3.0
    attention: str = "sol scheduled + sage"
    weight_dtype: str = "default"
    clip_device: str = "default"


class ModelStore:
    """``DATA_DIR/render_models.json``: which H3 files every render loads.

    The environment sets what a pod starts with; this lets Studio change it afterwards without editing the
    notebook or restarting. Read on every render, and an empty value means "use the one from the environment".
    """

    FIELDS = ("unet_name", "clip_name", "video_vae", "audio_vae")

    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "render_models.json")
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def stored(self) -> dict:
        data = self._load()
        return {key: str(data[key]) for key in self.FIELDS if isinstance(data.get(key), str) and data[key].strip()}

    def resolve(self, defaults: "ModelSettings") -> "ModelSettings":
        stored = self.stored()
        return dataclasses.replace(defaults, **stored) if stored else defaults

    def save(self, values: dict) -> dict:
        """Store the names given. "" clears one back to the environment's choice."""
        with self._lock:
            data = self._load()
            for key, value in values.items():
                if key not in self.FIELDS:
                    raise ValueError(f"No render model setting {key!r}; use one of: {', '.join(self.FIELDS)}.")
                if value is None:
                    continue
                if str(value).strip():
                    data[key] = str(value).strip()
                else:
                    data.pop(key, None)
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            os.replace(tmp, self.path)
        return self.stored()


@dataclass(frozen=True)
class Settings:
    token: str
    comfy_url: str = "http://127.0.0.1:8188"
    #: Public address of this gateway, used to build download and upload links.
    public_base_url: str = "http://127.0.0.1:8000"
    data_dir: str = "hawk_api_data"
    max_upload_mb: int = 2048
    link_ttl_seconds: int = 7 * 24 * 3600
    lora_cache_seconds: float = 60.0
    reconcile_seconds: float = 30.0
    planner_model: str = "xai/grok-4.3"
    atlas_url: str = "https://api.atlascloud.ai/v1"
    atlas_api_key: str = ""
    agent_model: str = "xai/grok-4.6"
    #: Writes chat summaries whatever the chat's model is: cheap, and good enough to condense.
    agent_summary_model: str = "deepseek-ai/deepseek-v4.1-flash"
    #: Summarise older messages once the conversation sent with each call passes this many tokens (estimated).
    agent_compact_tokens: int = 20_000
    #: Messages kept word for word after an automatic summary.
    agent_keep_messages: int = 10
    #: Text-to-image default: fast and cheap. Edits (reference images) always use Seedream edit.
    image_model: str = "z-image/turbo"
    #: auto = local Krea 2 when installed and idle, else image_model, else Seedream.
    image_engine: str = "auto"
    krea_unet: str = "krea2_turbo_fp8_scaled.safetensors"
    krea_clip: str = "qwen3vl_4b_fp8_scaled.safetensors"
    krea_vae: str = "qwen_image_vae.safetensors"
    qwen21_unet: str = "qwen_image_2.1_int8_convrot.safetensors"
    qwen21_clip: str = "qwen3vl_8b_bf16.safetensors"
    qwen21_vae: str = "qwen_image_2.1_vae_bf16.safetensors"
    zimage_unet: str = "z_image_turbo_nvfp4.safetensors"
    zimage_clip: str = "qwen_3_4b_fp4_mixed.safetensors"
    zimage_vae: str = "z_image_ae.safetensors"
    #: Local Krea 2 text-to-image attaches the go-to adult pair (SNOFS + Mystic XXX) unless the request names its
    #: own adult LoRA. HAWK_KREA_ADULT_DEFAULT=0 turns it off; edits of uploaded photos never get them.
    krea_adult_default: bool = True
    #: ComfyUI's input folder on this machine; imports copy files straight into it when set.
    comfy_input_dir: str = ""
    #: ComfyUI's output folder on this machine; videos are then served straight from disk (with byte ranges).
    comfy_output_dir: str = ""
    #: Mounted Google Drive (Colab: drive.mount("/content/drive")).
    drive_root: str = "/content/drive/MyDrive"
    max_import_files: int = 2000
    models: ModelSettings = field(default_factory=ModelSettings)

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "jobs.sqlite3")

    @property
    def loras_path(self) -> str:
        return os.path.join(self.data_dir, "loras.json")

    @classmethod
    def from_env(cls) -> "Settings":
        token = _env("HAWK_API_TOKEN")
        if len(token) < 16:
            raise SystemExit(
                "Set HAWK_API_TOKEN to a random secret of at least 16 characters, "
                "e.g. `export HAWK_API_TOKEN=$(openssl rand -hex 24)`."
            )
        defaults = ModelSettings()
        models = ModelSettings(
            unet_name=_env("HAWK_UNET", defaults.unet_name),
            clip_name=_env("HAWK_CLIP", defaults.clip_name),
            video_vae=_env("HAWK_VIDEO_VAE", defaults.video_vae),
            audio_vae=_env("HAWK_AUDIO_VAE", defaults.audio_vae),
            attention=_env("HAWK_ATTENTION", defaults.attention),
            weight_dtype=_env("HAWK_WEIGHT_DTYPE", defaults.weight_dtype),
        )
        return cls(
            token=token,
            comfy_url=_env("COMFY_URL", cls.comfy_url).rstrip("/"),
            public_base_url=_env("PUBLIC_BASE_URL", cls.public_base_url).rstrip("/"),
            data_dir=_env("DATA_DIR", cls.data_dir),
            max_upload_mb=int(_env("MAX_UPLOAD_MB", str(cls.max_upload_mb))),
            link_ttl_seconds=int(_env("LINK_TTL_SECONDS", str(cls.link_ttl_seconds))),
            planner_model=_env("HAWK_PLANNER_MODEL", cls.planner_model),
            atlas_url=_env("ATLAS_API_URL", cls.atlas_url).rstrip("/"),
            atlas_api_key=_env("ATLAS_API_KEY"),
            agent_model=_env("HAWK_AGENT_MODEL", cls.agent_model),
            agent_summary_model=_env("HAWK_AGENT_SUMMARY_MODEL", cls.agent_summary_model),
            agent_compact_tokens=int(_env("HAWK_AGENT_COMPACT_TOKENS", str(cls.agent_compact_tokens))),
            agent_keep_messages=max(2, int(_env("HAWK_AGENT_KEEP_MESSAGES", str(cls.agent_keep_messages)))),
            image_model=_env("HAWK_IMAGE_MODEL", cls.image_model),
            image_engine=_env("HAWK_IMAGE_ENGINE", cls.image_engine),
            krea_unet=_env("HAWK_KREA_UNET", cls.krea_unet),
            krea_clip=_env("HAWK_KREA_CLIP", cls.krea_clip),
            krea_vae=_env("HAWK_KREA_VAE", cls.krea_vae),
            qwen21_unet=_env("HAWK_QWEN21_UNET", cls.qwen21_unet),
            qwen21_clip=_env("HAWK_QWEN21_CLIP", cls.qwen21_clip),
            qwen21_vae=_env("HAWK_QWEN21_VAE", cls.qwen21_vae),
            zimage_unet=_env("HAWK_ZIMAGE_UNET", cls.zimage_unet),
            zimage_clip=_env("HAWK_ZIMAGE_CLIP", cls.zimage_clip),
            zimage_vae=_env("HAWK_ZIMAGE_VAE", cls.zimage_vae),
            krea_adult_default=_env("HAWK_KREA_ADULT_DEFAULT", "1") not in ("0", "false", "no", "off"),
            comfy_input_dir=_env("COMFY_INPUT_DIR"),
            comfy_output_dir=_env("COMFY_OUTPUT_DIR"),
            drive_root=_env("HAWK_DRIVE_ROOT", cls.drive_root),
            models=models,
        )
