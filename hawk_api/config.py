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
class LLMSettings:
    """Which service answers a planning or chat call, and as whom.

    The environment sets what a pod starts with and Studio changes it afterwards, the same arrangement as
    ModelSettings. An empty value everywhere means "use the one below it", so a blank field in Studio is a
    request to fall back rather than a value in its own right.
    """

    llm_provider: str = "atlas"
    atlas_api_key_override: str = ""
    openrouter_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    #: Which of atlas.ROUTING decides between the services serving one OpenRouter model id. Ignored on
    #: Atlas, which serves its own models and has nothing to choose between.
    openrouter_routing: str = "sticky"
    planner_model_override: str = ""
    agent_model_override: str = ""
    agent_summary_model_override: str = ""
    #: The characters' voices, kept apart from the director's tool-calling. Blank means "whatever the chat's
    #: own model is", which is how every chat behaved before the split.
    agent_prose_model_override: str = ""
    #: Which model looks at an image. Was a hardcoded pair of Atlas ids that 404ed on any other provider.
    agent_vision_model_override: str = ""


class LLMSettingsStore:
    """``DATA_DIR/llm_settings.json``: the provider, its key and any model overrides.

    Read on every call, so a change in Studio applies to the next request without a restart.

    **Not carried across a restart.** The Drive snapshot copies the database and nothing else, and this file
    lives on the runtime's own disk, so a restored pod is back on whatever the environment says. That is why
    the environment is a first-class source here rather than a one-time default: putting the key in a Colab
    secret survives, typing it into Studio does not.
    """

    FIELDS = ("llm_provider", "atlas_api_key_override", "openrouter_url", "openrouter_api_key",
              "openrouter_routing", "planner_model_override", "agent_model_override",
              "agent_summary_model_override", "agent_prose_model_override", "agent_vision_model_override")
    #: Written but never read back out: a key is set or replaced, not displayed or round-tripped.
    SECRETS = ("atlas_api_key_override", "openrouter_api_key")

    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "llm_settings.json")
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def stored(self) -> dict:
        """The settings that have a value. A blank one is left out, so it cannot replace the default it
        was meant to defer to -- an empty openrouter_url reaching AtlasClient becomes Atlas's own URL,
        which is how "OpenRouter" ends up talking to Atlas with an OpenRouter key."""
        data = self._load()
        return {key: str(data[key]).strip() for key in self.FIELDS
                if isinstance(data.get(key), str) and data[key].strip()}

    def resolve(self, defaults: "LLMSettings") -> "LLMSettings":
        stored = self.stored()
        return dataclasses.replace(defaults, **stored) if stored else defaults

    def hints(self) -> dict:
        """Enough of each stored key to recognise it, for a panel that must not be able to echo one back."""
        data = self._load()
        found = {}
        for key in self.SECRETS:
            value = str(data.get(key) or "").strip()
            found[f"{key}_hint"] = (f"{value[:3]}\u2026{value[-3:]}" if len(value) > 8 else "\u2026") if value else ""
        return found

    def save(self, values: dict) -> dict:
        """Store the values given. "" clears one back to the environment's choice, as in ModelStore."""
        with self._lock:
            data = self._load()
            for key, value in values.items():
                if key not in self.FIELDS:
                    raise ValueError(f"No LLM setting {key!r}; use one of: {', '.join(self.FIELDS)}.")
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
    #: Comma-separated, best first: the later ids run when the provider in force does not serve the earlier
    #: ones. A plan is 7-9k *output* tokens, which is where its cost is, so the cheaper model leads and grok
    #: stays as the rung to fall back on. grok-4.3 is last because it writes noticeably weaker plans.
    # deepseek-v4-pro cannot read images, and a plan with reference photos needs a model that can; v4.1-flash
    # can, so a plan with photos lands there instead of jumping straight to grok's price.
    planner_model: str = "deepseek/deepseek-v4-pro, deepseek/deepseek-v4.1-flash, xai/grok-4.6, xai/grok-4.3"
    atlas_url: str = "https://api.atlascloud.ai/v1"
    atlas_api_key: str = ""
    openrouter_api_key: str = ""
    #: Measured on a live pod: grok-4.6 bills $2.00/M in and $6.00/M out against deepseek-v4-pro's $0.955 and
    #: $1.911, and the director re-sends the whole prompt on every step of a chain, so this one setting moves
    #: the bill more than anything else here. deepseek-v4-pro drove a full generate -> inspect -> report chain
    #: correctly on that pod; grok stays behind it for anyone who wants it back.
    #: deepseek-v4.1-flash is the first fallback: v4-pro can spend its whole token budget reasoning and come
    #: back empty (finish_reason=length), and the same family's flash model answers the same prompt at a
    #: fraction of grok's price.
    agent_model: str = "deepseek/deepseek-v4-pro, deepseek/deepseek-v4.1-flash, x-ai/grok-4.6, xai/grok-4.3"
    #: The characters' voices. The same chain as agent_model on purpose: turning the prose split on must not by
    #: itself change how any existing chat sounds. An uncensored prose model is something to opt into -- and
    #: note that magnum-v4-72b bills its cached tokens at full price, so on a group chat, where nearly every
    #: token is a re-sent prompt, it costs more than grok-4.6 does.
    agent_prose_model: str = "deepseek/deepseek-v4-pro, deepseek/deepseek-v4.1-flash, x-ai/grok-4.6, xai/grok-4.3"
    #: Writes chat summaries whatever the chat's model is: cheap, and good enough to condense. Its cached
    #: tokens bill at $0.001/M, which is what makes compaction almost free.
    agent_summary_model: str = "deepseek/deepseek-v4.1-flash, deepseek-ai/deepseek-v4.1-flash, xai/grok-4.3"
    #: Looks at images for inspect_image when the chat's own model cannot see. Inspection sends pictures, so it
    #: is token-heavy: a model built for vision at $0.104/M beats a general one at $2.00/M.
    #: Measured on Atlas with a mismatched brief and with an adult nude: none of these refused, and
    #: qwen3.6-35b-a3b caught the same framing flaw grok-4.6 did, at about 1/7 the cost and 4x the speed
    #: (grok-4.3 and qwen3.5-flash passed it). qwen3-vl-32b is not on Atlas; it stays for OpenRouter.
    #: Only three candidates are tried per check, and a refusal still bills, so order matters.
    agent_vision_model: str = "qwen/qwen3.6-35b-a3b, xai/grok-4.3, xai/grok-4.6, qwen/qwen3-vl-32b-instruct"
    #: Summarise older messages once the conversation sent with each call passes this many tokens (estimated).
    agent_compact_tokens: int = 20_000
    #: Messages kept word for word after an automatic summary.
    agent_keep_messages: int = 10
    #: How much of the prompt recalled facts may fill. 500 is roughly 10-15 one-sentence facts.
    agent_recall_tokens: int = 500
    #: The kill switch for recall. False stops facts being injected but keeps extracting them, so turning it
    #: back on has data to work with rather than starting from nothing.
    agent_graph_recall: bool = True
    #: Text-to-image default: fast and cheap. Edits (reference images) always use Seedream edit.
    image_model: str = "z-image/turbo"
    #: auto = local Krea 2 when installed and idle, else image_model, else Seedream.
    image_engine: str = "auto"
    krea_unet: str = "krea2_turbo_fp8_scaled.safetensors"
    krea_clip: str = "qwen3vl_4b_fp8_scaled.safetensors"
    krea_vae: str = "qwen_image_vae.safetensors"
    qwen21_unet: str = "qwen_image_2.1_bf16.safetensors"
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
    llm_overrides: LLMSettings = field(default_factory=LLMSettings)

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
            openrouter_api_key=_env("OPENROUTER_API_KEY"),
            agent_model=_env("HAWK_AGENT_MODEL", cls.agent_model),
            agent_prose_model=_env("HAWK_AGENT_PROSE_MODEL", cls.agent_prose_model),
            agent_summary_model=_env("HAWK_AGENT_SUMMARY_MODEL", cls.agent_summary_model),
            agent_vision_model=_env("HAWK_AGENT_VISION_MODEL", cls.agent_vision_model),
            agent_compact_tokens=int(_env("HAWK_AGENT_COMPACT_TOKENS", str(cls.agent_compact_tokens))),
            agent_recall_tokens=int(_env("HAWK_AGENT_RECALL_TOKENS", str(cls.agent_recall_tokens))),
            agent_graph_recall=_env("HAWK_AGENT_GRAPH_RECALL", "1").strip().lower() not in ("0", "false", "no"),
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
            # A pod restored from a snapshot has its database back and llm_settings.json gone, so the
            # environment is the only place a provider choice survives a restart.
            llm_overrides=LLMSettings(
                llm_provider=_env("HAWK_LLM_PROVIDER", LLMSettings.llm_provider),
                openrouter_url=_env("OPENROUTER_URL", LLMSettings.openrouter_url),
                openrouter_routing=_env("OPENROUTER_ROUTING", LLMSettings.openrouter_routing),
                # The key is not read here: Settings.openrouter_api_key already holds OPENROUTER_API_KEY,
                # and the atlas property falls back to it. Two readers of one variable is one too many.
                planner_model_override=_env("HAWK_PLANNER_MODEL_OVERRIDE"),
                agent_model_override=_env("HAWK_AGENT_MODEL_OVERRIDE"),
                agent_summary_model_override=_env("HAWK_AGENT_SUMMARY_MODEL_OVERRIDE"),
                agent_prose_model_override=_env("HAWK_AGENT_PROSE_MODEL_OVERRIDE"),
                agent_vision_model_override=_env("HAWK_AGENT_VISION_MODEL_OVERRIDE"),
            ),
        )
