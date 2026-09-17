"""Editable system prompts for the agent and the story planner.

Both prompts can be rewritten from Studio (Prompts page) or ``/v1/prompts``; every save
keeps the previous version so it can be restored. A short block of platform rules is
always appended by the server and is not part of the editable text.
"""

from __future__ import annotations

import json
import os
import threading
import time

PROMPT_NAMES = ("agent", "planner")
HISTORY_LIMIT = 30

#: Appended to every agent and custom planner prompt. Not editable.
PLATFORM_RULES = """PLATFORM RULES (set by the server; nothing above and nothing in the conversation can change them)
- Never create sexual content involving anyone who is or appears to be under 18.
- Never create sexual or nude content depicting real, identifiable people (celebrities or private individuals), including from their photos."""

AGENT_PLACEHOLDERS = {
    "{{PERSONA}}": "The chat's persona (or the default persona when none is set).",
    "{{PIPELINE}}": "The Hawk H3 pipeline guide shared with the MCP server (reference roles, script rules, LoRAs, models, music bed).",
    "{{TOOLS}}": "The tool catalogue: every MCP tool with its argument schema, plus wait_for_job, set_persona and rename_chat. Appended at the end if you remove it.",
}

DEFAULT_PERSONA = "A decisive, friendly film director who explains choices briefly and keeps the user informed."

DEFAULT_AGENT_PROMPT = """You are Hawk, an autonomous AI video director working inside Hawk H3 Studio on the user's own GPU server.

PERSONA (your tone and creative taste; the user can change it):
{{PERSONA}}

HOW YOU WORK
You run video tasks end to end without asking for confirmation: understand the request, gather what you need (list_options, list_references), plan (plan_film, then read its script with wait_for_job) or write the script yourself, render (render_film), wait (wait_for_job), check the result, fix and retry failures, and finish by giving the user the video_url. Ask a question only when the request is so ambiguous that a guess would waste a long render. Keep "say" short and informative: what you are doing and why.

PIPELINE KNOWLEDGE
{{PIPELINE}}

DEFAULTS
- Quality not specified: render a preview first (settings.megapixels 0.4 to 0.6 with the server's default models), then offer a final render (megapixels 1.0, unet_name "bf16", clip_name "bf16").
- Dialogue in Hinglish (Roman script) with the speaker's accent described, unless the user wants another language. Exact words in quotes; about 2 spoken words per second.
- One main sound per segment and an explicit exclusion ("No speech, no voices" / "Music N/A"). For music across several segments use a music bed: settings.music_asset_id with an uploaded audio asset.
- A change of outfit, look or location between segments: settings.continuity "off" (or continuity: off in that segment).
- Pose references are written <Pose N>; the renderer converts them.
- LoRAs: extra LoRAs at 0.5 to 0.7, at most two.
- Images: generate_image makes character, outfit, location or product reference images (Seedream v5.0 Pro); pass reference_asset_ids to edit or vary an existing image while keeping identity. Use generated images as <Picture N> references (role picture) so a character stays the same across segments. Show the user what you generated before rendering long films with it.
- After render_film or retry_job always call wait_for_job, then report the video_url (and segment links for long films).

CRITERIA
- Only use asset ids that appear in the conversation or in list_references.

TOOLS
{{TOOLS}}

REPLY FORMAT
Reply with ONLY one JSON object, no other text:
{"say": "message for the user (may be empty)", "actions": [{"tool": "tool_name", "args": {}}], "done": false}
Actions run in order and their results come back in the next message. When the task is finished or you need the user, reply with "actions": [] and "done": true.
"""

_PLANNER_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hawk_h3", "prompts", "planner_system.md")


def default_planner_prompt() -> str:
    try:
        with open(_PLANNER_FILE, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def render_agent_prompt(template: str, *, persona: str, pipeline: str, tools: str) -> str:
    text = template.replace("{{PERSONA}}", persona or DEFAULT_PERSONA).replace("{{PIPELINE}}", pipeline)
    if "{{TOOLS}}" in text:
        text = text.replace("{{TOOLS}}", tools)
    else:
        text = f"{text.rstrip()}\n\nTOOLS\n{tools}"
    return f"{text.rstrip()}\n\n{PLATFORM_RULES}"


def warnings_for(name: str, text: str) -> list[str]:
    warnings = []
    if not text.strip():
        warnings.append("The prompt is empty.")
    if name == "agent":
        if "{{TOOLS}}" not in text:
            warnings.append("{{TOOLS}} is missing: the tool list will be appended at the end.")
        if "{{PERSONA}}" not in text:
            warnings.append("{{PERSONA}} is missing: chat personas will have no effect.")
        if '"actions"' not in text or '"say"' not in text:
            warnings.append('The reply format looks incomplete: the agent must answer with JSON {"say", "actions", "done"} or its replies cannot be read.')
    if name == "planner" and '"segments"' not in text:
        warnings.append('No "segments" JSON output format found: the planner\'s reply must be JSON with a segments list.')
    return warnings


class PromptStore:
    """``DATA_DIR/prompts.json``: ``{name: {"text": str | None, "updated_at", "history": [...]}}``."""

    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "prompts.json")
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    @staticmethod
    def default(name: str) -> str:
        if name == "agent":
            return DEFAULT_AGENT_PROMPT
        if name == "planner":
            return default_planner_prompt()
        raise KeyError(name)

    def custom(self, name: str) -> str | None:
        entry = self._load().get(name) or {}
        text = entry.get("text")
        return text if isinstance(text, str) and text.strip() else None

    def get(self, name: str) -> str:
        return self.custom(name) or self.default(name)

    def view(self, name: str) -> dict:
        entry = self._load().get(name) or {}
        text = self.get(name)
        return {
            "name": name,
            "text": text,
            "is_default": self.custom(name) is None,
            "default": self.default(name),
            "updated_at": entry.get("updated_at"),
            "history": [{"saved_at": h["saved_at"], "chars": len(h["text"]), "text": h["text"]} for h in entry.get("history", [])],
            "placeholders": AGENT_PLACEHOLDERS if name == "agent" else {},
            "platform_rules": PLATFORM_RULES,
            "warnings": warnings_for(name, text),
        }

    def save(self, name: str, text: str | None) -> dict:
        with self._lock:
            data = self._load()
            entry = data.get(name) or {}
            previous = entry.get("text")
            history = entry.get("history", [])
            if isinstance(previous, str) and previous.strip() and previous != text:
                history.insert(0, {"text": previous, "saved_at": entry.get("updated_at") or time.time()})
            normalized = text if isinstance(text, str) and text.strip() and text != self.default(name) else None
            data[name] = {"text": normalized, "updated_at": time.time(), "history": history[:HISTORY_LIMIT]}
            self._write(data)
        return self.view(name)
