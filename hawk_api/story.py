"""Story: co-write an original story with a model, turn each chapter into video, and grow one main film.

The user chats ideas with a co-writer (Gemma 4 31B, then Kimi K2.6, then DeepSeek V4.1 Flash by default). When a
chapter is ready the co-writer drafts it -- scenes, who is in them, the exact lines -- and the user can edit the
draft. Make video runs with no stops: the scenes become a story-planner job (with the cast's pictures and the
chapter's own images as references), the plan is rendered, and the render is appended to the story's main film,
which is every finished chapter joined in order. Redoing a chapter's video replaces it in the main film.

Stories, their chat and their chapters live in jobs.sqlite3 (so the database's Drive backup keeps them), and a
conversion a restart interrupted carries on from where it was.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid

from hawk_h3.script import ScriptError, max_segment_seconds, parse_script

from . import models_llm
from .atlas import AtlasError
from .jobs import NotFound, RequestError, Unavailable
from .manhwa import (CONTEXT_MAX, build_srt, concat_videos, film_to_drive, is_minor, media_seconds, parse_json,
                     rendered_file, shot_seconds, vet_face_image)
from .schemas import PlanRequest, ReferenceIn, RenderSettings, VideoRequest

log = logging.getLogger("hawk_api.story")

#: The co-writer, best first: Gemma 4, then DeepSeek V4.1 Flash (fast), then Kimi K2.6 (slow, reasons at length).
#: Gemma 4 is on OpenRouter only, so on Atlas the chain starts at DeepSeek.
DEFAULT_MODELS = "google/gemma-4-31b-it, deepseek-ai/deepseek-v4.1-flash, moonshotai/kimi-k2.6"
MODEL_CHOICES = ("google/gemma-4-31b-it", "moonshotai/kimi-k2.6", "deepseek-ai/deepseek-v4.1-flash")
#: The video plan's chain, settable apart from the co-writer's. Same order by default: DeepSeek writes a plan in
#: seconds when Gemma is rate-limited, and Kimi K2.6 -- minutes per plan, often cut off -- comes last.
PLAN_MODELS = "google/gemma-4-31b-it, deepseek-ai/deepseek-v4.1-flash, moonshotai/kimi-k2.6"
ROLES = ("cast", "extra", "narrator")
MAX_PICTURES = 9  # a render's picture limit
MAX_SCENES = 30
RECENT_MESSAGES = 60  # the chat since the last chapter that goes back to the model, at most
CHAT_MAX_TOKENS = 4000
DRAFT_MAX_TOKENS = 12000
POLL = 5.0
PLAN_WAIT = 30 * 60
RENDER_WAIT = 12 * 3600
ACTIVE = ("planning", "rendering", "appending")
SETTINGS_DEFAULTS = {"models": DEFAULT_MODELS, "plan_models": PLAN_MODELS, "aspect_ratio": "16:9", "segment_seconds": 10.0, "render": {},
                     "subtitles": "file",
                     # film: chapters are made by the film engine (storyboard, check, joined film) whenever the cast
                     # have pictures; classic: the story planner's single render, as before the engine
                     "video_engine": "film"}

CHAT_PROMPT = """\
You are the co-writer of a story that will become a film, one chapter at a time. The user gives you ideas; you
build on them: say what you like, add concrete detail (what we see, who does what, what they say), suggest where
it could go next, and ask at most one question. Keep the user's ideas -- they lead. Under 250 words, plain text.
When the user is ready, they press "Draft chapter" and you will turn this conversation into the chapter.

Every character is an adult (18 or older). Never put anyone under 18 in a sexual or romantic situation.

Title: {title}
Premise: {premise}
Style and tone: {style}
Cast:
{cast}
Story so far (finished chapters):
{chapters}
"""

DRAFT_PROMPT = """\
Turn the conversation below into chapter {number} of the story, ready to be filmed. Use what the user decided;
fill gaps the way the conversation points. Write 3-{max_scenes} scenes in order. Each scene is one continuous shot
of film:
- "description": what the camera sees -- who is where, what they do, their expressions, the camera angle -- in 2-4
  sentences, concrete and visual.
- "setting": where and when, the light.
- "characters": the names of everyone visible (use the cast's names).
- "dialogue": every line spoken in the scene, in order: {{"speaker": a name or "narration", "text": the exact line}}.
Every character is an adult (18 or older).

Title of the story: {title}
Premise: {premise}
Style and tone: {style}
Cast:
{cast}
Earlier chapters:
{chapters}

Conversation for this chapter:
{conversation}

Reply with only JSON:
{{"title": "chapter title", "summary": "2-4 sentences", "scenes": [{{"description": "", "setting": "",
"characters": [], "dialogue": [{{"speaker": "", "text": ""}}]}}]}}
"""


OUTLINE_ATTEMPTS = 4

OUTLINE_PROMPT = """\
Plan the next {count} chapter{plural} of this story, to be filmed one chapter at a time. Continue from where the
earlier chapters end (or begin the story if there are none), following the premise and its arc closely.

Title: {title}
Premise: {premise}
Style and tone: {style}
Cast so far:
{cast}
Earlier chapters:
{chapters}
{planned}
Write exactly {count} chapter{plural} in "chapters" -- no fewer. For each chapter: a short title and "beats" -- 3-6 sentences on what happens, where, who is there, how it moves
the arc forward, and one or two key lines of dialogue. List the characters too: everyone the chapters need, with a
description an image model can draw (adult age, build, face, hair, usual clothes) and a role: "cast" for people
who appear on screen, "narrator" for a voice-over. Every character is an adult (18 or older).

Reply with only JSON:
{{"characters": [{{"name": "", "description": "", "role": "cast"}}],
"chapters": [{{"title": "", "beats": ""}}]}}
"""

PORTRAIT_PROMPT = ("A photorealistic portrait photograph of {name}, an adult: {description}. Head and shoulders, "
                   "facing the camera, natural expression, soft even light, plain background, 85mm lens. Only one "
                   "person, no text.")

_CHAPTER_PREFIX = re.compile(r"^\s*(?:chapter|ch\.?)\s*\d+\s*[:.\-\u2013\u2014]\s*", re.I)


def _clean(value, limit: int = CONTEXT_MAX) -> str:
    return " ".join(str(value or "").split())[:limit]


def clean_scene(item: dict) -> dict:
    return {"description": _clean(item.get("description"), 1500), "setting": _clean(item.get("setting"), 400),
            "characters": [_clean(n, 80) for n in item.get("characters") or [] if _clean(n, 80)][:8],
            "dialogue": [{"speaker": _clean(d.get("speaker"), 80) or "narration", "text": str(d.get("text") or "").strip()[:600]}
                         for d in item.get("dialogue") or [] if isinstance(d, dict) and str(d.get("text") or "").strip()]}


def chapter_brief(story: dict, chapter: dict, earlier: list[dict], labels: list[str], cap: float) -> tuple[str, list[float]]:
    """The story planner's brief for one chapter: one segment per scene, each timed by its dialogue, the lines
    spoken word for word. Returns (brief, the seconds asked for per scene)."""
    scenes = chapter["scenes"]
    seconds = [shot_seconds(scene["dialogue"], cap) for scene in scenes]
    lines = [f"A live-action film: \"{story['title']}\", chapter {chapter['idx'] + 1}: {chapter.get('title') or ''}.",
             f"Write exactly {len(scenes)} segment{'s' if len(scenes) != 1 else ''}, one per scene below, in order, each "
             "with the \"duration\" given. Characters speak their lines word for word, in the order given; narration "
             "is a voice-over; a scene with no lines has no speech. All characters are adults. Leave "
             "references_to_make empty."]
    if labels:
        lines.append("Attached pictures: " + "; ".join(f"<Picture {n}> = {label}" for n, label in enumerate(labels, 1))
                     + ". Use a character's picture in every segment they are in, so they look the same throughout.")
        if any(label.startswith("first frame of scene") for label in labels):
            lines.append("A picture labelled 'first frame of scene N' is that segment's literal first frame (keyframe "
                         "completion): start the segment exactly on it and animate from there, keeping its pose, framing, "
                         "setting and clothing; the character pictures keep the faces.")
    if story.get("style"):
        lines.append(f"Style and tone: {story['style']}")
    if earlier:
        lines.append("Story so far: " + " ".join(f"Ch {c['idx'] + 1}: {c.get('summary') or c.get('title') or ''}"
                                                for c in earlier)[:3000])
    cast = [f"{c['name']}: {c.get('description') or ''}" for c in story.get("characters") or [] if c.get("role") != "narrator"]
    if cast:
        lines.append("Cast: " + "; ".join(cast))
    for n, (scene, sec) in enumerate(zip(scenes, seconds), 1):
        lines.append(f"\nScene {n} -- {sec:g} s" + (f", {scene['setting']}" if scene.get("setting") else "") + ":")
        lines.append(scene.get("description") or "")
        if scene.get("characters"):
            lines.append("People: " + ", ".join(scene["characters"]))
        for said in scene["dialogue"]:
            who = said["speaker"]
            lines.append(f'{"Narration (voice-over)" if who.lower() == "narration" else who}: "{said["text"]}"')
    return "\n".join(lines), seconds


# --------------------------------------------------------------------------- store


class StoryStore:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS stories (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS story_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, story_id TEXT NOT NULL, role TEXT NOT NULL, data TEXT NOT NULL,
        created_at REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS story_messages_story ON story_messages(story_id, id);
    CREATE TABLE IF NOT EXISTS story_chapters (
        id TEXT PRIMARY KEY, story_id TEXT NOT NULL, idx INTEGER NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS story_chapters_story ON story_chapters(story_id, idx);
    """

    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(self._SCHEMA)
            self._db.commit()

    def save_story(self, story: dict) -> None:
        story["updated_at"] = time.time()
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO stories VALUES (?, ?, ?, ?, ?)",
                             (story["id"], story["title"], json.dumps(story, ensure_ascii=False), story["created_at"],
                              story["updated_at"]))
            self._db.commit()

    def get_story(self, story_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM stories WHERE id = ?", (story_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def stories(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM stories ORDER BY updated_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete_story(self, story_id: str) -> None:
        with self._lock:
            for table, column in (("story_messages", "story_id"), ("story_chapters", "story_id"), ("stories", "id")):
                self._db.execute(f"DELETE FROM {table} WHERE {column} = ?", (story_id,))
            self._db.commit()

    def add_message(self, story_id: str, role: str, data: dict) -> dict:
        now = time.time()
        with self._lock:
            cursor = self._db.execute("INSERT INTO story_messages (story_id, role, data, created_at) VALUES (?, ?, ?, ?)",
                                      (story_id, role, json.dumps(data, ensure_ascii=False), now))
            self._db.commit()
        return {"id": cursor.lastrowid, "role": role, "created_at": now, **data}

    def messages(self, story_id: str, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, role, data, created_at FROM story_messages WHERE story_id = ? "
                                    "ORDER BY id DESC LIMIT ?", (story_id, limit)).fetchall()
        return [{"id": i, "role": r, "created_at": t, **json.loads(d)} for i, r, d, t in reversed(rows)]

    def save_chapter(self, chapter: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO story_chapters VALUES (?, ?, ?, ?, ?)",
                             (chapter["id"], chapter["story_id"], chapter["idx"], json.dumps(chapter, ensure_ascii=False),
                              chapter["created_at"]))
            self._db.commit()

    def chapters(self, story_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM story_chapters WHERE story_id = ? ORDER BY idx",
                                    (story_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_chapter(self, chapter_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM story_chapters WHERE id = ?", (chapter_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def delete_chapter(self, chapter_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM story_chapters WHERE id = ?", (chapter_id,))
            self._db.commit()

    def active_chapters(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM story_chapters").fetchall()
        return [c for c in (json.loads(row[0]) for row in rows) if c.get("status") in ACTIVE]


# --------------------------------------------------------------------------- manager


class StoryManager:
    def __init__(self, service, store: StoryStore | None = None):
        self.service = service
        self.store = store or StoryStore(service.settings.db_path)
        self._tasks: dict[str, asyncio.Task] = {}
        self._film_locks: dict[str, asyncio.Lock] = {}
        self._busy: dict[str, dict] = {}  # story id -> {"what": "chat" | "draft", "since"}: one co-writer call at a time
        self._jobs: dict[str, asyncio.Task] = {}  # background chat / draft calls, by story
        self._autopilots: dict[str, asyncio.Task] = {}
        self.poll = POLL
        self.boards = None  # the StoryboardManager; create_app sets it, so a chapter can have a storyboard

    async def stop(self) -> None:
        tasks = list(self._tasks.values()) + list(self._jobs.values()) + list(self._autopilots.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ----------------------------------------------------------------- stories

    def _story(self, story_id: str) -> dict:
        story = self.store.get_story(story_id)
        if story is None:
            raise NotFound(f"No story {story_id!r}.")
        return story

    def create_story(self, title: str, premise: str = "", style: str = "") -> dict:
        title = _clean(title, 120)
        if not title:
            raise RequestError("Give the story a title.")
        now = time.time()
        story = {"id": uuid.uuid4().hex[:12], "title": title, "premise": _clean(premise), "style": _clean(style, 600),
                 "characters": [], "settings": dict(SETTINGS_DEFAULTS), "film": None, "created_at": now}
        self.store.save_story(story)
        return self.view(story["id"])

    def update_story(self, story_id: str, *, title=None, premise=None, style=None, settings: dict | None = None) -> dict:
        story = self._story(story_id)
        if title is not None and _clean(title, 120):
            story["title"] = _clean(title, 120)
        if premise is not None:
            story["premise"] = _clean(premise)
        if style is not None:
            story["style"] = _clean(style, 600)
        if settings:
            merged = {**SETTINGS_DEFAULTS, **(story.get("settings") or {})}
            for key, value in settings.items():
                if key in SETTINGS_DEFAULTS and value is not None:
                    merged[key] = value
            if merged.get("video_engine") not in ("film", "classic"):
                raise RequestError("video_engine is film or classic.")
            if merged.get("render"):
                merged["render"] = {k: v for k, v in merged["render"].items()
                                    if k not in ("aspect_ratio", "default_seconds") and v not in (None, "")}
                try:
                    RenderSettings(**merged["render"])
                except ValueError as exc:
                    raise RequestError(str(exc).splitlines()[0]) from None
            story["settings"] = merged
        self.store.save_story(story)
        return self.view(story_id)

    async def delete_story(self, story_id: str) -> dict:
        """The story, its chat and its chapters. Rendered videos and the main film stay in Media."""
        self._story(story_id)
        for chapter in self.store.chapters(story_id):
            task = self._tasks.pop(chapter["id"], None)
            if task is not None:
                task.cancel()
        self.store.delete_story(story_id)
        return {"deleted": story_id}

    def list_stories(self) -> list[dict]:
        out = []
        for story in self.store.stories():
            chapters = self.store.chapters(story["id"])
            out.append({"id": story["id"], "title": story["title"], "premise": story.get("premise", ""),
                        "chapters": len(chapters), "done": sum(1 for c in chapters if c.get("status") == "done"),
                        "updated_at": story.get("updated_at")})
        return out

    def _image_view(self, asset_id: str) -> dict | None:
        asset = self.service.store.get_asset(asset_id) if asset_id else None
        if asset is None:
            return None
        shown = self.service.asset_view(asset)
        return {"id": asset_id, "thumb_url": shown.get("thumb_url"), "file_url": shown.get("file_url")}

    def _job_video(self, job_id: str | None) -> str | None:
        job = self.service.store.get_job(job_id) if job_id else None
        if not job or job.get("status") != "done":
            return None
        viewer = getattr(self.service, "job_view", None)
        return (viewer(job) or {}).get("video_url") if viewer else None

    def _asset_video(self, asset_id: str | None) -> str | None:
        asset = self.service.store.get_asset(asset_id) if asset_id else None
        return self.service.asset_view(asset).get("file_url") if asset else None

    def view(self, story_id: str) -> dict:
        self._renumber(story_id)
        story = self._story(story_id)
        characters = []
        for c in story.get("characters") or []:
            characters.append(dict(c, minor=is_minor(c["name"], {"appearance": c.get("description")}),
                                   images=[v for v in (self._image_view(i) for i in c.get("image_asset_ids") or []) if v]))
        chapters = []
        for c in self.store.chapters(story_id):
            chapters.append(dict(c, video_url=self._job_video(c.get("render_job_id")) or self._asset_video(c.get("film_asset_id")),
                                 image_views=[dict(i, **(self._image_view(i["asset_id"]) or {}))
                                              for i in c.get("images") or []]))
        film = dict(story["film"]) if story.get("film") else None
        if film:
            shown = self._image_view(film.get("asset_id"))  # any asset: the view only needs its urls
            film.update(file_url=(shown or {}).get("file_url"))
        return dict(story, characters=characters, chapters=chapters, film=film, busy=self._busy.get(story_id),
                    settings={**SETTINGS_DEFAULTS, **(story.get("settings") or {})}, model_choices=list(MODEL_CHOICES))

    # ----------------------------------------------------------------- cast

    def _character(self, story: dict, name: str) -> dict:
        found = next((c for c in story.get("characters") or [] if c["name"] == name), None)
        if found is None:
            raise NotFound(f"No character {name!r} in this story.")
        return found

    def add_character(self, story_id: str, name: str, description: str = "", role: str = "cast") -> dict:
        story = self._story(story_id)
        name = _clean(name, 80)
        if not name:
            raise RequestError("Give the character a name.")
        if any(c["name"] == name for c in story.get("characters") or []):
            raise RequestError(f"{name!r} is already in the cast.")
        if role not in ROLES:
            raise RequestError(f"role should be one of {', '.join(ROLES)}.")
        story.setdefault("characters", []).append({"name": name, "description": _clean(description), "role": role,
                                                   "image_asset_ids": []})
        self.store.save_story(story)
        return self.view(story_id)

    def edit_character(self, story_id: str, name: str, *, new_name=None, description=None, role=None) -> dict:
        story = self._story(story_id)
        character = self._character(story, name)
        if role is not None:
            if role not in ROLES:
                raise RequestError(f"role should be one of {', '.join(ROLES)}.")
            character["role"] = role
            if role == "narrator":
                character["image_asset_ids"] = []  # a voice has no face
        if description is not None:
            character["description"] = _clean(description)
        if character["image_asset_ids"] and is_minor(character["name"], {"appearance": character["description"]}):
            raise RequestError(f"{character['name']} now reads as under 18, so they can't keep a picture: remove it first.")
        new_name = _clean(new_name or "", 80)
        if new_name and new_name != name:
            if any(c["name"] == new_name for c in story["characters"]):
                raise RequestError(f"{new_name!r} is already in the cast.")
            character["name"] = new_name
            for chapter in self.store.chapters(story_id):  # the drafts follow the rename
                for scene in chapter.get("scenes") or []:
                    scene["characters"] = [new_name if n == name else n for n in scene.get("characters") or []]
                    for line in scene.get("dialogue") or []:
                        if line.get("speaker") == name:
                            line["speaker"] = new_name
                self.store.save_chapter(chapter)
        self.store.save_story(story)
        return self.view(story_id)

    def remove_character(self, story_id: str, name: str) -> dict:
        story = self._story(story_id)
        self._character(story, name)
        story["characters"] = [c for c in story["characters"] if c["name"] != name]
        self.store.save_story(story)
        return self.view(story_id)

    async def set_character_image(self, story_id: str, name: str, asset_id: str | None, *, add: bool = False) -> dict:
        """The character's picture (add=True keeps the ones they have and adds this one). Minors and the narrator
        get none; a picture of, or made from a photo of, a real identifiable person is refused."""
        story = self._story(story_id)
        character = self._character(story, name)
        if asset_id:
            if is_minor(character["name"], {"appearance": character.get("description")}):
                raise RequestError(f"{name} reads as under 18 and can't be given a picture.")
            if character.get("role") == "narrator":
                raise RequestError(f"{name} is the narrator: a voice, no picture. Make them cast first.")
            #await vet_face_image(self.service, asset_id)
            story = self._story(story_id)  # the check may have taken a while
            character = self._character(story, name)
            ids = character.get("image_asset_ids") or []
            character["image_asset_ids"] = (ids + [asset_id] if add else [asset_id])[-4:] if asset_id not in ids else ids
            if character.get("role") == "extra":
                character["role"] = "cast"
        else:
            character["image_asset_ids"] = []
        self.store.save_story(story)
        return self.view(story_id)

    # ----------------------------------------------------------------- the co-writer

    async def _models(self, story: dict, vision: bool) -> list[str]:
        chain = (story.get("settings") or {}).get("models") or DEFAULT_MODELS
        return [m for m in models_llm.resolve_many("planner", chain, await self.service.model_catalogue(),
                                                   vision=vision) if m]

    async def _ask(self, models: list[str], messages: list[dict], *, json_mode: bool, max_tokens: int,
                   purpose: str) -> tuple[str, str]:
        failures = []
        for model in models:
            try:
                reply, _usage = await self.service.atlas.chat(model, messages, json_mode=json_mode, max_tokens=max_tokens,
                                                              temperature=0.8, max_retries=1, purpose=purpose)
            except AtlasError as exc:
                failures.append(f"{model}: {exc}"[:240])
                continue
            if (reply or "").strip():
                return reply.strip(), model
            failures.append(f"{model}: empty reply")
        raise Unavailable("No co-writer model answered: " + (" | ".join(failures) or "none is configured."))

    def _cast_text(self, story: dict) -> str:
        rows = [f"- {c['name']} ({c.get('role', 'cast')}): {c.get('description') or '(not described yet)'}"
                for c in story.get("characters") or []]
        return "\n".join(rows) or "(no one yet -- characters come from the conversation)"

    def _chapters_text(self, story_id: str) -> str:
        rows = [f"Chapter {c['idx'] + 1}: {c.get('title') or ''} -- {c.get('summary') or ''}"
                for c in self.store.chapters(story_id)]
        return "\n".join(rows) or "(none yet: this is the beginning)"

    def _since_last_chapter(self, story_id: str) -> list[dict]:
        messages = self.store.messages(story_id)
        last = max((i for i, m in enumerate(messages) if m["role"] == "event"), default=-1)
        return [m for m in messages[last + 1:] if m["role"] in ("user", "assistant")][-RECENT_MESSAGES:]

    async def _image_parts(self, asset_ids: list[str]) -> list[dict]:
        parts = []
        for asset_id in asset_ids[:4]:
            try:
                data, mime = await self.service.asset_file(asset_id, 768)
            except Exception:
                continue
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
        return parts

    async def chat(self, story_id: str, text: str, attachments: list[str] | None = None) -> dict:
        """The user's idea and the co-writer's answer, both kept in the story's chat (waits for the answer)."""
        self._story(story_id)
        if not (text or "").strip() and not attachments:
            raise RequestError("Write an idea first.")
        self._claim(story_id, "chat")
        try:
            user, history, text, attachments = self._prepare_chat(story_id, text, attachments)
            answer = await self._answer(story_id, user, history, text, attachments)
        finally:
            self._busy.pop(story_id, None)
        return {"user": user, "assistant": answer}

    def _prepare_chat(self, story_id: str, text: str, attachments: list[str] | None):
        text = (text or "").strip()
        attachments = [a for a in attachments or [] if self.service.store.get_asset(a)]
        history = self._since_last_chapter(story_id)
        user = self.store.add_message(story_id, "user", {"text": text[:6000], "attachments": attachments})
        return user, history, text, attachments

    async def _answer(self, story_id: str, user: dict, history: list[dict], text: str, attachments: list[str]) -> dict:
        story = self._story(story_id)
        system = CHAT_PROMPT.format(title=story["title"], premise=story.get("premise") or "(open)",
                                    style=story.get("style") or "(not set)", cast=self._cast_text(story),
                                    chapters=self._chapters_text(story_id))
        messages = [{"role": "system", "content": system}]
        messages += [{"role": m["role"], "content": m.get("text") or ""} for m in history if m.get("text")]
        content: list[dict] = [{"type": "text", "text": text or "(see the pictures)"}] + await self._image_parts(attachments)
        messages.append({"role": "user", "content": content if len(content) > 1 else content[0]["text"]})
        reply, model = await self._ask(await self._models(story, vision=len(content) > 1), messages, json_mode=False,
                                       max_tokens=CHAT_MAX_TOKENS, purpose="story writer")
        return self.store.add_message(story_id, "assistant", {"text": reply[:8000], "model": model})

    def _claim(self, story_id: str, what: str) -> None:
        """One co-writer call per story at a time: a second press while one runs is refused, not duplicated."""
        busy = self._busy.get(story_id)
        if busy:
            raise RequestError(f"The co-writer is still {'answering' if busy['what'] == 'chat' else 'drafting a chapter'} "
                               f"(started {int(time.time() - busy['since'])} s ago). Wait for it.")
        self._busy[story_id] = {"what": what, "since": time.time()}

    def _background(self, story_id: str, what: str, work) -> None:
        async def run():
            try:
                await work
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # said in the chat, where the user is looking
                log.warning("story %s %s failed: %s", story_id, what, exc)
                self.store.add_message(story_id, "event", {"text": f"{'Reply' if what == 'chat' else 'Draft'} failed: "
                                                                   f"{str(exc)[:400]}", "error": True})
            finally:
                self._busy.pop(story_id, None)
                self._jobs.pop(story_id, None)
        self._jobs[story_id] = asyncio.create_task(run())

    def start_chat(self, story_id: str, text: str, attachments: list[str] | None = None) -> dict:
        """Save the user's idea and answer it in the background: a slow model must not outlast the tunnel's 100 s
        and leave the browser thinking it failed. Poll messages for the reply."""
        self._story(story_id)
        if not (text or "").strip() and not attachments:
            raise RequestError("Write an idea first.")
        self._claim(story_id, "chat")
        try:
            prepared = self._prepare_chat(story_id, text, attachments)
        except Exception:
            self._busy.pop(story_id, None)
            raise
        self._background(story_id, "chat", self._answer(story_id, *prepared))
        return {"user": prepared[0], "pending": True}

    def start_draft(self, story_id: str) -> dict:
        """Draft the next chapter in the background (it can take minutes); the chapter appears when it is done."""
        self._story(story_id)
        if not self._since_last_chapter(story_id):
            raise RequestError("Chat a few ideas first: the chapter is drafted from the conversation since the last one.")
        self._claim(story_id, "draft")
        self._background(story_id, "draft", self.draft_chapter(story_id, claimed=True))
        return {"drafting": True}

    def _next_idx(self, story_id: str) -> int:
        return max((c["idx"] for c in self.store.chapters(story_id)), default=-1) + 1

    def _renumber(self, story_id: str) -> None:
        """Chapters numbered 0, 1, 2... in their order (position, then when they were made), with no gaps or ties."""
        chapters = sorted(self.store.chapters(story_id), key=lambda c: (c["idx"], c.get("created_at") or 0))
        for n, chapter in enumerate(chapters):
            if chapter["idx"] != n:
                chapter["idx"] = n
                self.store.save_chapter(chapter)

    def messages(self, story_id: str) -> list[dict]:
        self._story(story_id)
        return self.store.messages(story_id)

    async def draft_chapter(self, story_id: str, *, claimed: bool = False, plan: str = "") -> dict:
        """The next chapter's scenes, drafted from the conversation since the last chapter -- or, for the autopilot,
        from that chapter's outline (``plan``). Saved as a draft."""
        if not claimed:
            self._claim(story_id, "draft")
        try:
            return await self._draft(story_id, plan)
        finally:
            if not claimed:
                self._busy.pop(story_id, None)

    async def _draft(self, story_id: str, plan: str = "") -> dict:
        story = self._story(story_id)
        conversation = self._since_last_chapter(story_id)
        if not conversation and not plan:
            raise RequestError("Chat a few ideas first: the chapter is drafted from the conversation since the last one.")
        self._renumber(story_id)
        number = self._next_idx(story_id)
        source = (f"PLAN FOR THIS CHAPTER: {plan}" if plan else
                  "\n".join(f"{m['role'].upper()}: {m.get('text') or ''}" for m in conversation))
        text = DRAFT_PROMPT.format(
            number=number + 1, max_scenes=min(MAX_SCENES, 12), title=story["title"],
            premise=story.get("premise") or "(open)", style=story.get("style") or "(not set)",
            cast=self._cast_text(story), chapters=self._chapters_text(story_id), conversation=source[:40000])
        reply, model = await self._ask(await self._models(story, vision=False), [{"role": "user", "content": text}],
                                       json_mode=True, max_tokens=DRAFT_MAX_TOKENS, purpose="story chapter")
        data = parse_json(reply) or {}
        scenes = [clean_scene(s) for s in data.get("scenes") or [] if isinstance(s, dict)][:MAX_SCENES]
        scenes = [s for s in scenes if s["description"] or s["dialogue"]]
        if not scenes:
            raise Unavailable(f"{model} did not write any scenes. Try Draft chapter again.")
        number = self._next_idx(story_id)  # again: another chapter may have arrived while this one was written
        title = _CHAPTER_PREFIX.sub("", _clean(data.get("title"), 160)).strip()
        chapter = {"id": uuid.uuid4().hex[:12], "story_id": story_id, "idx": number,
                   "title": title or f"Chapter {number + 1}",
                   "summary": _clean(data.get("summary"), 1200), "scenes": scenes, "images": [], "status": "draft",
                   "drafted_by": model, "created_at": time.time(), "history": []}
        self.store.save_chapter(chapter)
        self.store.add_message(story_id, "event", {"text": f"Chapter {chapter['idx'] + 1} drafted: {chapter['title']}",
                                                   "chapter_id": chapter["id"]})
        self._learn_cast(story_id, scenes)
        return chapter

    def _learn_cast(self, story_id: str, scenes: list[dict]) -> None:
        """New names in a draft join the cast as extras, so the user can describe them and give them a face."""
        story = self._story(story_id)
        known = {c["name"] for c in story.get("characters") or []}
        for scene in scenes:
            for name in scene["characters"] + [d["speaker"] for d in scene["dialogue"]]:
                if name and name not in known and name.lower() not in ("narration", "narrator"):
                    story.setdefault("characters", []).append({"name": name, "description": "", "role": "extra",
                                                               "image_asset_ids": []})
                    known.add(name)
        self.store.save_story(story)

    def edit_chapter(self, story_id: str, chapter_id: str, *, title=None, summary=None, scenes=None,
                     images: list[dict] | None = None) -> dict:
        chapter = self._chapter(story_id, chapter_id)
        if chapter.get("status") in ACTIVE:
            raise RequestError("This chapter's video is being made: edit it when that has finished.")
        if title is not None:
            chapter["title"] = _clean(title, 160) or chapter["title"]
        if summary is not None:
            chapter["summary"] = _clean(summary, 1200)
        if scenes is not None:
            cleaned = [clean_scene(s) for s in scenes if isinstance(s, dict)][:MAX_SCENES]
            if not [s for s in cleaned if s["description"] or s["dialogue"]]:
                raise RequestError("A chapter needs at least one scene.")
            chapter["scenes"] = cleaned
        if images is not None:
            kept = []
            for item in images[:MAX_PICTURES]:
                asset_id = str((item or {}).get("asset_id") or "")
                if not self.service.store.get_asset(asset_id):
                    raise NotFound(f"No asset {asset_id!r}.")
                kept.append({"asset_id": asset_id, "label": _clean(item.get("label"), 200)})
            chapter["images"] = kept
        self.store.save_chapter(chapter)
        return chapter

    async def add_chapter_image(self, story_id: str, chapter_id: str, asset_id: str, label: str = "") -> dict:
        """A supporting picture for a chapter (a place, an object, a look). Pictures of real people are refused."""
        chapter = self._chapter(story_id, chapter_id)
        await vet_face_image(self.service, asset_id)
        chapter = self._chapter(story_id, chapter_id)
        images = [i for i in chapter.get("images") or [] if i["asset_id"] != asset_id]
        return self.edit_chapter(story_id, chapter_id, images=images + [{"asset_id": asset_id, "label": label}])

    def delete_chapter(self, story_id: str, chapter_id: str) -> dict:
        chapter = self._chapter(story_id, chapter_id)
        if chapter.get("status") in ACTIVE:
            raise RequestError("This chapter's video is being made: wait for it, then delete.")
        self.store.delete_chapter(chapter_id)
        self._renumber(story_id)  # keep the numbering continuous
        return {"deleted": chapter_id, "film_outdated": bool(chapter.get("render_job_id"))}

    def _chapter(self, story_id: str, chapter_id: str) -> dict:
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None or chapter["story_id"] != story_id:
            raise NotFound(f"No chapter {chapter_id!r} in this story.")
        return chapter

    # ----------------------------------------------------------------- video

    def _storyboard_panels(self, chapter: dict) -> list[str]:
        """The chapter's storyboard panels, one per scene, once its board has them all; else none."""
        board_id = chapter.get("storyboard_id")
        if not board_id or self.boards is None:
            return []
        try:
            board = self.boards.board(board_id)
        except NotFound:
            return []
        panels = board.get("panels") or []
        if board.get("status") in ("drawing",) or not panels or not all(panels):
            return []
        return [p for p in panels if self.service.store.get_asset(p)]

    def _references(self, story: dict, chapter: dict) -> tuple[list[ReferenceIn], list[str]]:
        """The chapter's pictures, cast first (one per character in it), then its storyboard panels (each the first
        frame of its scene), then its own images: nine at most."""
        names = {n for scene in chapter["scenes"] for n in scene["characters"] + [d["speaker"] for d in scene["dialogue"]]}
        refs, labels = [], []
        for character in story.get("characters") or []:
            if character["name"] not in names or character.get("role") == "narrator" or not character.get("image_asset_ids"):
                continue
            if is_minor(character["name"], {"appearance": character.get("description")}):
                continue
            asset_id = next((i for i in character["image_asset_ids"] if self.service.store.get_asset(i)), None)
            if asset_id and len(refs) < MAX_PICTURES:
                label = f"{character['name']}" + (f" -- {character['description']}" if character.get("description") else "")
                refs.append(ReferenceIn(asset_id=asset_id, role="picture", label=label[:200]))
                labels.append(label[:200])
        for scene, panel in enumerate(self._storyboard_panels(chapter)[: len(chapter["scenes"])], 1):
            if len(refs) >= MAX_PICTURES:
                break
            label = f"first frame of scene {scene} (storyboard)"
            refs.append(ReferenceIn(asset_id=panel, role="picture", label=label))
            labels.append(label)
        for image in chapter.get("images") or []:
            if len(refs) >= MAX_PICTURES or not self.service.store.get_asset(image["asset_id"]):
                continue
            label = image.get("label") or "a picture for this chapter"
            refs.append(ReferenceIn(asset_id=image["asset_id"], role="picture", label=label[:200]))
            labels.append(label[:200])
        return refs, labels

    def _chapter_cast(self, story: dict, chapter: dict) -> list[dict]:
        """The chapter's people who have a picture: the storyboard's identity references."""
        names = {n for scene in chapter["scenes"] for n in scene["characters"] + [d["speaker"] for d in scene["dialogue"]]}
        return [c for c in story.get("characters") or []
                if c["name"] in names and c.get("role") != "narrator" and c.get("image_asset_ids")
                and not is_minor(c["name"], {"appearance": c.get("description")})
                and any(self.service.store.get_asset(i) for i in c["image_asset_ids"])]

    def make_storyboard(self, story_id: str, chapter_id: str) -> dict:
        """The chapter's scenes drawn as a storyboard (one $0.018 Seedream Flash sheet per 8 scenes), with the cast's
        pictures keeping their faces. Once drawn, Make video opens each scene on its panel."""
        if self.boards is None:
            raise Unavailable("Storyboards aren't available on this server.")
        story = self._story(story_id)
        chapter = self._chapter(story_id, chapter_id)
        if not chapter.get("scenes"):
            raise RequestError("This chapter has no scenes to draw.")
        if chapter.get("status") in ACTIVE:
            raise RequestError("This chapter's video is being made; storyboard it after.")
        cast = self._chapter_cast(story, chapter)[:3]
        if not cast:
            raise RequestError("Give this chapter's characters a picture first: the storyboard keeps their faces from it.")
        identity = [next(i for i in c["image_asset_ids"] if self.service.store.get_asset(i)) for c in cast]
        people = "; ".join(f"{c['name']}" + (f" -- {c['description']}" if c.get("description") else "") for c in cast)
        character = (f"The people, in the order of the reference images: {people}. All are adults."
                     if len(cast) > 1 else f"{people}. An adult.")
        shots = []
        for scene in chapter["scenes"]:
            who = ", ".join(scene.get("characters") or [])
            shot = (f"{scene['setting']}: " if scene.get("setting") else "") + (scene.get("description") or "")
            shots.append((shot + (f" ({who})" if who else "")).strip()[:600] or "a scene")
        settings = {**SETTINGS_DEFAULTS, **(story.get("settings") or {})}
        board = self.boards.create(title=f"{story['title']} - ch {chapter['idx'] + 1}"[:120], character=character,
                                   identity_asset_ids=identity, shots=shots, aspect_ratio=settings["aspect_ratio"],
                                   style=story.get("style") or "", link={"story_id": story_id, "chapter_id": chapter_id})
        chapter = self._set(chapter_id, storyboard_id=board["id"])
        return {"chapter": chapter, "storyboard": board}

    def drop_storyboard(self, story_id: str, chapter_id: str) -> dict:
        """Make video goes back to planning the chapter from its text alone. The board stays on the Storyboard page."""
        self._chapter(story_id, chapter_id)
        return self._set(chapter_id, storyboard_id=None)

    async def identity_pack(self, story_id: str, name: str, *, anatomy: bool = False) -> dict:
        """The character's reference views as one sheet; the front view and the face close-up become their first
        pictures, so every later image and video of them starts from a clean, full-body reference."""
        from .storyboard import identity_pack

        story = self._story(story_id)
        character = self._character(story, name)
        if character.get("role") == "narrator":
            raise RequestError(f"{name} is the narrator: a voice, no picture.")
        if is_minor(character["name"], {"appearance": character.get("description")}):
            raise RequestError(f"{name} reads as under 18 and can't be given pictures.")
        sources = [i for i in character.get("image_asset_ids") or [] if self.service.store.get_asset(i)][:3]
        if not sources:
            raise RequestError(f"Give {name} a picture first: the pack is drawn from it.")
        pack = await identity_pack(self.service, character["name"], character.get("description") or "", sources,
                                   story.get("style") or "", anatomy, collection=f"story/{story['title']}")
        views = {p["view"]: p["id"] for p in pack["panels"]}
        story = self._story(story_id)
        character = self._character(story, name)
        first = [views[v] for v in ("front", "face") if v in views]
        character["image_asset_ids"] = (first + [i for i in character.get("image_asset_ids") or [] if i not in first])[:4]
        self.store.save_story(story)
        return {"pack": pack, "story": self.view(story_id)}

    def make_video(self, story_id: str, chapter_id: str, *, redo: bool = False) -> dict:
        """Plan, render and append the chapter, with no stops. redo keeps the old video in the chapter's history
        and puts the new one in its place in the main film."""
        chapter = self._chapter(story_id, chapter_id)
        if chapter.get("status") in ACTIVE:
            raise RequestError("This chapter's video is already being made.")
        if not chapter.get("scenes"):
            raise RequestError("This chapter has no scenes.")
        has_video = chapter.get("render_job_id") or chapter.get("film_asset_id")
        if has_video and not redo:
            raise RequestError("This chapter already has a video: use Redo video to make a new one.")
        if has_video:
            chapter.setdefault("history", []).append({"plan_job_id": chapter.get("plan_job_id"),
                                                      "render_job_id": chapter.get("render_job_id"),
                                                      "board_id": chapter.get("board_id"),
                                                      "film_asset_id": chapter.get("film_asset_id"),
                                                      "replaced_at": time.time()})
        chapter.update(status="planning", plan_job_id=None, render_job_id=None, board_id=None, film_asset_id=None,
                       error="", started_at=time.time(), plan_retried=False)
        self.store.save_chapter(chapter)
        self._launch(chapter)
        return chapter

    def _launch(self, chapter: dict) -> None:
        self._tasks[chapter["id"]] = asyncio.create_task(self._convert(chapter["story_id"], chapter["id"]))

    async def resume(self, wait_for_gpu: float = 600.0) -> list[str]:
        """Carry on with every chapter -- and every autopilot -- a restart interrupted, once ComfyUI answers."""
        active = [c for c in self.store.active_chapters() if c["id"] not in self._tasks]
        pilots = [s["id"] for s in self.store.stories()
                  if (s.get("autopilot") or {}).get("status") == "running" and s["id"] not in self._autopilots]
        if not active and not pilots:
            return []
        deadline = time.monotonic() + wait_for_gpu
        while time.monotonic() < deadline:
            try:
                if await self.service.comfy.object_info("KSampler"):
                    break
            except Exception:
                pass
            await asyncio.sleep(10)
        for chapter in active:
            self._launch(chapter)
        for story_id in pilots:
            self._autopilots[story_id] = asyncio.create_task(self._autopilot(story_id))
        return [c["id"] for c in active] + pilots

    # ----------------------------------------------------------------- autopilot: premise to film

    def start_autopilot(self, story_id: str, *, chapters: int = 5, portraits: bool = True, videos: bool = True) -> dict:
        """From the premise alone: outline the next chapters and the cast, make the cast's pictures, then for each
        chapter draft its scenes and (with videos) make its video, appended to the main film. Saved as it goes, so a
        restart carries on; stop_autopilot halts it."""
        story = self._story(story_id)
        if not (story.get("premise") or "").strip():
            raise RequestError("Write the story's premise first: the autopilot plans the chapters from it.")
        if story_id in self._autopilots or (story.get("autopilot") or {}).get("status") == "running":
            raise RequestError("The autopilot is already running for this story.")
        story["autopilot"] = {"status": "running", "stage": "outlining", "count": max(1, min(int(chapters or 1), 30)),
                              "portraits": bool(portraits), "videos": bool(videos), "outline": [], "next": 0,
                              "errors": [], "started_at": time.time(), "finished_at": None}
        self.store.save_story(story)
        self._autopilots[story_id] = asyncio.create_task(self._autopilot(story_id))
        return story["autopilot"]

    async def stop_autopilot(self, story_id: str) -> dict:
        """Stop after the step it is on. Chapters, pictures and videos made so far are kept."""
        story = self._story(story_id)
        task = self._autopilots.pop(story_id, None)
        if task is not None:
            task.cancel()
            await asyncio.wait([task], timeout=30)
        story = self._story(story_id)
        if story.get("autopilot"):
            story["autopilot"].update(status="stopped", finished_at=time.time())
            self.store.save_story(story)
        return story.get("autopilot") or {}

    def _pilot(self, story_id: str, **fields) -> dict:
        story = self._story(story_id)
        pilot = story.setdefault("autopilot", {})
        pilot.update(fields)
        self.store.save_story(story)
        return pilot

    async def _autopilot(self, story_id: str) -> None:
        try:
            pilot = self._story(story_id)["autopilot"]
            # 1. the outline and the cast (kept, so a resume does not plan again)
            if not pilot.get("outline"):
                self._pilot(story_id, stage="outlining")
                pilot = self._pilot(story_id, outline=await self._outline(story_id, pilot["count"]))
            # 2. the cast's pictures
            if pilot.get("portraits") and not pilot.get("portraits_done"):
                self._pilot(story_id, stage="character pictures")
                await self._make_portraits(story_id)
                pilot = self._pilot(story_id, portraits_done=True)
            # 3. chapter by chapter: draft, then video (waited for, so the main film grows in order)
            for n in range(pilot.get("next", 0), len(pilot["outline"])):
                item = pilot["outline"][n]
                chapter = self.store.get_chapter(item.get("chapter_id") or "")
                if chapter is None:
                    self._pilot(story_id, stage=f"drafting chapter {n + 1} of {len(pilot['outline'])}")
                    while self._busy.get(story_id):  # a chat or draft the user started: let it finish first
                        await asyncio.sleep(self.poll)
                    chapter = await self.draft_chapter(story_id, plan=f"{item['title']}: {item['beats']}")
                    item["chapter_id"] = chapter["id"]
                    pilot = self._pilot(story_id, outline=pilot["outline"])
                if pilot.get("videos"):
                    chapter = self.store.get_chapter(chapter["id"])
                    if not (chapter.get("render_job_id") or chapter.get("film_asset_id")) and chapter.get("status") not in ACTIVE:
                        self._pilot(story_id, stage=f"video for chapter {n + 1} of {len(pilot['outline'])}")
                        self.make_video(story_id, chapter["id"])
                    while (self.store.get_chapter(chapter["id"]) or {}).get("status") in ACTIVE:
                        await asyncio.sleep(self.poll)
                    finished = self.store.get_chapter(chapter["id"]) or {}
                    if finished.get("status") == "failed":
                        pilot = self._pilot(story_id, errors=(pilot.get("errors") or []) +
                                            [f"Chapter {n + 1}: {finished.get('error') or 'failed'}"[:300]])
                pilot = self._pilot(story_id, next=n + 1)
            self._pilot(story_id, status="done", stage="finished", finished_at=time.time())
        except asyncio.CancelledError:
            raise  # a stop sets "stopped"; a shutdown leaves "running", so resume() carries on
        except Exception as exc:
            log.warning("story autopilot %s failed: %s", story_id, exc)
            pilot = self._story(story_id).get("autopilot") or {}
            self._pilot(story_id, status="failed", stage="failed", finished_at=time.time(),
                        errors=(pilot.get("errors") or []) + [f"{type(exc).__name__}: {exc}"[:400]])
        finally:
            self._autopilots.pop(story_id, None)

    async def _outline(self, story_id: str, count: int) -> list[dict]:
        """count chapters of outline. A model that lists fewer is asked again for the rest (a few times), each time
        shown what is already planned so the arc carries on."""
        story = self._story(story_id)
        models = await self._models(story, vision=False)
        outline: list[dict] = []
        people: list = []
        model = ""
        for _ in range(OUTLINE_ATTEMPTS):
            want = count - len(outline)
            planned = ""
            if outline:
                planned = ("Already planned to follow them (do not repeat these; continue after the last one):\n" +
                           "\n".join(f"- {o['title']}: {o['beats']}" for o in outline) + "\n")
            text = OUTLINE_PROMPT.format(count=want, plural="s" if want != 1 else "", title=story["title"],
                                         premise=story.get("premise") or "", style=story.get("style") or "(not set)",
                                         cast=self._cast_text(story), chapters=self._chapters_text(story_id),
                                         planned=planned)
            try:
                reply, model = await self._ask(models, [{"role": "user", "content": text}], json_mode=True,
                                               max_tokens=DRAFT_MAX_TOKENS, purpose="story outline")
            except Exception:
                if outline:  # keep what is planned rather than lose it
                    break
                raise
            data = parse_json(reply) or {}
            people += [p for p in data.get("characters") or [] if isinstance(p, dict)]
            outline += [{"title": _CHAPTER_PREFIX.sub("", _clean(c.get("title"), 160)), "beats": _clean(c.get("beats"), 2000)}
                        for c in data.get("chapters") or [] if isinstance(c, dict) and _clean(c.get("beats"))][:want]
            if len(outline) >= count:
                break
        if not outline:
            raise Unavailable(f"{model or 'The co-writer'} did not write an outline.")
        if len(outline) < count:
            pilot = self._story(story_id).get("autopilot") or {}
            self._pilot(story_id, errors=(pilot.get("errors") or []) +
                        [f"The outline has {len(outline)} of {count} chapters: the co-writer would not plan more."])
        data = {"characters": people}
        story = self._story(story_id)
        known = {c["name"]: c for c in story.get("characters") or []}
        for person in data.get("characters") or []:
            if not isinstance(person, dict) or not _clean(person.get("name"), 80):
                continue
            name, role = _clean(person["name"], 80), person.get("role") if person.get("role") in ROLES else "cast"
            if name in known:  # the user's own description and role always win
                if not known[name].get("description"):
                    known[name]["description"] = _clean(person.get("description"))
                continue
            known[name] = {"name": name, "description": _clean(person.get("description")), "role": role,
                           "image_asset_ids": []}
            story.setdefault("characters", []).append(known[name])
        self.store.save_story(story)
        return outline

    async def _make_portraits(self, story_id: str) -> None:
        """A picture made on this GPU from the description, for every cast character who has none. A character
        who reads as under 18, or the narrator, gets none."""
        story = self._story(story_id)
        for character in story.get("characters") or []:
            if (character.get("role") != "cast" or character.get("image_asset_ids")
                    or is_minor(character["name"], {"appearance": character.get("description")})):
                continue
            prompt = PORTRAIT_PROMPT.format(name=character["name"], description=character.get("description") or "as described")
            try:
                result = await self._generate(prompt)
            except Exception as exc:  # the chapters still get made, just without this face
                pilot = self._story(story_id).get("autopilot") or {}
                self._pilot(story_id, errors=(pilot.get("errors") or []) + [f"Picture of {character['name']}: {exc}"[:300]])
                continue
            ids = [a["id"] for a in result.get("assets") or []]
            if ids:
                update = getattr(self.service, "update_asset", None)
                if update:
                    update(ids[0], collection=f"story/{story['title']}/cast", add_tags=["story", "cast"])
                fresh = self._story(story_id)
                for c in fresh.get("characters") or []:
                    if c["name"] == character["name"] and not c.get("image_asset_ids"):
                        c["image_asset_ids"] = [ids[0]]
                self.store.save_story(fresh)

    async def _generate(self, prompt: str) -> dict:
        for attempt in range(11):
            try:
                return await self.service.generate_images(prompt, engine="qwen21")
            except RequestError as exc:
                if "busy" not in str(exc).lower() or attempt == 10:
                    raise
            await asyncio.sleep(60 if self.poll >= 1 else 0)
        raise RequestError("ComfyUI stayed busy.")

    def _set(self, chapter_id: str, **fields) -> dict:
        chapter = self.store.get_chapter(chapter_id) or {}
        chapter.update(fields)
        if chapter.get("id"):
            self.store.save_chapter(chapter)
        return chapter

    async def _wait(self, job_id: str, seconds: float) -> dict:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            job = self.service.store.get_job(job_id) or {}
            if job.get("status") in ("done", "failed", "cancelled") or not job:
                return job
            await asyncio.sleep(self.poll)
        return {"status": "timeout"}

    def _uses_engine(self, story: dict, chapter: dict) -> bool:
        """The film engine makes the chapter's video when it is here, the story hasn't asked for the classic
        planner, and the chapter's cast have pictures for it to keep (it needs identity pictures)."""
        if self.boards is None or (story.get("settings") or {}).get("video_engine") == "classic":
            return False
        return bool(chapter.get("board_id")) or bool(self._chapter_cast(story, chapter))

    def _chapter_film_board(self, story: dict, chapter: dict, settings: dict) -> dict:
        """The chapter as a film: a shot per scene (dialogue word for word), the cast as the identity, and the
        storyboard's panels when the chapter has a finished one (else Flash draws them)."""
        cast = self._chapter_cast(story, chapter)[:3]
        identity = [next(i for i in c["image_asset_ids"] if self.service.store.get_asset(i)) for c in cast]
        people = "; ".join(f"{c['name']}" + (f" -- {c['description']}" if c.get("description") else "") for c in cast)
        character = (f"The people, in the order of the reference images: {people}. All are adults."
                     if len(cast) > 1 else f"{people}. An adult.")
        shots, motions, dialogue = [], [], []
        for scene in chapter["scenes"]:
            who = ", ".join(scene.get("characters") or [])
            shot = (f"{scene['setting']}: " if scene.get("setting") else "") + (scene.get("description") or "")
            shots.append((shot + (f" ({who})" if who else "")).strip()[:600] or "a scene")
            motions.append((scene.get("description") or shot)[:600])
            dialogue.append([f'{"Narration (voice-over)" if d["speaker"].lower() == "narration" else d["speaker"]}: "{d["text"]}"'
                             for d in scene.get("dialogue") or []])
        panels = self._storyboard_panels(chapter)
        render = RenderSettings(**{**(settings.get("render") or {}), "aspect_ratio": settings["aspect_ratio"],
                                   "default_seconds": float(settings["segment_seconds"]), "continuity": "off"})
        return self.boards.create(
            title=f"{story['title']} - ch {chapter['idx'] + 1}"[:120], character=character, identity_asset_ids=identity,
            shots=shots, motions=motions, dialogue=dialogue, aspect_ratio=settings["aspect_ratio"],
            style=story.get("style") or "", review="auto", check=not panels,
            panel_asset_ids=panels if len(panels) == len(shots) else None,
            link={"story_id": story["id"], "chapter_id": chapter["id"]},
            then_render={"settings": render.model_dump(), "segment_seconds": float(settings["segment_seconds"]),
                         "mood": "", "music": ""})

    async def _convert_film(self, story_id: str, chapter_id: str) -> None:
        story = self._story(story_id)
        chapter = self._chapter(story_id, chapter_id)
        settings = {**SETTINGS_DEFAULTS, **(story.get("settings") or {})}
        if not chapter.get("board_id"):  # a resume keeps the board it had
            board = self._chapter_film_board(story, chapter, settings)
            chapter = self._set(chapter_id, board_id=board["id"], status="rendering")
        deadline = time.monotonic() + RENDER_WAIT
        while True:
            board = self.boards.board(chapter["board_id"])
            if board["status"] in ("done", "failed"):
                break
            if time.monotonic() > deadline:
                raise RequestError("The chapter's film did not finish in time.")
            await asyncio.sleep(self.poll)
        if board["status"] != "done":
            raise RequestError(f"The chapter's film did not finish: {board.get('error') or 'failed'}")
        self._set(chapter_id, film_asset_id=board["render"]["film_asset_id"], status="appending")
        await self.rebuild_film(story_id)
        self._set(chapter_id, status="done", error="", finished_at=time.time())

    async def _convert(self, story_id: str, chapter_id: str) -> None:
        try:
            story = self._story(story_id)
            chapter = self._chapter(story_id, chapter_id)
            if self._uses_engine(story, chapter) and not chapter.get("render_job_id"):
                await self._convert_film(story_id, chapter_id)
                return
            settings = {**SETTINGS_DEFAULTS, **(story.get("settings") or {})}
            # 1. the plan (reused on a resume)
            if not chapter.get("render_job_id"):
                plan_id = chapter.get("plan_job_id")
                if not plan_id or not self.service.store.get_job(plan_id):
                    refs, labels = self._references(story, chapter)
                    earlier = [c for c in self.store.chapters(story_id) if c["idx"] < chapter["idx"]]
                    cap = min(max_segment_seconds(), 15.0)
                    brief, seconds = chapter_brief(story, chapter, earlier, labels, cap)
                    try:
                        request = PlanRequest(story=brief, segment_count=min(len(chapter["scenes"]), 40),
                                              segment_seconds=float(settings["segment_seconds"]),
                                              aspect_ratio=settings["aspect_ratio"],
                                              model=settings.get("plan_models") or PLAN_MODELS,
                                              references=refs)
                    except ValueError as exc:
                        raise RequestError(str(exc).splitlines()[0]) from None
                    plan_id = (await self.service.create_plan(request))["id"]
                    chapter = self._set(chapter_id, plan_job_id=plan_id, planned_seconds=seconds, status="planning")
                plan = await self._wait(plan_id, PLAN_WAIT)
                if plan.get("status") in ("failed", "cancelled") and not chapter.get("plan_retried"):
                    self._set(chapter_id, plan_retried=True)
                    await self.service.retry(plan_id)
                    plan = await self._wait(plan_id, PLAN_WAIT)
                if plan.get("status") != "done":
                    raise RequestError(f"The plan did not finish: {plan.get('error') or plan.get('status')}")
                # 2. the render
                render = RenderSettings(**{**(settings.get("render") or {}), "aspect_ratio": settings["aspect_ratio"],
                                           "default_seconds": float(settings["segment_seconds"])})
                job = await self.service.create_video(VideoRequest(plan_job_id=plan_id, settings=render))
                chapter = self._set(chapter_id, render_job_id=job["id"], status="rendering")
            done = await self._wait(chapter["render_job_id"], RENDER_WAIT)
            if done.get("status") != "done":
                raise RequestError(f"The render did not finish: {done.get('error') or done.get('status')}")
            # 3. the main film
            self._set(chapter_id, status="appending")
            await self.rebuild_film(story_id)
            self._set(chapter_id, status="done", error="", finished_at=time.time())
        except asyncio.CancelledError:
            raise  # a shutdown: the status stays active and resume() carries on after the restart
        except Exception as exc:
            log.warning("story chapter %s failed: %s", chapter_id, exc)
            self._set(chapter_id, status="failed", error=str(exc)[:600], finished_at=time.time())
        finally:
            self._tasks.pop(chapter_id, None)

    async def rebuild_film(self, story_id: str) -> dict | None:
        """The main film: every chapter with a finished video, in order, joined (stream copy when they match),
        saved in Media under story/<title>, copied to Drive, with subtitles from the chapters' lines."""
        lock = self._film_locks.setdefault(story_id, asyncio.Lock())
        async with lock:
            story = self._story(story_id)
            chapters = [c for c in self.store.chapters(story_id)
                        if (c.get("film_asset_id") and self.service.store.get_asset(c["film_asset_id"]))
                        or (c.get("render_job_id")
                            and (self.service.store.get_job(c["render_job_id"]) or {}).get("status") == "done")]
            if not chapters:
                return None
            if not shutil.which("ffmpeg"):
                raise Unavailable("ffmpeg is not installed on the server; run the notebook's ffmpeg cell.")
            work = tempfile.mkdtemp(prefix="hawk_story_")
            try:
                files = [await self._chapter_file(c, os.path.join(work, f"ch_{n:03d}.mp4"))
                         for n, c in enumerate(chapters, 1)]
                out = os.path.join(work, "story.mp4")
                if len(files) == 1:
                    shutil.copyfile(files[0], out)
                    copied = True
                else:
                    copied = await asyncio.to_thread(concat_videos, files, out)
                slug = re.sub(r"[^0-9A-Za-z]+", "-", story["title"]).strip("-").lower() or "story"
                name = f"{slug}_ch1-{len(chapters)}.mp4"
                with open(out, "rb") as handle:
                    asset = await self.service.add_asset(
                        name, handle, "video/mp4", os.path.getsize(out), collection=f"story/{story['title']}",
                        tags=["story", "film"], local_path=out, dedupe=False,
                        source={"type": "generated", "engine": "ffmpeg", "generator": "story film", "story_id": story_id,
                                "jobs": [c.get("render_job_id") or c.get("film_asset_id") for c in chapters]})
                drive = await film_to_drive(self.service, out, name)
                srt = ""
                if (story.get("settings") or {}).get("subtitles", "file") != "off":
                    srt = await self._srt(chapters, files)
                    if srt:
                        path = self._srt_path(story_id)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w", encoding="utf-8") as handle:
                            handle.write(srt)
                        if drive:
                            await film_to_drive(self.service, path, name.rsplit(".", 1)[0] + ".srt")
            finally:
                shutil.rmtree(work, ignore_errors=True)
            story = self._story(story_id)
            story["film"] = {"asset_id": asset["id"], "chapters": [c["id"] for c in chapters], "drive": drive,
                             "stream_copy": copied, "subtitles": bool(srt), "updated_at": time.time()}
            self.store.save_story(story)
            return story["film"]

    async def _chapter_file(self, chapter: dict, target: str) -> str:
        """A chapter's finished video on disk: its film from the engine, or its render."""
        if chapter.get("film_asset_id"):
            asset = self.service.store.get_asset(chapter["film_asset_id"])
            finder = getattr(self.service, "local_asset_path", None)
            local = finder(asset) if asset and finder else None
            if local:
                return local
            with open(target, "wb") as handle:
                handle.write(await self.service.asset_bytes(asset))
            return target
        return await rendered_file(self.service, chapter["render_job_id"], target)

    async def _srt(self, chapters: list[dict], files: list[str]) -> str:
        parts = []
        for chapter, path in zip(chapters, files):
            job = self.service.store.get_job(chapter.get("render_job_id") or "") or {}
            plan = self.service.store.get_job(chapter.get("plan_job_id") or "") or {}
            lines = [scene["dialogue"] for scene in chapter["scenes"]]
            seconds: list[float] = []
            try:
                script = parse_script(job.get("script") or plan.get("script") or "")
                default = float(((job.get("request") or {}).get("settings") or {}).get("default_seconds") or 10)
                seconds = [float(seg.duration or default) for seg in script.segments]
            except (ScriptError, TypeError, ValueError):
                pass
            if len(seconds) == len(lines):
                segments = [{"seconds": s, "lines": l} for s, l in zip(seconds, lines)]
            else:
                segments = [{"seconds": sum(seconds) or 10.0, "lines": [x for l in lines for x in l]}]
            parts.append({"duration": await asyncio.to_thread(media_seconds, path), "segments": segments})
        return build_srt(parts)

    def _srt_path(self, story_id: str) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(self.service.settings.db_path)), "story_subtitles",
                            f"{story_id}.srt")

    def subtitles(self, story_id: str) -> str:
        self._story(story_id)
        path = self._srt_path(story_id)
        if not os.path.isfile(path):
            raise NotFound("This story's film has no subtitles yet.")
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
