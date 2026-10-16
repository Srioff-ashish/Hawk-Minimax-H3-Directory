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

#: The co-writer, best first. Gemma 4 is on OpenRouter only, so on Atlas the chain starts at Kimi.
DEFAULT_MODELS = "google/gemma-4-31b-it, moonshotai/kimi-k2.6, deepseek-ai/deepseek-v4.1-flash"
MODEL_CHOICES = ("google/gemma-4-31b-it", "moonshotai/kimi-k2.6", "deepseek-ai/deepseek-v4.1-flash")
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
SETTINGS_DEFAULTS = {"models": DEFAULT_MODELS, "aspect_ratio": "16:9", "segment_seconds": 10.0, "render": {},
                     "subtitles": "file"}

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
        self.poll = POLL

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
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

    def view(self, story_id: str) -> dict:
        story = self._story(story_id)
        characters = []
        for c in story.get("characters") or []:
            characters.append(dict(c, minor=is_minor(c["name"], {"appearance": c.get("description")}),
                                   images=[v for v in (self._image_view(i) for i in c.get("image_asset_ids") or []) if v]))
        chapters = []
        for c in self.store.chapters(story_id):
            chapters.append(dict(c, video_url=self._job_video(c.get("render_job_id")),
                                 image_views=[dict(i, **(self._image_view(i["asset_id"]) or {}))
                                              for i in c.get("images") or []]))
        film = dict(story["film"]) if story.get("film") else None
        if film:
            shown = self._image_view(film.get("asset_id"))  # any asset: the view only needs its urls
            film.update(file_url=(shown or {}).get("file_url"))
        return dict(story, characters=characters, chapters=chapters, film=film,
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
            await vet_face_image(self.service, asset_id)
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
        """The user's idea and the co-writer's answer, both kept in the story's chat."""
        story = self._story(story_id)
        text = (text or "").strip()
        if not text and not attachments:
            raise RequestError("Write an idea first.")
        attachments = [a for a in attachments or [] if self.service.store.get_asset(a)]
        history = self._since_last_chapter(story_id)
        user = self.store.add_message(story_id, "user", {"text": text[:6000], "attachments": attachments})
        system = CHAT_PROMPT.format(title=story["title"], premise=story.get("premise") or "(open)",
                                    style=story.get("style") or "(not set)", cast=self._cast_text(story),
                                    chapters=self._chapters_text(story_id))
        messages = [{"role": "system", "content": system}]
        messages += [{"role": m["role"], "content": m.get("text") or ""} for m in history if m.get("text")]
        content: list[dict] = [{"type": "text", "text": text or "(see the pictures)"}] + await self._image_parts(attachments)
        messages.append({"role": "user", "content": content if len(content) > 1 else content[0]["text"]})
        reply, model = await self._ask(await self._models(story, vision=len(content) > 1), messages, json_mode=False,
                                       max_tokens=CHAT_MAX_TOKENS, purpose="story writer")
        answer = self.store.add_message(story_id, "assistant", {"text": reply[:8000], "model": model})
        return {"user": user, "assistant": answer}

    def messages(self, story_id: str) -> list[dict]:
        self._story(story_id)
        return self.store.messages(story_id)

    async def draft_chapter(self, story_id: str) -> dict:
        """The conversation since the last chapter, turned into the next chapter's scenes. Saved as a draft."""
        story = self._story(story_id)
        conversation = self._since_last_chapter(story_id)
        if not conversation:
            raise RequestError("Chat a few ideas first: the chapter is drafted from the conversation since the last one.")
        chapters = self.store.chapters(story_id)
        text = DRAFT_PROMPT.format(
            number=len(chapters) + 1, max_scenes=min(MAX_SCENES, 12), title=story["title"],
            premise=story.get("premise") or "(open)", style=story.get("style") or "(not set)",
            cast=self._cast_text(story), chapters=self._chapters_text(story_id),
            conversation="\n".join(f"{m['role'].upper()}: {m.get('text') or ''}" for m in conversation)[:40000])
        reply, model = await self._ask(await self._models(story, vision=False), [{"role": "user", "content": text}],
                                       json_mode=True, max_tokens=DRAFT_MAX_TOKENS, purpose="story chapter")
        data = parse_json(reply) or {}
        scenes = [clean_scene(s) for s in data.get("scenes") or [] if isinstance(s, dict)][:MAX_SCENES]
        scenes = [s for s in scenes if s["description"] or s["dialogue"]]
        if not scenes:
            raise Unavailable(f"{model} did not write any scenes. Try Draft chapter again.")
        chapter = {"id": uuid.uuid4().hex[:12], "story_id": story_id, "idx": len(chapters),
                   "title": _clean(data.get("title"), 160) or f"Chapter {len(chapters) + 1}",
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
        for n, other in enumerate(self.store.chapters(story_id)):  # keep the numbering continuous
            if other["idx"] != n:
                other["idx"] = n
                self.store.save_chapter(other)
        return {"deleted": chapter_id, "film_outdated": bool(chapter.get("render_job_id"))}

    def _chapter(self, story_id: str, chapter_id: str) -> dict:
        chapter = self.store.get_chapter(chapter_id)
        if chapter is None or chapter["story_id"] != story_id:
            raise NotFound(f"No chapter {chapter_id!r} in this story.")
        return chapter

    # ----------------------------------------------------------------- video

    def _references(self, story: dict, chapter: dict) -> tuple[list[ReferenceIn], list[str]]:
        """The chapter's pictures, cast first (one per character in it), then its own images: nine at most."""
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
        for image in chapter.get("images") or []:
            if len(refs) >= MAX_PICTURES or not self.service.store.get_asset(image["asset_id"]):
                continue
            label = image.get("label") or "a picture for this chapter"
            refs.append(ReferenceIn(asset_id=image["asset_id"], role="picture", label=label[:200]))
            labels.append(label[:200])
        return refs, labels

    def make_video(self, story_id: str, chapter_id: str, *, redo: bool = False) -> dict:
        """Plan, render and append the chapter, with no stops. redo keeps the old video in the chapter's history
        and puts the new one in its place in the main film."""
        chapter = self._chapter(story_id, chapter_id)
        if chapter.get("status") in ACTIVE:
            raise RequestError("This chapter's video is already being made.")
        if not chapter.get("scenes"):
            raise RequestError("This chapter has no scenes.")
        if chapter.get("render_job_id") and not redo:
            raise RequestError("This chapter already has a video: use Redo video to make a new one.")
        if chapter.get("render_job_id"):
            chapter.setdefault("history", []).append({"plan_job_id": chapter.get("plan_job_id"),
                                                      "render_job_id": chapter.get("render_job_id"),
                                                      "replaced_at": time.time()})
        chapter.update(status="planning", plan_job_id=None, render_job_id=None, error="", started_at=time.time(),
                       plan_retried=False)
        self.store.save_chapter(chapter)
        self._launch(chapter)
        return chapter

    def _launch(self, chapter: dict) -> None:
        self._tasks[chapter["id"]] = asyncio.create_task(self._convert(chapter["story_id"], chapter["id"]))

    async def resume(self, wait_for_gpu: float = 600.0) -> list[str]:
        """Carry on with every chapter a restart interrupted, once ComfyUI answers."""
        active = [c for c in self.store.active_chapters() if c["id"] not in self._tasks]
        if not active:
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
        return [c["id"] for c in active]

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

    async def _convert(self, story_id: str, chapter_id: str) -> None:
        try:
            story = self._story(story_id)
            chapter = self._chapter(story_id, chapter_id)
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
                                              aspect_ratio=settings["aspect_ratio"], model=settings["models"],
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
                        if c.get("render_job_id")
                        and (self.service.store.get_job(c["render_job_id"]) or {}).get("status") == "done"]
            if not chapters:
                return None
            if not shutil.which("ffmpeg"):
                raise Unavailable("ffmpeg is not installed on the server; run the notebook's ffmpeg cell.")
            work = tempfile.mkdtemp(prefix="hawk_story_")
            try:
                files = [await rendered_file(self.service, c["render_job_id"], os.path.join(work, f"ch_{n:03d}.mp4"))
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
                                "jobs": [c["render_job_id"] for c in chapters]})
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

    async def _srt(self, chapters: list[dict], files: list[str]) -> str:
        parts = []
        for chapter, path in zip(chapters, files):
            job = self.service.store.get_job(chapter["render_job_id"]) or {}
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
