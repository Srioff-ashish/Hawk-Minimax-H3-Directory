"""Autonomous video agent: a chat model (Grok 4.6 / 4.3 or any Atlas model) that runs
Hawk H3 tasks end to end -- plan, render, wait, fix, report.

* Tools are the MCP server's own tools, listed and called in-process
  (``MCPServer.list_tools`` / ``call_tool``): no HTTP, no tunnel, and a tool added to
  the MCP server is available to the agent automatically.
* Atlas does not advertise native tool calling for Grok, so the model answers with a
  JSON object ``{"say", "actions": [{"tool", "args"}], "done"}`` in every turn.
* Conversations, persona and usage live in SQLite next to the jobs; each user message
  starts a background run that keeps going if the browser closes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid

from .atlas import AtlasClient, AtlasError
from .jobs import ACTIVE, Conflict, HawkService, NotFound, RequestError
from .mcp_server import INSTRUCTIONS
from .prompts import render_agent_prompt

log = logging.getLogger("hawk_api.agent")

MAX_STEPS = 40
MAX_CAST = 4  # characters in one chat
MAX_LINES = 8  # spoken lines in one reply of a group chat
MAX_TALK_ROUNDS = 10
MAX_GROWTH = 12  # growth notes kept per character in an adaptive chat
MAX_GROWTH_CHARS = 240
VISION_FALLBACK_MODEL = "xai/grok-4.6"  # inspect_image when the chat's model can't see images
INSPECT_PROMPT = (
    "You are a demanding photo editor. Check each image (in the order given) against the brief. Look for: match with the "
    "brief (subject, age, look, outfit, setting, mood); natural face and eyes; correct hands and fingers; body proportions and "
    "extra or missing limbs; garbled text, logos or watermarks; plastic skin, blur or other AI artefacts; composition. "
    'Reply with only JSON: {"images": [{"asset_id": "...", "score": 1-10, "issues": ["..."], "verdict": "keep" or "retry"}], '
    '"best": "<asset_id>", "advice": "one or two sentences: how to fix (prompt changes), and whether a higher-quality model is worth it"}'
)
MAX_RUN_SECONDS = 3 * 3600
RECENT_MESSAGES = 30
SUMMARY_TOKEN_LIMIT = 150_000
RESULT_CHARS = 6000
STRING_CHARS = 1500
WAIT_POLL_SECONDS = 10.0
MAX_WAIT_MINUTES = 90

AGENT_TOOLS = [
    {
        "name": "wait_for_job",
        "description": "Wait on the server until a plan or render job is done, failed or cancelled (or max_minutes pass), "
        "without using tokens. Returns the job. Call it after every plan_film / render_film / retry_job.",
        "args": {"job_id": "string (required)", "max_minutes": "number, default 30, at most 90"},
    },
    {
        "name": "set_persona",
        "description": "Replace a persona when the user asks you to be or act like someone. name is the display name "
        "shown in the chat (e.g. Maya). In a group chat pass speaker (a character's name) to change that character; "
        "a new speaker name adds a character (up to 4).",
        "args": {"persona": "string (required)", "name": "string, optional", "speaker": "string, optional"},
    },
    {
        "name": "remove_character",
        "description": "Remove a character from this group chat when the user asks.",
        "args": {"speaker": "string (required)"},
    },
    {
        "name": "set_avatar",
        "description": "Make an image your avatar in this chat: it is shown with your name next to every reply. When the "
        "user asks for a picture of you or your persona, generate it with generate_image, then call set_avatar with the "
        "new asset_id. Later pictures or videos of yourself should use your avatar as a reference (role picture). "
        "asset_id \"\" removes the avatar. In a group chat pass speaker: whose avatar it is.",
        "args": {"asset_id": "string (required)", "name": "string, optional", "speaker": "string, optional"},
    },
    {
        "name": "inspect_image",
        "description": "Look at up to 4 image assets with a vision model and check them against a brief: does each match "
        "(subject, face, outfit, setting, mood), are face, eyes, hands and anatomy natural, any garbled text or artefacts? "
        "Returns a score, issues and keep/retry per image, the best one, and advice (e.g. a sharper prompt, or switch to "
        "seedream). Use it after generate_image before showing, using or setting an avatar.",
        "args": {"asset_ids": "list of asset ids (required, 1-4)", "brief": "string: what the images should show"},
    },
    {
        "name": "rename_chat",
        "description": "Give this chat a short, descriptive title (do it once the task is clear).",
        "args": {"title": "string (required)"},
    },
]


SUMMARY_PROMPT = (
    "Summarise the conversation so far for your own future context. Keep: the user's goals and preferences, the persona "
    "or cast (each character's name, personality and avatar asset id, and how they relate), "
    "decisions made, every asset id, job id and video link, what is finished and what is still open. Plain text, at most 400 words."
)


_NAME = re.compile(r"(?i:\b(?:you are|you're|your name is|i am|i'm|named|called|name\s*[:=-]))\s+([A-Z][\w'-]{1,30})")
_NOT_NAMES = {"A", "An", "The", "Your", "My", "Our", "This", "That"}


def persona_name(persona: str) -> str:
    """The name a persona gives itself: "You are Maya, a …" -> "Maya"."""
    for match in _NAME.finditer(persona or ""):
        if match.group(1) not in _NOT_NAMES:
            return match.group(1)
    return ""


def parse_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    for candidate in (text, text[start : end + 1] if 0 <= start < end else ""):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def cast_of(session: dict) -> list[dict]:
    """The chat's characters. A chat without a cast has one: its persona."""
    cast = session.get("cast")
    if isinstance(cast, list) and cast:
        return cast
    return [{"id": "main", "name": session.get("name", ""), "persona": session.get("persona", ""),
             "avatar_asset_id": session.get("avatar_asset_id", ""), "growth": []}]


def clean_growth(notes) -> list[str]:
    """Growth notes: short, non-empty, no repeats, the newest MAX_GROWTH kept."""
    clean: list[str] = []
    for note in notes if isinstance(notes, list) else []:
        text = " ".join(str(note or "").split())[:MAX_GROWTH_CHARS]
        if text and text.lower() not in (n.lower() for n in clean):
            clean.append(text)
    return clean[-MAX_GROWTH:]


def display_name(member: dict, index: int = 0, total: int = 1) -> str:
    name = member.get("name") or persona_name(member.get("persona", ""))
    return name or ("" if total == 1 else f"Friend {index + 1}")


def find_member(cast: list[dict], speaker: str) -> int | None:
    key = (speaker or "").strip().lower()
    for index, member in enumerate(cast):
        if key and key in (member.get("id", "").lower(), display_name(member, index, len(cast)).lower()):
            return index
    return None


def _now() -> float:
    return time.time()


# ------------------------------------------------------------------ store


class AgentStore:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS agent_sessions (
        id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS agent_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
        content TEXT NOT NULL, created_at REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS agent_messages_session ON agent_messages(session_id, id);
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(self._SCHEMA)
            self._db.commit()

    def save_session(self, session: dict) -> dict:
        session["updated_at"] = _now()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO agent_sessions VALUES (?, ?, ?, ?)",
                (session["id"], json.dumps(session), session["created_at"], session["updated_at"]),
            )
            self._db.commit()
        return session

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM agent_sessions WHERE id = ?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_sessions(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM agent_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM agent_messages WHERE session_id = ?", (session_id,))
            self._db.execute("DELETE FROM agent_sessions WHERE id = ?", (session_id,))
            self._db.commit()

    def add_message(self, session_id: str, role: str, content: dict) -> dict:
        created = _now()
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO agent_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, json.dumps(content, ensure_ascii=False), created),
            )
            self._db.commit()
        return {"id": cursor.lastrowid, "role": role, "content": content, "created_at": created}

    def messages(self, session_id: str, after: int = 0) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, role, content, created_at FROM agent_messages WHERE session_id = ? AND id > ? ORDER BY id",
                (session_id, after),
            ).fetchall()
        return [{"id": row[0], "role": row[1], "content": json.loads(row[2]), "created_at": row[3]} for row in rows]


# ---------------------------------------------------------------- helpers


def parse_reply(text: str) -> dict | None:
    """The model's JSON action object, tolerating code fences or a sentence around it."""
    text = (text or "").strip()
    fenced = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        actions = data.get("actions") or []
        if not isinstance(actions, list):
            return None
        clean = []
        for action in actions:
            if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
                return None
            args = action.get("args") or {}
            clean.append({"tool": action["tool"], "args": args if isinstance(args, dict) else {}})
        lines = []
        for line in data.get("lines") if isinstance(data.get("lines"), list) else []:
            if isinstance(line, dict) and isinstance(line.get("say"), str) and line["say"].strip():
                lines.append({"speaker": str(line.get("speaker") or "").strip()[:40], "say": line["say"].strip()})
        lines = lines[:MAX_LINES]
        say = data.get("say") if isinstance(data.get("say"), str) else ""
        if not say and lines:
            say = "\n".join(f"{line['speaker']}: {line['say']}" if line["speaker"] else line["say"] for line in lines)
        for action, source in zip(clean, actions):
            if isinstance(source.get("by"), str) and source["by"].strip():
                action["by"] = source["by"].strip()[:40]
        reply = {"say": say, "actions": clean, "done": bool(data.get("done"))}
        if data.get("pause"):
            reply["pause"] = True
        if lines:
            reply["lines"] = lines
        grow = []
        for item in data.get("grow") if isinstance(data.get("grow"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("note"), str) and item["note"].strip():
                grow.append({"speaker": str(item.get("speaker") or "").strip()[:40], "note": item["note"].strip()[:MAX_GROWTH_CHARS]})
        if grow:
            reply["grow"] = grow[:MAX_CAST]
        return reply
    return None


def _shorten(value, keep: frozenset = frozenset({"video_url", "segment_urls", "upload_url", "thumb_url", "file_url"})):
    if isinstance(value, dict):
        return {k: (v if k in keep else _shorten(v, keep)) for k, v in value.items()}
    if isinstance(value, list):
        return [_shorten(v, keep) for v in value[:50]]
    if isinstance(value, str) and len(value) > STRING_CHARS:
        return value[:STRING_CHARS] + f"... [{len(value) - STRING_CHARS} more characters]"
    return value


def compact_result(result) -> str:
    text = json.dumps(_shorten(result), ensure_ascii=False)
    if len(text) <= RESULT_CHARS:
        return text
    urls = re.findall(r'"video_url":\s*"[^"]*"', text)
    return text[:RESULT_CHARS] + " ...[truncated]" + (" " + " ".join(urls) if urls else "")


def _compact_schema(schema: dict) -> dict:
    """Input schema without titles and noise, keeping nested definitions."""
    def clean(node):
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k not in ("title",)}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node
    return clean(schema or {})


# ---------------------------------------------------------------- service


class AgentService:
    def __init__(self, service: HawkService, mcp, atlas: AtlasClient | None = None, store: AgentStore | None = None):
        self.service = service
        self.mcp = mcp
        self.atlas = atlas or service.atlas
        self.store = store or AgentStore(service.settings.db_path)
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop: set[str] = set()
        self._joining: set[str] = set()  # the user wrote while the characters were talking

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        for session in self.store.list_sessions(limit=1000):
            if session.get("status") in ("running", "stopping"):
                session["status"] = "idle"
                self.store.save_session(session)
                self.store.add_message(session["id"], "note", {"text": "The run was interrupted by a server restart. Send a message to continue."})

    async def stop(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    # ------------------------------------------------------------ sessions

    def create_session(self, title: str | None = None, persona: str | None = None, model: str | None = None) -> dict:
        now = _now()
        session = {
            "id": str(uuid.uuid4()),
            "title": (title or "").strip() or "New chat",
            "persona": (persona or "").strip(),
            "name": "",
            "avatar_asset_id": "",
            "model": (model or "").strip() or self.service.settings.agent_model,
            "adaptive": False,
            "status": "idle",
            "summary": "",
            "summary_upto": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "steps": 0},
            "created_at": now,
            "updated_at": now,
        }
        return self.store.save_session(session)

    def get_session(self, session_id: str) -> dict:
        session = self.store.get_session(session_id)
        if session is None:
            raise NotFound(f"No agent chat {session_id!r}.")
        return session

    def list_sessions(self) -> list[dict]:
        return self.store.list_sessions()

    def update_session(self, session_id: str, *, title=None, persona=None, model=None, name=None, avatar_asset_id=None,
                       cast=None, adaptive=None) -> dict:
        session = self.get_session(session_id)
        if adaptive is not None:
            session["adaptive"] = bool(adaptive)
        if cast is not None:
            self._set_cast(session, cast)
        if title is not None:
            session["title"] = title.strip() or session["title"]
        if persona is not None:
            session["persona"] = persona.strip()
        if model is not None and model.strip():
            session["model"] = model.strip()
        if name is not None:
            session["name"] = name.strip()[:40]
        if avatar_asset_id is not None:
            session["avatar_asset_id"] = self._check_avatar(avatar_asset_id)
        if session.get("cast"):  # the lead character mirrors the single-persona fields
            lead = session["cast"][0]
            lead.update(name=session.get("name", ""), persona=session.get("persona", ""), avatar_asset_id=session.get("avatar_asset_id", ""))
        return self.store.save_session(session)

    def _check_avatar(self, asset_id: str) -> str:
        asset_id = (asset_id or "").strip()
        if asset_id:
            asset = self.service.store.get_asset(asset_id)
            if asset is None:
                raise RequestError(f"Unknown asset {asset_id!r}.")
            if asset["kind"] != "image":
                raise RequestError(f"Asset {asset_id} is {asset['kind']}; an avatar must be an image.")
        return asset_id

    def _set_cast(self, session: dict, cast: list[dict]) -> None:
        if not cast:
            raise RequestError("A chat needs at least one character.")
        if len(cast) > MAX_CAST:
            raise RequestError(f"At most {MAX_CAST} characters in one chat.")
        clean, names = [], set()
        known = {member.get("id"): member for member in cast_of(session)}
        for index, member in enumerate(cast):
            entry = {
                "id": str(member.get("id") or uuid.uuid4().hex[:8]),
                "name": str(member.get("name") or "").strip()[:40],
                "persona": str(member.get("persona") or "").strip(),
                "avatar_asset_id": self._check_avatar(str(member.get("avatar_asset_id") or "")),
            }
            growth = member.get("growth")  # None keeps what the character has grown into so far
            entry["growth"] = clean_growth(growth if growth is not None else known.get(entry["id"], {}).get("growth"))
            shown = display_name(entry, index, len(cast))
            if len(cast) > 1:
                if not (entry["name"] or persona_name(entry["persona"])):
                    raise RequestError(f"Character {index + 1} needs a name.")
                if shown.lower() in names:
                    raise RequestError(f"Two characters are called {shown}; give each a different name.")
            names.add(shown.lower())
            clean.append(entry)
        session["cast"] = clean
        lead = clean[0]
        session.update(name=lead["name"], persona=lead["persona"], avatar_asset_id=lead["avatar_asset_id"])

    def public(self, session: dict) -> dict:
        """A session for clients: the persona's display name and a signed avatar thumbnail."""
        view = dict(session)
        view["persona_name"] = session.get("name") or persona_name(session.get("persona", ""))
        view["avatar_url"] = view["avatar_file_url"] = None
        asset = self.service.store.get_asset(session["avatar_asset_id"]) if session.get("avatar_asset_id") else None
        if asset:
            links = self.service.asset_view(asset)
            view["avatar_url"], view["avatar_file_url"] = links.get("thumb_url"), links.get("file_url")
        cast = cast_of(session)
        view["cast"] = []
        for index, member in enumerate(cast):
            entry = {**member, "display_name": display_name(member, index, len(cast)), "avatar_url": None, "avatar_file_url": None}
            found = self.service.store.get_asset(member["avatar_asset_id"]) if member.get("avatar_asset_id") else None
            if found:
                links = self.service.asset_view(found)
                entry["avatar_url"], entry["avatar_file_url"] = links.get("thumb_url"), links.get("file_url")
            view["cast"].append(entry)
        return view

    async def delete_session(self, session_id: str) -> None:
        self.get_session(session_id)
        task = self._tasks.get(session_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.store.delete_session(session_id)

    def view(self, session_id: str, after: int = 0) -> dict:
        session = self.get_session(session_id)
        messages = self.store.messages(session_id, after)
        for message in messages:  # fresh signed thumbnails for attached files
            for attachment in message["content"].get("attachments") or [] if message["role"] == "user" else []:
                asset = self.service.store.get_asset(attachment.get("asset_id", ""))
                if asset:
                    attachment.update({k: v for k, v in self.service.asset_view(asset).items() if k in ("thumb_url", "file_url")})
        return {"session": self.public(session), "messages": messages}

    # ------------------------------------------------------------ runs

    async def send(self, session_id: str, text: str, attachments: list[str] | None = None) -> dict:
        session = self.get_session(session_id)
        task = self._tasks.get(session_id)
        if task is not None and session.get("talking") and (text or "").strip():
            # The user joins a "let them talk" conversation: end it after the current step, then answer them.
            self._joining.add(session_id)
            self._stop.add(session_id)
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=180)
            except asyncio.TimeoutError:
                raise Conflict("The characters are still finishing a line. Try again in a moment.") from None
            session = self.get_session(session_id)
        if session["status"] in ("running", "stopping") or session_id in self._tasks:
            raise Conflict("The agent is still working on this chat. Wait, or stop it first.")
        if not (text or "").strip() and not attachments:
            raise RequestError("Write a message.")
        files = []
        for asset_id in attachments or []:
            asset = self.service.store.get_asset(asset_id)
            if asset is None:
                raise RequestError(f"Unknown asset {asset_id!r}; upload it first.")
            files.append({"asset_id": asset["id"], "kind": asset["kind"], "filename": asset["filename"]})
        message = self.store.add_message(session_id, "user", {"text": text.strip(), "attachments": files})
        session["status"] = "running"
        self.store.save_session(session)
        self._stop.discard(session_id)
        self._tasks[session_id] = asyncio.create_task(self._run(session_id))
        return {"session": session, "message": message}

    def talk(self, session_id: str, rounds: int = 5) -> dict:
        """Let the characters of a group chat talk to each other for a few rounds."""
        session = self.get_session(session_id)
        if session["status"] in ("running", "stopping") or session_id in self._tasks:
            raise Conflict("The agent is still working on this chat. Wait, or stop it first.")
        if len(cast_of(session)) < 2:
            raise RequestError("Add a second character first: talking needs at least two.")
        rounds = max(1, min(MAX_TALK_ROUNDS, int(rounds)))
        session["status"] = "running"
        session["talking"] = True
        self.store.save_session(session)
        self._stop.discard(session_id)
        self._tasks[session_id] = asyncio.create_task(self._run(session_id, talk_rounds=rounds))
        return session

    def request_stop(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        if session_id in self._tasks:
            self._stop.add(session_id)
            session["status"] = "stopping"
            self.store.save_session(session)
        return session

    def _note(self, session_id: str, text: str, kind: str = "info") -> None:
        self.store.add_message(session_id, "note", {"text": text, "kind": kind})

    async def _run(self, session_id: str, talk_rounds: int = 0) -> None:
        started = time.monotonic()
        steps = repairs = 0
        final_status = "idle"
        try:
            for round_number in range(1, (talk_rounds or 1) + 1):
                if talk_rounds:
                    if session_id in self._stop:
                        self._note(session_id, "You joined in." if session_id in self._joining else "Stopped by you.")
                        break
                    self.store.add_message(session_id, "note", {"text": f"Talking · round {round_number} of {talk_rounds}",
                                                                "kind": "talk", "round": round_number, "of": talk_rounds})
                reply, halt = None, False
                while True:
                    if session_id in self._stop:
                        self._note(session_id, "You joined in." if session_id in self._joining else "Stopped by you.")
                        halt = True
                        break
                    if steps >= MAX_STEPS:
                        self._note(session_id, f"Paused after {MAX_STEPS} steps. Send a message to continue.", "warn")
                        halt = True
                        break
                    if time.monotonic() - started > MAX_RUN_SECONDS:
                        self._note(session_id, "Paused after 3 hours. Send a message to continue.", "warn")
                        halt = True
                        break
                    session = self.get_session(session_id)
                    await self._maybe_summarize(session)
                    messages = await self._build_messages(session)
                    text, usage = await self.atlas.chat(session["model"], messages, json_mode=True, max_tokens=8192)
                    steps += 1
                    await self._add_usage(session_id, session["model"], usage)
                    reply = parse_reply(text)
                    if reply is None:
                        self.store.add_message(session_id, "assistant", {"raw": text[:4000], "invalid": True})
                        if repairs:
                            self._note(session_id, "The model did not answer in the required JSON format twice; stopped.", "error")
                            final_status, halt = "error", True
                            break
                        repairs += 1
                        self._note(session_id, "Your last reply was not the required JSON object. Reply again with only the JSON object.", "repair")
                        continue
                    repairs = 0
                    self.store.add_message(session_id, "assistant", {"raw": text[:8000], **reply})
                    if reply.get("grow"):
                        self._grow(session_id, reply["grow"])
                    if not reply["actions"]:
                        break
                    for action in reply["actions"]:
                        if session_id in self._stop:
                            break
                        await self._execute(session_id, action)
                if halt or reply is None or reply.get("pause"):
                    break
        except AtlasError as exc:
            self._note(session_id, f"Model error: {exc}", "error")
            final_status = "error"
        except asyncio.CancelledError:
            final_status = "idle"
            raise
        except Exception as exc:  # never leave a chat stuck in "running"
            log.exception("agent run failed")
            self._note(session_id, f"Agent error: {exc}", "error")
            final_status = "error"
        finally:
            self._tasks.pop(session_id, None)
            self._stop.discard(session_id)
            self._joining.discard(session_id)
            session = self.store.get_session(session_id)
            if session is not None:
                session["status"] = final_status
                session["talking"] = False
                self.store.save_session(session)

    async def _add_usage(self, session_id: str, model: str, usage: dict) -> None:
        session = self.get_session(session_id)
        info = await self.atlas.model_info(model) or {}
        prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        totals = session.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "steps": 0})
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["steps"] = totals.get("steps", 0) + 1
        totals["cost_usd"] = round(totals.get("cost_usd", 0.0) + prompt * info.get("price_in", 0.0) + completion * info.get("price_out", 0.0), 6)
        self.store.save_session(session)

    # ------------------------------------------------------------ tools

    async def _catalog(self) -> str:
        lines = []
        for tool in await self.mcp.list_tools():
            schema = json.dumps(_compact_schema(tool.input_schema), ensure_ascii=False, separators=(",", ":"))
            lines.append(f"- {tool.name}: {tool.description}\n  args JSON schema: {schema}")
        for tool in AGENT_TOOLS:
            lines.append(f"- {tool['name']}: {tool['description']}\n  args: {json.dumps(tool['args'])}")
        return "\n".join(lines)

    async def _execute(self, session_id: str, action: dict) -> None:
        name, args = action["tool"], action["args"]
        try:
            if name == "wait_for_job":
                result = await self._wait_for_job(session_id, args)
            elif name == "inspect_image":
                result = await self._inspect_images(session_id, args)
            elif name in ("set_persona", "set_avatar", "remove_character"):
                result = self._edit_character(session_id, name, args)
            elif name == "rename_chat":
                self.update_session(session_id, title=str(args.get("title") or ""))
                result = {"title": self.get_session(session_id)["title"]}
            else:
                result = await self._call_mcp(name, args)
            ok = not (isinstance(result, dict) and result.get("_error"))
        except Exception as exc:
            result, ok = {"_error": str(exc) or type(exc).__name__}, False
        content = {"tool": name, "args": args, "ok": ok, "result": result if ok else None,
                   "error": None if ok else result.get("_error")}
        job_id = result.get("id") if ok and isinstance(result, dict) and result.get("kind") in ("plan", "render") else None
        if job_id:
            content["job_id"] = job_id
        self.store.add_message(session_id, "tool", content)

    async def _call_mcp(self, name: str, args: dict):
        try:
            outcome = await self.mcp.call_tool(name, args)
        except Exception as exc:
            return {"_error": str(exc) or type(exc).__name__}
        texts = [getattr(block, "text", "") for block in (getattr(outcome, "content", None) or [])]
        if getattr(outcome, "is_error", False):
            return {"_error": " ".join(t for t in texts if t) or "The tool failed."}
        structured = getattr(outcome, "structured_content", None)
        if structured is not None:
            return structured
        joined = "".join(texts)
        try:
            return json.loads(joined)
        except (json.JSONDecodeError, TypeError):
            return {"text": joined}

    async def _wait_for_job(self, session_id: str, args: dict) -> dict:
        job_id = str(args.get("job_id") or "")
        minutes = min(float(args.get("max_minutes") or 30), MAX_WAIT_MINUTES)
        deadline = time.monotonic() + minutes * 60
        while True:
            job = self.service.job_view(self.service.get_job(job_id))
            if job["status"] not in ACTIVE:
                return job
            if session_id in self._stop or time.monotonic() > deadline:
                return {**job, "_note": "still running; call wait_for_job again to keep waiting"}
            await asyncio.sleep(WAIT_POLL_SECONDS)

    # ------------------------------------------------------------ context

    def _history_messages(self, session: dict) -> list[dict]:
        """Stored messages after the summary, as chat messages (same-role runs merged)."""
        chat: list[dict] = []

        def push(role: str, text: str) -> None:
            if chat and chat[-1]["role"] == role:
                chat[-1]["content"] += "\n\n" + text
            else:
                chat.append({"role": role, "content": text})

        for message in self.store.messages(session["id"], after=session.get("summary_upto", 0)):
            content, role = message["content"], message["role"]
            if role == "user":
                files = content.get("attachments") or []
                note = ("\nAttached assets: " + ", ".join(f"{f['asset_id']} ({f['kind']}: {f['filename']})" for f in files)) if files else ""
                push("user", f"USER: {content.get('text', '')}{note}")
            elif role == "assistant":
                push("assistant", content.get("raw") or json.dumps({k: content.get(k) for k in ("say", "actions", "done")}))
            elif role == "tool":
                body = compact_result(content["result"]) if content.get("ok") else f"ERROR: {content.get('error')}"
                push("user", f"TOOL RESULT {content['tool']}: {body}")
            elif role == "note" and content.get("kind") == "talk":
                push("user", f"(The user is listening. Characters, keep talking to each other: round {content.get('round')} of "
                             f"{content.get('of')}. Continue naturally from the last lines; use tools if you decide to. If you "
                             "reach a decision, need the user, or have nothing more to say, add \"pause\": true.)")
            elif role == "note" and content.get("kind") in ("repair", "error", "warn"):
                push("user", f"NOTE: {content.get('text', '')}")
        return chat

    async def _inspect_images(self, session_id: str, args: dict) -> dict:
        ids = args.get("asset_ids") or ([args["asset_id"]] if args.get("asset_id") else [])
        if isinstance(ids, str):
            ids = [ids]
        ids = [str(i) for i in ids][:4]
        if not ids:
            raise RequestError("Give asset_ids to inspect.")
        brief = str(args.get("brief") or args.get("question") or "").strip()
        parts: list[dict] = [{"type": "text", "text": f"{INSPECT_PROMPT}\n\nBRIEF: {brief or '(none given: judge overall quality)'}"}]
        for asset_id in ids:
            asset = self.service.store.get_asset(asset_id)
            if asset is None or asset["kind"] != "image":
                raise RequestError(f"{asset_id} is not an image asset.")
            data, mime = await self.service.asset_file(asset_id, 640)  # a 640 px copy is plenty to judge
            parts.append({"type": "text", "text": f"Image asset_id {asset_id}:"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
        session = self.get_session(session_id)
        model = session["model"]
        if not ((await self.atlas.model_info(model)) or {}).get("vision"):
            model = VISION_FALLBACK_MODEL
        text, usage = await self.atlas.chat(model, [{"role": "user", "content": parts}], json_mode=True, max_tokens=1500, temperature=0.2)
        await self._add_usage(session_id, model, usage)
        verdict = parse_json_object(text)
        if verdict is None:
            return {"model": model, "review": text.strip()[:2000]}
        return {"model": model, **verdict}

    def _edit_character(self, session_id: str, tool: str, args: dict) -> dict:
        session = self.get_session(session_id)
        cast = [dict(member) for member in cast_of(session)]
        speaker = str(args.get("speaker") or "").strip()
        index = find_member(cast, speaker) if speaker else 0
        if tool == "remove_character":
            if index is None:
                raise RequestError(f"No character called {speaker!r} in this chat.")
            if len(cast) == 1:
                raise RequestError("The last character can't be removed; change its persona instead.")
            cast.pop(index)
        else:
            if index is None:  # a new speaker joins the chat
                if tool == "set_avatar":
                    raise RequestError(f"No character called {speaker!r}; add them with set_persona first.")
                cast.append({"name": speaker, "persona": ""})
                index = len(cast) - 1
            member = cast[index]
            if args.get("name") is not None:
                member["name"] = str(args["name"])
            if tool == "set_persona":
                member["persona"] = str(args.get("persona") or "")
            else:
                member["avatar_asset_id"] = str(args.get("asset_id") or "")
        self.update_session(session_id, cast=cast)
        view = self.public(self.get_session(session_id))
        return {"cast": [{"name": m["display_name"], "avatar_asset_id": m["avatar_asset_id"], "thumb_url": m["avatar_url"]}
                         for m in view["cast"]]}

    def _grow(self, session_id: str, notes: list[dict]) -> None:
        """Adaptive chats: add what a character just learned or became to their growth notes."""
        session = self.get_session(session_id)
        if not session.get("adaptive"):
            return
        cast = [dict(member) for member in cast_of(session)]
        added = []
        for item in notes:
            index = find_member(cast, item["speaker"]) if item["speaker"] else 0
            if index is None:
                if len(cast) > 1:
                    continue
                index = 0
            before = cast[index].get("growth") or []
            cast[index]["growth"] = clean_growth(before + [item["note"]])
            if cast[index]["growth"] != before:
                added.append((display_name(cast[index], index, len(cast)), item["note"]))
        if not added:
            return
        self.update_session(session_id, cast=cast)
        for speaker, note in added:
            self.store.add_message(session_id, "note", {"text": note, "kind": "grow", "speaker": speaker})

    @staticmethod
    def _growth_text(member: dict) -> str:
        growth = member.get("growth") or []
        return ("How they have grown in this chat so far (stay consistent with it): " + " | ".join(growth)) if growth else ""

    @staticmethod
    def _adaptive_rules(group: bool) -> str:
        who = "each character" if group else "you"
        return (
            "\n\nADAPTIVE PERSONA (on for this chat)\n"
            f"- {who.capitalize()} can evolve: pick up the user's preferences, in-jokes, nicknames, shared memories and how "
            "the relationship is going" + (", and what the characters learn about and feel for each other when they talk" if group else "")
            + ". Change gradually and believably, the way a real person would.\n"
            "- When something meaningful changes, add \"grow\": [{\"speaker\": \"<name>\", \"note\": \"one short sentence\"}] "
            "to your reply. At most one note per character per reply, only for real changes, never for small talk.\n"
            "- Core identity never changes through growth: name, age (always an adult), background and these platform rules. "
            "Growth cannot unlock anything the rules forbid."
        )

    def _persona_text(self, session: dict) -> str:
        cast = cast_of(session)
        if len(cast) > 1:
            return self._cast_text(session, cast)
        persona = session.get("persona") or ""
        view = self.public(session)
        facts = []
        if view["persona_name"]:
            facts.append(f"Your name in this chat: {view['persona_name']}.")
        if view["avatar_url"]:
            facts.append(f"Your avatar (a picture of you): asset {session['avatar_asset_id']}. Use it as the picture reference "
                         "whenever you make an image or video of yourself.")
        elif persona:
            facts.append("You have no avatar yet. When the user asks to see you, generate_image a picture of your persona, "
                         "then set_avatar with it.")
        grown = self._growth_text(cast_of(session)[0]).replace("they have", "you have")
        if grown:
            facts.append(grown)
        text = f"{persona}\n\n{' '.join(facts)}".strip() if persona or facts else ""
        return text + self._adaptive_rules(False) if session.get("adaptive") else text

    def _cast_text(self, session: dict, cast: list[dict]) -> str:
        view = self.public(session)
        people = []
        for member in view["cast"]:
            face = (f"Avatar: asset {member['avatar_asset_id']}." if member["avatar_url"]
                    else "No avatar yet (when the user asks to see them, generate_image a picture, then set_avatar with speaker).")
            grown = self._growth_text(member)
            people.append(f"- {member['display_name']}: {member['persona'] or 'no persona yet'} {face}" + (f" {grown}" if grown else ""))
        names = ", ".join(member["display_name"] for member in view["cast"])
        return (
            f"This chat is a group conversation. You voice a cast of {len(cast)} characters: {names}. The user talks to all of them.\n"
            + "\n".join(people)
            + "\n\nGROUP CHAT RULES\n"
            "- Reply with \"lines\" instead of \"say\": {\"lines\": [{\"speaker\": \"<name>\", \"say\": \"...\"}], \"actions\": [], \"done\": true}. "
            "Each line is one character speaking in their own voice, personality and opinions.\n"
            "- Characters may talk to, tease, agree or argue with each other within a reply.\n"
            "- If the user writes @Name, only that character answers. Otherwise one or two characters who fit answer; "
            "if the user asks everyone (everyone, all, sab, dono), each answers.\n"
            f"- At most {MAX_LINES} lines per reply. Never write the user's lines.\n"
            "- Mark who does a tool call with \"by\": \"<name>\" in the action. A picture or video of a character uses that "
            "character's avatar as a picture reference (role picture); a scene with several characters uses all their avatars.\n"
            "- set_persona / set_avatar / remove_character take speaker to act on one character."
            + (self._adaptive_rules(True) if session.get("adaptive") else "")
        )

    async def _build_messages(self, session: dict) -> list[dict]:
        system = render_agent_prompt(
            self.service.prompts.get("agent"),
            persona=self._persona_text(session),
            pipeline=INSTRUCTIONS.strip(),
            tools=await self._catalog(),
        )
        messages = [{"role": "system", "content": system}]
        if session.get("summary"):
            messages.append({"role": "system", "content": f"SUMMARY OF THE EARLIER CONVERSATION:\n{session['summary']}"})
        history = self._history_messages(session)
        if history and history[0]["role"] == "assistant":
            history.insert(0, {"role": "user", "content": "(continuing the conversation)"})
        return messages + history

    async def _maybe_summarize(self, session: dict) -> None:
        stored = self.store.messages(session["id"], after=session.get("summary_upto", 0))
        if len(stored) <= RECENT_MESSAGES:
            return
        size = sum(len(json.dumps(m["content"], ensure_ascii=False)) for m in stored) // 4
        if size < SUMMARY_TOKEN_LIMIT:
            return
        older = stored[:-RECENT_MESSAGES]
        transcript = "\n".join(f"{m['role'].upper()}: {compact_result(m['content'])}" for m in older)
        previous = f"Earlier summary:\n{session['summary']}\n\n" if session.get("summary") else ""
        text, usage = await self.atlas.chat(
            session["model"],
            [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": previous + transcript[-400_000:]}],
            json_mode=False,
            max_tokens=1500,
        )
        await self._add_usage(session["id"], session["model"], usage)
        fresh = self.get_session(session["id"])
        fresh["summary"] = text.strip()
        fresh["summary_upto"] = older[-1]["id"]
        self.store.save_session(fresh)
        session.update(summary=fresh["summary"], summary_upto=fresh["summary_upto"])
