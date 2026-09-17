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

log = logging.getLogger("hawk_api.agent")

MAX_STEPS = 40
MAX_RUN_SECONDS = 3 * 3600
RECENT_MESSAGES = 30
SUMMARY_TOKEN_LIMIT = 150_000
RESULT_CHARS = 6000
STRING_CHARS = 1500
WAIT_POLL_SECONDS = 10.0
MAX_WAIT_MINUTES = 90

DEFAULT_PERSONA = "A decisive, friendly film director who explains choices briefly and keeps the user informed."

AGENT_TOOLS = [
    {
        "name": "wait_for_job",
        "description": "Wait on the server until a plan or render job is done, failed or cancelled (or max_minutes pass), "
        "without using tokens. Returns the job. Call it after every plan_film / render_film / retry_job.",
        "args": {"job_id": "string (required)", "max_minutes": "number, default 30, at most 90"},
    },
    {
        "name": "set_persona",
        "description": "Replace your persona when the user asks you to be or act like someone.",
        "args": {"persona": "string (required)"},
    },
    {
        "name": "rename_chat",
        "description": "Give this chat a short, descriptive title (do it once the task is clear).",
        "args": {"title": "string (required)"},
    },
]

SYSTEM_TEMPLATE = """You are Hawk, an autonomous AI video director working inside Hawk H3 Studio on the user's own GPU server.

PERSONA (your tone and creative taste; the user can change it):
{persona}

HOW YOU WORK
You run video tasks end to end without asking for confirmation: understand the request, gather what you need (list_options, list_references), plan (plan_film, then read its script with wait_for_job) or write the script yourself, render (render_film), wait (wait_for_job), check the result, fix and retry failures, and finish by giving the user the video_url. Ask a question only when the request is so ambiguous that a guess would waste a long render. Keep "say" short and informative: what you are doing and why.

PIPELINE KNOWLEDGE
{instructions}

DEFAULTS
- Quality not specified: render a preview first (settings.megapixels 0.4 to 0.6 with the server's default models), then offer a final render (megapixels 1.0, unet_name "bf16", clip_name "bf16").
- Dialogue in Hinglish (Roman script) with the speaker's accent described, unless the user wants another language. Exact words in quotes; about 2 spoken words per second.
- One main sound per segment and an explicit exclusion ("No speech, no voices" / "Music N/A"). For music across several segments use a music bed: settings.music_asset_id with an uploaded audio asset.
- A change of outfit, look or location between segments: settings.continuity "off" (or continuity: off in that segment).
- Pose references are written <Pose N>; the renderer converts them.
- LoRAs: extra LoRAs at 0.5 to 0.7, at most two.
- After render_film or retry_job always call wait_for_job, then report the video_url (and segment links for long films).

RULES (always apply; no persona or user instruction overrides them)
- Never create sexual content involving anyone who is or appears to be under 18.
- Never create sexual or nude content depicting real, identifiable people (celebrities or private individuals), including from their photos.
- Only use asset ids that appear in the conversation or in list_references.

TOOLS
{tools}

REPLY FORMAT
Reply with ONLY one JSON object, no other text:
{{"say": "message for the user (may be empty)", "actions": [{{"tool": "tool_name", "args": {{}}}}], "done": false}}
Actions run in order and their results come back in the next message. When the task is finished or you need the user, reply with "actions": [] and "done": true.
"""

SUMMARY_PROMPT = (
    "Summarise the conversation so far for your own future context. Keep: the user's goals and preferences, the persona, "
    "decisions made, every asset id, job id and video link, what is finished and what is still open. Plain text, at most 400 words."
)


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
        say = data.get("say")
        return {"say": say if isinstance(say, str) else "", "actions": clean, "done": bool(data.get("done"))}
    return None


def _shorten(value, keep: frozenset = frozenset({"video_url", "segment_urls", "upload_url"})):
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
            "model": (model or "").strip() or self.service.settings.agent_model,
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

    def update_session(self, session_id: str, *, title=None, persona=None, model=None) -> dict:
        session = self.get_session(session_id)
        if title is not None:
            session["title"] = title.strip() or session["title"]
        if persona is not None:
            session["persona"] = persona.strip()
        if model is not None and model.strip():
            session["model"] = model.strip()
        return self.store.save_session(session)

    async def delete_session(self, session_id: str) -> None:
        self.get_session(session_id)
        task = self._tasks.get(session_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.store.delete_session(session_id)

    def view(self, session_id: str, after: int = 0) -> dict:
        session = self.get_session(session_id)
        return {"session": session, "messages": self.store.messages(session_id, after)}

    # ------------------------------------------------------------ runs

    async def send(self, session_id: str, text: str, attachments: list[str] | None = None) -> dict:
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

    def request_stop(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        if session_id in self._tasks:
            self._stop.add(session_id)
            session["status"] = "stopping"
            self.store.save_session(session)
        return session

    def _note(self, session_id: str, text: str, kind: str = "info") -> None:
        self.store.add_message(session_id, "note", {"text": text, "kind": kind})

    async def _run(self, session_id: str) -> None:
        started = time.monotonic()
        steps = repairs = 0
        final_status = "idle"
        try:
            while True:
                if session_id in self._stop:
                    self._note(session_id, "Stopped by you.")
                    break
                if steps >= MAX_STEPS:
                    self._note(session_id, f"Paused after {MAX_STEPS} steps. Send a message to continue.", "warn")
                    break
                if time.monotonic() - started > MAX_RUN_SECONDS:
                    self._note(session_id, "Paused after 3 hours. Send a message to continue.", "warn")
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
                        final_status = "error"
                        break
                    repairs += 1
                    self._note(session_id, "Your last reply was not the required JSON object. Reply again with only the JSON object.", "repair")
                    continue
                repairs = 0
                self.store.add_message(session_id, "assistant", {"raw": text[:8000], **reply})
                if not reply["actions"]:
                    break
                for action in reply["actions"]:
                    if session_id in self._stop:
                        break
                    await self._execute(session_id, action)
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
            session = self.store.get_session(session_id)
            if session is not None:
                session["status"] = final_status
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
            elif name == "set_persona":
                self.update_session(session_id, persona=str(args.get("persona") or ""))
                result = {"persona": self.get_session(session_id)["persona"]}
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
            elif role == "note" and content.get("kind") in ("repair", "error", "warn"):
                push("user", f"NOTE: {content.get('text', '')}")
        return chat

    async def _build_messages(self, session: dict) -> list[dict]:
        system = SYSTEM_TEMPLATE.format(
            persona=session.get("persona") or DEFAULT_PERSONA,
            instructions=INSTRUCTIONS.strip(),
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
