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
import math
import os
import random
import re
import sqlite3
import threading
import time
import uuid

from . import cast_talk
from . import image_engines
from . import models_llm
from .atlas import AtlasClient, AtlasError
from .cast_talk import USER_KEY, clean_feelings, visible_to
from .jobs import ACTIVE, IMAGE_PRICES, Conflict, HawkService, NotFound, RequestError
from .mcp_server import INSTRUCTIONS
from .prompts import PLATFORM_RULES, render_agent_prompt

log = logging.getLogger("hawk_api.agent")

MAX_STEPS = 40
#: Args a subject's name could plausibly appear in. Matching the whole args blob instead let a lora
#: filename or a negative prompt decide who owned a picture.
SUBJECT_FIELDS = ("prompt", "script", "story", "text", "brief")
USER_OWNER_WORDS = ("user", "you", "me", "myself")
MAX_CAST = 4  # characters in one chat
MAX_LINES = 8  # spoken lines in one reply of a group chat
MAX_TALK_ROUNDS = 10
MAX_TALK_MAKES = 5  # things any one character may have made during one "let them talk" (default 2)
MAX_TALK_MAKES_TOTAL = 8  # and across the whole cast, so a big room can't order a dozen renders at once
MAKE_STEPS = 8  # model steps the director gets to make one thing
CHARACTER_KEEP_MESSAGES = 12  # a character's memory keeps this many recent messages word for word
TURN_MAX_TOKENS = 4000  # room for reasoning models; a turn itself is short
MAX_GROWTH = 12  # hard cap on growth notes per character in an adaptive chat
MERGE_GROWTH_AT = 8  # at this many notes, the model merges them into a few denser ones
MERGED_GROWTH = 4
MAX_GROWTH_CHARS = 240
# inspect_image tries the chat's model (when it sees images), then these, until one returns a usable verdict: some models
# refuse or garble reviews of certain images (adult content in particular).
#: Superseded by the "vision" role in models_llm.ROLES, which resolves against the provider's own catalogue.
#: Kept as the last-resort chain so a pod with no /models access behaves as it always did.
VISION_FALLBACK_MODELS = ("xai/grok-4.6", "xai/grok-4.3")
# Reasoning models (DeepSeek V4.x) think before the verdict; too small a budget ends the reply empty (finish_reason=length).
INSPECT_MAX_TOKENS = 8000
PASS_SCORE = 6  # an inspected batch whose best image scores below this counts as a failed take
# After a failed take, engine "auto" moves one rung up for the rest of the run (until the user writes again).
# The rungs come from HawkService.image_ladder, so the agent and the gateway can never disagree about the order.
INSPECT_PROMPT = (
    "You are a demanding photo editor. Check each image (in the order given) against the brief. Look for: match with the "
    "brief (subject, age, look, outfit, setting, mood); natural face and eyes; correct hands and fingers; body proportions and "
    "extra or missing limbs; garbled text, logos or watermarks; plastic skin, blur or other AI artefacts; composition. "
    'Reply with only JSON: {"images": [{"asset_id": "...", "score": 1-10, "issues": ["..."], "verdict": "keep" or "retry"}], '
    '"best": "<asset_id>", "advice": "one or two sentences: how to fix (prompt changes), and whether a higher-quality model is worth it"}'
)
MAX_RUN_SECONDS = 3 * 3600
MANUAL_KEEP_MESSAGES = 4  # the Compact button keeps this many messages word for word
SUMMARY_MAX_TOKENS = 6000  # room for reasoning models to think before writing the summary
# Tools whose argument schema is large are listed by description only until the chat uses them (describe_tool shows it).
BIG_SCHEMA_TOKENS = 250

#: relation -> (the family it supersedes within, what it means, the node kinds its object may have).
#:
#: Deliberately four, not nine. The first cut had located_in, status_of, knows_about, wants and agreed as well,
#: and on the pod the model put nearly everything in them: "located_in :: Asset c7eab94fc3ea, portrait.png",
#: "wants :: User asked Riya for tomorrow's plan". The sentences were fine; the relation was usually wrong,
#: which matters because the relation picks the family and the family is what makes a new fact supersede an old
#: one. Wrong families mean contradictions silently never resolve.
#:
#: What survived is what a graph is uniquely good at: who someone is to someone else, how they feel, and who
#: made or owns what. The situational facts those five were catching -- plans, wants, what is still open --
#: are what the prose summary already carries, in order, which a set of triples cannot do anyway.
RELATIONS = {
    "relation_to": ("relation", "who someone is to someone else", ("person",)),
    "feels_toward": ("feeling", "how someone feels about someone", ("person",)),
    "owns": ("owns", "what belongs to someone", ("asset", "thing")),
    "made": ("made", "what someone created", ("asset", "thing")),
}
#: Facts taken from one compaction. A cheap summary model will invent them given room, so the cap is in the
#: prompt and here.
MAX_FACTS = 20
#: Live facts kept per chat. Nothing pruned them before, and a three-message chat on the pod produced 32, which
#: extrapolates badly over a year. The lowest-scoring live edges are marked stale rather than deleted, so the
#: panel can still show what was dropped and a revival is one update.
MAX_LIVE_FACTS = 400
#: A node's identity shapes: how many characters an asset id has, and the kinds a fact may name.
NODE_KINDS = ("person", "place", "thing", "event", "asset", "topic")
ALWAYS_FULL_TOOLS = frozenset({"generate_image"})
# What survives in tool results from earlier turns: ids, links, outcomes and verdicts; raw details are dropped.
BRIEF_KEYS = frozenset({
    "id", "asset_id", "job_id", "kind", "status", "engine", "model", "error", "video_url", "score", "verdict", "best",
    "advice", "issues", "title", "name", "note", "engine_note", "cost_usd", "filename", "message", "progress",
    "duration", "segments", "assets", "images", "cast",
})
BRIEF_STRING_CHARS = 300
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
        "description": "Set who a character is. To create several characters, call it once per character with speaker "
        "set to that character's name (a new speaker adds a character, up to 4; in a fresh chat the first one replaces "
        "the unnamed default). Pass speaker to change an existing character too. Without speaker (and a name that "
        "isn't in the cast) it changes the lead, i.e. you. name is the display name shown in the chat (e.g. Maya).",
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
        "name": "describe_tool",
        "description": "Show the full argument schema of a tool that the catalogue lists without one (large tools such as "
        "render_film and plan_film). Call it once before the first use of such a tool in a chat.",
        "args": {"name": "string (required): the tool's name"},
    },
    {
        "name": "rename_chat",
        "description": "Give this chat a short, descriptive title (do it once the task is clear).",
        "args": {"title": "string (required)"},
    },
]


SUMMARY_PROMPT = (
    "Summarise the conversation so far for your own future context. Keep: the user's goals and preferences, the persona "
    "or cast (each character's name, personality and avatar asset id, and how they relate), how each character feels about "
    "the user and about each other and the moments that caused it, "
    "decisions made, every asset id, job id and video link, what is finished and what is still open. Plain text, at most 400 words."
)

#: Appended to both summarisers. They already make exactly one call each, so the facts ride along in it and the
#: graph costs no extra request -- which was the requirement, not an optimisation.
FACTS_TAIL = (
    "\n\nReply with ONE JSON object and nothing else:\n"
    '{{"summary": "<the summary above, as plain text>", "facts": [{{"s": "<who>", "r": "<a relation below>", '
    '"o": "<who or what>", "v": <-1..1>, "f": "<one short sentence>"}}]}}\n'
    "Use ONLY these relations, for what they say and nothing else:\n{relations}\n"
    "These are for lasting things about people: who they are to each other, how they feel, what they made or "
    "own. They are NOT for what happened, what was decided, what someone wants next or where the shoot is -- "
    "all of that belongs in the summary above, where the order it happened in can be kept. If a fact does not "
    "fit one of the relations exactly, leave it out rather than forcing it into the nearest one. Name people "
    'exactly as the conversation does and use "user" for the user. At most {limit} facts, and an empty list is '
    "a perfectly good answer -- most compactions should produce only a few."
)
#: One example each, because the names alone were being read as "the nearest verb".
RELATION_EXAMPLES = {
    "relation_to": '{"s": "Riya", "r": "relation_to", "o": "Tiya", "f": "Riya is Tiya\'s older sister."}',
    "feels_toward": '{"s": "Riya", "r": "feels_toward", "o": "user", "v": -0.6, "f": "Riya finds the user\'s '
                    'last-minute changes exhausting."}',
    "owns": '{"s": "Tiya", "r": "owns", "o": "a1b2c3d4e5f6", "f": "Asset a1b2c3d4e5f6 is Tiya\'s avatar."}',
    "made": '{"s": "Riya", "r": "made", "o": "a1b2c3d4e5f6", "f": "Riya made the rooftop portrait a1b2c3d4e5f6."}',
}


def facts_tail() -> str:
    lines = [f"- {name}: {meaning}\n    e.g. {RELATION_EXAMPLES[name]}"
             for name, (_, meaning, _) in RELATIONS.items()]
    return FACTS_TAIL.format(relations="\n".join(lines), limit=MAX_FACTS)


def read_summary_and_facts(text: str) -> tuple[str, list[dict]]:
    """A summariser's reply as (prose summary, facts).

    The property that makes asking for facts safe: a reply that is not the JSON object we asked for is treated
    as the prose summary and yields no facts, which is byte-for-byte what happened before facts existed. A model
    that ignores the format gives the old product, never a broken one -- so a cheap or unfamiliar summary model
    can never cost a chat its memory.
    """
    raw = (text or "").strip()
    data = parse_json_object(raw)
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str) or not data["summary"].strip():
        return raw, []
    facts = []
    for item in data.get("facts") if isinstance(data.get("facts"), list) else []:
        if isinstance(item, dict) and str(item.get("r") or "").strip().lower() in RELATIONS:
            facts.append(item)  # the relation is real; whether the object suits it is checked in _write_fact
    return data["summary"].strip(), facts[:MAX_FACTS]

MERGE_GROWTH_PROMPT = (
    "You keep a character's memory of how they have changed during one chat. Merge the growth notes below into at most "
    "{limit} short notes (one sentence each). Keep every lasting change in feelings, attitudes towards the user and the "
    "other characters, habits and preferences; the strongest emotional shifts matter most, even if they are old. Merge "
    "repeats, keep the latest state when notes conflict, drop pure events or plans unless they explain a feeling. "
    "Write in the same language style as the notes. Never add anything new. "
    'Reply with only JSON: {{"notes": ["...", "..."]}}'
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
    """A character by id or name. Exact first; then a unique match on a first name or the start of a name, so
    "Sonia" finds "Sonia Mausi" (models shorten names)."""
    key = " ".join((speaker or "").strip().lstrip("@").lower().split())
    if not key:
        return None
    names = [display_name(member, index, len(cast)).lower() for index, member in enumerate(cast)]
    for index, member in enumerate(cast):
        if key in (member.get("id", "").lower(), names[index]):
            return index
    for test in (lambda name: name.split()[0] == key.split()[0],  # same first name: "Sonia" / "Nisha ji"
                 lambda name: name.startswith(key) or key.startswith(name)):  # a shortened name: "Ana"
        hits = [index for index, name in enumerate(names) if name and test(name)]
        if len(hits) == 1:
            return hits[0]
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
        content TEXT NOT NULL, created_at REAL NOT NULL, excluded INTEGER NOT NULL DEFAULT 0);
    CREATE INDEX IF NOT EXISTS agent_messages_session ON agent_messages(session_id, id);

    /* What a chat knows, as entities and the relations between them, so that "who is Riya to Tiya" does not
       cost a re-read of the whole conversation and survives both a restart and a change of chat model. These
       live in jobs.sqlite3, which the Drive snapshot already copies whole, so they need no new snapshot work. */
    CREATE TABLE IF NOT EXISTS agent_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        owner TEXT NOT NULL DEFAULT '',        -- '' = said aloud, what the director knows; else a cast member id
        key TEXT NOT NULL,                     -- 'member:<id>' | 'user' | an asset id | a casefolded label
        kind TEXT NOT NULL,                    -- person|place|thing|event|asset|topic
        label TEXT NOT NULL, props TEXT NOT NULL DEFAULT '{}',
        mentions INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE UNIQUE INDEX IF NOT EXISTS agent_nodes_key ON agent_nodes(session_id, owner, key);

    CREATE TABLE IF NOT EXISTS agent_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        owner TEXT NOT NULL DEFAULT '', src INTEGER NOT NULL, dst INTEGER NOT NULL,
        rel TEXT NOT NULL, family TEXT NOT NULL,   -- family = the group a new fact supersedes within
        fact TEXT NOT NULL,                        -- the one sentence recall injects
        weight REAL NOT NULL DEFAULT 1.0,          -- -1..+1 for feels_toward
        state TEXT NOT NULL DEFAULT 'live',        -- live|stale; a superseded fact is kept, never deleted
        since_message INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE UNIQUE INDEX IF NOT EXISTS agent_edges_triple ON agent_edges(session_id, owner, src, rel, dst);
    CREATE INDEX IF NOT EXISTS agent_edges_src ON agent_edges(session_id, owner, src, state);
    CREATE INDEX IF NOT EXISTS agent_edges_dst ON agent_edges(session_id, owner, dst, state);
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(self._SCHEMA)
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(agent_messages)")}
            if "excluded" not in columns:  # a database from before messages could be forgotten
                self._db.execute("ALTER TABLE agent_messages ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0")
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
            self._db.execute("DELETE FROM agent_edges WHERE session_id = ?", (session_id,))
            self._db.execute("DELETE FROM agent_nodes WHERE session_id = ?", (session_id,))
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

    def messages(self, session_id: str, after: int = 0, include_hidden: bool = False,
                 before: int = 0, limit: int = 0) -> list[dict]:
        """The chat's messages, always oldest first. Forgotten ones (excluded) are left out unless
        include_hidden: every caller that builds a model prompt uses the default, so a forgotten message is
        never sent to a model again.

        ``limit`` returns the *newest* that many of whatever matched, because a chat is read from its end:
        opening a year-old conversation should not mean building a thousand messages the reader will scroll
        straight past. ``before`` then pages backwards from the oldest one already held. Callers that want
        everything -- every prompt builder, and the poll for new messages -- pass neither and are unaffected.
        """
        clauses, args = ["session_id = ?", "id > ?"], [session_id, after]
        if before:
            clauses.append("id < ?")
            args.append(before)
        if not include_hidden:
            clauses.append("excluded = 0")
        order = " ORDER BY id DESC LIMIT ?" if limit else " ORDER BY id"
        if limit:
            args.append(limit)
        with self._lock:
            rows = self._db.execute(
                "SELECT id, role, content, created_at, excluded FROM agent_messages WHERE "
                + " AND ".join(clauses) + order, args,
            ).fetchall()
        if limit:
            rows = rows[::-1]  # the newest N, handed back in reading order
        messages = [{"id": row[0], "role": row[1], "content": json.loads(row[2]), "created_at": row[3]} for row in rows]
        if include_hidden:
            for message, row in zip(messages, rows):
                message["excluded"] = bool(row[4])
        return messages

    def has_messages_before(self, session_id: str, message_id: int) -> bool:
        """Whether anything older than this is still in the chat, so a reader can be offered it."""
        with self._lock:
            return self._db.execute(
                "SELECT 1 FROM agent_messages WHERE session_id = ? AND id < ? LIMIT 1",
                (session_id, message_id),
            ).fetchone() is not None

    def get_message(self, session_id: str, message_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, role, content, created_at, excluded FROM agent_messages WHERE session_id = ? AND id = ?",
                (session_id, message_id),
            ).fetchone()
        return {"id": row[0], "role": row[1], "content": json.loads(row[2]), "created_at": row[3],
                "excluded": bool(row[4])} if row else None

    def set_excluded(self, session_id: str, message_ids: list[int], value: bool) -> None:
        with self._lock:
            self._db.executemany(
                "UPDATE agent_messages SET excluded = ? WHERE session_id = ? AND id = ?",
                [(1 if value else 0, session_id, message_id) for message_id in message_ids],
            )
            self._db.commit()

    # ------------------------------------------------------------ the graph

    def upsert_node(self, session_id: str, owner: str, key: str, kind: str, label: str) -> int:
        """A node's row id, created or touched. ``key`` is the caller's already-resolved identity."""
        now = _now()
        with self._lock:
            row = self._db.execute("SELECT id, label FROM agent_nodes WHERE session_id = ? AND owner = ? AND key = ?",
                                   (session_id, owner, key)).fetchone()
            if row:
                # The label follows the newest naming, so a character renamed mid-chat reads correctly, but the
                # key does not move: the identity is what other rows point at.
                self._db.execute("UPDATE agent_nodes SET mentions = mentions + 1, updated_at = ?, label = ? WHERE id = ?",
                                 (now, label or row[1], row[0]))
                self._db.commit()
                return int(row[0])
            cursor = self._db.execute(
                "INSERT INTO agent_nodes (session_id, owner, key, kind, label, props, mentions, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, '{}', 1, ?, ?)",
                (session_id, owner, key, kind, label, now, now))
            self._db.commit()
            return int(cursor.lastrowid)

    def nodes(self, session_id: str, owners: tuple[str, ...]) -> list[dict]:
        marks = ",".join("?" for _ in owners)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id, owner, key, kind, label, mentions FROM agent_nodes WHERE session_id = ? AND owner IN ({marks})",
                (session_id, *owners)).fetchall()
        return [{"id": r[0], "owner": r[1], "key": r[2], "kind": r[3], "label": r[4], "mentions": r[5]} for r in rows]

    def node_by_key(self, session_id: str, owner: str, key: str) -> int | None:
        with self._lock:
            row = self._db.execute("SELECT id FROM agent_nodes WHERE session_id = ? AND owner = ? AND key = ?",
                                   (session_id, owner, key)).fetchone()
        return int(row[0]) if row else None

    def put_edge(self, session_id: str, owner: str, src: int, dst: int, rel: str, family: str, fact: str,
                 weight: float = 1.0, since_message: int = 0) -> int:
        """Write one fact, superseding any live fact in the same family between the same two nodes.

        Last writer wins, and the old row is marked stale rather than deleted: keeping it is what lets a
        character say "she used to trust me", and it is the undo path when a summary is forgotten. Contradiction
        needs no model call -- it is the family that decides what a new fact replaces.
        """
        now = _now()
        with self._lock:
            self._db.execute(
                "UPDATE agent_edges SET state = 'stale', updated_at = ? WHERE session_id = ? AND owner = ? AND src = ?"
                " AND dst = ? AND family = ? AND rel <> ? AND state = 'live'",
                (now, session_id, owner, src, dst, family, rel))
            self._db.execute(
                "INSERT INTO agent_edges (session_id, owner, src, dst, rel, family, fact, weight, state,"
                " since_message, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'live', ?, ?, ?)"
                " ON CONFLICT(session_id, owner, src, rel, dst) DO UPDATE SET"
                " fact = excluded.fact, weight = excluded.weight, state = 'live',"
                " since_message = excluded.since_message, updated_at = excluded.updated_at",
                (session_id, owner, src, dst, rel, family, fact, weight, since_message, now, now))
            row = self._db.execute("SELECT id FROM agent_edges WHERE session_id = ? AND owner = ? AND src = ?"
                                   " AND rel = ? AND dst = ?", (session_id, owner, src, rel, dst)).fetchone()
            self._db.commit()
        return int(row[0]) if row else 0

    def edges_around(self, session_id: str, owners: tuple[str, ...], node_ids: list[int],
                     live_only: bool = True) -> list[dict]:
        """Every edge touching these nodes, in either direction. One query per hop of the walk."""
        if not node_ids:
            return []
        owner_marks = ",".join("?" for _ in owners)
        node_marks = ",".join("?" for _ in node_ids)
        state = " AND state = 'live'" if live_only else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT id, owner, src, dst, rel, family, fact, weight, state, since_message FROM agent_edges"
                f" WHERE session_id = ? AND owner IN ({owner_marks}){state}"
                f" AND (src IN ({node_marks}) OR dst IN ({node_marks}))",
                (session_id, *owners, *node_ids, *node_ids)).fetchall()
        keys = ("id", "owner", "src", "dst", "rel", "family", "fact", "weight", "state", "since_message")
        return [dict(zip(keys, row)) for row in rows]

    def all_edges(self, session_id: str, live_only: bool = False) -> list[dict]:
        """The whole graph for one chat, for the panel that lets a wrong fact be found and removed."""
        state = " AND e.state = 'live'" if live_only else ""
        with self._lock:
            rows = self._db.execute(
                "SELECT e.id, e.owner, e.rel, e.family, e.fact, e.weight, e.state, e.since_message,"
                " s.label, d.label FROM agent_edges e"
                " JOIN agent_nodes s ON s.id = e.src JOIN agent_nodes d ON d.id = e.dst"
                f" WHERE e.session_id = ?{state} ORDER BY e.updated_at DESC", (session_id,)).fetchall()
        keys = ("id", "owner", "rel", "family", "fact", "weight", "state", "since_message", "src_label", "dst_label")
        return [dict(zip(keys, row)) for row in rows]

    def delete_edge(self, session_id: str, edge_id: int) -> bool:
        with self._lock:
            cursor = self._db.execute("DELETE FROM agent_edges WHERE session_id = ? AND id = ?", (session_id, edge_id))
            self._db.commit()
        return cursor.rowcount > 0

    def trim_facts(self, session_id: str, keep: int) -> int:
        """Keep a chat's live facts under a cap, oldest and least-mentioned first.

        Nothing pruned before, and a three-message chat on the pod produced 32 facts. Dropped rows go to
        'stale' rather than being deleted, so the panel still shows them and reviving one is an update: the
        same treatment a superseded fact gets, for the same reason.
        """
        with self._lock:
            live = self._db.execute(
                "SELECT COUNT(*) FROM agent_edges WHERE session_id = ? AND state = 'live'", (session_id,)).fetchone()[0]
            if live <= keep:
                return 0
            cursor = self._db.execute(
                "UPDATE agent_edges SET state = 'stale' WHERE id IN ("
                "  SELECT e.id FROM agent_edges e JOIN agent_nodes s ON s.id = e.src"
                "  WHERE e.session_id = ? AND e.state = 'live'"
                "  ORDER BY e.since_message ASC, s.mentions ASC, e.updated_at ASC LIMIT ?)",
                (session_id, live - keep))
            self._db.commit()
        return cursor.rowcount

    def drop_facts_since(self, session_id: str, since_message: int) -> int:
        """Forget the facts a compaction produced, and revive whatever they superseded.

        Called when messages are forgotten: a fact extracted from a reply that no longer exists should not
        outlive it, and the family it displaced must not be left with nothing live in it.
        """
        with self._lock:
            families = self._db.execute(
                "SELECT DISTINCT owner, src, dst, family FROM agent_edges WHERE session_id = ? AND since_message >= ?",
                (session_id, since_message)).fetchall()
            cursor = self._db.execute("DELETE FROM agent_edges WHERE session_id = ? AND since_message >= ?",
                                      (session_id, since_message))
            for owner, src, dst, family in families:
                alive = self._db.execute(
                    "SELECT 1 FROM agent_edges WHERE session_id = ? AND owner = ? AND src = ? AND dst = ?"
                    " AND family = ? AND state = 'live' LIMIT 1", (session_id, owner, src, dst, family)).fetchone()
                if alive:
                    continue
                newest = self._db.execute(
                    "SELECT id FROM agent_edges WHERE session_id = ? AND owner = ? AND src = ? AND dst = ?"
                    " AND family = ? ORDER BY updated_at DESC LIMIT 1",
                    (session_id, owner, src, dst, family)).fetchone()
                if newest:
                    self._db.execute("UPDATE agent_edges SET state = 'live' WHERE id = ?", (newest[0],))
            self._db.commit()
        return cursor.rowcount

    def delete_messages(self, session_id: str, message_ids: list[int]) -> None:
        with self._lock:
            self._db.executemany(
                "DELETE FROM agent_messages WHERE session_id = ? AND id = ?",
                [(session_id, message_id) for message_id in message_ids],
            )
            self._db.commit()


# ---------------------------------------------------------------- helpers


def split_spoken_lines(say: str, cast: list[dict]) -> list[dict]:
    """A group-chat reply that came back as plain "say", split into lines by the names it uses.

    Models that ignore the "lines" format usually still write "Nisha: ...\nSonia Mausi: ...". Recovering the speakers
    keeps the bubbles attributed; anything that names nobody is left alone for the caller to mark as the narrator."""
    if not say or len(cast) < 2:
        return []
    names = [display_name(member, index, len(cast)) for index, member in enumerate(cast)]
    pattern = "|".join(re.escape(name) for name in names if name)
    if not pattern:
        return []
    hits = list(re.finditer(rf"^[ \t*_]*({pattern})[ \t]*[:\u2014-][ \t]*", say, re.IGNORECASE | re.MULTILINE))
    lines = []
    for index, hit in enumerate(hits):
        end = hits[index + 1].start() if index + 1 < len(hits) else len(say)
        spoken = say[hit.end():end].strip()
        member = find_member(cast, hit.group(1))
        if spoken and member is not None:
            lines.append({"speaker": names[member], "say": spoken})
    return lines[:MAX_LINES]


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
            raw = source.get("of")  # who is in the picture, as opposed to who took it
            wanted = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
            of = [str(n).strip()[:40] for n in wanted if isinstance(n, str) and str(n).strip()]
            if of:
                action["of"] = of[:MAX_CAST]
        reply = {"say": say, "actions": clean, "done": bool(data.get("done"))}
        if data.get("pause"):
            reply["pause"] = True
        if lines:
            reply["lines"] = lines
        grow = []
        for item in data.get("grow") if isinstance(data.get("grow"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("note"), str) and item["note"].strip():
                entry = {"speaker": str(item.get("speaker") or "").strip()[:40], "note": item["note"].strip()[:MAX_GROWTH_CHARS]}
                if isinstance(item.get("about"), str) and item["about"].strip():
                    entry["about"] = item["about"].strip()[:40]
                grow.append(entry)
        if grow:
            reply["grow"] = grow[:MAX_CAST]
        return reply
    return None


def _shorten(value, keep: frozenset = frozenset({"video_url", "segment_urls", "upload_url", "thumb_url", "file_url"}),
             limit: int = STRING_CHARS):
    if isinstance(value, dict):
        return {k: (v if k in keep else _shorten(v, keep, limit)) for k, v in value.items()}
    if isinstance(value, list):
        return [_shorten(v, keep, limit) for v in value[:50]]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more characters]"
    return value


def compact_result(result) -> str:
    text = json.dumps(_shorten(result), ensure_ascii=False)
    if len(text) <= RESULT_CHARS:
        return text
    urls = re.findall(r'"video_url":\s*"[^"]*"', text)
    return text[:RESULT_CHARS] + " ...[truncated]" + (" " + " ".join(urls) if urls else "")


def brief_result(value, depth: int = 0):
    """An earlier turn's tool result cut down to what later turns refer to (ids, links, outcomes, verdicts)."""
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if key not in BRIEF_KEYS or item in (None, "", [], {}):
                continue
            brief = brief_result(item, depth + 1)
            if brief not in (None, "", [], {}):
                kept[key] = brief
        return kept
    if isinstance(value, list):
        return [b for b in (brief_result(item, depth + 1) for item in value[:12]) if b not in (None, "", [], {})]
    if isinstance(value, str) and len(value) > BRIEF_STRING_CHARS:
        return value[:BRIEF_STRING_CHARS] + "..."
    return value


def estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m["content"]) if isinstance(m["content"], str) else len(json.dumps(m["content"])) for m in messages) // 4


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
        self._atlas = atlas
        self.store = store or AgentStore(service.settings.db_path)
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop: set[str] = set()
        self._joining: set[str] = set()  # the user wrote while the characters were talking
        self._failed_engines: dict[str, set[str]] = {}  # per chat, this run: image engines whose takes failed inspection
        self._awaiting_choice: dict[str, dict] = {}  # per chat: inspection wants a retake, the user has not picked yet
        self._makes: dict[str, set[asyncio.Task]] = {}  # per chat: pictures still rendering beside the talk
        self._final: dict[str, str] = {}  # per running chat: the status it ends with
        settings = service.settings
        self.compact_tokens = settings.agent_compact_tokens  # summarise older messages above this (estimated tokens)
        self.keep_messages = settings.agent_keep_messages

    def _llm(self):
        """The LLM settings in force. Read through the service when it keeps a store, else straight off the
        pod's own settings: AgentService is built against a stand-in in places, and opening a chat should not
        fail because the thing it was handed has no store."""
        store = getattr(self.service, "llm_settings", None)
        overrides = self.service.settings.llm_overrides
        return store.resolve(overrides) if store is not None else overrides

    async def _catalogue(self) -> list[dict]:
        """The chat models the provider in force actually serves, or [] when it cannot be asked.

        [] is not "no models": models_llm.resolve reads it as "do not second-guess the setting", so a /models
        call that fails leaves every role sending exactly the id it was configured with.
        """
        try:
            return await self.atlas.list_models()
        except AtlasError:
            return []

    def _configured_chain(self, role: str, session: dict | None = None) -> str:
        """The chain a role is set to, before it meets the provider's catalogue."""
        llm = self._llm()
        settings = self.service.settings
        return {
            "director": (session or {}).get("model") or llm.agent_model_override or settings.agent_model,
            "prose": (session or {}).get("prose_model") or llm.agent_prose_model_override
            or settings.agent_prose_model or (session or {}).get("model") or settings.agent_model,
            "summary": llm.agent_summary_model_override or settings.agent_summary_model,
            "vision": llm.agent_vision_model_override or settings.agent_vision_model,
        }.get(role, "")

    async def model_for(self, role: str, session: dict | None = None) -> str:
        """The id to send for one role, resolved against the provider that is switched on.

        Every role used to hold a bare id, so switching provider silently pointed five of them at models the
        new service does not serve. See hawk_api/models_llm.py.
        """
        return models_llm.resolve(role, self._configured_chain(role, session), await self._catalogue())

    async def models_for(self, role: str, session: dict | None = None) -> list[str]:
        """Every id worth trying for a role, best first.

        The vision ladder walks the whole list looking for a model that will agree to look at the picture.
        The other roles use the tail as a failover: the chain was always meant to answer "the provider would
        not serve this one", and a provider that errors mid-run is the same question asked later.
        """
        return models_llm.resolve_many(role, self._configured_chain(role, session), await self._catalogue())

    @property
    def summary_model(self) -> str:
        """The configured summary chain, unresolved. Kept for callers that only report the setting."""
        return self._llm().agent_summary_model_override or self.service.settings.agent_summary_model

    @property
    def atlas(self) -> AtlasClient:
        return self._atlas or self.service.atlas

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

    def create_session(self, title: str | None = None, persona: str | None = None, model: str | None = None,
                       prose_model: str | None = None) -> dict:
        now = _now()
        # Stored as configured, not resolved: resolution happens per call against whichever provider is
        # switched on then, so a chat made today still works after the provider changes under it.
        default_model = self._llm().agent_model_override or self.service.settings.agent_model
        session = {
            "id": str(uuid.uuid4()),
            "title": (title or "").strip() or "New chat",
            "persona": (persona or "").strip(),
            "name": "",
            "avatar_asset_id": "",
            "model": (model or "").strip() or default_model,
            # Blank means "follow the pod's prose setting, then this chat's own model" -- how every chat
            # behaved before the director and the characters were given separate models.
            "prose_model": (prose_model or "").strip(),
            "adaptive": False,
            "whispers": False,
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

    def update_session(self, session_id: str, *, title=None, persona=None, model=None, prose_model=None, name=None,
                       avatar_asset_id=None, cast=None, adaptive=None, whispers=None) -> dict:
        session = self.get_session(session_id)
        if adaptive is not None:
            session["adaptive"] = bool(adaptive)
        if whispers is not None:
            session["whispers"] = bool(whispers)
        if cast is not None:
            self._set_cast(session, cast)
        if title is not None:
            session["title"] = title.strip() or session["title"]
        if persona is not None:
            session["persona"] = persona.strip()
        if model is not None and model.strip():
            session["model"] = model.strip()
        if prose_model is not None:  # blank clears it back to the pod's setting, so it is not guarded on strip()
            session["prose_model"] = prose_model.strip()
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
            before = known.get(entry["id"], {})
            growth = member.get("growth")  # None keeps what the character has grown into so far
            entry["growth"] = clean_growth(growth if growth is not None else before.get("growth"))
            feelings = member.get("feelings")  # the same for private feelings about the user and the others
            entry["feelings"] = clean_feelings(feelings if feelings is not None else before.get("feelings"))
            if before.get("memory"):  # the character's own memory of the chat (talk mode)
                entry["memory"], entry["memory_upto"] = before["memory"], before.get("memory_upto", 0)
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
            entry["feelings_view"] = [{"key": key, "about": self._about_name(cast, key), "notes": notes}
                                      for key, notes in clean_feelings(member.get("feelings")).items()]
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

    # ------------------------------------------------------- forgetting

    def _forget_group(self, session_id: str, message_id: int) -> tuple[dict, list[int]]:
        """A message and what belongs with it: an assistant reply takes the tool results it produced."""
        target = self.store.get_message(session_id, message_id)
        if target is None:
            raise NotFound(f"Message {message_id} is not in this chat.")
        ids = [message_id]
        if target["role"] == "assistant":
            for message in self.store.messages(session_id, after=message_id, include_hidden=True):
                if message["role"] not in ("tool", "note"):
                    break
                if message["role"] == "tool":
                    ids.append(message["id"])
        return target, ids

    def _rebuild_after_forgetting(self, session_id: str, ids: list[int]) -> bool:
        """Compaction may already have folded these messages into the chat summary or a character's memory. Clear
        whatever covers them, so the next call rebuilds it from what is left instead of repeating forgotten text."""
        session = self.get_session(session_id)
        oldest = min(ids)
        rebuilt = False
        # A fact extracted from a reply must not outlive the reply. Anything superseded by a dropped fact comes
        # back live, so a family is never left with nothing in it.
        if self.store.drop_facts_since(session_id, oldest):
            rebuilt = True
        if session.get("summary") and oldest <= session.get("summary_upto", 0):
            session.update(summary="", summary_upto=0)
            rebuilt = True
        cast = [dict(member) for member in cast_of(session)]
        for member in cast:
            if member.get("memory") and oldest <= member.get("memory_upto", 0):
                member.update(memory="", memory_upto=0)
                rebuilt = True
        if rebuilt:
            session["cast"] = cast
            self.store.save_session(session)
        return rebuilt

    def forget_message(self, session_id: str, message_id: int, mode: str = "hide") -> dict:
        """Take a message out of the conversation the models see. "hide" keeps the bubble in Studio, greyed out and
        restorable; "purge" removes it for good. An assistant reply takes its tool results with it."""
        self.get_session(session_id)
        if session_id in self._tasks:
            raise Conflict("The agent is working; forget once it has finished.")
        if mode not in ("hide", "purge"):
            raise RequestError('mode must be "hide" or "purge".')
        _, ids = self._forget_group(session_id, message_id)
        rebuilt = self._rebuild_after_forgetting(session_id, ids)
        if mode == "purge":
            self.store.delete_messages(session_id, ids)
        else:
            self.store.set_excluded(session_id, ids, True)
        return {"session": self.get_session(session_id), "forgotten": ids, "mode": mode, "rebuilt": rebuilt}

    def restore_message(self, session_id: str, message_id: int) -> dict:
        """Put a hidden message back into the conversation the models see."""
        self.get_session(session_id)
        if session_id in self._tasks:
            raise Conflict("The agent is working; restore once it has finished.")
        _, ids = self._forget_group(session_id, message_id)
        self.store.set_excluded(session_id, ids, False)
        return {"session": self.get_session(session_id), "restored": ids}

    def forget_last(self, session_id: str) -> dict:
        """The Forget last reply button: hide the newest reply and the tool results that came with it."""
        self.get_session(session_id)
        if session_id in self._tasks:  # checked before looking for a reply: a busy chat is a conflict, not a 404
            raise Conflict("The agent is working; forget once it has finished.")
        last = next((m for m in reversed(self.store.messages(session_id)) if m["role"] == "assistant"), None)
        if last is None:
            raise NotFound("There is no reply to forget in this chat yet.")
        return self.forget_message(session_id, last["id"], "hide")

    def view(self, session_id: str, after: int = 0, before: int = 0, limit: int = 0) -> dict:
        """The chat as Studio draws it. ``limit`` opens on the last page rather than the whole transcript and
        ``before`` walks back from there; ``has_older`` says whether there is another page to walk to."""
        session = self.get_session(session_id)
        messages = self.store.messages(session_id, after, include_hidden=True, before=before, limit=limit)
        for message in messages:  # fresh signed thumbnails for attached files
            for attachment in message["content"].get("attachments") or [] if message["role"] == "user" else []:
                asset = self.service.store.get_asset(attachment.get("asset_id", ""))
                if asset:
                    attachment.update({k: v for k, v in self.service.asset_view(asset).items() if k in ("thumb_url", "file_url")})
        # Only meaningful for a paged read: an unpaged one already holds everything after "after".
        older = bool(messages) and (before or limit) and self.store.has_messages_before(session_id, messages[0]["id"])
        return {"session": self.public(session), "messages": messages, "has_older": bool(older)}

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
        content = {"text": text.strip(), "attachments": files}
        whisper = self._whisper_target(session, text)
        if whisper:
            content["private_to"] = whisper
        message = self.store.add_message(session_id, "user", content)
        session["status"] = "running"
        self.store.save_session(session)
        self._stop.discard(session_id)
        self._failed_engines.pop(session_id, None)
        self._awaiting_choice.pop(session_id, None)
        self._tasks[session_id] = asyncio.create_task(self._run(session_id))
        return {"session": session, "message": message}

    def talk(self, session_id: str, rounds: int = 5, makes: int = 2) -> dict:
        """Let the characters of a group chat talk to each other for a few rounds.

        makes is per character, not for the room: a quiet character should still be able to ask for something
        after a talkative one has had its fill. MAX_TALK_MAKES_TOTAL caps the cast as a whole.
        """
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
        self._failed_engines.pop(session_id, None)
        self._awaiting_choice.pop(session_id, None)
        makes = max(0, min(MAX_TALK_MAKES, int(makes)))
        self._tasks[session_id] = asyncio.create_task(self._run(session_id, talk_rounds=rounds, makes=makes))
        return session

    def _whisper_target(self, session: dict, text: str) -> str:
        """With whispers on, "@Name ..." at the start of a message is private to that character (its id)."""
        cast = cast_of(session)
        match = re.match(r"\s*@([\w'-]+)", text or "")
        if not session.get("whispers") or len(cast) < 2 or not match:
            return ""
        index = find_member(cast, match.group(1))
        return cast[index]["id"] if index is not None else ""

    @staticmethod
    def _about_name(cast: list[dict], key: str) -> str:
        if key == USER_KEY:
            return "the user"
        for index, member in enumerate(cast):
            if member.get("id") == key:
                return display_name(member, index, len(cast))
        return key

    # ------------------------------------------------------------ what a chat knows

    def _node_key(self, session: dict, label: str) -> tuple[str, str, str]:
        """(key, kind, label) for something a fact names.

        Identity reuses what already exists rather than inventing a second notion of who someone is:
        find_member for the cast, cast_talk's USER_KEY for the user, an asset id as itself. Anything else is a
        casefolded label. Getting this wrong is how a fact about Riya ends up filed under Tiya, so nothing here
        guesses -- see _merge_key for the one near-miss rule, which refuses when it is ambiguous.
        """
        text = " ".join(str(label or "").strip().lstrip("@").split())
        if not text:
            return "", "", ""
        cast = cast_of(session)
        index = find_member(cast, text)
        if index is not None:
            member = cast[index]
            return f"member:{member.get('id') or index}", "person", display_name(member, index, len(cast))
        if text.lower() in ("user", "the user", "you", "me"):
            return USER_KEY, "person", "the user"
        if re.fullmatch(r"[0-9a-f]{12,36}", text.lower()):
            return text.lower(), "asset", text
        return text.casefold(), "topic", text

    def _merge_key(self, session_id: str, owner: str, key: str, kind: str) -> str:
        """The existing key this one is plainly the same as, else itself.

        Only a prefix or suffix match against **exactly one** node of the same kind counts, deliberately the
        same rule as cast_talk.mentioned_all: "a first name two characters share names neither". Ambiguity makes
        a new node, because two nodes for one person is a smaller bug than one node for two people.
        """
        if key in (USER_KEY, "") or key.startswith("member:"):
            return key
        candidates = [n["key"] for n in self.store.nodes(session_id, (owner,))
                      if n["kind"] == kind and n["key"] != key
                      and (n["key"].startswith(key) or key.startswith(n["key"])
                           or n["key"].endswith(key) or key.endswith(n["key"]))]
        return candidates[0] if len(candidates) == 1 else key

    def _node_for(self, session: dict, owner: str, label: str) -> int:
        key, kind, shown = self._node_key(session, label)
        if not key:
            return 0
        key = self._merge_key(session["id"], owner, key, kind)
        return self.store.upsert_node(session["id"], owner, key, kind, shown)

    def _write_fact(self, session: dict, owner: str, subject: str, rel: str, obj: str, fact: str,
                    weight: float = 1.0, since_message: int = 0) -> bool:
        rel = str(rel or "").strip().lower()
        if rel not in RELATIONS or not str(fact or "").strip():
            return False
        family, _, kinds = RELATIONS[rel]
        src = self._node_for(session, owner, subject)
        dst = self._node_for(session, owner, obj)
        if not src or not dst or src == dst:
            return False
        # The object has to be the kind the relation is about. This is what catches a model that reached for the
        # nearest verb: "located_in" an asset, "relation_to" a plan. No model call, and it refuses rather than
        # files the fact in a family where it will never be contradicted.
        _, kind, _ = self._node_key(session, obj)
        if kind and kind not in kinds:
            return False
        try:
            weight = max(-1.0, min(1.0, float(weight)))
        except (TypeError, ValueError):
            weight = 1.0
        self.store.put_edge(session["id"], owner, src, dst, rel, family,
                            " ".join(str(fact).split())[:240], weight, since_message)
        self.store.trim_facts(session["id"], MAX_LIVE_FACTS)
        return True

    def _seed_graph(self, session: dict) -> None:
        """Give an existing chat a usable graph for no tokens at all.

        Every cast member, the user and every avatar become nodes, and the feelings dict -- already one short
        sentence per entry -- becomes feels_toward edges. This is why no LLM backfill is needed for the chats
        that already exist: the per-chat cost is exactly what the graph is meant to remove.
        """
        if session.get("graph_seeded"):
            return
        cast = cast_of(session)
        self._node_for(session, "", "user")
        for index, member in enumerate(cast):
            name = display_name(member, index, len(cast))
            if name:
                self._node_for(session, "", name)
            if member.get("avatar_asset_id"):
                self._write_fact(session, "", name, "owns", member["avatar_asset_id"],
                                 f"{name}'s avatar is asset {member['avatar_asset_id']}.")
            for key, notes in clean_feelings(member.get("feelings")).items():
                about = self._about_name(cast, key)
                text = " ".join(notes)
                if name and about and text:
                    self._write_fact(session, member.get("id", ""), name, "feels_toward", about,
                                     f"{name} on {about}: {text}")
        session["graph_seeded"] = True
        self.store.save_session(session)

    def _recall_text(self, session: dict, owner: str, seed_text: str) -> str:
        """Facts about whoever this turn is about, within a token budget. No model call.

        Seeded from the names the turn mentions, then a two-hop walk out from them. Bounded on purpose: the
        point of the graph is to answer "who is this person to me" without re-reading the conversation, and an
        unbounded dump would cost more than the prose it is meant to replace.
        """
        if not self.service.settings.agent_graph_recall:
            return ""
        # A character reads what was said aloud plus its own private view. The director reads only what was said
        # aloud: it already gets every character's private feelings through _state_text, exactly as it always
        # did, so pulling them in again here would duplicate them and cost tokens twice.
        owners = ("",) if not owner else ("", owner)
        nodes = self.store.nodes(session["id"], owners)
        if not nodes:
            return ""
        text = f" {' '.join(str(seed_text or '').split()).casefold()} "
        seeds: list[int] = []
        for node in nodes:
            label = node["label"].casefold()
            if node["key"] == USER_KEY or (label and (f" {label} " in text or f" {label}'" in text or f" {label}," in text)):
                seeds.append(node["id"])
        if owner:
            index = find_member(cast_of(session), owner)
            if index is not None:
                cast = cast_of(session)
                me = self.store.node_by_key(session["id"], "", f"member:{cast[index].get('id') or index}")
                if me:
                    seeds.append(me)
        seeds = list(dict.fromkeys(seeds))
        if not seeds:
            return ""

        by_id = {node["id"]: node for node in nodes}
        scored: dict[int, tuple[float, str]] = {}
        frontier = seeds
        newest = max((e["since_message"] for e in self.store.edges_around(session["id"], owners, seeds)), default=0) or 1
        for hop in (1, 2):
            edges = self.store.edges_around(session["id"], owners, frontier)
            reached: list[int] = []
            for edge in edges:
                other = edge["dst"] if edge["src"] in frontier else edge["src"]
                node = by_id.get(other)
                score = (2.0 if hop == 1 else 0.0) + min(1.0, edge["since_message"] / newest) \
                    + 0.5 * math.log1p(node["mentions"] if node else 1) + abs(edge["weight"])
                if edge["id"] not in scored or scored[edge["id"]][0] < score:
                    scored[edge["id"]] = (score, edge["fact"])
                reached.append(other)
            frontier = [n for n in dict.fromkeys(reached) if n not in seeds][:6]
            if not frontier:
                break

        budget = max(0, int(self.service.settings.agent_recall_tokens))
        lines, used = [], 0
        for _, fact in sorted(scored.values(), key=lambda pair: -pair[0]):
            cost = len(fact) // 4 + 2
            if used + cost > budget:
                break
            lines.append(f"- {fact}")
            used += cost
        if not lines:
            return ""
        return "WHAT YOU ALREADY KNOW ABOUT WHO IS IN THIS TURN\n" + "\n".join(lines)

    def graph_view(self, session_id: str) -> dict:
        """What this chat knows, for the panel. Being able to read it is what makes recall safe to leave on."""
        session = self.get_session(session_id)
        cast = cast_of(session)
        rows = []
        for edge in self.store.all_edges(session_id):
            rows.append({**edge, "private_to": self._about_name(cast, edge["owner"]) if edge["owner"] else ""})
        return {"facts": rows, "nodes": len(self.store.nodes(session_id, ("", *(m.get("id", "") for m in cast)))),
                "recall_on": bool(self.service.settings.agent_graph_recall)}

    def forget_fact(self, session_id: str, edge_id: int) -> dict:
        self.get_session(session_id)
        if not self.store.delete_edge(session_id, edge_id):
            raise NotFound(f"No fact {edge_id} in this chat.")
        return self.graph_view(session_id)

    def request_stop(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        if session_id in self._tasks:
            self._stop.add(session_id)
            session["status"] = "stopping"
            self.store.save_session(session)
        return session

    def _note(self, session_id: str, text: str, kind: str = "info") -> None:
        self.store.add_message(session_id, "note", {"text": text, "kind": kind})

    async def _run(self, session_id: str, talk_rounds: int = 0, makes: int = 2) -> None:
        self._final[session_id] = "idle"
        try:
            if talk_rounds:
                await self._talk(session_id, talk_rounds, makes)
                # The conversation ran ahead of its pictures; wait for them before the chat says it is done,
                # so Studio keeps polling at its working cadence until the last one has landed.
                pending = self._makes.get(session_id) or set()
                if pending:
                    await asyncio.gather(*list(pending), return_exceptions=True)
            else:
                last_user = next((m for m in reversed(self.store.messages(session_id)) if m["role"] == "user"), None)
                private = (last_user or {}).get("content", {}).get("private_to", "") if last_user else ""
                await self._director_turn(session_id, MAX_STEPS, private_to=private)
        except AtlasError as exc:
            self._note(session_id, f"Model error: {exc}", "error")
            self._final[session_id] = "error"
        except asyncio.CancelledError:
            self._final[session_id] = "idle"
            raise
        except Exception as exc:  # never leave a chat stuck in "running"
            log.exception("agent run failed")
            self._note(session_id, f"Agent error: {exc}", "error")
            self._final[session_id] = "error"
        finally:
            for task in list(self._makes.pop(session_id, set())):
                task.cancel()
            self._tasks.pop(session_id, None)
            self._stop.discard(session_id)
            self._joining.discard(session_id)
            session = self.store.get_session(session_id)
            if session is not None:
                session["status"] = self._final.pop(session_id, "idle")
                session["talking"] = False
                self.store.save_session(session)

    def _halted(self, session_id: str, started: float, steps: int, max_steps: int) -> bool:
        """Stop, step and time limits, checked before every model call (notes say why it stopped)."""
        if session_id in self._stop:
            self._note(session_id, "You joined in." if session_id in self._joining else "Stopped by you.")
            return True
        if steps >= max_steps:
            if max_steps == MAX_STEPS:
                self._note(session_id, f"Paused after {MAX_STEPS} steps. Send a message to continue.", "warn")
            return True
        if time.monotonic() - started > MAX_RUN_SECONDS:
            self._note(session_id, "Paused after 3 hours. Send a message to continue.", "warn")
            return True
        return False

    async def _director_call(self, session_id: str, session: dict, messages: list[dict]) -> tuple[str, str, dict]:
        """One director step, walking down the configured chain when the provider errors.

        The director was the only model call on the whole path with no fallback at all: the prose turn, both
        summarisers, the growth merge and the vision ladder each catch AtlasError and try the next id, while
        this one let it escape to _run, which ends the entire turn with "Model error". A single rate limit
        was therefore worth more here than anywhere else -- and it is what makes a free-tier model, which is
        rate limited by definition, unusable for the busiest role in the chat.
        """
        models = await self.models_for("director", session)
        for position, model in enumerate(models):
            try:
                text, usage = await self.atlas.chat(model, messages, json_mode=True, max_tokens=8192)
            except AtlasError as exc:
                if position + 1 >= len(models):
                    raise
                log.warning("director step with %s failed: %s", model, exc)
                self._note(session_id, f"{model} could not answer ({exc}); trying {models[position + 1]}.", "warn")
                continue
            return model, text, usage
        raise AtlasError("No director model is available: the provider lists none of the configured ids.")

    async def _director_turn(self, session_id: str, max_steps: int, private_to: str = "", asked_by: str = "",
                             subjects: tuple[str, ...] = ()) -> dict | None:
        """The full agent (persona or cast, tools, pipeline) works until it replies without actions. Returns the last
        reply, or None when it stopped. private_to: the user whispered to one character, so the answer is private too.
        asked_by: the character who asked for this while they were talking, so what it makes belongs to them."""
        started = time.monotonic()
        steps = repairs = 0
        reply = None
        while True:
            if self._halted(session_id, started, steps, max_steps):
                return None
            session = self.get_session(session_id)
            await self._maybe_summarize(session)
            messages = await self._build_messages(session)
            model, text, usage = await self._director_call(session_id, session, messages)
            steps += 1
            call = await self._add_usage(session_id, model, usage)
            reply = parse_reply(text)
            if reply is None:
                self.store.add_message(session_id, "assistant", {"raw": text[:4000], "invalid": True, "usage": call})
                if repairs:
                    self._note(session_id, "The model did not answer in the required JSON format twice; stopped.", "error")
                    self._final[session_id] = "error"
                    return None
                repairs += 1
                self._note(session_id, "Your last reply was not the required JSON object. Reply again with only the JSON object.", "repair")
                continue
            repairs = 0
            cast = cast_of(session)
            if len(cast) > 1 and not reply.get("lines") and reply.get("say"):
                recovered = split_spoken_lines(reply["say"], cast)
                if recovered:  # the model wrote "Nisha: ..." in one block instead of using "lines"
                    reply["lines"] = recovered
                else:
                    reply["narrator"] = True  # nobody is speaking: the UI shows it without a character's name
            content = {"raw": text[:8000], **reply, "usage": call}
            if private_to:
                content["private_to"] = private_to
            self.store.add_message(session_id, "assistant", content)
            if reply.get("grow"):
                await self._grow(session_id, reply["grow"])
            if not reply["actions"]:
                return reply
            for action in reply["actions"]:
                if session_id in self._stop:
                    break
                await self._execute(session_id, action, asked_by, subjects)
            if (self._awaiting_choice.get(session_id) or {}).pop("blocked", False):
                self._note(session_id, "Waiting for you to pick a take before generating again.", "warn")
                return reply

    # ------------------------------------------------------------ let them talk

    async def _talk(self, session_id: str, rounds: int, makes: int) -> None:
        """The characters talk to each other: each turn is one character's own model call (see cast_talk)."""
        started = time.monotonic()
        steps = 0
        spoke: list[int] = []
        spent: dict[int, int] = {}  # makes used, per character
        last_index, last_text = self._last_speaker(session_id)
        for round_number in range(1, rounds + 1):
            if session_id in self._stop:
                self._note(session_id, "You joined in." if session_id in self._joining else "Stopped by you.")
                return
            self.store.add_message(session_id, "note", {"text": f"Talking · round {round_number} of {rounds}",
                                                        "kind": "talk", "round": round_number, "of": rounds})
            for _ in range(len(cast_of(self.get_session(session_id)))):
                if self._halted(session_id, started, steps, MAX_STEPS):
                    return
                session = self.get_session(session_id)
                cast = cast_of(session)
                names = [display_name(m, i, len(cast)) for i, m in enumerate(cast)]
                index = cast_talk.next_speaker(cast, names, last_index, last_text, spoke)
                turn = await self._character_turn(session, index)
                steps += 1
                if turn is None:
                    self._note(session_id, f"{names[index]} couldn't answer in the expected format; the talk stopped.", "error")
                    self._final[session_id] = "error"
                    return
                spoke.append(index)
                last_index, last_text = index, turn["say"]
                if turn.get("grow"):
                    await self._grow(session_id, [{"speaker": names[index], **item} for item in turn["grow"]])
                if turn.get("make"):
                    # The budget is each character's own: one enthusiastic character used to spend the whole
                    # talk's allowance before a quieter one ever asked for anything.
                    if spent.get(index, 0) >= makes:
                        self._note(session_id, f"{names[index]} wanted something made, but has already had {makes} this talk.", "warn")
                    elif sum(spent.values()) >= MAX_TALK_MAKES_TOTAL:
                        self._note(session_id, f"{names[index]} wanted something made, but this talk has made {MAX_TALK_MAKES_TOTAL} already.", "warn")
                    else:
                        spent[index] = spent.get(index, 0) + 1
                        subjects = tuple(self._subjects_for(cast, names, index, turn))
                        act = turn.get("act", "")
                        if act in ("selfie", "snap", "group_shot"):
                            # It said what it was taking and of whom, so no director turn is needed and the
                            # picture can render while the others keep talking.
                            self.store.add_message(session_id, "note", {"text": turn["make"], "kind": "make",
                                                                        "speaker": names[index], "act": act,
                                                                        "of": list(subjects)})
                            self._spawn_make(session_id, names[index], turn["make"], act, subjects)
                        else:
                            await self._make_for(session_id, names[index], turn["make"], act=act, of=subjects)
                if turn.get("pause") and set(spoke) >= set(range(len(cast))):
                    return  # a pause counts once everyone has had a say; before that the talk goes on

    def _last_speaker(self, session_id: str) -> tuple[int | None, str]:
        """Who spoke last (a character's index, or None for the user) and what they said, to pick the first speaker."""
        session = self.get_session(session_id)
        cast = cast_of(session)
        for message in reversed(self.store.messages(session_id)):
            content = message["content"]
            if message["role"] == "user":
                return None, content.get("text", "")
            if message["role"] == "assistant" and content.get("lines"):
                line = content["lines"][-1]
                return find_member(cast, line.get("speaker", "")), line.get("say", "")
        return None, ""

    async def _character_turn(self, session: dict, index: int) -> dict | None:
        """One character speaks: a small call with only what this character knows."""
        session_id = session["id"]
        cast = cast_of(session)
        member = cast[index]
        await self._remember(session, member["id"])
        session = self.get_session(session_id)
        cast = cast_of(session)
        member = cast[index]
        names = [display_name(m, i, len(cast)) for i, m in enumerate(cast)]
        name = names[index]
        others = "\n".join(f"- {names[i]}: {(m.get('persona') or 'no persona yet')[:300]}" for i, m in enumerate(cast) if i != index)
        feelings = clean_feelings(member.get("feelings"))
        feeling_lines = [f"- About {self._about_name(cast, key)}: " + " ".join(notes) for key, notes in feelings.items()]
        grown = self._growth_text(member).replace("they have", "you have")
        # Identical for every character in this chat and every turn of it, so it leads and stays cached.
        rules = cast_talk.CHARACTER_RULES_PROMPT.format(
            words=cast_talk.TURN_WORDS,
            adaptive=cast_talk.ADAPTIVE_TURN if session.get("adaptive") else "",
            make_field=', "make": "optional", "act": "selfie|snap|share|group_shot", "of": ["names"]',
            grow_field=', "grow": []' if session.get("adaptive") else "",
            rules=PLATFORM_RULES,
        )
        # Who this one is, sent below the conversation: it is what differs between one character and the next,
        # and recall changes it every turn on top of that.
        self_text = cast_talk.CHARACTER_SELF_PROMPT.format(
            name=name,
            persona=member.get("persona") or f"{name}, a character in this chat.",
            growth=f"\n{grown}" if grown else "",
            others=others,
            listening=" They are listening to you talk right now and may join in.",
            feelings="\n".join(feeling_lines) or "(none yet)",
            memory="\n\n".join(part for part in (
                member.get("memory") or "",
                self._recall_text(session, member["id"], self._recent_text(session)),
            ) if part) or "(nothing older: everything is in the conversation below)",
        )
        transcript = self._transcript(session, member["id"], names)
        messages = [{"role": "system", "content": rules},
                    {"role": "user", "content": f"THE CONVERSATION SO FAR (most recent last):\n{transcript or '(nothing yet)'}"},
                    {"role": "system", "content": self_text}]
        def record(turn: dict, call: dict | None) -> dict:
            line = {"speaker": name, "say": turn["say"]}
            if turn.get("to"):
                line["to"] = turn["to"]
            content = {"lines": [line], "say": f"{name}: {turn['say']}", "actions": [], "done": True, "talk": True,
                       "speaker_id": member["id"], "raw": json.dumps({"lines": [line]}, ensure_ascii=False), "usage": call}
            self.store.add_message(session_id, "assistant", content)
            return turn

        # The characters speak with the prose model and the director keeps its own. A model chosen for voice is
        # often worse at formatting, so the last attempt moves to the director's model, which has to be good at
        # JSON to drive the tools at all.
        prose = await self.model_for("prose", session)
        director = await self.model_for("director", session)
        attempts = [prose, prose, director if director != prose else prose]
        text, call, failure = "", None, None
        for model in attempts:
            try:
                text, usage = await self.atlas.chat(model, messages, json_mode=True, max_tokens=TURN_MAX_TOKENS, temperature=0.8)
            except AtlasError as exc:
                # A refusal, a rate limit, or an empty reply -- deepseek-v4-pro returned one on the pod and it
                # ended the whole talk, because the error escaped before any of the parsing fallbacks below
                # could run. A model that says nothing once usually says something on the next attempt.
                log.warning("character turn with %s failed: %s", model, exc)
                failure = exc
                continue
            call = await self._add_usage(session_id, model, usage)
            turn = cast_talk.parse_turn(text)
            if turn is not None:
                return record(turn, call)
            messages.append({"role": "assistant", "content": text[:2000]})
            messages.append({"role": "user", "content": 'Reply with only the JSON object: {"say": "...", "to": "..."}'})
        if failure is not None and not text:
            raise failure
        # Both strict attempts are spent. Read the last reply as plain speech rather than dropping the turn:
        # a character that says nothing looks to the user exactly like a broken chat.
        salvaged = cast_talk.spoken_anyway(text, names)
        return record(salvaged, call) if salvaged is not None else None

    def _transcript(self, session: dict, member_id: str, names: list[str], upto: int | None = None) -> str:
        """The chat as one character heard it, after its memory (whispers to others left out), newest last."""
        cast = cast_of(session)
        member = next((m for m in cast if m.get("id") == member_id), {})
        lines = []
        for message in self.store.messages(session["id"], after=member.get("memory_upto", 0)):
            if upto is not None and message["id"] > upto:
                break
            if not visible_to(message, member_id):
                continue
            lines.extend(self._script_lines(message, member_id))
        return "\n".join(lines[-60:])

    def _script_lines(self, message: dict, member_id: str) -> list[str]:
        content, role = message["content"], message["role"]
        if role == "user":
            text = content.get("text", "")
            files = content.get("attachments") or []
            if files:
                text += " [shared " + ", ".join(f"{f['kind']} {f['asset_id']}" for f in files) + "]"
            return [f"User (whispering only to you): {text}" if content.get("private_to") == member_id else f"User: {text}"]
        if role == "assistant" and not content.get("invalid"):
            if content.get("lines"):
                return [f"{line.get('speaker') or 'Agent'}: {line.get('say', '')}" for line in content["lines"]]
            return [content["say"]] if content.get("say") else []
        if role == "tool" and content.get("ok") and content.get("tool") in ("generate_image", "render_film", "wait_for_job", "set_avatar"):
            brief = brief_result(content.get("result"))
            ids = [a.get("id") for a in (brief.get("assets") or []) if isinstance(a, dict)] if isinstance(brief, dict) else []
            what = f"images {', '.join(ids)}" if ids else json.dumps(brief, ensure_ascii=False)[:200]
            return [f"[{content['tool']}: {what}]"]
        if role == "note" and content.get("kind") == "make":
            # The characters read this line back, so it has to say who took it and who is in it: a selfie
            # and a candid of someone else are different events to react to.
            of = ", ".join(str(n) for n in (content.get("of") or []) if n)
            kind = {"selfie": "took a selfie", "snap": f"took a photo of {of}" if of else "took a photo",
                    "group_shot": f"took a group photo of {of}" if of else "took a group photo",
                    "share": "shared a picture"}.get(content.get("act"), "asked for this to be made")
            return [f"[{content.get('speaker')} {kind}: {content.get('text', '')}]"]
        return []

    def _resolve_owner(self, session: dict, cast: list[dict], names: list[str], who: str) -> dict | None:
        """A name (a character's, or the user's) as an owner record."""
        key = (who or "").strip().lstrip("@").lower()
        if not key:
            return None
        if key in USER_OWNER_WORDS:
            return {"name": "You", "member": USER_KEY, "session": session["id"]}
        index = find_member(cast, who)
        if index is None:
            return None
        return {"name": names[index], "member": cast[index].get("id", ""), "session": session["id"]}

    def _attribution(self, session: dict, action: dict, asked_by: str = "",
                     subjects: tuple[str, ...] = ()) -> dict:
        """Who took this, and who is in it.

        "by" is the photographer and "of" the subjects. They are the same person for a selfie and opposite
        ones for a candid, and conflating them is how "Maya, take a photo of Riya" ended up as Riya's photo.

        by: marked on the call -> the character who asked for it -> the one character the prompt names -> nobody.
        of: marked on the call -> the subjects the caller already knows -> everyone the prompt names.

        Whoever asked outranks a name in the description, because asking is evidence of authorship and being
        described is evidence of being photographed. Nobody is invented: an owner we cannot work out is left
        unset rather than drawn at random, since a guess is indistinguishable from a fact once it is stored.
        """
        cast = cast_of(session)
        names = [display_name(member, index, len(cast)) for index, member in enumerate(cast)]
        # Only the fields a person's name would actually appear in: the whole args blob also carries lora
        # filenames, negatives and engine ids, and "riya_v2.safetensors" is not a mention of Riya.
        args = action.get("args") or {}
        text = " ".join(str(args.get(field) or "") for field in SUBJECT_FIELDS)
        named = cast_talk.mentioned_all(text, names)

        of: list[dict] = []
        for who in (action.get("of") or ()) or subjects or ():
            found = self._resolve_owner(session, cast, names, str(who))
            if found and not any(p["member"] == found["member"] for p in of):
                of.append({"name": found["name"], "member": found["member"]})
        if not of:
            of = [{"name": names[i], "member": cast[i].get("id", "")} for i in named]

        by = self._resolve_owner(session, cast, names, str(action.get("by") or ""))
        how = "marked" if by else ""
        if by is None and asked_by:
            by, how = self._resolve_owner(session, cast, names, asked_by), "asked"
        if by is None and len(named) == 1:
            by, how = self._resolve_owner(session, cast, names, names[named[0]]), "named"
        if by is None and len(named) > 1:
            # A picture of several of them was taken by one of them: whose phone it was is genuinely
            # arbitrary, so the draw is the answer rather than a stand-in for one.
            by, how = self._resolve_owner(session, cast, names, names[random.choice(named)]), "group"
        if by:
            by["how"] = how
        return {"by": by, "of": of[:cast_talk.MAX_SUBJECTS]}

    def _owner_for(self, session: dict, action: dict, asked_by: str = "",
                   subjects: tuple[str, ...] = ()) -> dict | None:
        """Who owns what this call makes. See _attribution."""
        return self._attribution(session, action, asked_by, subjects)["by"]

    def _take_ownership(self, session: dict, action: dict, result, asked_by: str = "",
                        subjects: tuple[str, ...] = ()) -> dict | None:
        """Mark the assets (and the render job) a call produced with who took them and who is in them."""
        if not isinstance(result, dict):
            return None
        assets = [a for a in result.get("assets") or [] if a.get("id") or a.get("asset_id")]
        job = result["id"] if result.get("kind") in ("plan", "render") and result.get("id") else None
        if not (assets or job):  # a call that made nothing (inspect_image, list_references) belongs to nobody
            return None
        found = self._attribution(session, action, asked_by, subjects)
        owner, of = found["by"], found["of"]
        if not owner and not of:
            return None
        for asset in assets:
            try:
                self.service.update_asset(asset.get("id") or asset["asset_id"], owner=owner, of=of)
                if owner:
                    asset["by"] = owner
                if of:
                    asset["of"] = of
            except (NotFound, RequestError):
                continue
        if job and owner:
            self.service.set_job_owner(job, owner)
        return owner


    def _subjects_for(self, cast: list[dict], names: list[str], speaker_index: int, turn: dict) -> list[str]:
        """Who is in what this character asked for, resolved to real cast names.

        Named outright if they said so, else whoever the description names. A selfie is of the speaker; a
        snap deliberately is not, because a candid is by one person and of another -- the case guessing from
        the description always got backwards.
        """
        found = []
        for who in turn.get("of") or ():
            index = find_member(cast, str(who))
            if index is not None and names[index] not in found:
                found.append(names[index])
        if not found:
            found = [names[i] for i in cast_talk.mentioned_all(turn.get("make") or "", names)]
        act = turn.get("act")
        if act == "selfie" and names[speaker_index] not in found:
            found.insert(0, names[speaker_index])
        elif act == "group_shot" and not found:
            found = list(names)
        return found[:cast_talk.MAX_SUBJECTS]

    def _avatars_for(self, session: dict, names: list[str]) -> list[str]:
        """(name, avatar asset id) for each named character that has one, and the names of those that do not.

        Both halves are returned because a character without an avatar used to be dropped in silence: a group
        shot of two would go to the engine with one reference, come back with one likeness and the other face
        invented, and nothing anywhere said why. The caller names the ones it holds and says which it does not.
        """
        cast = cast_of(session)
        found: list[tuple[str, str]] = []
        without: list[str] = []
        for name in names:
            index = find_member(cast, name)
            if index is None:
                continue
            avatar = str(cast[index].get("avatar_asset_id") or "")
            if not avatar:
                without.append(name)
            elif not any(avatar == held for _, held in found):
                found.append((name, avatar))
        return found[:cast_talk.MAX_SUBJECTS], without[:cast_talk.MAX_SUBJECTS]

    async def _snap_for(self, session_id: str, speaker: str, request: str, act: str, of: tuple[str, ...]) -> None:
        """A character took a picture: make it straight away, without waking the director.

        A turn that says what it is making and who is in it has already answered everything the director
        would have reasoned out, so this calls the image engine itself. That is what lets it run beside the
        conversation: the director rewrites the chat's summary as it works, and a second one doing that while
        the characters are still talking would lose whichever of the two saved the session last. This writes
        no session state at all -- only messages, which are append-only.
        """
        session = self.get_session(session_id)
        faces, without = self._avatars_for(session, list(of))
        refs = [asset_id for _, asset_id in faces]
        # Nothing labels a reference for the engine, so the prompt has to: see cast_talk.reference_key.
        request = cast_talk.reference_key(request, faces, without)
        if without:
            self._note(session_id, f"No avatar for {', '.join(without)}, so "
                       + ("her face was" if len(without) == 1 else "their faces were")
                       + " drawn from the description rather than kept. Set one with set_avatar.", "warn")
        action = {"tool": "generate_image", "args": {"prompt": request, "reference_asset_ids": refs},
                  "by": speaker, "of": list(of)}
        try:
            result = await self.service.generate_images(request, reference_asset_ids=refs, engine="auto")
            ok, error = True, None
        except Exception as exc:
            result, ok, error = None, False, str(exc) or type(exc).__name__
        content = {"tool": "generate_image", "args": action["args"], "ok": ok,
                   "result": result, "error": error, "act": act}
        if ok:
            owner = self._take_ownership(self.get_session(session_id), action, result, speaker, tuple(of))
            if owner:
                content["by"] = owner
        self.store.add_message(session_id, "tool", content)

    def _spawn_make(self, session_id: str, speaker: str, request: str, act: str, of: tuple[str, ...]) -> None:
        """Start making it and let the conversation carry on. _run waits for these before it goes idle."""
        task = asyncio.create_task(self._snap_for(session_id, speaker, request, act, of))
        self._makes.setdefault(session_id, set()).add(task)
        task.add_done_callback(lambda t: self._makes.get(session_id, set()).discard(t))

    async def _make_for(self, session_id: str, speaker: str, request: str, *, act: str = "",
                        of: tuple[str, ...] = ()) -> None:
        """A character asked for something to be made: the director, with its tools, makes it and presents it.

        The speaker and the subjects travel with the request, so what comes back is attributed from what the
        character actually said rather than from guessing at names in the description.
        """
        note = {"text": request, "kind": "make", "speaker": speaker}
        if act:
            note["act"] = act
        if of:
            note["of"] = list(of)
        self.store.add_message(session_id, "note", note)
        await self._director_turn(session_id, MAKE_STEPS, asked_by=speaker, subjects=tuple(of))

    async def _remember(self, session: dict, member_id: str, force: bool = False) -> bool:
        """Fold older parts of the chat into one character's own memory (its point of view), with the cheap model."""
        cast = cast_of(session)
        member = next((m for m in cast if m.get("id") == member_id), None)
        if member is None:
            return False
        heard = [m for m in self.store.messages(session["id"], after=member.get("memory_upto", 0))
                 if visible_to(m, member_id) and m["role"] in ("user", "assistant", "tool")]
        keep = MANUAL_KEEP_MESSAGES if force else CHARACTER_KEEP_MESSAGES
        if len(heard) <= keep:
            return False
        names = [display_name(m, i, len(cast)) for i, m in enumerate(cast)]
        if not force and len(self._transcript(session, member_id, names)) // 4 < self.compact_tokens // 2:
            return False
        older_upto = heard[-keep - 1]["id"]
        older = self._transcript(session, member_id, names, upto=older_upto)
        name = names[cast.index(member)]
        previous = f"Your earlier memory:\n{member['memory']}\n\n" if member.get("memory") else ""
        # No facts asked for here, on purpose. This ran once per character and owned everything it extracted
        # privately, so one line said aloud became three rows: the shared one and a private copy for each
        # character who heard it. On the pod a three-message chat produced 32 facts where 6 were the content.
        # The private view worth keeping -- how one character feels about another -- is already written for
        # free by _grow, correctly owned, with no model call at all.
        request = [{"role": "system", "content": cast_talk.MEMORY_PROMPT.format(name=name)},
                   {"role": "user", "content": f"{previous}What you heard since:\n{older}"}]
        for model in dict.fromkeys((await self.model_for("summary"), await self.model_for("director", session))):
            try:
                text, usage = await self.atlas.chat(model, request, json_mode=False, max_tokens=SUMMARY_MAX_TOKENS, max_retries=1)
            except AtlasError as exc:
                log.warning("character memory with %s failed: %s", model, exc)
                continue
            await self._add_usage(session["id"], model, usage)
            memory = (text or "").strip()
            fresh = self.get_session(session["id"])
            cast = [dict(m) for m in cast_of(fresh)]
            for entry in cast:
                if entry.get("id") == member_id:
                    entry["memory"], entry["memory_upto"] = memory, older_upto
            fresh["cast"] = cast
            self.store.save_session(fresh)
            return True
        return False

    async def _add_usage(self, session_id: str, model: str, usage: dict) -> dict:
        """Add one model call to the chat's totals; returns that call's usage. The cost is an estimate at Atlas list
        prices, with cached input tokens (a repeated prompt prefix) at the model's cheaper cache-read price."""
        session = self.get_session(session_id)
        info = await self.atlas.model_info(model) or {}
        prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int((details.get("cached_tokens") if isinstance(details, dict) else 0) or usage.get("prompt_cache_hit_tokens") or 0)
        cached = min(cached, prompt)
        # Reasoning tokens are billed as output but are not in the reply, so a chat on a thinking model looks
        # inexplicably expensive until they are shown.
        out_details = usage.get("completion_tokens_details") or {}
        reasoning = int((out_details.get("reasoning_tokens") if isinstance(out_details, dict) else 0) or 0)
        price_in = info.get("price_in", 0.0)
        cost = ((prompt - cached) * price_in + cached * info.get("price_cache", price_in)
                + completion * info.get("price_out", 0.0))
        totals = session.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "steps": 0})
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["cached_tokens"] = totals.get("cached_tokens", 0) + cached
        totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + reasoning
        totals["steps"] = totals.get("steps", 0) + 1
        totals["cost_usd"] = round(totals.get("cost_usd", 0.0) + cost, 6)
        self.store.save_session(session)
        return {"model": model, "in": prompt, "out": completion, "cached": cached,
                "reasoning": reasoning, "cost_usd": round(cost, 6)}

    # ------------------------------------------------------------ tools

    def _tools_in_use(self, session: dict | None) -> set[str]:
        """Tools this chat has used, ever: their full schemas stay in the catalogue.

        Deliberately cumulative. It used to read only the messages after the summary watermark, so a compaction
        could take a tool's schema back *out* of the catalogue -- which is both a cache problem (the prefix
        shrinks, and everything after the change is re-read at full price) and arguably a correctness one, since
        the agent is told to call describe_tool once per chat and would have no way of knowing the answer had
        been withdrawn. The set is kept on the session and only ever grows; a few hundred resident tokens buys a
        prefix that a compaction cannot move.
        """
        if session is None:
            return set()
        seen = {str(name) for name in (session.get("tools_seen") or []) if name}
        before = set(seen)
        for message in self.store.messages(session["id"]):
            if message["role"] == "tool":
                content = message["content"]
                if content.get("tool"):
                    seen.add(str(content["tool"]))
                if content.get("tool") == "describe_tool":
                    named = str((content.get("args") or {}).get("name") or "")
                    if named:
                        seen.add(named)
        if seen != before:
            session["tools_seen"] = sorted(seen)
            self.store.save_session(session)
        return seen

    async def _catalog(self, session: dict | None = None) -> str:
        """Every tool with its description. Large argument schemas appear only once the chat uses the tool, so a chat
        about images doesn't pay for render_film's schema on every call (session None: everything in full)."""
        used = self._tools_in_use(session)
        lines = []
        for tool in await self.mcp.list_tools():
            schema = json.dumps(_compact_schema(tool.input_schema), ensure_ascii=False, separators=(",", ":"))
            if session is not None and len(schema) // 4 > BIG_SCHEMA_TOKENS and tool.name not in ALWAYS_FULL_TOOLS | used:
                lines.append(f"- {tool.name}: {tool.description}\n  args: large; call describe_tool {{\"name\": \"{tool.name}\"}} "
                             "once before using it")
                continue
            lines.append(f"- {tool.name}: {tool.description}\n  args JSON schema: {schema}")
        for tool in AGENT_TOOLS:
            lines.append(f"- {tool['name']}: {tool['description']}\n  args: {json.dumps(tool['args'])}")
        return "\n".join(lines)

    async def _step_up_engine(self, session_id: str, args: dict) -> str | None:
        """The engine for a generate_image call with engine "auto" once a take failed inspection this run: the next
        rung above the highest one that failed, on the ladder for what this call is doing."""
        failed = self._failed_engines.get(session_id)
        if not failed or str(args.get("engine") or args.get("model") or "auto").strip().lower() != "auto":
            return None
        ladder = await self.service.ready_image_engines("edit" if args.get("reference_asset_ids") else "generate")
        if args.get("loras"):  # LoRAs (adult ones included) only run on the local engines, so stay among those
            ladder = [engine for engine in ladder if image_engines.get(engine).lora_family]
            # ... and only those that hold these particular files: stepping up onto an engine of another
            # family refuses the retake outright rather than making a better picture.
            try:
                families = await self.service.local_images.families_for_loras(
                    [(spec or {}).get("name") if isinstance(spec, dict) else spec for spec in args["loras"]])
            except Exception:
                families = None
            if families is not None:
                ladder = [engine for engine in ladder if image_engines.get(engine).lora_family in families]
        if not ladder:
            return None
        # an id this ladder doesn't hold (an older chat, or an engine since switched off) is ignored, not a crash
        top = max((ladder.index(engine) for engine in failed if engine in ladder), default=-1)
        stepped = ladder[min(top + 1, len(ladder) - 1)]
        if stepped == ladder[0]:
            return None  # no step up means nothing to say
        if not image_engines.get(stepped).local and self.service.image_engines.confirm_paid():
            # Retrying on a paid engine spends the user's money on a judgement call the model made by
            # itself. It has to show what it already has and be told to go ahead.
            return None
        return stepped

    async def _paid_rung(self, session_id: str, args: dict) -> str | None:
        """The paid engine a failed take would have escalated to, when confirmation is holding it back."""
        failed = self._failed_engines.get(session_id)
        if not failed or not self.service.image_engines.confirm_paid():
            return None
        ladder = await self.service.ready_image_engines("edit" if args.get("reference_asset_ids") else "generate")
        top = max((ladder.index(engine) for engine in failed if engine in ladder), default=-1)
        if top < 0 or top + 1 >= len(ladder):
            return None
        nxt = ladder[top + 1]
        return nxt if not image_engines.get(nxt).local else None

    def _retake_args(self, asset_ids: list[str]) -> dict:
        """What another take of these images would be, read off the assets themselves.

        _step_up_engine picks the ladder from its args -- "edit" when there are references, local-only when
        there are LoRAs -- and inspect_image is told none of that, only which images to look at. Inventing
        {"engine": "auto"} therefore sent every edit down the generate ladder, where it could be offered
        z-image/turbo, which takes no references at all. Generated assets record what made them.
        """
        args: dict = {"engine": "auto"}
        refs: list[str] = []
        loras: list = []
        for asset_id in asset_ids:
            source = (self.service.store.get_asset(asset_id) or {}).get("source") or {}
            for ref in source.get("references") or []:
                if str(ref) not in refs:
                    refs.append(str(ref))
            loras = loras or list(source.get("loras") or [])
        if refs:
            args["reference_asset_ids"] = refs
        if loras:  # adult LoRAs included: only the local engines run them, so a retake must stay there
            args["loras"] = loras
        return args

    def _record_takes(self, session_id: str, verdict: dict) -> bool:
        """A batch that inspection rejects (every image "retry", or the best below PASS_SCORE) marks its engine failed.

        Returns whether it was rejected, because that is also the moment the user gets to choose.
        """
        images = [item for item in verdict.get("images") or [] if isinstance(item, dict)]
        if not images:
            return False
        scores = [float(item["score"]) for item in images if isinstance(item.get("score"), (int, float))]
        rejected = all(str(item.get("verdict", "")).lower() == "retry" for item in images) or (scores and max(scores) < PASS_SCORE)
        if not rejected:
            return False
        for item in images:
            asset = self.service.store.get_asset(str(item.get("asset_id") or ""))
            source = (asset or {}).get("source") or {}
            # Images made from now on say which engine made them; older ones are traced from the model name.
            engine = source.get("engine") or image_engines.id_for_generator(str(source.get("generator") or ""))
            if engine:
                self._failed_engines.setdefault(session_id, set()).add(engine)
        return True

    async def _execute(self, session_id: str, action: dict, asked_by: str = "",
                       subjects: tuple[str, ...] = ()) -> None:
        name, args = action["tool"], action["args"]
        stepped = await self._step_up_engine(session_id, args) if name == "generate_image" else None
        if stepped:
            args = {**args, "engine": stepped}
        try:
            if name == "generate_image" and session_id in self._awaiting_choice:
                # A model that argues with the gate would otherwise spend the whole step budget on refusals.
                self._awaiting_choice[session_id]["blocked"] = True
                raise RequestError(
                    "Inspection rejected the last take and the user has not chosen yet. Show them the images from "
                    "inspect_image, say what is wrong and what another take would cost, and end your turn with "
                    '"done": true. This tool works again as soon as they reply.')
            if name == "wait_for_job":
                result = await self._wait_for_job(session_id, args)
            elif name == "describe_tool":
                result = await self._describe_tool(args)
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
        if stepped and ok and isinstance(result, dict):
            result = {**result, "engine_note": f"An earlier take failed inspection, so engine auto used {stepped} this time."}
        owner = self._take_ownership(self.get_session(session_id), action, result, asked_by, subjects) if ok else None
        content = {"tool": name, "args": args, "ok": ok, "result": result if ok else None,
                   "error": None if ok else result.get("_error")}
        if owner:
            content["by"] = owner
        job_id = result.get("id") if ok and isinstance(result, dict) and result.get("kind") in ("plan", "render") else None
        if job_id:
            content["job_id"] = job_id
        self.store.add_message(session_id, "tool", content)

    async def _describe_tool(self, args: dict) -> dict:
        name = str(args.get("name") or "").strip()
        for tool in await self.mcp.list_tools():
            if tool.name == name:  # _tools_in_use sees this call, so the catalogue shows the schema from the next step on
                return {"name": name, "note": "Its full args JSON schema is now in your tool catalogue."}
        raise RequestError(f"No tool called {name!r}.")

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
        """Stored messages after the summary, as chat messages (same-role runs merged). Turns before the user's latest
        message are sent trimmed (free, no model call): tool results keep ids, links, outcomes and verdicts, and long
        action arguments such as image prompts are cut; the current turn is sent in full."""
        chat: list[dict] = []
        stored = self.store.messages(session["id"], after=session.get("summary_upto", 0))
        current = max((index for index, m in enumerate(stored) if m["role"] == "user"), default=0)

        def push(role: str, text: str) -> None:
            if chat and chat[-1]["role"] == role:
                chat[-1]["content"] += "\n\n" + text
            else:
                chat.append({"role": role, "content": text})

        for index, message in enumerate(stored):
            content, role = message["content"], message["role"]
            old = index < current
            if role == "user":
                files = content.get("attachments") or []
                note = ("\nAttached assets: " + ", ".join(f"{f['asset_id']} ({f['kind']}: {f['filename']})" for f in files)) if files else ""
                whisper = content.get("private_to")
                if whisper:
                    to = self._about_name(cast_of(session), whisper)
                    push("user", f"USER (whispering privately to {to}: only {to} hears this and only {to} answers, in a private "
                                 f"line; the others must not learn it from anyone but {to}): {content.get('text', '')}{note}")
                else:
                    push("user", f"USER: {content.get('text', '')}{note}")
            elif role == "assistant":
                raw = content.get("raw") or json.dumps({k: content.get(k) for k in ("say", "actions", "done")})
                if old and len(raw) > 1500 and not content.get("invalid"):
                    reply = {k: v for k, v in content.items() if k not in ("raw", "usage")}
                    reply["actions"] = [{**action, "args": _shorten(action.get("args") or {}, frozenset(), BRIEF_STRING_CHARS)}
                                        for action in reply.get("actions") or []]
                    raw = json.dumps(reply, ensure_ascii=False)
                push("assistant", raw)
            elif role == "tool":
                if not content.get("ok"):
                    body = f"ERROR: {str(content.get('error'))[:BRIEF_STRING_CHARS] if old else content.get('error')}"
                elif old:
                    body = json.dumps(brief_result(content["result"]), ensure_ascii=False) or "{}"
                else:
                    body = compact_result(content["result"])
                push("user", f"TOOL RESULT {content['tool']}: {body}")
            elif role == "note" and content.get("kind") == "make":
                latest = not any(m["role"] == "assistant" for m in stored[index + 1:])
                who = content.get("speaker") or "A character"
                if latest:
                    push("user", f"(STAGE DIRECTION, not the user: while the characters talk, {who} asked for this to be made: "
                                 f"{content.get('text', '')}. Make it now with your tools (the characters' avatars as picture "
                                 f"references where they appear, and \"by\": \"{who}\" on the calls so it is {who}'s own), "
                                 f"then reply with one short line from {who} presenting it, with \"done\": true.)")
                else:
                    push("user", f"(Earlier, {who} asked for this to be made: {content.get('text', '')})")
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
        # The chat's own model first when it can see, then the vision role. That role used to be a hardcoded
        # pair of Atlas ids, so on any other provider both 404ed and a text-only chat could not inspect at all.
        chat_model = await self.model_for("director", session)
        models = [chat_model] if ((await self.atlas.model_info(chat_model)) or {}).get("vision") else []
        models += [m for m in await self.models_for("vision", session) if m not in models]
        failures = []
        for model in models:
            try:
                text, usage = await self.atlas.chat(model, [{"role": "user", "content": parts}], json_mode=True,
                                                    max_tokens=INSPECT_MAX_TOKENS, temperature=0.2, max_retries=1)
            except AtlasError as exc:
                failures.append(f"{model}: {exc}")
                continue
            await self._add_usage(session_id, model, usage)
            verdict = parse_json_object(text)
            if not isinstance(verdict, dict) or not isinstance(verdict.get("images"), list):
                failures.append(f"{model}: no usable verdict ({text.strip()[:160]!r})")
                continue
            images = [item for item in verdict["images"] if isinstance(item, dict)]
            for index, item in enumerate(images):  # some models echo a made-up id; the images come back in the order sent
                if str(item.get("asset_id")) not in ids and index < len(ids):
                    item["asset_id"] = ids[index]
            if str(verdict.get("best")) not in ids and images:
                best = max(images, key=lambda item: item.get("score") if isinstance(item.get("score"), (int, float)) else 0)
                verdict["best"] = best["asset_id"]
            verdict["images"] = images
            rejected = self._record_takes(session_id, verdict)
            result = {"model": model, **verdict}
            if failures:
                result["skipped_models"] = failures
            step_args = self._retake_args(ids)
            nxt = await self._step_up_engine(session_id, step_args)
            paid = None
            if nxt:
                result["next_engine"] = f"The next generate_image with engine auto will use {nxt}."
            else:
                paid = await self._paid_rung(session_id, step_args)
                if paid:
                    spec = image_engines.get(paid)
                    price = IMAGE_PRICES.get(spec.price_key, 0.0)
                    result["needs_approval"] = (
                        f"The free engines on this GPU have been tried. The next one is {spec.label}, which costs "
                        f"about ${price:g} an image. Show the user the best take you already have, say what is wrong "
                        f"with it, and ask whether to spend that or try locally again. Only call generate_image with "
                        f'engine "{paid}" once they have said yes; otherwise retry with engine "auto".')
            if rejected and self.service.image_engines.pick_takes():
                result["choose"] = self._offer_choice(session_id, verdict, nxt, paid)
            return result
        log.warning("inspect_image failed on every model: %s", failures)
        raise RequestError("No vision model could inspect these images (" + "; ".join(f[:300] for f in failures) + "). "
                           "Don't regenerate only because inspection is unavailable: show the images and let the user judge. "
                           "For a retake without LoRAs use engine \"seedream\"; LoRA images (adult ones included) only come "
                           "from the local engines.")

    def _offer_choice(self, session_id: str, verdict: dict, stepped: str | None, paid: str | None) -> dict:
        """What inspection rejected, handed back to the user to decide on.

        Another take costs either time on this GPU or money at Atlas, and inspection is a model's opinion, not
        the user's -- so a rejected batch stops here and they pick from what exists. Until they answer, the gate
        in ``_execute`` refuses another generate_image, because a prose instruction alone is one the model can
        talk itself out of.
        """
        assets = []
        for item in verdict.get("images") or []:
            asset = self.service.store.get_asset(str(item.get("asset_id") or ""))
            if asset is None:
                continue
            shown = self.service.asset_view(asset)
            assets.append({"asset_id": asset["id"], "score": item.get("score"),
                           "verdict": item.get("verdict"), "issues": item.get("issues") or [],
                           "thumb_url": shown.get("thumb_url"), "file_url": shown.get("file_url")})
        options = [{"id": "keep", "label": "Use this one",
                    "reply": "Keep image {asset_id} as it is. Don't take it again."}]
        if len(assets) > 1:
            # Judging a whole batch one take at a time is the tedious part; keeping them all is one answer.
            options.append({"id": "keep_all", "label": f"Keep all {len(assets)}",
                            "reply": "Keep all of these images as they are: {asset_ids}. Don't take them again."})
        if stepped:
            options.append({"id": "retry", "label": f"Try again on {image_engines.get(stepped).label}",
                            "reply": 'Try again with engine "auto".'})
        elif not paid:
            options.append({"id": "retry", "label": "Try again", "reply": 'Try again with engine "auto".'})
        if paid:
            spec = image_engines.get(paid)
            price = IMAGE_PRICES.get(spec.price_key, 0.0)
            options.append({"id": paid, "label": f"Spend about ${price:g} on {spec.label}",
                            "reply": f'Go ahead and use engine "{paid}" for the next take.'})
        self._awaiting_choice[session_id] = {"best": str(verdict.get("best") or "")}
        return {"assets": assets, "best": str(verdict.get("best") or ""), "options": options,
                "note": "Stop here with \"done\": true. Show the user these images, say what is wrong with them and "
                        "what each choice costs, and wait. Another generate_image will be refused until they answer."}

    def _edit_character(self, session_id: str, tool: str, args: dict) -> dict:
        session = self.get_session(session_id)
        cast = [dict(member) for member in cast_of(session)]
        speaker = str(args.get("speaker") or "").strip()
        name = str(args.get("name") or "").strip()
        if not speaker and name and tool != "remove_character" and find_member(cast, name) is not None:
            speaker = name  # a character named without speaker: edit that one, not the lead
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
                    names = ", ".join(display_name(m, i, len(cast)) or "(unnamed lead)" for i, m in enumerate(cast))
                    raise RequestError(f"No character called {speaker!r} (the cast is: {names}); add them with set_persona first.")
                if not display_name(cast[0], 0, 1):
                    # The chat's lead has no name yet (a fresh chat): the first named character takes its place
                    # instead of joining next to an unnamed "character 1".
                    cast[0] = {"id": cast[0].get("id") or uuid.uuid4().hex[:8], "name": speaker, "persona": "",
                               "avatar_asset_id": "", "growth": [], "feelings": {}}
                    index = 0
                else:
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

    async def _grow(self, session_id: str, notes: list[dict]) -> None:
        """Adaptive chats: add what a character just became (about "self") to its growth notes, or how it now feels
        about the user or another character to its private feelings (group chats)."""
        session = self.get_session(session_id)
        if not session.get("adaptive"):
            return
        cast = [dict(member) for member in cast_of(session)]
        added, merge = [], []
        for item in notes:
            index = find_member(cast, item["speaker"]) if item.get("speaker") else 0
            if index is None:
                if len(cast) > 1:
                    continue
                index = 0
            member = cast[index]
            about = str(item.get("about") or "self").strip()
            key = None
            if len(cast) > 1 and about.lower() not in ("self", "me", "myself", ""):
                target = find_member(cast, about)
                key = USER_KEY if about.lower() in ("user", "you", "the user") else (cast[target]["id"] if target not in (None, index) else None)
            if key is None:
                before = member.get("growth") or []
                member["growth"] = clean_growth(before + [item["note"]])
                if member["growth"] != before:
                    added.append((member.get("id", ""), display_name(member, index, len(cast)), item["note"], ""))
                    if len(member["growth"]) >= MERGE_GROWTH_AT:
                        merge.append((member["id"], None))
                continue
            feelings = clean_feelings(member.get("feelings"))
            before = feelings.get(key, [])
            feelings = clean_feelings({**feelings, key: before + [item["note"]]})
            if feelings.get(key) != before:
                member["feelings"] = feelings
                added.append((member.get("id", ""), display_name(member, index, len(cast)), item["note"],
                              self._about_name(cast, key)))
                if len(feelings[key]) >= cast_talk.MAX_FEELINGS:
                    merge.append((member["id"], key))
        if not added:
            return
        updated = self.update_session(session_id, cast=cast)
        # The same note, as a relation, for nothing: it already carries who felt it and about whom. Written here
        # rather than at compaction so the graph is useful before a chat is ever long enough to compact, and so
        # the relation outlives the prose note that _merge_growth will eventually fold away.
        self._seed_graph(updated)
        for member_id, speaker, note, about in added:
            # Owned by the character it belongs to, never by "" (said aloud). _grow keeps these in that
            # character's own feelings and growth, and the other characters' prompts do not show them: writing
            # them as public facts would hand Riya's private view of Tiya straight to Tiya through recall.
            # Only a feeling about someone else becomes a fact. A note about who a character is becoming has
            # nobody on the other end of it, so there is no relation for it to be -- and it is already carried,
            # in full, by that character's own growth notes in the prompt.
            if about:
                self._write_fact(updated, member_id, speaker, "feels_toward", about, f"{speaker} on {about}: {note}")
        for _, speaker, note, about in added:
            content = {"text": note, "kind": "grow", "speaker": speaker}
            if about:
                content["about"] = about
            self.store.add_message(session_id, "note", content)
        for member_id, key in merge:
            await self._merge_growth(session_id, member_id, key)

    async def _merge_growth(self, session_id: str, member_id: str, key: str | None = None) -> None:
        """Fold a character's growth notes (or its feelings about one person, key) into a few denser ones, so early
        shifts aren't pushed out by later ones. Written by the cheap summary model."""
        session = self.get_session(session_id)
        cast = [dict(member) for member in cast_of(session)]
        member = next((m for m in cast if m.get("id") == member_id), None)
        if member is None:
            return
        name = display_name(member, cast.index(member), len(cast)) or "the agent"
        notes = member.get("growth") or [] if key is None else clean_feelings(member.get("feelings")).get(key, [])
        limit = MERGED_GROWTH if key is None else cast_talk.MERGED_FEELINGS
        about = "" if key is None else f" These notes are how {name} feels about {self._about_name(cast, key)}."
        listing = "\n".join(f"- {note}" for note in notes)
        request = [{"role": "system", "content": MERGE_GROWTH_PROMPT.format(limit=limit)},
                   {"role": "user", "content": f"Character: {name}. Persona: {member.get('persona', '')}{about}\n\nGrowth notes, oldest first:\n{listing}"}]
        text = None
        for model in dict.fromkeys((await self.model_for("summary"), await self.model_for("director", session))):
            try:
                text, usage = await self.atlas.chat(model, request, json_mode=True, max_tokens=3000, temperature=0.2, max_retries=1)
            except AtlasError as exc:
                log.warning("merging growth notes with %s failed: %s", model, exc)
                continue
            await self._add_usage(session_id, model, usage)
            break
        merged = clean_growth((parse_json_object(text or "") or {}).get("notes"))[:limit]
        if not merged:
            return  # the notes stay as they are; clean_growth / clean_feelings still cap them
        if key is None:
            member["growth"] = merged
        else:
            member["feelings"] = {**clean_feelings(member.get("feelings")), key: merged}
        self.update_session(session_id, cast=cast)
        what = "growth" if key is None else f"feelings about {self._about_name(cast, key)}"
        self.store.add_message(session_id, "note", {"text": f"Condensed {name}'s {what} into {len(merged)} notes.",
                                                    "kind": "grow-merge", "speaker": name})

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
            "- When a character really changes, add \"grow\": [{\"speaker\": \"<name>\", "
            + ("\"about\": \"self\" | \"user\" | \"<another character's name>\", " if group else "")
            + "\"note\": \"one short sentence\"}] to your reply" + (" (about: who the feeling is towards; self for who they "
            "are becoming)" if group else "") + ". A note records who they are becoming, not what happened: a feeling, an attitude towards the user "
            "or another character, a habit, a preference. Good: \"Feels sidelined by the user and hides it behind jokes.\" "
            "Bad: \"Sent the recce list.\" Most replies need no note; roughly one per character per conversation, only "
            "when something shifts. Never for small talk, plans or tasks.\n"
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
        # Growth is deliberately NOT here. It changes almost every turn, and anything inside the persona block
        # sits near the top of a ~6k prompt that is otherwise identical between calls: one word of growth moving
        # invalidates the provider's cache for everything after it. It goes in the CHARACTER STATE message that
        # _build_messages puts below the stable prefix instead.
        text = f"{persona}\n\n{' '.join(facts)}".strip() if persona or facts else ""
        return text + self._adaptive_rules(False) if session.get("adaptive") else text

    def _cast_text(self, session: dict, cast: list[dict]) -> str:
        view = self.public(session)
        people = []
        for member in view["cast"]:
            face = (f"Avatar: asset {member['avatar_asset_id']}." if member["avatar_url"]
                    else "No avatar yet (when the user asks to see them, generate_image a picture, then set_avatar with speaker).")
            # Growth and private feelings are not here on purpose: see _state_text. They change nearly every
            # turn in an adaptive chat, and this block is the part of the prompt worth keeping byte-identical.
            people.append(f"- {member['display_name']}: {member['persona'] or 'no persona yet'} {face}")
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
            "- Mark who does a tool call with \"by\": \"<name>\" in the action: what it makes belongs to that character, and the library shows it as theirs. A picture or video of a character uses that character's avatar as a picture reference (role picture); a scene with several characters uses all their avatars, and one of them owns it.\n"
            "- set_persona / set_avatar / remove_character take speaker to act on one character.\n"
            "- Private feelings belong to one character: that character acts on them, but the others don't know them "
            "unless they have been said aloud in the chat."
            + ("\n- Whispers are on: a message marked as whispered to one character is heard only by that character. Only "
               "they answer, and nobody else refers to it unless that character tells them." if session.get("whispers") else "")
            + (self._adaptive_rules(True) if session.get("adaptive") else "")
        )

    def _recent_text(self, session: dict) -> str:
        """The last few messages as plain words, only to decide which names this turn is about."""
        parts = []
        for message in self.store.messages(session["id"])[-4:]:
            content = message["content"]
            if isinstance(content, dict):
                parts.append(str(content.get("text") or content.get("say") or ""))
                for line in content.get("lines") or []:
                    parts.append(str((line or {}).get("say") or ""))
        return " ".join(part for part in parts if part)[-4000:]

    def _state_text(self, session: dict) -> str:
        """How the characters have grown and what they privately feel, as its own block.

        This used to live inside {{PERSONA}}, at the top of a prompt that is otherwise the same on every call of
        a chat. Providers cache a prompt by its leading bytes and stop at the first one that differs, so a single
        growth note landing there threw away the cached ~6k prefix and the whole instruction block was re-read at
        full price -- up to eight times per user message. Below the stable prefix it invalidates nothing above it.
        """
        cast = cast_of(session)
        group = len(cast) > 1
        blocks: list[str] = []
        if group:
            view = self.public(session)
            for member, raw in zip(view["cast"], cast):
                grown = self._growth_text(raw)
                felt = "; ".join(f"about {self._about_name(cast, key)}: " + " ".join(notes)
                                 for key, notes in clean_feelings(raw.get("feelings")).items())
                if not grown and not felt:
                    continue
                line = f"- {member['display_name']}:"
                if grown:
                    line += f" {grown}"
                if felt:
                    line += f" Private feelings ({member['display_name']} only): {felt}."
                blocks.append(line)
        else:
            grown = self._growth_text(cast[0]).replace("they have", "you have")
            if grown:
                blocks.append(grown)
        if not blocks:
            return ""
        return "CHARACTER STATE (the newest part of who they are; it changes as the chat goes on)\n" + "\n".join(blocks)

    async def _build_messages(self, session: dict) -> list[dict]:
        system = render_agent_prompt(
            self.service.prompts.get("agent"),
            persona=self._persona_text(session),
            pipeline=INSTRUCTIONS.strip(),
            tools=await self._catalog(session),
        )
        cast = cast_of(session)
        if len(cast) > 1:
            # The editable prompt ends with the single-voice REPLY FORMAT ({"say": ...}); some models follow whatever
            # comes last, so a group chat restates its own format after it and that one wins.
            names = ", ".join(display_name(member, index, len(cast)) for index, member in enumerate(cast))
            system += (
                "\n\nREPLY FORMAT IN THIS GROUP CHAT (this replaces the format above)\n"
                '{"lines": [{"speaker": "<one of: ' + names + '>", "say": "what that character says"}], '
                '"actions": [{"tool": "tool_name", "args": {}}], "done": false}\n'
                "Never reply with \"say\" here: every spoken word belongs to a named character, and a reply with no "
                "\"lines\" reaches the user with no name on it."
            )
        # The order is the whole point, and it is decided by how a prompt cache actually works: a provider
        # matches the longest identical *prefix* and re-reads everything from the first byte that differs. So
        # the stable instruction block comes first, then the summary (which changes only at a compaction), then
        # the history, which only ever grows -- each call extends the last one and the whole thing stays warm.
        #
        # The volatile block goes LAST, below the history, not between the summary and it. Measured on the pod:
        # with it above the history, a turn that changed it re-read the entire conversation and the cached share
        # fell from 98% to 1%. Below the history it invalidates only itself, and being last also puts the facts
        # nearest the model's attention, which is where recall wants to be anyway.
        messages = [{"role": "system", "content": system}]
        if session.get("summary"):
            messages.append({"role": "system", "content": f"SUMMARY OF THE EARLIER CONVERSATION:\n{session['summary']}"})
        history = self._history_messages(session)
        if history and history[0]["role"] == "assistant":
            history.insert(0, {"role": "user", "content": "(continuing the conversation)"})
        state = self._state_text(session)
        recall = self._recall_text(session, "", self._recent_text(session))
        tail = "\n\n".join(part for part in (state, recall) if part)
        return messages + history + ([{"role": "system", "content": tail}] if tail else [])

    async def _maybe_summarize(self, session: dict, force: bool = False) -> bool:
        """Fold older messages into the chat's summary with one call to the cheap summary model: automatically once the
        conversation sent with each call passes compact_tokens, or now (force, the Compact button). The newest messages
        stay word for word; everything stays in the database and in Studio. True when it summarised."""
        stored = self.store.messages(session["id"], after=session.get("summary_upto", 0))
        keep = MANUAL_KEEP_MESSAGES if force else self.keep_messages
        spoken = [index for index, m in enumerate(stored) if m["role"] != "note"]  # notes don't count towards keep
        if len(spoken) <= keep:
            return False
        if not force and estimate_tokens(self._history_messages(session)) < self.compact_tokens:
            return False
        older = stored[: spoken[-keep]]
        lines = []
        for m in older:
            content = m["content"]
            if m["role"] == "tool":
                body = json.dumps(brief_result(content.get("result")), ensure_ascii=False) if content.get("ok") else f"ERROR: {content.get('error')}"
                lines.append(f"TOOL {content.get('tool')}: {body}")
            elif m["role"] == "assistant":
                reply = {k: v for k, v in content.items() if k not in ("raw", "usage", "invalid")}
                lines.append("ASSISTANT: " + json.dumps(_shorten(reply, frozenset(), BRIEF_STRING_CHARS), ensure_ascii=False))
            else:
                lines.append(f"{m['role'].upper()}: {json.dumps(content, ensure_ascii=False)}")
        transcript = "\n".join(lines)
        previous = f"Earlier summary:\n{session['summary']}\n\n" if session.get("summary") else ""
        # The same single call now also returns the facts for the graph. json_mode is on, but nothing depends on
        # it: read_summary_and_facts treats an unparseable reply as the prose summary it always was.
        request = [{"role": "system", "content": SUMMARY_PROMPT + facts_tail()},
                   {"role": "user", "content": previous + transcript[-400_000:]}]
        text = usage = model = None
        for model in dict.fromkeys((await self.model_for("summary"), await self.model_for("director", session))):
            try:
                text, usage = await self.atlas.chat(model, request, json_mode=True, max_tokens=SUMMARY_MAX_TOKENS, max_retries=1)
                break
            except AtlasError as exc:
                log.warning("summary with %s failed: %s", model, exc)
        if text is None:
            if force:
                raise RequestError("Could not summarise the chat right now; try again in a moment.")
            return False
        await self._add_usage(session["id"], model, usage)
        summary, facts = read_summary_and_facts(text)
        fresh = self.get_session(session["id"])
        fresh["summary"] = summary
        fresh["summary_upto"] = older[-1]["id"]
        self.store.save_session(fresh)
        session.update(summary=fresh["summary"], summary_upto=fresh["summary_upto"])
        # owner "" : this is the conversation as everyone heard it, which is what the director knows.
        self._seed_graph(fresh)
        for fact in facts:
            self._write_fact(fresh, "", fact.get("s"), fact.get("r"), fact.get("o"),
                             fact.get("f"), fact.get("v", 1.0), fresh["summary_upto"])
        return True

    async def context_tokens(self, session: dict) -> int:
        """Estimated input tokens of the next model call: instructions, tool catalogue, summary and history."""
        return estimate_tokens(await self._build_messages(session))

    async def compact(self, session_id: str) -> dict:
        """The Compact button: summarise everything but the last few messages now."""
        session = self.get_session(session_id)
        if session_id in self._tasks:
            raise Conflict("The agent is working; compact once it has finished.")
        before = await self.context_tokens(session)
        done = await self._maybe_summarize(session, force=True)
        remembered = []
        cast = cast_of(self.get_session(session_id))
        if len(cast) > 1:  # each character's own memory, used when they talk among themselves
            for index, member in enumerate(cast):
                if await self._remember(self.get_session(session_id), member["id"], force=True):
                    remembered.append(display_name(member, index, len(cast)))
        after = await self.context_tokens(self.get_session(session_id))
        text = (f"Compacted: each call now sends about {after:,} tokens instead of {before:,}." if done
                else "Nothing to compact yet: the chat is already short." if not remembered else "The chat itself is already short.")
        if remembered:
            text += f" Condensed the memories of {', '.join(remembered)}."
        self.store.add_message(session_id, "note", {"text": text, "kind": "compact", "before": before, "after": after})
        return {"session": self.get_session(session_id), "compacted": done or bool(remembered), "before": before,
                "after": after, "memories": remembered}
