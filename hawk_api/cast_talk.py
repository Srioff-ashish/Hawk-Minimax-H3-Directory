"""Group chats where every character is its own speaker ("Let them talk").

Each turn is one small model call for one character: its persona, its growth, its private feelings about the
user and the others, its own memory of older parts of the chat, and the recent conversation it could hear. No
tool catalogue: a character that wants something made asks for it ("make"), and the director (the full agent
with tools) makes it. Whispers: with whispers on, a user message starting with @Name is private to that
character, and so is the reply.
"""

from __future__ import annotations

import json
import re

MAX_FEELINGS = 6  # notes per relationship before they are condensed
MERGED_FEELINGS = 3
USER_KEY = "user"
TURN_WORDS = 60  # a spoken turn stays short, like a real conversation

CHARACTER_TURN_PROMPT = """You are {name}, one of the characters in a group chat. Stay fully in character.

WHO YOU ARE
{persona}{growth}

WHO ELSE IS HERE
{others}
- The user: a real person in this chat.{listening}

YOUR PRIVATE FEELINGS (only you know these; the others can't see them unless you say them aloud)
{feelings}

YOUR MEMORY OF EARLIER IN THIS CHAT (your own point of view)
{memory}

HOW TO SPEAK
- It is your turn. Say one short turn (1-3 sentences, at most about {words} words) as {name} only: never write anyone else's lines.
- Talk to whoever the conversation calls for: usually the other characters. Address the user only when they spoke to you, asked something, or the moment really calls for it.
- React to what was just said, and move the conversation forward: every turn adds something new (an opinion, a question, a story, a tease, a disagreement, a decision). Never repeat or rephrase a point that was already made, by you or anyone else.
- Bring your own opinions, moods and quirks; tease, agree, argue or change the subject the way {name} would.
- If the user asked you all something, work it out among yourselves (argue, compare, persuade) before anyone turns back to the user with an answer.
- Match the chat's language and style (Hinglish in Roman script if that is how it is going).
- If you want an image or video made (a look to try, a photo of yourself, a scene), add "make" with a full description; the director makes it and everyone sees it. Only when it matters to the conversation.
- Add "pause": true only when the conversation has really run its course or cannot go on without the user; not just because someone could ask the user.{adaptive}

Reply with only JSON: {{"say": "...", "to": "<a name, user, or all>"{make_field}{grow_field}, "pause": false}}

{rules}"""

ADAPTIVE_TURN = """
- You can change, gradually and believably. When a moment really shifts how you feel or who you are becoming, add "grow": [{{"about": "self" | "user" | "<a character's name>", "note": "one short sentence from your point of view"}}]. A note records a feeling, attitude, habit or preference, not an event. Good: "Feels Riya always steals the spotlight and pretends not to mind." Bad: "Talked about lehengas." Most turns need none.
- Your core identity never changes: name, age (always an adult), background and the platform rules."""

MEMORY_PROMPT = (
    "You are {name}. Write your own memory of the conversation below, from your point of view, for your future self. "
    "Keep: what happened and who said what that mattered to you, how you felt and feel about the user and each other "
    "character, promises, plans, jokes and running themes, and every asset id and link exactly. You only know what you "
    "heard; don't invent the others' thoughts. First person, plain text, at most 250 words."
)


def clean_feelings(feelings, limit: int = 12, chars: int = 240) -> dict[str, list[str]]:
    """{about: [notes]} with about "user" or a character id; short, no repeats, newest kept."""
    clean: dict[str, list[str]] = {}
    for about, notes in (feelings or {}).items() if isinstance(feelings, dict) else []:
        kept: list[str] = []
        for note in notes if isinstance(notes, list) else []:
            text = " ".join(str(note or "").split())[:chars]
            if text and text.lower() not in (n.lower() for n in kept):
                kept.append(text)
        if kept and str(about).strip():
            clean[str(about).strip()[:40]] = kept[-limit:]
    return clean


def visible_to(message: dict, member_id: str | None) -> bool:
    """Whispers are heard only by the character they were for (member_id None: the director, who hears all)."""
    private = message["content"].get("private_to") if isinstance(message.get("content"), dict) else None
    return not private or member_id is None or private == member_id


def mentioned(text: str, cast: list[dict], names: list[str], exclude: int | None = None) -> int | None:
    """The first character named in text (full name, a first name only they have, or @Name), other than exclude."""
    firsts = [name.split()[0].lower() if name else "" for name in names]
    best = None
    for index, name in enumerate(names):
        if index == exclude or not name:
            continue
        forms = [name]
        if len(firsts[index]) >= 3 and firsts.count(firsts[index]) == 1:
            forms.append(name.split()[0])  # "Sonia" for "Sonia Mausi"
        for form in forms:
            match = re.search(rf"(?<![\w@])@?{re.escape(form)}\b", text or "", re.IGNORECASE)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), index)
    return best[1] if best else None


def next_speaker(cast: list[dict], names: list[str], last_index: int | None, last_text: str, spoke: list[int]) -> int:
    """Who talks next: the character just named or asked; else whoever has waited longest (never the same twice)."""
    named = mentioned(last_text, cast, names, exclude=last_index)
    if named is not None:
        return named
    order = [i for i in range(len(cast)) if i != last_index] or [0]
    waited = {i: (len(spoke) - 1 - max((n for n, s in enumerate(spoke) if s == i), default=-10_000)) for i in order}
    return max(order, key=lambda i: (waited[i], -i))


def parse_turn(text: str) -> dict | None:
    """A character's turn: {"say", "to", "make"?, "grow"?, "pause"?}."""
    text = (text or "").strip()
    fenced = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    for candidate in (text, text[start : end + 1] if 0 <= start < end else ""):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("say"), str) or not data["say"].strip():
            continue
        turn = {"say": data["say"].strip(), "to": str(data.get("to") or "").strip()[:40]}
        if isinstance(data.get("make"), str) and data["make"].strip():
            turn["make"] = data["make"].strip()[:1500]
        grow = []
        for item in data.get("grow") if isinstance(data.get("grow"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("note"), str) and item["note"].strip():
                grow.append({"about": str(item.get("about") or "self").strip()[:40], "note": item["note"].strip()[:240]})
        if grow:
            turn["grow"] = grow[:3]
        if data.get("pause"):
            turn["pause"] = True
        return turn
    return None
