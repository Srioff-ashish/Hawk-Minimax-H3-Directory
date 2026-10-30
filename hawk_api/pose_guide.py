"""The sex-position guide (guides/poses.md), for whoever writes an explicit prompt: the planner, the agent and MCP.

Video and image models know no position names, so a prompt that says "full nelson" gets whatever the model
guesses. The guide spells each position out as body geometry, camera and motion. The planner gets the relevant
entries added to its brief when the brief is sexual; the agent and MCP clients look them up with pose_guide.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guides", "poses.md")
#: A brief that says any of these is a sex scene even when it names no position. Words with an everyday meaning
#: ("cock" the bird, "Dick" the name, "cum laude", a plot that "thrusts") are left out.
SEXUAL = re.compile(r"\b(sex|sexual|having sex|\w*fuck\w*|penetrat\w*|anal|vagina\w*|vulva|pussy|penis|his cock|"
                    r"his dick|blowjob|handjob|titjob|oral sex|cunnilingus|fellatio|nsfw|porn\w*|orgasm\w*|cumming|"
                    r"cumshot|creampie|intercourse|rides him|riding him|sucks him|double penetration|threesome)\b", re.IGNORECASE)
#: Position names that are also everyday words: a butterfly tattoo, a candle-lit room, a cowgirl at a rodeo, a doggy
#: in the park. They pick a position only in a brief that is already sexual; on their own they mean nothing here.
EVERYDAY = frozenset({"butterfly", "candle", "carry", "riding", "cowgirl", "doggy", "pretzel", "flatiron", "face down",
                      "bent over", "legs up", "one leg up", "lotus", "ballerina", "bellarina", "bridge", "the bridge",
                      "chairman", "the chairman", "caboose", "seashell", "sea shell", "tabletop", "table top",
                      "on the table", "corkscrew", "wheelbarrow", "leap frog", "leapfrog", "valedictorian",
                      "pinball wizard", "upstanding citizen", "stand and deliver", "on all fours", "woman on top",
                      "man on top", "missionary", "half folded", "carrying", "69", "sixty nine",
                      "piledriver", "pile driver", "anvil", "deep impact", "knees to chest", "knees out",
                      "standing split", "standing face to face", "standing against the wall", "frog legs",
                      "closed leg", "legs on shoulders", "on his lap", "bullseye", "bull s eye", "bulls eye", "king of clubs",
                      "sandwich", "sidesaddle", "wishbone", "northern exposure", "southern exposure", "standard dp",
                      "reverse right angle", "missionary double"})
#: An everyday name followed by one of these describes a thing, not a position: "candle-lit", "butterfly tattoo".
NOT_A_POSE = re.compile(r"^(?:-|\s+)(?:lit|light|lights|tattoo|tattoos|print|necklace|earrings?|stand|shop|wings?|"
                        r"clip|sleeves?|garden|holder|flame|wax|race|hat|boots?)\b", re.IGNORECASE)


def _as_pose(text: str, name: str) -> bool:
    """Whether an everyday name is used as a position somewhere in the text, not only as a thing."""
    pattern = r"\b" + r"[\s\-]+".join(map(re.escape, name.split())) + r"\b"
    return any(not NOT_A_POSE.match(text[m.end():]) for m in re.finditer(pattern, text, re.IGNORECASE))


#: When a sexual brief names no position, the planner gets these in full and only the names of the rest:
#: the whole guide is too long to send with every plan.
CORE = ("missionary", "doggy", "cowgirl", "reverse-cowgirl", "prone-bone", "standing-doggy", "wall-lift", "lotus")


#: Words too common in position names to pick one out by themselves.
FILLER = frozenset({"with", "from", "under", "position", "style", "legs", "knees", "missionary", "variations"})


@dataclass(frozen=True)
class Pose:
    key: str
    name: str
    aliases: tuple[str, ...]
    text: str  # the entry's lines after the heading, as written

    def view(self) -> dict:
        return {"key": self.key, "name": self.name, "aliases": list(self.aliases), "guide": self.text}


_cache: tuple[float, str, list[Pose]] | None = None


def load() -> tuple[str, list[Pose]]:
    """(the rules every position shares, the positions). Re-read when the file changes, so it can be edited live."""
    global _cache
    stamp = os.path.getmtime(PATH)
    if _cache and _cache[0] == stamp:
        return _cache[1], _cache[2]
    with open(PATH, encoding="utf-8") as handle:
        text = handle.read()
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    rules, poses = "", []
    for section in sections[1:]:
        heading, _, body = section.partition("\n")
        if " — " not in heading:  # "Rules for every position"
            rules = body.strip()
            continue
        key, name = (part.strip() for part in heading.split(" — ", 1))
        lines = body.strip().splitlines()
        aliases = ()
        if lines and lines[0].startswith("aliases:"):
            aliases = tuple(a.strip().lower() for a in lines[0][len("aliases:"):].split(",") if a.strip())
            lines = lines[1:]
        poses.append(Pose(key, name, aliases, "\n".join(lines).strip()))
    _cache = (stamp, rules, poses)
    return rules, poses


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def mentioned(text: str, asked: bool = False) -> list[Pose]:
    """Positions a brief names, by key or alias. The longest name wins where one contains another, so "standing
    full nelson" picks the standing entry rather than every full nelson."""
    _, poses = load()
    haystack = f" {_norm(text)} "
    found = {pose.key: [n for n in {_norm(pose.key), *(_norm(a) for a in pose.aliases)} if n and f" {n} " in haystack]
             for pose in poses}
    # A brief is sexual when it says so, or when it names a position by a name nothing else means ("mating press").
    # asked: the text is a position lookup (the pose_guide tool), so every name means the position.
    sexual = asked or bool(SEXUAL.search(text or "")) or any(n not in EVERYDAY for names in found.values() for n in names)
    matched: dict[str, str] = {}  # pose key -> the longest of its names found in the text
    for pose in poses:
        names = [n for n in found[pose.key] if n not in EVERYDAY or asked or (sexual and _as_pose(text, n))]
        if names:
            matched[pose.key] = max(names, key=len)
    # A name found only inside a longer found name does not count: "full nelson" inside "standing full nelson".
    keep = {key for key, name in matched.items()
            if not any(other != key and name != longer and f" {name} " in f" {longer} "
                       for other, longer in matched.items())}
    return [pose for pose in poses if pose.key in keep]


def lookup(query: str | None = None) -> dict:
    """The pose_guide tool: one position (or a few) by name, or the index when no name is given."""
    rules, poses = load()
    if not query or not query.strip():
        return {"rules": rules,
                "positions": [{"key": p.key, "name": p.name, "aliases": list(p.aliases)} for p in poses],
                "note": "Call pose_guide again with a position's name for its full description."}
    found = mentioned(query, asked=True)
    if not found:  # a partial word: "nelson", "cowgirl"
        words = {w for w in _norm(query).split() if len(w) > 3 and w not in FILLER}
        found = [p for p in poses if words & set(_norm(" ".join((p.key, p.name, *p.aliases))).split())]
    if not found:
        return {"rules": rules, "error": f"No position matches {query!r}.",
                "positions": [p.name for p in poses]}
    return {"rules": rules, "positions": [p.view() for p in found]}


def _field(pose: Pose, name: str) -> str:
    return next((line[len(name) + 1:].strip() for line in pose.text.splitlines() if line.startswith(f"{name}:")), "")


def planner_note(brief: str) -> str:
    """What the planner gets beside a sexual brief: the shared rules plus the positions it names, or the core
    positions (and the rest by name) when it names none. "" for any other brief -- a butterfly tattoo or a
    candle-lit room is not a position."""
    found = mentioned(brief)
    if not found and not SEXUAL.search(brief or ""):
        return ""
    rules, poses = load()
    if found:
        body = "\n\n".join(f"{p.name}\n{p.text}" for p in found)
    else:  # no position named: the core ones' ready sentence and pitfall, and the rest by name to keep the brief short
        core = [p for p in poses if p.key in CORE]
        body = "\n\n".join(f"{p.name}\n{_field(p, 'prompt')}\nTags: {_field(p, 'tags')}\nWatch: {_field(p, 'watch')}" for p in core)
        body += ("\n\nOther positions (name one in the brief to get its full geometry): "
                 + ", ".join(p.name for p in poses if p.key not in CORE) + ".")
    return ("POSITION GUIDE (the video model knows no position names: write each position as this geometry, one "
            f"position per segment):\n{rules}\n\n{body}")
