"""The character library: a person saved once and used everywhere -- storyboards, Story, Manhwa, MCP and the agent.

A character is their fixed facts (what every prompt must keep: age, ethnicity, figure), up to three identity pictures
(the face and body every panel and render carries), an optional identity pack (front, back, side, face views drawn
as one sheet), prompt notes (wording that fixes a model's habit, e.g. "teardrop breasts with natural sag"), and a
default platform. Asking for "Rebel" anywhere then brings all of that along, so nothing is typed twice and nothing
is guessed from a picture.

Saved in DATA_DIR/characters.json.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid

from .jobs import NotFound, RequestError

MAX_IDENTITY = 3
PLATFORMS = ("general", "instagram", "fanvue")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


class CharacterStore:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "characters.json")
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
            json.dump(data, handle, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    def all(self) -> list[dict]:
        return sorted(self._load().values(), key=lambda c: c.get("name", "").lower())

    def get(self, character_id: str) -> dict | None:
        return self._load().get(character_id)

    def save(self, character: dict) -> dict:
        with self._lock:
            data = self._load()
            character["updated_at"] = time.time()
            data[character["id"]] = character
            self._write(data)
        return character

    def delete(self, character_id: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(character_id, None)
            self._write(data)


class CharacterManager:
    def __init__(self, service):
        self.service = service
        self.store = CharacterStore(service.settings.data_dir)

    # ---- reading
    def list(self) -> list[dict]:
        return [self.view(c) for c in self.store.all()]

    def find(self, name_or_id: str) -> dict:
        """By id, or by name ignoring case and punctuation."""
        key = (name_or_id or "").strip()
        found = self.store.get(key)
        if found:
            return found
        slug = _slug(key)
        for character in self.store.all():
            if _slug(character["name"]) == slug:
                return character
        raise NotFound(f"No character {name_or_id!r} in the library. Save them first (save_character).")

    def view(self, character: dict | str) -> dict:
        character = self.find(character) if isinstance(character, str) else character
        get = self.service.store.get_asset
        out = dict(character)
        out["identity_views"] = [self.service.asset_view(get(a)) for a in character.get("identity_asset_ids", []) if get(a)]
        out["pack_views"] = {view: self.service.asset_view(get(a)) for view, a in (character.get("pack") or {}).items() if get(a)}
        return out

    def resolve(self, names: list[str]) -> dict:
        """What a film needs from its people: one character text (facts and prompt notes, names first) and their
        identity pictures -- at most three in all, one per person first so everyone is carried."""
        people = [self.find(n) for n in names or [] if (n or "").strip()]
        if not people:
            raise RequestError("Name at least one character from the library.")
        lines = []
        for person in people:
            text = f"{person['name']}: {person['facts'].strip()}"
            if person.get("notes", "").strip():
                text += f" {person['notes'].strip()}"
            lines.append(text)
        identity, rest = [], []
        for person in people:
            ids = [a for a in person.get("identity_asset_ids") or [] if self.service.store.get_asset(a)]
            if ids:
                identity.append(ids[0])
                rest.extend(ids[1:])
        identity = (identity + rest)[:MAX_IDENTITY]
        if not identity:
            raise RequestError(f"{', '.join(p['name'] for p in people)} ha{'ve' if len(people) > 1 else 's'} no identity picture yet.")
        platform = next((p.get("platform") for p in people if p.get("platform") not in (None, "", "general")), "general")
        return {"names": [p["name"] for p in people], "character": " ".join(lines), "identity_asset_ids": identity,
                "platform": platform, "people": len(people)}

    # ---- writing
    def _check(self, name: str, facts: str) -> None:
        from .manhwa import is_minor

        if not name.strip():
            raise RequestError("Give the character a name.")
        if not facts.strip():
            raise RequestError("Write the character's fixed facts: age, ethnicity, figure, hair -- what every prompt keeps.")
        if is_minor(name, {"appearance": facts}):
            raise RequestError(f"{name} reads as under 18: the library is for adult characters only.")

    def _images(self, ids: list[str] | None) -> list[str]:
        ids = list(ids or [])
        if len(ids) > MAX_IDENTITY:
            raise RequestError(f"At most {MAX_IDENTITY} identity pictures.")
        for asset_id in ids:
            asset = self.service.store.get_asset(asset_id)
            if asset is None or asset.get("kind") != "image":
                raise RequestError(f"No image asset {asset_id!r}.")
        return ids

    def create(self, name: str, facts: str, identity_asset_ids: list[str] | None = None, notes: str = "",
               platform: str = "general") -> dict:
        self._check(name, facts)
        if platform not in PLATFORMS:
            raise RequestError(f"platform should be one of: {', '.join(PLATFORMS)}.")
        try:
            self.find(name)
            raise RequestError(f"There is already a character called {name}: edit them instead.")
        except NotFound:
            pass
        character = {"id": uuid.uuid4().hex[:12], "name": name.strip()[:80], "facts": facts.strip()[:2000],
                     "identity_asset_ids": self._images(identity_asset_ids), "notes": notes.strip()[:1000],
                     "platform": platform, "pack": {}, "created_at": time.time()}
        return self.view(self.store.save(character))

    def update(self, name_or_id: str, **fields) -> dict:
        character = self.find(name_or_id)
        name = fields.get("name") if fields.get("name") is not None else character["name"]
        facts = fields.get("facts") if fields.get("facts") is not None else character["facts"]
        self._check(name, facts)
        character.update(name=name.strip()[:80], facts=facts.strip()[:2000])
        if fields.get("notes") is not None:
            character["notes"] = fields["notes"].strip()[:1000]
        if fields.get("platform") is not None:
            if fields["platform"] not in PLATFORMS:
                raise RequestError(f"platform should be one of: {', '.join(PLATFORMS)}.")
            character["platform"] = fields["platform"]
        if fields.get("identity_asset_ids") is not None:
            character["identity_asset_ids"] = self._images(fields["identity_asset_ids"])
        return self.view(self.store.save(character))

    def delete(self, name_or_id: str) -> dict:
        character = self.find(name_or_id)
        self.store.delete(character["id"])
        return {"deleted": character["name"], "note": "Their pictures and pack stay in Media."}

    async def make_pack(self, name_or_id: str, *, anatomy: bool = False, style: str = "") -> dict:
        """The identity pack drawn from their identity pictures ($0.018); the views are kept on the character."""
        from .storyboard import identity_pack

        character = self.find(name_or_id)
        if not character.get("identity_asset_ids"):
            raise RequestError(f"Give {character['name']} an identity picture first: the pack is drawn from it.")
        pack = await identity_pack(self.service, character["name"], f"{character['facts']} {character.get('notes', '')}",
                                   character["identity_asset_ids"], style, anatomy, collection=f"characters/{character['name']}")
        character = self.find(character["id"])
        character["pack"] = {p["view"]: p["id"] for p in pack["panels"]}
        self.store.save(character)
        return {"character": self.view(character), "pack": pack}
