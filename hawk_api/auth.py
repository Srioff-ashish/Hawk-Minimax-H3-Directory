"""Token checks and signed links. Pure Python.

Three ways in:

* ``Authorization: Bearer <token>`` -- REST clients, Claude Desktop/Code, xAI remote MCP.
* ``/t/<token>/...`` path prefix -- claude.ai custom connectors, which cannot send headers.
* ``?exp=&sig=`` signed links -- downloads and the upload page, so a chat user can open
  them in a browser without ever seeing the token.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from urllib.parse import urlencode


def token_matches(expected: str, supplied: str | None) -> bool:
    return bool(supplied) and hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))


def bearer(header: str | None) -> str | None:
    if header and header[:7].lower() == "bearer ":
        return header[7:].strip()
    return None


def split_path_token(path: str) -> tuple[str | None, str]:
    """``/t/<token>/v1/jobs`` -> (``<token>``, ``/v1/jobs``); other paths pass through."""
    if not path.startswith("/t/"):
        return None, path
    token, _, rest = path[3:].partition("/")
    return token, "/" + rest


def _signature(secret: str, path: str, expires: int) -> str:
    message = f"{path}\n{expires}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()[:40]


def sign_path(secret: str, path: str, ttl_seconds: int, now: float | None = None) -> str:
    expires = int((time.time() if now is None else now) + ttl_seconds)
    if ttl_seconds >= 86400:
        # Round up to a whole UTC day so a file keeps one URL all day: browsers and CDNs can cache it.
        expires = int(math.ceil(expires / 86400) * 86400)
    return f"{path}?{urlencode({'exp': expires, 'sig': _signature(secret, path, expires)})}"


def signature_valid(secret: str, path: str, expires: str | None, signature: str | None, now: float | None = None) -> bool:
    try:
        expires_at = int(expires or "")
    except ValueError:
        return False
    if expires_at < (time.time() if now is None else now):
        return False
    return hmac.compare_digest(_signature(secret, path, expires_at), signature or "")
