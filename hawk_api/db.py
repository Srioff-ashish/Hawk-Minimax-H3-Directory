"""One SQLite connection per database file, shared by every store that uses it.

The job, agent, story and manhwa stores (and the boards and characters) all live in ``jobs.sqlite3``. Each used to
open its own connection with its own lock, so two of them writing at once could fail with "database is locked",
and nothing set WAL, so a long read blocked every writer. Here they share one connection and one re-entrant lock:
writes are serialised in this process, WAL lets the Drive snapshot read while the API writes, and busy_timeout
waits out the snapshot's own short locks instead of failing.
"""

from __future__ import annotations

import os
import sqlite3
import threading

_shared: dict[str, tuple[sqlite3.Connection, threading.RLock]] = {}
_guard = threading.Lock()


def connect(path: str) -> tuple[sqlite3.Connection, threading.RLock]:
    """The shared (connection, lock) for this file, opened and tuned on first use."""
    key = os.path.realpath(path)
    with _guard:
        found = _shared.get(key)
        if found is not None:
            try:
                found[0].execute("SELECT 1")
                return found
            except sqlite3.ProgrammingError:  # closed by its owner: open a fresh one
                _shared.pop(key, None)
        os.makedirs(os.path.dirname(key) or ".", exist_ok=True)
        conn = sqlite3.connect(key, check_same_thread=False, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:  # a filesystem without shared memory (some FUSE mounts): keep the default
            pass
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        found = (conn, threading.RLock())
        _shared[key] = found
        return found


def forget(path: str) -> None:
    """Close and drop the shared connection (tests that delete their data directory)."""
    key = os.path.realpath(path)
    with _guard:
        found = _shared.pop(key, None)
    if found is not None:
        found[0].close()
