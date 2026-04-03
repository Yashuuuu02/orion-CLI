"""Orion session & message persistence (sqlite3)."""

from __future__ import annotations

import random
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path.home() / ".orion"
DB_PATH = DATA_DIR / "sessions.db"

ADJECTIVES = [
    "brave", "swift", "quiet", "bright", "calm",
    "dark", "eager", "fierce", "gentle", "hollow",
]
NOUNS = [
    "falcon", "river", "storm", "ember", "ridge",
    "cedar", "dusk", "flare", "grove", "haven",
]

_CREATE_SESSIONS = """\
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    cwd        TEXT NOT NULL
);
"""

_CREATE_MESSAGES = """\
CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class SessionManager:
    """Manages sessions and messages in a local SQLite database."""

    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(_CREATE_SESSIONS)
        self.conn.execute(_CREATE_MESSAGES)
        self.conn.commit()

    # ── sessions ─────────────────────────────────────────────────────────

    def create_session(self, cwd: str) -> dict:
        """Create a new session with a random two-word name."""
        session = {
            "id": str(uuid.uuid4()),
            "name": f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cwd": cwd,
        }
        self.conn.execute(
            "INSERT INTO sessions (id, name, created_at, cwd) VALUES (?, ?, ?, ?)",
            (session["id"], session["name"], session["created_at"], session["cwd"]),
        )
        self.conn.commit()
        return session

    def get_latest_session(self) -> dict | None:
        """Return the most recent session, or ``None``."""
        cur = self.conn.execute(
            "SELECT id, name, created_at, cwd FROM sessions ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(("id", "name", "created_at", "cwd"), row))

    def get_all_sessions(self) -> list[dict]:
        """Return every session, newest first."""
        cur = self.conn.execute(
            "SELECT id, name, created_at, cwd FROM sessions ORDER BY created_at DESC"
        )
        return [dict(zip(("id", "name", "created_at", "cwd"), r)) for r in cur.fetchall()]

    # ── messages ─────────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str) -> dict:
        """Append a message to the given session."""
        msg = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (msg["id"], msg["session_id"], msg["role"], msg["content"], msg["created_at"]),
        )
        self.conn.commit()
        return msg

    def get_messages(self, session_id: str) -> list[dict]:
        """Return all messages for a session, oldest first."""
        cur = self.conn.execute(
            "SELECT id, session_id, role, content, created_at "
            "FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        return [
            dict(zip(("id", "session_id", "role", "content", "created_at"), r))
            for r in cur.fetchall()
        ]
