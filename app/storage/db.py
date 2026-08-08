"""
Archivio locale dei colloqui (SQLite, nessun dato inviato online).

Ogni colloquio salvato include: dati candidato, trascrizione completa,
report generato dall'AI e data/ora.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS interviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_name TEXT NOT NULL,
    role TEXT,
    created_at TEXT NOT NULL,
    duration_seconds REAL,
    detected_language TEXT,
    transcript TEXT,
    report TEXT
);
"""


@dataclass
class Interview:
    id: Optional[int]
    candidate_name: str
    role: str
    created_at: str
    duration_seconds: float
    detected_language: str
    transcript: str
    report: str


@contextmanager
def _connect():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(SCHEMA)


def save_interview(interview: Interview) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO interviews
                (candidate_name, role, created_at, duration_seconds,
                 detected_language, transcript, report)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interview.candidate_name,
                interview.role,
                interview.created_at or datetime.now().isoformat(timespec="seconds"),
                interview.duration_seconds,
                interview.detected_language,
                interview.transcript,
                interview.report,
            ),
        )
        return cur.lastrowid


def list_interviews() -> List[Interview]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM interviews ORDER BY created_at DESC"
        ).fetchall()
        return [Interview(**dict(row)) for row in rows]


def get_interview(interview_id: int) -> Optional[Interview]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM interviews WHERE id = ?", (interview_id,)
        ).fetchone()
        return Interview(**dict(row)) if row else None


def delete_interview(interview_id: int):
    with _connect() as conn:
        conn.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))
