"""
Archivio locale dei colloqui (SQLite).

Nessun dato lascia il computer: trascrizioni e report restano in un
singolo file dentro %APPDATA%, che l'utente puo' copiare o cancellare
quando vuole.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app import config

log = logging.getLogger(__name__)

_SCHEMA = """
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

# Colonne aggiunte dopo la prima versione: vengono create al volo sui
# database gia' esistenti, cosi' un aggiornamento non perde i dati.
_MIGRATIONS = {
    "segments_json": "ALTER TABLE interviews ADD COLUMN segments_json TEXT",
    "used_llm": "ALTER TABLE interviews ADD COLUMN used_llm INTEGER DEFAULT 0",
    "notes": "ALTER TABLE interviews ADD COLUMN notes TEXT",
}

_init_lock = threading.Lock()
_initialised = False


@dataclass
class Interview:
    candidate_name: str
    role: str = ""
    created_at: str = ""
    duration_seconds: float = 0.0
    detected_language: str = ""
    transcript: str = ""
    report: str = ""
    segments: list[dict] = field(default_factory=list)
    used_llm: bool = False
    notes: str = ""
    id: Optional[int] = None

    @property
    def display_date(self) -> str:
        try:
            return datetime.fromisoformat(self.created_at).strftime("%d/%m/%Y %H:%M")
        except Exception:
            return self.created_at


@contextmanager
def _connect():
    conn = sqlite3.connect(str(config.DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(force: bool = False) -> None:
    global _initialised
    with _init_lock:
        if _initialised and not force:
            return
        with _connect() as conn:
            # Modalita' WAL: lettura dall'interfaccia e scrittura dai
            # processi di lavoro possono avvenire insieme senza bloccarsi.
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except Exception:
                log.debug("PRAGMA journal_mode non applicabile", exc_info=True)
            conn.executescript(_SCHEMA)
            existing = {
                row["name"] for row in conn.execute("PRAGMA table_info(interviews)")
            }
            for column, statement in _MIGRATIONS.items():
                if column not in existing:
                    log.info("Aggiorno il database: aggiungo la colonna '%s'", column)
                    conn.execute(statement)
        _initialised = True


def _with_recovery(operation):
    """
    Esegue un'operazione ricreando lo schema se la tabella e' sparita.

    Capita quando l'utente elimina il file del database mentre il
    programma e' aperto, cosa del tutto legittima visto che gli diciamo
    che i suoi dati stanno in un unico file.
    """
    try:
        return operation()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            raise
        log.warning("Database ricreato dopo la rimozione del file")
        init_db(force=True)
        return operation()


def _row_to_interview(row: sqlite3.Row) -> Interview:
    data = dict(row)
    segments: list[dict] = []
    raw = data.get("segments_json")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                segments = parsed
        except Exception:
            log.debug("Segmenti non leggibili per il colloquio %s", data.get("id"))

    # SQLite non impone il tipo delle colonne: un valore inatteso non
    # deve far sparire l'intero elenco dei colloqui.
    try:
        duration = float(data.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    return Interview(
        id=data.get("id"),
        candidate_name=str(data.get("candidate_name") or ""),
        role=str(data.get("role") or ""),
        created_at=str(data.get("created_at") or ""),
        duration_seconds=duration,
        detected_language=str(data.get("detected_language") or ""),
        transcript=str(data.get("transcript") or ""),
        report=str(data.get("report") or ""),
        segments=segments,
        used_llm=bool(data.get("used_llm")),
        notes=str(data.get("notes") or ""),
    )


def save_interview(interview: Interview) -> int:
    init_db()
    created = interview.created_at or datetime.now().isoformat(timespec="seconds")

    def _operation() -> int:
        with _connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO interviews
                    (candidate_name, role, created_at, duration_seconds,
                     detected_language, transcript, report, segments_json,
                     used_llm, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interview.candidate_name,
                    interview.role,
                    created,
                    interview.duration_seconds,
                    interview.detected_language,
                    interview.transcript,
                    interview.report,
                    json.dumps(interview.segments, ensure_ascii=False),
                    1 if interview.used_llm else 0,
                    interview.notes,
                ),
            )
            return int(cursor.lastrowid)

    return _with_recovery(_operation)


def update_report(interview_id: int, report: str, used_llm: bool) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE interviews SET report = ?, used_llm = ? WHERE id = ?",
            (report, 1 if used_llm else 0, interview_id),
        )


def update_notes(interview_id: int, notes: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE interviews SET notes = ? WHERE id = ?", (notes, interview_id)
        )


def list_interviews(limit: int = 500) -> list[Interview]:
    init_db()

    def _operation() -> list[Interview]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM interviews "
                "ORDER BY datetime(created_at) DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        interviews = []
        for row in rows:
            # Una riga illeggibile non deve nascondere tutte le altre.
            try:
                interviews.append(_row_to_interview(row))
            except Exception:
                log.warning("Colloquio non leggibile, lo salto", exc_info=True)
        return interviews

    return _with_recovery(_operation)


def get_interview(interview_id: int) -> Optional[Interview]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM interviews WHERE id = ?", (interview_id,)
        ).fetchone()
        return _row_to_interview(row) if row else None


def delete_interview(interview_id: int) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))
