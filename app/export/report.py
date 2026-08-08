"""
Esportazione del report di colloquio in formato Word (.docx) e testo (.txt).

L'export in PDF viene ottenuto stampando il .docx (Word/LibreOffice hanno
sempre "salva come PDF"), per non aggiungere dipendenze pesanti extra
all'MVP. Se in futuro serve un export PDF diretto si puo' aggiungere
reportlab senza toccare il resto dell'app.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app import config
from app.storage.db import Interview


def _safe_filename(name: str) -> str:
    keep = "-_ "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return cleaned.replace(" ", "_") or "colloquio"


def export_txt(interview: Interview) -> Path:
    filename = f"{_safe_filename(interview.candidate_name)}_{interview.id}.txt"
    path = config.EXPORTS_DIR / filename
    content = (
        f"Candidato: {interview.candidate_name}\n"
        f"Posizione: {interview.role}\n"
        f"Data: {interview.created_at}\n"
        f"Lingua rilevata: {interview.detected_language}\n"
        f"\n--- TRASCRIZIONE ---\n{interview.transcript}\n"
        f"\n--- REPORT ---\n{interview.report}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def export_docx(interview: Interview) -> Path:
    from docx import Document
    from docx.shared import Pt

    doc = Document()

    title = doc.add_heading(f"Report colloquio - {interview.candidate_name}", level=1)

    meta = doc.add_paragraph()
    meta.add_run(f"Posizione: {interview.role}\n").bold = False
    meta.add_run(f"Data: {interview.created_at}\n")
    meta.add_run(f"Lingua rilevata: {interview.detected_language}\n")
    meta.add_run(f"Durata: {round((interview.duration_seconds or 0) / 60, 1)} min")

    doc.add_heading("Report generato dall'AI", level=2)
    for block in (interview.report or "").split("\n"):
        if block.strip():
            doc.add_paragraph(block.strip())

    doc.add_heading("Trascrizione completa", level=2)
    p = doc.add_paragraph(interview.transcript or "")
    for run in p.runs:
        run.font.size = Pt(9)

    filename = f"{_safe_filename(interview.candidate_name)}_{interview.id}.docx"
    path = config.EXPORTS_DIR / filename
    doc.save(str(path))
    return path
