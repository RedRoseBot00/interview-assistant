"""
QThread worker per le operazioni lente (download modelli, generazione
report), cosi' l'interfaccia grafica non si blocca mai.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from app.models import download as model_download
from app.summarization.llm import LocalLLM, ReportResult


class ModelDownloadWorker(QThread):
    progress = Signal(str, int, int)  # nome modello, scaricati, totale
    finished_ok = Signal()
    failed = Signal(str)

    def run(self):
        try:
            model_download.ensure_models_ready(
                on_progress=lambda name, done, total: self.progress.emit(
                    name, done, total
                )
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class ReportGenerationWorker(QThread):
    finished_ok = Signal(object)  # ReportResult
    failed = Signal(str)

    def __init__(self, transcript: str, candidate_name: str, role: str,
                 detected_language: str, report_language: str, parent=None):
        super().__init__(parent)
        self.transcript = transcript
        self.candidate_name = candidate_name
        self.role = role
        self.detected_language = detected_language
        self.report_language = report_language

    def run(self):
        try:
            llm = LocalLLM()
            result: ReportResult = llm.generate_report(
                transcript=self.transcript,
                candidate_name=self.candidate_name,
                role=self.role,
                detected_language=self.detected_language,
                report_language=self.report_language,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
