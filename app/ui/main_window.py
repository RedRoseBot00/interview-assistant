"""
Finestra principale dell'applicazione: schermata colloquio live,
report generato, e storico colloqui salvati.
"""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app import config
from app.export.report import export_docx, export_txt
from app.storage import db
from app.ui.session import InterviewSession
from app.ui.workers import ModelDownloadWorker, ReportGenerationWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{config.APP_DISPLAY_NAME} v{config.APP_VERSION}")
        self.resize(1000, 700)

        self.session: InterviewSession | None = None
        self.current_report_text = ""
        self.current_interview_id: int | None = None

        db.init_db()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(self._build_interview_tab(), "Colloquio")
        self.tabs.addTab(self._build_history_tab(), "Storico")

        self._refresh_history()
        self._check_models_on_startup()

    # ------------------------------------------------------------------
    # Tab "Colloquio"
    # ------------------------------------------------------------------
    def _build_interview_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # --- Riga dati candidato ---
        form_row = QHBoxLayout()
        self.candidate_input = QLineEdit()
        self.candidate_input.setPlaceholderText("Nome candidato")
        self.role_input = QLineEdit()
        self.role_input.setPlaceholderText("Posizione / ruolo")
        form_row.addWidget(QLabel("Candidato:"))
        form_row.addWidget(self.candidate_input)
        form_row.addWidget(QLabel("Posizione:"))
        form_row.addWidget(self.role_input)
        layout.addLayout(form_row)

        # --- Barra di stato/progresso download modelli ---
        self.status_label = QLabel("Pronto.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)

        # --- Pulsanti principali ---
        button_row = QHBoxLayout()
        self.start_button = QPushButton("Inizia colloquio")
        self.stop_button = QPushButton("Termina e genera report")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_interview)
        self.stop_button.clicked.connect(self._stop_interview)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        layout.addLayout(button_row)

        # --- Trascrizione live + report affiancati ---
        splitter = QSplitter(Qt.Horizontal)

        transcript_box = QWidget()
        transcript_layout = QVBoxLayout(transcript_box)
        transcript_layout.addWidget(QLabel("Trascrizione live"))
        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        transcript_layout.addWidget(self.transcript_view)
        splitter.addWidget(transcript_box)

        report_box = QWidget()
        report_layout = QVBoxLayout(report_box)
        report_layout.addWidget(QLabel("Report AI"))
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        report_layout.addWidget(self.report_view)

        export_row = QHBoxLayout()
        self.export_docx_button = QPushButton("Esporta Word (.docx)")
        self.export_txt_button = QPushButton("Esporta testo (.txt)")
        self.export_docx_button.setEnabled(False)
        self.export_txt_button.setEnabled(False)
        self.export_docx_button.clicked.connect(self._export_docx)
        self.export_txt_button.clicked.connect(self._export_txt)
        export_row.addWidget(self.export_docx_button)
        export_row.addWidget(self.export_txt_button)
        report_layout.addLayout(export_row)

        splitter.addWidget(report_box)
        layout.addWidget(splitter)

        return widget

    # ------------------------------------------------------------------
    # Tab "Storico"
    # ------------------------------------------------------------------
    def _build_history_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._show_history_item)
        layout.addWidget(self.history_list, 1)

        self.history_detail = QTextEdit()
        self.history_detail.setReadOnly(True)
        layout.addWidget(self.history_detail, 2)

        return widget

    def _refresh_history(self):
        self.history_list.clear()
        for interview in db.list_interviews():
            label = f"{interview.created_at} - {interview.candidate_name} ({interview.role})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, interview.id)
            self.history_list.addItem(item)

    def _show_history_item(self, item: QListWidgetItem):
        interview_id = item.data(Qt.UserRole)
        interview = db.get_interview(interview_id)
        if not interview:
            return
        self.history_detail.setPlainText(
            f"Candidato: {interview.candidate_name}\n"
            f"Posizione: {interview.role}\n"
            f"Data: {interview.created_at}\n\n"
            f"--- REPORT ---\n{interview.report}\n\n"
            f"--- TRASCRIZIONE ---\n{interview.transcript}"
        )

    # ------------------------------------------------------------------
    # Download modelli al primo avvio
    # ------------------------------------------------------------------
    def _check_models_on_startup(self):
        from app.models.download import llm_model_present

        if llm_model_present():
            return

        self.status_label.setText(
            "Primo avvio: download del modello AI locale in corso (una tantum, "
            "richiede una connessione internet)..."
        )
        self.progress_bar.setVisible(True)
        self.start_button.setEnabled(False)

        self._download_worker = ModelDownloadWorker()
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished_ok.connect(self._on_download_done)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.start()

    def _on_download_progress(self, name: str, done: int, total: int):
        if total > 0:
            pct = int(done / total * 100)
            self.progress_bar.setValue(pct)
            mb_done = done / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self.status_label.setText(
                f"Download modello AI ({name}): {mb_done:.0f} MB / {mb_total:.0f} MB"
            )

    def _on_download_done(self):
        self.progress_bar.setVisible(False)
        self.status_label.setText("Modelli pronti. Puoi iniziare un colloquio.")
        self.start_button.setEnabled(True)

    def _on_download_failed(self, message: str):
        self.progress_bar.setVisible(False)
        self.status_label.setText("Download del modello fallito.")
        self.start_button.setEnabled(True)
        QMessageBox.warning(self, "Download fallito", message)

    # ------------------------------------------------------------------
    # Ciclo di vita del colloquio
    # ------------------------------------------------------------------
    def _start_interview(self):
        if not self.candidate_input.text().strip():
            QMessageBox.information(
                self, "Dati mancanti", "Inserisci almeno il nome del candidato."
            )
            return

        self.transcript_view.clear()
        self.report_view.clear()
        self.export_docx_button.setEnabled(False)
        self.export_txt_button.setEnabled(False)

        settings = config.DEFAULT_SETTINGS
        self.session = InterviewSession(
            whisper_model_size=settings["whisper_model_size"],
            capture_mic=settings["capture_microphone"],
            capture_system=settings["capture_system_audio"],
        )
        self.session.segment_received.connect(self._append_transcript)
        self.session.model_loading.connect(
            lambda: self.status_label.setText("Caricamento modello di trascrizione...")
        )
        self.session.model_ready.connect(
            lambda: self.status_label.setText("Colloquio in corso, registrazione attiva.")
        )
        self.session.error_occurred.connect(self._show_error)
        self.session.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _append_transcript(self, text: str):
        self.transcript_view.append(text)

    def _stop_interview(self):
        if not self.session:
            return
        self.stop_button.setEnabled(False)
        self.status_label.setText("Generazione report in corso...")

        duration = self.session.elapsed_seconds
        transcript = self.session.full_transcript
        language = self.session.last_language

        self.session.stop()
        self.session.close()

        self._pending_duration = duration
        self._pending_transcript = transcript
        self._pending_language = language

        self._report_worker = ReportGenerationWorker(
            transcript=transcript,
            candidate_name=self.candidate_input.text().strip(),
            role=self.role_input.text().strip(),
            detected_language=language,
            report_language=config.DEFAULT_SETTINGS["report_language"],
        )
        self._report_worker.finished_ok.connect(self._on_report_done)
        self._report_worker.failed.connect(self._on_report_failed)
        self._report_worker.start()

    def _on_report_done(self, result):
        self.report_view.setPlainText(result.text)
        self.current_report_text = result.text

        interview = db.Interview(
            id=None,
            candidate_name=self.candidate_input.text().strip(),
            role=self.role_input.text().strip(),
            created_at=datetime.now().isoformat(timespec="seconds"),
            duration_seconds=self._pending_duration,
            detected_language=self._pending_language,
            transcript=self._pending_transcript,
            report=result.text,
        )
        self.current_interview_id = db.save_interview(interview)
        self._refresh_history()

        self.status_label.setText("Report generato e colloquio salvato nello storico.")
        self.start_button.setEnabled(True)
        self.export_docx_button.setEnabled(True)
        self.export_txt_button.setEnabled(True)

    def _on_report_failed(self, message: str):
        self.status_label.setText("Errore nella generazione del report.")
        self.start_button.setEnabled(True)
        QMessageBox.warning(self, "Errore report", message)

    def _show_error(self, message: str):
        QMessageBox.warning(self, "Errore", message)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _current_interview(self):
        if self.current_interview_id is None:
            return None
        return db.get_interview(self.current_interview_id)

    def _export_docx(self):
        interview = self._current_interview()
        if not interview:
            return
        path = export_docx(interview)
        QMessageBox.information(self, "Esportato", f"Report salvato in:\n{path}")

    def _export_txt(self):
        interview = self._current_interview()
        if not interview:
            return
        path = export_txt(interview)
        QMessageBox.information(self, "Esportato", f"Report salvato in:\n{path}")
