"""
Finestra principale dell'applicazione.

La schermata del colloquio segue lo schema richiesto: a sinistra la
trascrizione dal vivo, al centro la videochiamata, a destra la scheda
del candidato con le note. Tutto il resto (report, archivio,
impostazioni) sta in schede separate, per non affollare la vista che
si usa mentre si parla con la persona.
"""
from __future__ import annotations

import html
import logging
import os
import subprocess
import sys
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app import compat, config, platform_detect, settings
from app.audio import echo as echo_module
from app.export.report import export_docx, export_txt
from app.storage import db
from app.ui import theme
from app.ui.call_view import CallView, best_call_window, list_windows
from app.ui.session import InterviewSession
from app.ui.theme import Card, Chip, LevelMeter, StatusDot
from app.ui.workers import PlatformProbe, ReportWorker, StartupWorker

log = logging.getLogger(__name__)

TAB_INTERVIEW = 0
TAB_REPORT = 1


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{config.APP_DISPLAY_NAME} — assistente per colloqui")
        self.resize(1440, 880)
        self.setMinimumSize(1150, 720)

        self.session: InterviewSession | None = None
        self.startup_worker: StartupWorker | None = None
        self.report_worker: ReportWorker | None = None
        self.platform_probe: PlatformProbe | None = None

        # Qt distrugge l'oggetto interno di un thread quando l'ultimo
        # riferimento sparisce: se accadesse mentre e' in esecuzione, il
        # programma verrebbe chiuso all'istante.
        self._live_workers: set = set()

        self.is_recording = False
        self.engine_ready = False
        self.closing = False
        self.current_interview_id: int | None = None
        self._pending: dict = {}
        self._warnings: list[str] = []
        self._streamed_report = False
        self._stats_cache = ""

        db.init_db()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addWidget(self._build_platform_bar())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_interview_tab(), "Colloquio")
        self.tabs.addTab(self._build_report_tab(), "Report")
        self.tabs.addTab(self._build_history_tab(), "Storico")
        self.tabs.addTab(self._build_settings_tab(), "Impostazioni")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._tick)
        self.elapsed_timer.start(1000)

        self.platform_timer = QTimer(self)
        self.platform_timer.timeout.connect(self._refresh_platform)
        self.platform_timer.start(8000)

        self._refresh_platform()
        self._refresh_history()
        self._apply_always_on_top(settings.get("always_on_top", False))
        QTimer.singleShot(600, self._auto_select_call_window)
        self._start_preparation()

    # ==================================================================
    # Intestazione e barra piattaforme
    # ==================================================================
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("HeaderBar")
        row = QHBoxLayout(header)
        row.setContentsMargins(22, 0, 22, 0)
        row.setSpacing(12)

        self.header_dot = StatusDot(theme.SUCCESS, 12)
        row.addWidget(self.header_dot)

        title = QLabel(config.APP_DISPLAY_NAME)
        title.setObjectName("HeaderTitle")
        row.addWidget(title)
        row.addStretch(1)

        version = QLabel(f"v{config.APP_VERSION}")
        version.setObjectName("HeaderVersion")
        row.addWidget(version)
        return header

    def _build_platform_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("PlatformBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(22, 8, 22, 8)
        row.setSpacing(8)

        hint = QLabel("Compatibile con:")
        hint.setObjectName("PlatformHint")
        row.addWidget(hint)
        for label in platform_detect.SUPPORTED_LABELS:
            row.addWidget(Chip(label))
        row.addStretch(1)

        self.echo_chip = QLabel("")
        self.echo_chip.setProperty("class", "Chip")
        self.echo_chip.setVisible(False)
        row.addWidget(self.echo_chip)

        self.platform_dot = StatusDot(theme.TEXT_MUTED, 9)
        row.addWidget(self.platform_dot)
        self.platform_label = QLabel("Rilevamento in corso...")
        self.platform_label.setProperty("class", "Muted")
        row.addWidget(self.platform_label)
        return bar

    # ==================================================================
    # Scheda "Colloquio"
    # ==================================================================
    def _build_interview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)

        layout.addLayout(self._build_toolbar())
        layout.addWidget(self._build_status_strip())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_transcript_card())
        splitter.addWidget(self._build_call_card())
        splitter.addWidget(self._build_candidate_column())
        splitter.setSizes([400, 620, 360])
        layout.addWidget(splitter, 1)
        return page

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.record_button = QPushButton("Avvia colloquio")
        self.record_button.setObjectName("PrimaryStart")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self._toggle_recording)
        row.addWidget(self.record_button)

        self.captions_button = QPushButton("Sottotitoli live: attivi")
        self.captions_button.setCheckable(True)
        self.captions_button.setChecked(True)
        self.captions_button.toggled.connect(self._toggle_captions)
        row.addWidget(self.captions_button)

        self.on_top_button = QPushButton("Sempre in primo piano")
        self.on_top_button.setCheckable(True)
        self.on_top_button.setChecked(bool(settings.get("always_on_top", False)))
        self.on_top_button.toggled.connect(self._apply_always_on_top)
        row.addWidget(self.on_top_button)

        row.addStretch(1)

        # Indicatori di livello compatti: rispondono alla domanda piu'
        # frequente durante una registrazione, "mi sta sentendo?".
        self.level_meters: dict[str, LevelMeter] = {}
        self.source_labels: dict[str, QLabel] = {}
        for speaker, fallback in (
            (config.SPEAKER_RECRUITER, "Tu"),
            (config.SPEAKER_CANDIDATE, "Candidato"),
        ):
            name = QLabel(
                settings.get(
                    "label_recruiter"
                    if speaker == config.SPEAKER_RECRUITER
                    else "label_candidate",
                    fallback,
                )
            )
            theme.bold(name)
            name.setStyleSheet(f"color: {theme.SPEAKER_COLORS[speaker]};")
            name.setToolTip("Sorgente audio non ancora attiva")
            self.source_labels[speaker] = name
            row.addWidget(name)

            meter = LevelMeter(theme.SPEAKER_COLORS[speaker])
            meter.setFixedWidth(90)
            self.level_meters[speaker] = meter
            row.addWidget(meter)

        row.addSpacing(10)
        self.recording_dot = StatusDot(theme.TEXT_MUTED, 11)
        row.addWidget(self.recording_dot)
        self.timer_label = QLabel("00:00")
        # La dimensione arriva dal foglio di stile (#TimerLabel): un
        # setFont qui verrebbe comunque sovrascritto dal QSS globale.
        self.timer_label.setObjectName("TimerLabel")
        row.addWidget(self.timer_label)
        return row

    def _build_status_strip(self) -> QWidget:
        container = QFrame()
        container.setProperty("class", "Card")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(6)

        self.status_label = QLabel("Preparazione in corso...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(f"color: {theme.WARNING};")
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)
        return container

    def _build_transcript_card(self) -> QWidget:
        card = Card("Trascrizione live")

        legend = QHBoxLayout()
        legend.setSpacing(14)
        for speaker, fallback in (
            (config.SPEAKER_RECRUITER, "Tu"),
            (config.SPEAKER_CANDIDATE, "Candidato"),
        ):
            key = (
                "label_recruiter"
                if speaker == config.SPEAKER_RECRUITER
                else "label_candidate"
            )
            legend.addWidget(StatusDot(theme.SPEAKER_COLORS[speaker], 9))
            label = QLabel(settings.get(key, fallback))
            label.setProperty("class", "Muted")
            legend.addWidget(label)
        legend.addStretch(1)
        card.body().addLayout(legend)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText(
            "La trascrizione comparira' qui, riga per riga, mentre il "
            "colloquio prosegue."
        )
        card.add(self.transcript_view, 1)
        return card

    def _build_call_card(self) -> QWidget:
        card = Card("Videochiamata")

        picker = QHBoxLayout()
        picker.setSpacing(8)
        self.window_combo = QComboBox()
        self.window_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window_combo.activated.connect(self._on_window_selected)
        picker.addWidget(self.window_combo, 1)

        refresh_button = QPushButton("Aggiorna")
        refresh_button.setToolTip("Rileggi l'elenco delle finestre aperte")
        refresh_button.clicked.connect(lambda: self._refresh_window_list(True))
        picker.addWidget(refresh_button)
        card.body().addLayout(picker)

        self.call_view = CallView()
        card.add(self.call_view, 1)

        hint = QLabel(
            "L'anteprima e' solo visiva: per parlare, condividere lo schermo o "
            "riagganciare usa la finestra originale della videochiamata."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "Muted")
        card.add(hint)
        return card

    def _build_candidate_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        candidate_card = Card("Candidato")
        self.candidate_input = QLineEdit()
        self.candidate_input.setPlaceholderText("Nome e cognome")
        self.role_input = QLineEdit()
        self.role_input.setPlaceholderText("Posizione, es. Sviluppatore full-stack")
        self.experience_input = QLineEdit()
        self.experience_input.setPlaceholderText("Esperienza, es. 5 anni")
        self.skills_input = QLineEdit()
        self.skills_input.setPlaceholderText("Competenze separate da virgola")
        self.skills_input.textChanged.connect(self._refresh_skill_chips)

        for label, widget in (
            ("Nome", self.candidate_input),
            ("Posizione", self.role_input),
            ("Esperienza", self.experience_input),
            ("Competenze", self.skills_input),
        ):
            caption = QLabel(label)
            caption.setProperty("class", "Muted")
            candidate_card.add(caption)
            candidate_card.add(widget)

        self.skills_row = QHBoxLayout()
        self.skills_row.setSpacing(6)
        self.skills_row.addStretch(1)
        candidate_card.body().addLayout(self.skills_row)
        layout.addWidget(candidate_card)

        notes_card = Card("Note")
        self.stats_label = QLabel("Le statistiche compariranno durante il colloquio.")
        self.stats_label.setWordWrap(True)
        self.stats_label.setProperty("class", "Muted")
        notes_card.add(self.stats_label)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Appunti durante il colloquio.\n"
            "Vengono salvati insieme alla trascrizione e finiscono nel report."
        )
        notes_card.add(self.notes_edit, 1)
        layout.addWidget(notes_card, 1)
        return column

    # ==================================================================
    # Scheda "Report"
    # ==================================================================
    def _build_report_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        card = Card("Report del colloquio")
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setPlaceholderText(
            "Al termine del colloquio qui comparira' il report generato "
            "dall'intelligenza artificiale locale: sintesi, punti di forza, "
            "aree di attenzione, competenze citate e domande di approfondimento."
        )
        card.add(self.report_view, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self.report_button = QPushButton("Genera report")
        self.report_button.setObjectName("AccentButton")
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self._generate_report)
        buttons.addWidget(self.report_button)

        self.export_docx_button = QPushButton("Esporta Word")
        self.export_docx_button.clicked.connect(lambda: self._export("docx"))
        self.export_txt_button = QPushButton("Esporta testo")
        self.export_txt_button.clicked.connect(lambda: self._export("txt"))
        for button in (self.export_docx_button, self.export_txt_button):
            button.setEnabled(False)
            buttons.addWidget(button)

        buttons.addStretch(1)
        self.delete_report_button = QPushButton("Elimina report")
        self.delete_report_button.setObjectName("DangerButton")
        self.delete_report_button.setEnabled(False)
        self.delete_report_button.setToolTip(
            "Cancella il report generato. La trascrizione del colloquio "
            "resta salvata e il report puo' essere rigenerato."
        )
        self.delete_report_button.clicked.connect(self._delete_report)
        buttons.addWidget(self.delete_report_button)
        card.body().addLayout(buttons)

        layout.addWidget(card, 1)
        return page

    # ==================================================================
    # Scheda "Storico"
    # ==================================================================
    def _build_history_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        left = Card("Colloqui salvati")
        self.history_list = QListWidget()
        self.history_list.currentItemChanged.connect(self._show_history_item)
        left.add(self.history_list, 1)

        buttons = QHBoxLayout()
        self.history_export_button = QPushButton("Esporta Word")
        self.history_export_button.clicked.connect(self._export_from_history)
        self.history_delete_button = QPushButton("Elimina")
        self.history_delete_button.clicked.connect(self._delete_from_history)
        buttons.addWidget(self.history_export_button)
        buttons.addWidget(self.history_delete_button)
        buttons.addStretch(1)
        left.body().addLayout(buttons)
        layout.addWidget(left, 2)

        right = Card("Dettaglio")
        self.history_detail = QTextEdit()
        self.history_detail.setReadOnly(True)
        self.history_detail.setPlaceholderText(
            "Seleziona un colloquio per rivederne report e trascrizione."
        )
        right.add(self.history_detail, 1)
        layout.addWidget(right, 3)
        return page

    # ==================================================================
    # Scheda "Impostazioni"
    # ==================================================================
    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        left = Card("Audio e trascrizione")
        self.model_combo = QComboBox()
        for size in config.WHISPER_MODEL_SIZES:
            self.model_combo.addItem(
                {
                    "tiny": "Minimo — velocissimo, meno preciso",
                    "base": "Base — veloce, consigliato su PC modesti",
                    "small": "Piccolo — piu' preciso, richiede 4 core o piu'",
                    "medium": "Medio — molto preciso, solo su PC potenti",
                }[size],
                size,
            )
        self._select_data(
            self.model_combo,
            settings.get("whisper_model_size"),
            settings.DEFAULTS["whisper_model_size"],
        )

        self.language_combo = QComboBox()
        self.language_combo.addItem("Rilevamento automatico", "auto")
        for code, name in (
            ("it", "Italiano"), ("en", "Inglese"), ("es", "Spagnolo"),
            ("fr", "Francese"), ("de", "Tedesco"), ("pt", "Portoghese"),
        ):
            self.language_combo.addItem(name, code)
        self._select_data(
            self.language_combo,
            settings.get("transcription_language"),
            settings.DEFAULTS["transcription_language"],
        )

        self.echo_combo = QComboBox()
        self.echo_combo.addItem(
            "Disattivata — corretta se si usano le cuffie", echo_module.MODE_OFF
        )
        self.echo_combo.addItem(
            "Automatica — scarta l'eco degli altoparlanti (consigliata)",
            echo_module.MODE_AUTO,
        )
        self.echo_combo.addItem(
            "Avanzata — cancella l'eco e conserva le sovrapposizioni di voce",
            echo_module.MODE_CANCEL,
        )
        self._select_data(
            self.echo_combo,
            settings.get("echo_mode"),
            settings.DEFAULTS["echo_mode"],
        )

        self.mic_check = QCheckBox("Registra il microfono (la tua voce)")
        self.mic_check.setChecked(bool(settings.get("capture_microphone", True)))
        self.system_check = QCheckBox(
            "Registra l'audio del computer (la voce del candidato)"
        )
        self.system_check.setChecked(bool(settings.get("capture_system_audio", True)))

        for label, widget in (
            ("Modello di trascrizione", self.model_combo),
            ("Lingua del colloquio", self.language_combo),
            ("Gestione dell'eco senza cuffie", self.echo_combo),
        ):
            caption = QLabel(label)
            caption.setProperty("class", "Muted")
            left.add(caption)
            left.add(widget)
        left.add(self.mic_check)
        left.add(self.system_check)

        echo_help = QLabel(
            "Senza cuffie la voce del candidato esce dalle casse e rientra nel "
            "microfono: la stessa frase verrebbe attribuita a entrambi. La "
            "modalita' automatica riconosce e scarta queste ripetizioni; quella "
            "avanzata le sottrae dal segnale, cosi' resta udibile anche chi "
            "interviene mentre l'altro parla, al prezzo di un po' di calcolo in piu'."
        )
        echo_help.setWordWrap(True)
        echo_help.setProperty("class", "Muted")
        left.add(echo_help)
        left.body().addStretch(1)
        outer.addWidget(left, 1)

        right = Card("Prestazioni e diagnostica")
        self.cpu_combo = QComboBox()
        self.cpu_combo.addItem("Automatica (consigliata)", "auto")
        self.cpu_combo.addItem("Compatibilita' — piu' lenta, sempre stabile", "compatible")
        self.cpu_combo.addItem("Prestazioni — piu' veloce, richiede CPU recente", "fast")
        self._select_data(
            self.cpu_combo,
            settings.get("cpu_mode"),
            settings.DEFAULTS["cpu_mode"],
        )
        caption = QLabel("Modalita' di calcolo")
        caption.setProperty("class", "Muted")
        right.add(caption)
        right.add(self.cpu_combo)

        self.hardware_label = QLabel(compat.describe_cpu())
        self.hardware_label.setWordWrap(True)
        self.hardware_label.setProperty("class", "Muted")
        right.add(self.hardware_label)

        cores = os.cpu_count() or 2
        if cores <= 2:
            weak = QLabel(
                f"Questo computer ha {cores} core: con il modello 'Piccolo' la "
                "trascrizione accumulera' ritardo. Scegli 'Base' per seguire il "
                "parlato in tempo reale."
            )
            weak.setWordWrap(True)
            weak.setStyleSheet(f"color: {theme.WARNING};")
            right.add(weak)

        self.retest_button = QPushButton("Ripeti il test di compatibilita'")
        self.retest_button.clicked.connect(self._retest_engine)
        right.add(self.retest_button)

        logs_button = QPushButton("Apri la cartella dei log")
        logs_button.clicked.connect(self._open_logs)
        right.add(logs_button)

        save_button = QPushButton("Salva impostazioni")
        save_button.setObjectName("AccentButton")
        save_button.clicked.connect(self._save_settings)
        right.add(save_button)
        right.body().addStretch(1)
        outer.addWidget(right, 1)
        return page

    # ==================================================================
    # Anteprima della videochiamata
    # ==================================================================
    def _refresh_window_list(self, notify: bool = False) -> None:
        current = self.call_view.source.handle if self.call_view.source else None
        windows = list_windows(exclude=int(self.winId()))

        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        self.window_combo.addItem("Nessuna anteprima", 0)
        for info in windows:
            self.window_combo.addItem(info.label, info.handle)
        if current:
            index = self.window_combo.findData(current)
            self.window_combo.setCurrentIndex(index if index >= 0 else 0)
        self.window_combo.blockSignals(False)

        if notify and not windows:
            QMessageBox.information(
                self,
                "Nessuna finestra disponibile",
                "Non ho trovato finestre da mostrare. Apri Teams, Zoom o la "
                "scheda del browser con la videochiamata, poi premi Aggiorna.",
            )

    def _auto_select_call_window(self) -> None:
        self._refresh_window_list()
        if not self.call_view.available:
            return
        best = best_call_window(exclude=int(self.winId()))
        if best is None:
            return
        if self.call_view.set_source(best):
            index = self.window_combo.findData(best.handle)
            if index >= 0:
                self.window_combo.setCurrentIndex(index)
            log.info("Anteprima agganciata a '%s'", best.title)

    def _on_window_selected(self, index: int) -> None:
        handle = self.window_combo.itemData(index)
        if not handle:
            self.call_view.clear()
            return
        for info in list_windows(exclude=int(self.winId())):
            if info.handle == handle:
                if not self.call_view.set_source(info):
                    QMessageBox.information(
                        self,
                        "Anteprima non disponibile",
                        "Non e' stato possibile agganciare questa finestra. "
                        "Verifica che non sia ridotta a icona e riprova.",
                    )
                return
        self._refresh_window_list()

    def _on_tab_changed(self, index: int) -> None:
        # La miniatura della videochiamata viene disegnata dal sistema
        # sopra la nostra finestra: va nascosta quando si cambia scheda,
        # altrimenti resterebbe visibile sopra il report.
        self.call_view.set_visible_thumbnail(index == TAB_INTERVIEW)
        if index == TAB_INTERVIEW:
            self.call_view.refresh()

    def moveEvent(self, event):  # noqa: N802 - firma imposta da Qt
        super().moveEvent(event)
        if hasattr(self, "call_view"):
            self.call_view.refresh()

    def resizeEvent(self, event):  # noqa: N802 - firma imposta da Qt
        super().resizeEvent(event)
        if hasattr(self, "call_view"):
            self.call_view.refresh()

    # ==================================================================
    # Preparazione all'avvio
    # ==================================================================
    def _keep_alive(self, worker) -> None:
        self._live_workers.add(worker)
        worker.finished.connect(lambda w=worker: self._live_workers.discard(w))

    def _start_preparation(self) -> None:
        if self.startup_worker is not None and self.startup_worker.isRunning():
            return

        self.record_button.setEnabled(False)
        self.retest_button.setEnabled(False)
        self.status_label.setText("Verifica dei componenti...")

        self.startup_worker = StartupWorker(settings.get("whisper_model_size", "small"))
        self._keep_alive(self.startup_worker)
        self.startup_worker.stage.connect(self.status_label.setText)
        self.startup_worker.progress.connect(self._on_download_progress)
        self.startup_worker.finished_ok.connect(self._on_ready)
        self.startup_worker.failed.connect(self._on_preparation_failed)
        self.startup_worker.start()

    def _on_download_progress(self, component: str, done: int, total: int) -> None:
        self.progress_bar.setVisible(True)
        if total > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(min(100, int(done / total * 100)))
            self.status_label.setText(
                f"Download del modello di {component}: "
                f"{done / 1048576:.0f} MB di {total / 1048576:.0f} MB"
            )
        else:
            self.progress_bar.setRange(0, 0)

    def _on_ready(self, compatible_mode: bool, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.engine_ready = True
        self.record_button.setEnabled(True)
        self.retest_button.setEnabled(True)
        self.status_label.setText(
            "Pronto. Inserisci il nome del candidato e avvia il colloquio."
        )
        if message:
            self._add_warning(message)

    def _on_preparation_failed(self, message: str, detail: str) -> None:
        self.progress_bar.setVisible(False)
        self.engine_ready = False
        self.record_button.setEnabled(False)
        self.retest_button.setEnabled(True)
        self.status_label.setText(message)
        self.header_dot.set_color(theme.DANGER)

        text = message
        if compat.is_emulated():
            text += (
                "\n\nQuesto computer usa un processore ARM64 con emulazione x64: "
                "e' una causa possibile."
            )
        text += f"\n\nI dettagli tecnici sono nel file di log:\n{config.LOG_DIR}"
        QMessageBox.critical(self, "Motore di trascrizione non disponibile", text)
        if detail:
            log.error("Dettaglio del test fallito: %s", detail[:4000])

    # ==================================================================
    # Registrazione
    # ==================================================================
    def _toggle_recording(self) -> None:
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if not self.engine_ready:
            return
        if not self.candidate_input.text().strip():
            QMessageBox.information(
                self,
                "Dati mancanti",
                "Inserisci almeno il nome del candidato prima di iniziare.",
            )
            self.candidate_input.setFocus()
            return

        # Un report del colloquio precedente ancora in corso continuerebbe
        # a scrivere nella schermata del nuovo, e non verrebbe mai
        # salvato: per l'interfaccia quel colloquio non esiste piu'.
        if self.report_worker is not None and self.report_worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Report ancora in corso",
                "E' ancora in corso la generazione del report del colloquio "
                "precedente. Avviando adesso un nuovo colloquio quel report "
                "andrebbe perso.\n\nVuoi interromperlo e procedere?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            self.report_worker.cancel()
            self.report_worker.wait(5000)
            self.report_worker = None

        self._release_session()

        self.transcript_view.clear()
        self.report_view.clear()
        self._warnings.clear()
        self._pending = {}
        self.warning_label.setVisible(False)
        self.echo_chip.setVisible(False)
        self.current_interview_id = None
        self.export_docx_button.setEnabled(False)
        self.export_txt_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.delete_report_button.setEnabled(False)
        # I campi del candidato precedente vanno ripuliti: altrimenti le
        # sue note e le sue competenze finirebbero salvate sul colloquio
        # della persona successiva.
        self.notes_edit.clear()
        self.experience_input.clear()
        self.skills_input.clear()

        self.session = InterviewSession(
            whisper_model_size=settings.get("whisper_model_size", "small"),
            capture_microphone=bool(settings.get("capture_microphone", True)),
            capture_system_audio=bool(settings.get("capture_system_audio", True)),
            language=settings.get("transcription_language", "auto"),
            echo_mode=settings.get("echo_mode", echo_module.MODE_AUTO),
            # Con parent=None l'ultimo riferimento all'oggetto poteva
            # restare in un thread di servizio, e il distruttore Qt
            # sarebbe stato eseguito fuori dal thread grafico: causa
            # nota di chiusure anomale sporadiche.
            parent=self,
        )
        self.session.segment_received.connect(self._on_segment)
        self.session.level_changed.connect(self._on_level)
        self.session.status_changed.connect(self.status_label.setText)
        self.session.warning_raised.connect(self._add_warning)
        self.session.error_raised.connect(self._add_warning)
        self.session.session_started.connect(self._on_session_started)
        self.session.start_failed.connect(self._on_session_start_failed)
        self.session.stopped.connect(self._on_session_stopped)

        self.record_button.setEnabled(False)
        self.status_label.setText("Avvio della registrazione...")
        self.session.start()

    def _on_session_started(self, sources: dict) -> None:
        self.is_recording = True
        self.record_button.setEnabled(True)
        self.record_button.setText("Termina colloquio")
        self.record_button.setObjectName("StopButton")
        self._restyle(self.record_button)
        self.recording_dot.set_color(theme.DANGER)

        for speaker, label in self.source_labels.items():
            device = sources.get(speaker)
            label.setToolTip(device or "Sorgente non attiva")
            label.setEnabled(bool(device))
        self.status_label.setText("Registrazione in corso.")

    def _on_session_start_failed(self, message: str) -> None:
        self.is_recording = False
        self.record_button.setEnabled(True)
        self.status_label.setText("Avvio non riuscito.")
        self._release_session()
        QMessageBox.warning(self, "Impossibile avviare la registrazione", message)

    def _release_session(self) -> None:
        if self.session is None:
            return
        try:
            self.session.disconnect()
        except (RuntimeError, TypeError):
            pass
        # deleteLater fa distruggere l'oggetto dal ciclo di eventi
        # grafico. Senza, l'ultimo riferimento poteva restare nel thread
        # che ha eseguito l'arresto, e distruggere un oggetto Qt fuori
        # dal proprio thread ha esito imprevedibile.
        try:
            self.session.deleteLater()
        except RuntimeError:
            pass
        self.session = None

    def _stop_recording(self) -> None:
        if not self.session:
            return
        self.record_button.setEnabled(False)
        self.status_label.setText(
            "Chiusura della registrazione: attendo le ultime frasi..."
        )
        self.recording_dot.set_color(theme.TEXT_MUTED)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        # L'arresto e' asincrono: smaltire la coda puo' richiedere
        # decine di secondi e bloccare qui l'interfaccia mostrerebbe
        # all'utente una finestra "non risponde" proprio mentre il
        # colloquio non e' ancora stato salvato.
        self.session.stop()

    def _on_session_stopped(self) -> None:
        session = self.session
        if session is None:
            return

        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

        self._pending = {
            "duration": session.elapsed_seconds,
            "segments": session.segments(),
            "transcript": session.transcript(self._labels()),
            "language": session.detected_language or "it",
        }
        self._release_session()

        self.is_recording = False
        self.record_button.setText("Avvia colloquio")
        self.record_button.setObjectName("PrimaryStart")
        self._restyle(self.record_button)
        self.record_button.setEnabled(True)
        for meter in self.level_meters.values():
            meter.set_level(0.0)

        if self.closing:
            self.close()
            return

        self._save_current_interview()

        if settings.get("auto_generate_report", True):
            self.tabs.setCurrentIndex(TAB_REPORT)
            self._generate_report()
        else:
            self.report_button.setEnabled(True)
            self.status_label.setText("Colloquio salvato. Puoi generare il report.")

    def _save_current_interview(self) -> None:
        interview = db.Interview(
            candidate_name=self.candidate_input.text().strip() or "Senza nome",
            role=self.role_input.text().strip(),
            created_at=datetime.now().isoformat(timespec="seconds"),
            duration_seconds=self._pending.get("duration", 0.0),
            detected_language=self._pending.get("language", ""),
            transcript=self._pending.get("transcript", ""),
            report="",
            segments=self._pending.get("segments", []),
            notes=self.notes_edit.toPlainText().strip(),
            experience=self.experience_input.text().strip(),
            skills=self.skills_input.text().strip(),
        )
        try:
            self.current_interview_id = db.save_interview(interview)
            self._refresh_history()
            self.export_docx_button.setEnabled(True)
            self.export_txt_button.setEnabled(True)
        except Exception:
            log.exception("Salvataggio del colloquio non riuscito")
            self._add_warning(
                "Il colloquio non e' stato salvato nell'archivio: controlla i log."
            )

    # ==================================================================
    # Report
    # ==================================================================
    def _generate_report(self) -> None:
        if self.report_worker is not None and self.report_worker.isRunning():
            QMessageBox.information(
                self,
                "Report gia' in corso",
                "E' gia' in corso la generazione di un report: attendi che finisca.",
            )
            return

        if not self._pending.get("segments"):
            QMessageBox.information(
                self,
                "Nessuna trascrizione",
                "Non e' stata registrata alcuna frase: non c'e' nulla da riassumere.",
            )
            # Senza questa riga il pulsante restava grigio per sempre e
            # l'utente non aveva modo di capire perche'.
            self.report_button.setEnabled(True)
            return

        self.report_button.setEnabled(False)
        # Anche l'eliminazione va bloccata: cancellare mentre il report
        # si sta scrivendo lo faceva ricomparire pochi secondi dopo,
        # perche' il processo in corso lo riscriveva a schermo e
        # nell'archivio.
        self.delete_report_button.setEnabled(False)
        self.status_label.setText(
            "Generazione del report in corso: richiede qualche minuto."
        )
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self.report_worker = ReportWorker(
            transcript=self._pending.get("transcript", ""),
            segments=self._pending.get("segments", []),
            labels=self._labels(),
            candidate_name=self.candidate_input.text().strip(),
            role=self.role_input.text().strip(),
            duration_seconds=self._pending.get("duration", 0.0),
            detected_language=self._pending.get("language", "it"),
            report_language=settings.get("report_language", "auto"),
        )
        self._keep_alive(self.report_worker)
        self.report_worker.finished_ok.connect(self._on_report_ready)
        self.report_worker.failed.connect(self._on_report_failed)
        self.report_worker.partial.connect(self._on_report_partial)
        self.report_view.clear()
        self._streamed_report = False
        self.report_worker.start()

    def _on_report_partial(self, pezzo: str) -> None:
        """Mostra il report mentre viene scritto, invece che alla fine."""
        # Un worker precedente puo' essere ancora vivo e continuare a
        # inviare testo: scriverlo qui lo mescolerebbe al report attuale.
        if self.sender() is not self.report_worker:
            return
        if not self._streamed_report:
            self._streamed_report = True
            self.status_label.setText("Il report si sta scrivendo...")
            self.progress_bar.setVisible(False)
        cursor = self.report_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(pezzo)
        scrollbar = self.report_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_report_ready(self, result) -> None:
        if self.sender() is not self.report_worker:
            log.info("Report di un colloquio precedente ignorato")
            return
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        # Il testo definitivo sostituisce quello mostrato durante la
        # scrittura: e' identico, ma cosi' eventuali frammenti persi per
        # strada non restano nella schermata.
        self.report_view.setPlainText(result.text)
        self.report_button.setEnabled(True)
        self.delete_report_button.setEnabled(bool(result.text.strip()))

        if self.current_interview_id is not None:
            try:
                db.update_report(self.current_interview_id, result.text, result.used_llm)
                self._refresh_history()
            except Exception:
                log.exception("Aggiornamento del report non riuscito")

        if result.used_llm:
            self.status_label.setText("Report generato e colloquio salvato.")
        else:
            self.status_label.setText("Report di riserva generato.")
            self._add_warning(
                "Il modello linguistico non e' stato utilizzabile: e' stato "
                f"prodotto un resoconto essenziale dalla trascrizione. {result.warning}"
            )

    def _delete_report(self) -> None:
        """Cancella il report mantenendo il colloquio e la sua trascrizione."""
        if not self.report_view.toPlainText().strip():
            return
        if self.report_worker is not None and self.report_worker.isRunning():
            QMessageBox.information(
                self,
                "Report in corso",
                "Il report si sta ancora scrivendo: attendi che finisca "
                "prima di eliminarlo.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Eliminare il report?",
            "Il report verra' cancellato. La trascrizione del colloquio resta "
            "salvata e potrai generare un nuovo report quando vuoi.\n\nProcedere?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self.report_view.clear()
        self.delete_report_button.setEnabled(False)

        if self.current_interview_id is not None:
            try:
                db.update_report(self.current_interview_id, "", False)
                self._refresh_history()
            except Exception:
                log.exception("Cancellazione del report non riuscita")
                self._add_warning(
                    "Il report e' stato tolto dalla schermata ma non "
                    "dall'archivio: controlla i log."
                )
        self.status_label.setText("Report eliminato. Puoi generarne uno nuovo.")

    def _on_report_failed(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.report_button.setEnabled(True)
        self.status_label.setText("Errore nella generazione del report.")
        QMessageBox.warning(self, "Report non generato", message)

    # ==================================================================
    # Aggiornamenti dell'interfaccia
    # ==================================================================
    def _labels(self) -> dict[str, str]:
        return {
            config.SPEAKER_RECRUITER: settings.get("label_recruiter", "Tu"),
            config.SPEAKER_CANDIDATE: settings.get("label_candidate", "Candidato"),
        }

    def _on_segment(self, segment: dict) -> None:
        if not self.captions_button.isChecked():
            return
        speaker = segment.get("speaker", "")
        label = self._labels().get(speaker, speaker)
        color = theme.SPEAKER_COLORS.get(speaker, theme.TEXT)
        stamp = datetime.now().strftime("%H:%M")
        text = html.escape(segment.get("text", ""))

        cursor = self.transcript_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertHtml(
            f'<div style="margin-bottom:10px">'
            f'<span style="color:{theme.TEXT_MUTED};font-size:11px">[{stamp}] </span>'
            f'<span style="color:{color};font-weight:600">{html.escape(label)}</span>'
            f'<br><span style="color:{theme.TEXT}">{text}</span>'
            f"</div><br>"
        )
        scrollbar = self.transcript_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_level(self, speaker: str, level: float) -> None:
        meter = self.level_meters.get(speaker)
        if meter is not None:
            meter.set_level(level)

    def _add_warning(self, message: str) -> None:
        if not message or message in self._warnings:
            return
        self._warnings.append(message)
        self.warning_label.setText(" • ".join(self._warnings[-3:]))
        self.warning_label.setVisible(True)
        log.warning("Avviso mostrato all'utente: %s", message)

    def _tick(self) -> None:
        if self.session is None:
            seconds = int(self._pending.get("duration", 0.0))
            self.timer_label.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")
            return

        seconds = int(self.session.elapsed_seconds)
        self.timer_label.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")

        if self.is_recording:
            stats = self.session.statistics()
            testo = (
                f"Interventi registrati: {stats['segments']} — "
                f"domande poste: {stats['questions']} — "
                f"parlato del candidato: {stats['candidate_share']}%"
            )
            velocita = self.session.realtime_factor
            if velocita:
                testo += f" — trascrizione a {velocita:.1f}× il tempo reale"
            # Riscrivere l'etichetta solo quando cambia evita un
            # ridisegno al secondo, che su un computer modesto si nota.
            if testo != self._stats_cache:
                self._stats_cache = testo
                self.stats_label.setText(testo)

            if self.session.speakers_detected and not self.echo_chip.isVisible():
                self.echo_chip.setText("Altoparlanti rilevati — eco gestita")
                self.echo_chip.setVisible(True)

            arretrato = self.session.pending_chunks
            if arretrato >= 4:
                self._add_warning(
                    "La trascrizione non sta al passo del parlato: nelle "
                    "impostazioni puoi scegliere un modello piu' leggero."
                )

    def _refresh_platform(self) -> None:
        if self.platform_probe is not None and self.platform_probe.isRunning():
            return
        self.platform_probe = PlatformProbe()
        self._keep_alive(self.platform_probe)
        self.platform_probe.result.connect(self._on_platform_status)
        self.platform_probe.start()

    def _on_platform_status(self, status) -> None:
        self.platform_label.setText(status.summary)
        self.platform_dot.set_color(
            theme.SUCCESS if status.is_active else theme.TEXT_MUTED
        )

    def _refresh_skill_chips(self, text: str) -> None:
        while self.skills_row.count() > 1:
            item = self.skills_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for raw in text.split(",")[:8]:
            skill = raw.strip()
            if skill:
                self.skills_row.insertWidget(
                    self.skills_row.count() - 1, Chip(skill, "SkillChip")
                )

    def _toggle_captions(self, enabled: bool) -> None:
        self.captions_button.setText(
            "Sottotitoli live: attivi" if enabled else "Sottotitoli live: nascosti"
        )
        if not enabled:
            self.transcript_view.setPlaceholderText(
                "Sottotitoli nascosti. La registrazione e la trascrizione "
                "proseguono: utile mentre condividi lo schermo."
            )

    def _apply_always_on_top(self, enabled: bool) -> None:
        settings.set("always_on_top", bool(enabled))
        self.setWindowFlag(Qt.WindowStaysOnTopHint, bool(enabled))
        self.show()
        if not hasattr(self, "call_view"):
            return
        # Su Windows cambiare i flag di una finestra principale la fa
        # ricreare: l'identificativo nativo cambia. L'anteprima della
        # videochiamata era pero' agganciata a quello vecchio, quindi
        # spariva e non tornava piu'. Va riagganciata alla finestra
        # nuova, non semplicemente ridisegnata.
        sorgente = getattr(self.call_view, "source", None)
        if sorgente is not None:
            self.call_view.set_source(sorgente)
        else:
            self.call_view.refresh()

    @staticmethod
    def _select_data(combo, value, fallback) -> None:
        """
        Seleziona un valore nel menu a tendina, con ripiego esplicito.

        Il vecchio max(0, indice) ripiegava sul PRIMO elemento: se il
        file delle impostazioni conteneva un valore non piu' previsto,
        la schermata mostrava "Rilevamento automatico" mentre il
        programma continuava a usare di nascosto il valore vecchio.
        """
        index = combo.findData(value)
        if index < 0:
            log.warning(
                "Valore '%s' non presente nel menu: uso '%s'", value, fallback
            )
            index = max(0, combo.findData(fallback))
        combo.setCurrentIndex(index)

    @staticmethod
    def _restyle(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # ==================================================================
    # Esportazione e storico
    # ==================================================================
    def _export(self, fmt: str) -> None:
        if self.current_interview_id is None:
            return
        # Un'eccezione dentro uno slot Qt non fa cadere il programma ma
        # non produce nemmeno alcun effetto: senza questa protezione il
        # pulsante sembrava semplicemente non funzionare.
        try:
            interview = db.get_interview(self.current_interview_id)
        except Exception:
            log.exception("Lettura del colloquio non riuscita")
            self._add_warning(
                "Non e' stato possibile leggere il colloquio dall'archivio."
            )
            return
        if interview is None:
            return
        interview.notes = self.notes_edit.toPlainText().strip()
        interview.experience = self.experience_input.text().strip()
        interview.skills = self.skills_input.text().strip()
        try:
            db.update_notes(
                interview.id,
                interview.notes,
                experience=interview.experience,
                skills=interview.skills,
            )
        except Exception:
            log.exception("Salvataggio delle note non riuscito")
        self._write_export(interview, fmt)

    def _write_export(self, interview, fmt: str) -> None:
        try:
            if fmt == "docx":
                path = export_docx(interview, self._labels())
            else:
                path = export_txt(interview, self._labels())
        except Exception as exc:
            log.exception("Esportazione non riuscita")
            QMessageBox.warning(
                self, "Esportazione non riuscita", f"Non e' stato possibile salvare: {exc}"
            )
            return

        answer = QMessageBox.question(
            self,
            "Esportazione completata",
            f"File salvato in:\n{path}\n\nVuoi aprire la cartella?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self._open_folder(path.parent)

    def _refresh_history(self) -> None:
        try:
            colloqui = db.list_interviews()
        except Exception:
            log.exception("Lettura dell'archivio non riuscita")
            self._add_warning("L'archivio dei colloqui non e' leggibile.")
            return
        self.history_list.clear()
        for interview in colloqui:
            item = QListWidgetItem(
                f"{interview.display_date}  —  {interview.candidate_name}"
                + (f"  ({interview.role})" if interview.role else "")
            )
            item.setData(Qt.UserRole, interview.id)
            self.history_list.addItem(item)

    def _selected_history_interview(self):
        item = self.history_list.currentItem()
        if item is None:
            return None
        try:
            return db.get_interview(item.data(Qt.UserRole))
        except Exception:
            log.exception("Lettura del colloquio selezionato non riuscita")
            self._add_warning("Il colloquio selezionato non e' leggibile.")
            return None

    def _show_history_item(self, current, _previous=None) -> None:
        if current is None:
            self.history_detail.clear()
            return
        try:
            interview = db.get_interview(current.data(Qt.UserRole))
        except Exception:
            log.exception("Lettura del colloquio non riuscita")
            self.history_detail.setPlainText(
                "Questo colloquio non e' leggibile dall'archivio. "
                "Il dettaglio dell'errore e' nel file di log."
            )
            return
        if interview is None:
            return

        labels = self._labels()
        lines = [
            f"Candidato: {interview.candidate_name}",
            f"Posizione: {interview.role or 'non indicata'}",
            f"Data: {interview.display_date}",
            f"Durata: {int(interview.duration_seconds // 60)} min",
            "",
            "=== REPORT ===",
            interview.report or "(report non generato)",
        ]
        if interview.notes:
            lines += ["", "=== NOTE ===", interview.notes]
        lines += ["", "=== TRASCRIZIONE ==="]
        if interview.segments:
            lines += [
                f"{labels.get(s.get('speaker', ''), s.get('speaker', ''))}: {s.get('text', '')}"
                for s in interview.segments
            ]
        else:
            lines.append(interview.transcript)
        self.history_detail.setPlainText("\n".join(lines))

    def _export_from_history(self) -> None:
        interview = self._selected_history_interview()
        if interview is None:
            QMessageBox.information(
                self, "Nessuna selezione", "Seleziona prima un colloquio dall'elenco."
            )
            return
        self._write_export(interview, "docx")

    def _delete_from_history(self) -> None:
        interview = self._selected_history_interview()
        if interview is None:
            return
        answer = QMessageBox.question(
            self,
            "Eliminare il colloquio?",
            f"Il colloquio con {interview.candidate_name} verra' eliminato "
            "definitivamente dall'archivio. Procedere?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        db.delete_interview(interview.id)
        if self.current_interview_id == interview.id:
            self.current_interview_id = None
        self._refresh_history()
        self.history_detail.clear()

    # ==================================================================
    # Impostazioni
    # ==================================================================
    def _save_settings(self) -> None:
        previous_model = settings.get("whisper_model_size", "small")
        previous_cpu = settings.get("cpu_mode", "auto")
        new_model = self.model_combo.currentData()
        new_cpu = self.cpu_combo.currentData()

        settings.set_many(
            {
                "whisper_model_size": new_model,
                "transcription_language": self.language_combo.currentData(),
                "echo_mode": self.echo_combo.currentData(),
                "capture_microphone": self.mic_check.isChecked(),
                "capture_system_audio": self.system_check.isChecked(),
                "cpu_mode": new_cpu,
            }
        )

        if new_model != previous_model or new_cpu != previous_cpu:
            settings.set_many({"engine_selftest": "", "engine_selftest_size": ""})
            QMessageBox.information(
                self,
                "Impostazioni salvate",
                "Chiudi e riapri l'applicazione per applicare le nuove "
                "impostazioni del motore di trascrizione.",
            )
        else:
            QMessageBox.information(
                self, "Impostazioni salvate", "Le modifiche sono state applicate."
            )

    def _retest_engine(self) -> None:
        settings.set_many({"engine_selftest": "", "engine_selftest_size": ""})
        self._start_preparation()

    def _open_logs(self) -> None:
        self._open_folder(config.LOG_DIR)

    @staticmethod
    def _open_folder(path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(str(path))  # noqa: S606 - apertura cartella richiesta
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            log.exception("Apertura della cartella non riuscita")

    # ==================================================================
    def closeEvent(self, event):  # noqa: N802 - firma imposta da Qt
        if self.is_recording and not self.closing:
            answer = QMessageBox.question(
                self,
                "Colloquio in corso",
                "La registrazione e' ancora attiva. Chiudere comunque? "
                "Il colloquio non verra' salvato.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        if self.session is not None:
            self.closing = True
            self.status_label.setText("Chiusura in corso...")
            try:
                self.session.stop()
            except Exception:
                log.exception("Chiusura della sessione non riuscita")
                self.session = None
            else:
                if self.session is not None and self.session.is_stopping:
                    event.ignore()
                    return

        self.elapsed_timer.stop()
        self.platform_timer.stop()
        self.call_view.clear()

        for worker in (self.report_worker, self.startup_worker, self.platform_probe):
            if worker is None or not worker.isRunning():
                continue
            if hasattr(worker, "cancel"):
                worker.cancel()
            else:
                worker.requestInterruption()
            if not worker.wait(5000):
                log.warning(
                    "Attesa prolungata di %s alla chiusura", type(worker).__name__
                )
                # Mai un'attesa senza limite di tempo: bloccherebbe il
                # thread grafico e Windows dichiarerebbe la finestra
                # "non risponde", senza alcuna via d'uscita per l'utente.
                if not worker.wait(3000):
                    log.error(
                        "%s non si e' fermato: lo termino", type(worker).__name__
                    )
                    worker.terminate()
                    worker.wait(2000)

        event.accept()
