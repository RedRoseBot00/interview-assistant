"""
Collegamento tra i moduli di base (audio e trascrizione, che girano su
thread Python) e l'interfaccia Qt, che puo' essere aggiornata solo dal
thread grafico.

Tutto passa attraverso i segnali Qt, che gestiscono automaticamente il
cambio di thread. Anche l'avvio e l'arresto avvengono fuori dal thread
grafico: aprire i dispositivi audio e smaltire la coda di trascrizione
puo' richiedere parecchi secondi, durante i quali Windows dichiarerebbe
la finestra "non risponde".
"""
from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, Signal

from app import config
from app.audio.capture import AudioError, AudioRecorder
from app.transcription.engine import TranscriptionEngine, TranscriptSegment

log = logging.getLogger(__name__)

LEVEL_EMIT_INTERVAL = 0.08  # secondi: evita di inondare l'interfaccia


class InterviewSession(QObject):
    segment_received = Signal(dict)
    level_changed = Signal(str, float)
    status_changed = Signal(str)
    warning_raised = Signal(str)
    error_raised = Signal(str)
    session_started = Signal(dict)     # {etichetta: nome dispositivo}
    start_failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        whisper_model_size: str,
        capture_microphone: bool = True,
        capture_system_audio: bool = True,
        language: str = "auto",
        echo_mode: str = "auto",
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._last_level_emit: dict[str, float] = {}

        self._lifecycle = threading.Lock()
        self._starting = False
        self._stopping = False
        self._stop_requested = False
        self.clean_shutdown = True

        self.recorder = AudioRecorder(
            capture_microphone=capture_microphone,
            capture_system_audio=capture_system_audio,
            on_level=self._handle_level,
            on_error=self._handle_audio_error,
        )
        self.engine = TranscriptionEngine(
            model_size=whisper_model_size,
            language=language,
            echo_mode=echo_mode,
            on_segment=self._handle_segment,
            on_status=self._handle_status,
            on_error=self._handle_engine_error,
        )

    # ------------------------------------------------------------------
    # Callback provenienti dai thread audio/trascrizione
    # ------------------------------------------------------------------
    def _handle_level(self, speaker: str, level: float) -> None:
        now = time.monotonic()
        if now - self._last_level_emit.get(speaker, 0.0) < LEVEL_EMIT_INTERVAL:
            return
        self._last_level_emit[speaker] = now
        self.level_changed.emit(speaker, level)

    def _handle_segment(self, segment: TranscriptSegment) -> None:
        self.segment_received.emit(segment.to_dict())

    def _handle_status(self, message: str) -> None:
        self.status_changed.emit(message)

    def _handle_engine_error(self, exc: Exception) -> None:
        self.error_raised.emit(str(exc))

    def _handle_audio_error(self, speaker: str, exc: Exception) -> None:
        self.warning_raised.emit(
            f"La sorgente audio '{speaker}' si e' interrotta: {exc}"
        )

    # ------------------------------------------------------------------
    # Avvio
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lifecycle:
            if self._starting or self._stop_requested:
                return
            self._starting = True
        threading.Thread(
            target=self._start_blocking, name="session-start", daemon=True
        ).start()

    def _aborted(self) -> bool:
        with self._lifecycle:
            return self._stop_requested

    def _start_blocking(self) -> None:
        try:
            if self._aborted():
                return

            self.status_changed.emit("Apertura dei dispositivi audio...")
            self.recorder.start()

            # L'utente puo' aver chiuso la finestra mentre aprivamo i
            # dispositivi: in quel caso smontiamo subito, invece di
            # lasciare thread che nessuno fermera' piu'.
            if self._aborted():
                self.recorder.stop()
                return

            for message in self.recorder.warnings:
                self.warning_raised.emit(message)

            self._started_at = time.monotonic()
            self.engine.start(self.recorder.audio_queue)

            if self._aborted():
                self.engine.stop(timeout=5)
                self.recorder.stop()
                return

            self.session_started.emit(dict(self.recorder.active_sources))
        except AudioError as exc:
            log.warning("Avvio non riuscito: %s", exc)
            self._cleanup_after_failed_start()
            self.start_failed.emit(str(exc))
        except Exception as exc:
            log.exception("Avvio della sessione non riuscito")
            self._cleanup_after_failed_start()
            self.start_failed.emit(
                f"Errore imprevisto all'avvio della registrazione: {exc}"
            )
        finally:
            with self._lifecycle:
                self._starting = False

    def _cleanup_after_failed_start(self) -> None:
        """Chiude tutto cio' che era gia' stato aperto prima dell'errore."""
        for closer in (lambda: self.engine.stop(timeout=5), self.recorder.stop):
            try:
                closer()
            except Exception:
                log.exception("Pulizia dopo avvio fallito non riuscita")

    # ------------------------------------------------------------------
    # Arresto
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """
        Avvia l'arresto e ritorna subito: al termine viene emesso il
        segnale 'stopped'. Solo a quel punto la trascrizione e'
        completa e i dati possono essere salvati.
        """
        with self._lifecycle:
            self._stop_requested = True
            if self._stopping:
                return
            self._stopping = True
        self._stopped_at = time.monotonic()
        threading.Thread(
            target=self._stop_blocking, name="session-stop", daemon=True
        ).start()

    def _stop_blocking(self) -> None:
        clean = True
        try:
            if not self.engine.stop():
                clean = False
                self.warning_raised.emit(
                    "La trascrizione non si e' conclusa del tutto: le ultime "
                    "frasi potrebbero mancare dal report."
                )
        except Exception:
            clean = False
            log.exception("Errore nella chiusura del motore di trascrizione")

        try:
            if not self.recorder.stop():
                clean = False
        except Exception:
            clean = False
            log.exception("Errore nella chiusura dei dispositivi audio")

        for message in self.recorder.warnings:
            self.warning_raised.emit(message)

        self.clean_shutdown = clean
        with self._lifecycle:
            self._stopping = False
        self.stopped.emit()

    @property
    def is_stopping(self) -> bool:
        with self._lifecycle:
            return self._stopping

    # ------------------------------------------------------------------
    # Dati della sessione
    # ------------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        end = self._stopped_at or time.monotonic()
        return max(0.0, end - self._started_at)

    @property
    def detected_language(self) -> str:
        return self.engine.detected_language or ""

    @property
    def pending_chunks(self) -> int:
        return self.engine.backlog

    @property
    def speakers_detected(self) -> bool:
        """True quando l'app ha riconosciuto l'eco degli altoparlanti."""
        return self.engine.speakers_detected

    def segments(self) -> list[dict]:
        return self.engine.segments_as_dicts()

    def transcript(self, labels: dict[str, str]) -> str:
        return self.engine.full_transcript(labels)

    def statistics(self) -> dict:
        """Numeri utili da mostrare accanto alle note."""
        segments = self.segments()
        recruiter = [s for s in segments if s["speaker"] == config.SPEAKER_RECRUITER]
        candidate = [s for s in segments if s["speaker"] == config.SPEAKER_CANDIDATE]
        recruiter_words = sum(len(s["text"].split()) for s in recruiter)
        candidate_words = sum(len(s["text"].split()) for s in candidate)
        total = recruiter_words + candidate_words
        return {
            "questions": sum(1 for s in recruiter if "?" in s["text"]),
            "recruiter_words": recruiter_words,
            "candidate_words": candidate_words,
            "candidate_share": round(100 * candidate_words / total) if total else 0,
            "segments": len(segments),
        }
