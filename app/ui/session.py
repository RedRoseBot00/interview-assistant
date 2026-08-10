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
        self._start_thread: threading.Thread | None = None
        self.clean_shutdown = True
        self._warnings_shown = 0
        # Vero quando l'interfaccia non c'e' piu' e i segnali vanno
        # lasciati cadere invece che emessi.
        self._detached = False

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

    # Questi tre metodi vengono chiamati DA ALTRI THREAD, e possono
    # arrivare dopo che la finestra ha gia' buttato via la sessione: se
    # il riconoscimento vocale e' rimasto indietro, il suo thread e'
    # fermo dentro una chiamata non interrompibile e finisce il lavoro
    # con calma, molto dopo che l'utente ha premuto "Termina". A quel
    # punto l'oggetto grafico sottostante non esiste piu' e Qt solleva
    # "Internal C++ object already deleted": era questo a far morire di
    # colpo il thread di trascrizione, portandosi via tutte le frasi
    # ancora da elaborare. Un segnale che non trova piu' nessuno non e'
    # un guasto: e' semplicemente troppo tardi, e va lasciato cadere.
    def _handle_segment(self, segment: TranscriptSegment) -> None:
        if self._detached:
            return
        try:
            self.segment_received.emit(segment.to_dict())
        except RuntimeError:
            self.detach()

    def _handle_status(self, message: str) -> None:
        if self._detached:
            return
        try:
            self.status_changed.emit(message)
        except RuntimeError:
            self.detach()

    def _handle_engine_error(self, exc: Exception) -> None:
        if self._detached:
            return
        try:
            self.error_raised.emit(str(exc))
        except RuntimeError:
            self.detach()

    def _handle_audio_error(self, speaker: str, exc: Exception) -> None:
        if self._detached:
            return
        try:
            self.warning_raised.emit(
                f"La sorgente audio '{speaker}' si e' interrotta: {exc}"
            )
        except RuntimeError:
            self.detach()

    def detach(self) -> None:
        """
        Smette di parlare con l'interfaccia.

        Da chiamare prima di distruggere la sessione, e automaticamente
        alla prima avvisaglia che l'oggetto grafico non c'e' piu'.
        """
        self._detached = True
        try:
            self.engine.detach_callbacks()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Avvio
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lifecycle:
            if self._starting or self._stop_requested:
                return
            self._starting = True
            # Il riferimento al thread va conservato: l'arresto deve
            # poterlo attendere. Senza, chi chiudeva la finestra durante
            # l'apertura dei dispositivi faceva partire i due percorsi in
            # parallelo, e il colloquio veniva salvato vuoto mentre il
            # motore di trascrizione si avviava a sessione gia' conclusa.
            self._start_thread = threading.Thread(
                target=self._start_blocking, name="session-start", daemon=True
            )
        self._start_thread.start()

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
            self._warnings_shown = len(self.recorder.warnings)

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
        """
        Chiude tutto cio' che era gia' stato aperto prima dell'errore.

        L'ordine e' lo stesso dell'arresto normale — prima le sorgenti
        audio, poi il motore — per la stessa ragione: le ultime frasi
        nascono proprio quando gli stream si chiudono, e con il motore
        gia' fermo finirebbero in una coda che nessuno legge.
        """
        for closer in (self.recorder.stop, lambda: self.engine.stop(timeout=5)):
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

        # Prima di toccare qualunque cosa, lasciamo finire l'avvio se e'
        # ancora in corso. _stop_requested e' gia' impostato, quindi
        # quell'avvio si fermera' da solo al primo controllo; qui
        # aspettiamo soltanto che abbia smesso di usare i dispositivi.
        avvio = self._start_thread
        if avvio is not None and avvio.is_alive():
            log.info("Attendo la conclusione dell'avvio prima di fermare")
            avvio.join(timeout=20)
            if avvio.is_alive():
                clean = False
                log.error("L'avvio della sessione non si e' concluso")

        # L'ordine conta: prima si fermano le sorgenti audio, poi il
        # motore. Chiudendo il motore per primo, l'ultima frase del
        # colloquio — che il rilevatore di voce consegna proprio quando
        # gli stream si chiudono — finirebbe in una coda che nessuno
        # legge piu', e sparirebbe dal report.
        try:
            if not self.recorder.stop():
                clean = False
        except Exception:
            clean = False
            log.exception("Errore nella chiusura dei dispositivi audio")

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

        # Gli avvisi gia' mostrati all'avvio non vanno ripetuti.
        nuovi = self.recorder.warnings[self._warnings_shown :]
        self._warnings_shown = len(self.recorder.warnings)
        for message in nuovi:
            self.warning_raised.emit(message)

        # Frasi abbandonate perche' la trascrizione era troppo indietro
        # quando il colloquio e' finito. Vanno dette: mancano dal report,
        # e chi le ha pronunciate merita di saperlo.
        abbandonate = int(getattr(self.engine, "dropped_on_stop", 0) or 0)
        if abbandonate:
            self.warning_raised.emit(
                f"{abbandonate} frasi non sono state trascritte: al termine "
                "del colloquio la trascrizione era ancora troppo indietro. "
                "Con un modello piu' leggero non succede."
            )

        # Il modello di trascrizione occupa centinaia di megabyte e non
        # serve piu' a nulla: il colloquio e' finito, e subito dopo il
        # programma ne carica un altro da due gigabyte per scrivere il
        # report. Su una macchina con quattro gigabyte di memoria — cioe'
        # su quasi ogni macchina virtuale — tenerli in memoria insieme
        # significa spostare mezza applicazione sul disco.
        #
        # Non basta smettere di usarlo: l'oggetto sessione e il thread
        # di avvio si tengono l'un l'altro, e finche' quel cerchio non si
        # spezza nessuno dei due viene liberato. Qui lo si spezza e si
        # chiede esplicitamente di fare pulizia.
        self._start_thread = None
        try:
            self.engine.release_model()
        except Exception:
            log.debug("Modello di trascrizione non liberato", exc_info=True)

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
    def pending_seconds(self) -> float:
        """Secondi di parlato ancora da trascrivere."""
        return float(getattr(self.engine, "backlog_seconds", 0.0))

    @property
    def suggested_model(self) -> str:
        """Modello consigliato quando quello scelto non regge."""
        return str(getattr(self.engine, "suggested_model", "") or "")

    @property
    def call_cost_seconds(self) -> float:
        """Secondi impiegati in media per trascrivere una frase."""
        return float(getattr(self.engine, "_call_cost", 0.0) or 0.0)

    @property
    def dropped_on_stop(self) -> int:
        """Frasi abbandonate perche' l'arretrato era troppo grande."""
        return int(getattr(self.engine, "dropped_on_stop", 0) or 0)

    @property
    def speakers_detected(self) -> bool:
        """True quando l'app ha riconosciuto l'eco degli altoparlanti."""
        return self.engine.speakers_detected

    @property
    def realtime_factor(self) -> float:
        """Quante volte il tempo reale riesce a trascrivere il computer."""
        return self.engine.realtime_factor

    def segments(self) -> list[dict]:
        return self.engine.segments_as_dicts()

    def transcript(self, labels: dict[str, str]) -> str:
        return self.engine.full_transcript(labels)

    def statistics(self) -> dict:
        """
        Numeri utili da mostrare accanto alle note.

        Il calcolo e' tenuto aggiornato dal motore man mano che le frasi
        arrivano: qui non si scorre nulla. Prima veniva rifatto da capo
        una volta al secondo su tutti i segmenti, e dopo mezz'ora di
        colloquio l'interfaccia cominciava a farsi pesante.
        """
        return self.engine.statistics()
