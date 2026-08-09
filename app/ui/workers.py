"""
Operazioni lente eseguite fuori dal thread grafico: preparazione dei
modelli all'avvio e generazione del report finale.
"""
from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QThread, Signal

from app import diagnostics, settings
from app.models import download as model_download

log = logging.getLogger(__name__)


class StartupWorker(QThread):
    """
    Prepara l'applicazione: scarica i modelli mancanti e verifica che il
    motore di trascrizione sia eseguibile su questo processore.
    """

    progress = Signal(str, int, int)   # componente, byte scaricati, totale
    stage = Signal(str)                # messaggio di stato leggibile
    finished_ok = Signal(bool, str)    # modalita' compatibilita', messaggio
    failed = Signal(str, str)          # messaggio, dettaglio tecnico

    def __init__(self, whisper_size: str, parent=None):
        super().__init__(parent)
        self.whisper_size = whisper_size
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """
        Interrompe davvero il lavoro in corso.

        Il solo requestInterruption() di Qt non basta: viene consultato
        soltanto fra una fase e l'altra, mentre lo scaricamento dei
        modelli dura minuti. Chi chiudeva la finestra durante il primo
        avvio si ritrovava la finestra bloccata finche' il download non
        finiva — o per sempre, se la rete si impuntava.
        """
        self._cancel.set()
        self.requestInterruption()

    def _stopped(self) -> bool:
        return self._cancel.is_set() or self.isInterruptionRequested()

    def run(self) -> None:  # noqa: D401 - metodo di QThread
        try:
            missing = model_download.missing_models(self.whisper_size)
            if missing:
                self.stage.emit(
                    "Primo avvio: scaricamento dei modelli AI ("
                    + ", ".join(missing)
                    + "). Serve una connessione a internet, una sola volta."
                )
                model_download.ensure_models_ready(
                    self.whisper_size,
                    on_progress=lambda name, done, total: self.progress.emit(
                        name, done, total
                    ),
                    should_stop=self._stopped,
                )

            if self._stopped():
                return

            state = settings.get("engine_selftest", "")
            tested_size = settings.get("engine_selftest_size", "")
            if state in ("ok", "ok-compatible") and tested_size == self.whisper_size:
                self.finished_ok.emit(state == "ok-compatible", "")
                return

            self.stage.emit("Verifica della compatibilita' con il processore...")
            result = diagnostics.run_transcription_selftest(
                self.whisper_size, should_stop=self._stopped
            )
            if self._stopped():
                return

            settings.set_many(
                {
                    "engine_selftest": result.state,
                    "engine_selftest_size": self.whisper_size,
                }
            )

            if self._stopped():
                return

            if result.ok:
                message = ""
                if result.compatible_mode:
                    message = (
                        "Attivata la modalita' compatibilita': questo processore "
                        "non supporta le istruzioni piu' veloci, quindi la "
                        "trascrizione sara' piu' lenta ma stabile."
                    )
                self.finished_ok.emit(result.compatible_mode, message)
            else:
                self.failed.emit(
                    "Il motore di trascrizione non riesce ad avviarsi su questo "
                    "computer. Le funzioni di registrazione sono disattivate.",
                    result.detail,
                )
        except model_download.DownloadCancelled:
            log.info("Preparazione interrotta dall'utente")
        except model_download.DownloadError as exc:
            self.failed.emit(str(exc), "")
        except Exception as exc:
            log.exception("Preparazione dell'applicazione non riuscita")
            if not self._stopped():
                self.failed.emit(f"Preparazione non riuscita: {exc}", "")


class ReportWorker(QThread):
    """
    Genera il report di fine colloquio in un processo isolato.

    Il processo figlio e' annullabile: senza questa possibilita', alla
    chiusura della finestra resterebbe un processo invisibile a
    consumare tutti i core del computer per diversi minuti.
    """

    finished_ok = Signal(object)   # ReportResult
    failed = Signal(str)
    partial = Signal(str)          # testo prodotto finora, pezzo per pezzo

    def __init__(
        self,
        transcript: str,
        segments: list[dict],
        labels: dict,
        candidate_name: str,
        role: str,
        duration_seconds: float,
        detected_language: str,
        report_language: str,
        parent=None,
    ):
        super().__init__(parent)
        self._args = dict(
            transcript=transcript,
            segments=segments,
            labels=labels,
            candidate_name=candidate_name,
            role=role,
            duration_seconds=duration_seconds,
            detected_language=detected_language,
            report_language=report_language,
        )

    def cancel(self) -> None:
        self.requestInterruption()
        try:
            from app.summarization.llm import cancel_running_generation

            cancel_running_generation()
        except Exception:
            log.debug("Annullamento della generazione non riuscito", exc_info=True)

    def run(self) -> None:  # noqa: D401 - metodo di QThread
        try:
            from app.summarization.llm import generate_report

            def _partial(pezzo: str) -> None:
                if not self.isInterruptionRequested():
                    self.partial.emit(pezzo)

            result = generate_report(on_partial=_partial, **self._args)
            if not self.isInterruptionRequested():
                self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("Generazione del report non riuscita")
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class PlatformProbe(QThread):
    """
    Rileva le applicazioni di videochiamata attive.

    Scorrere l'elenco dei processi costa qualche decina di millisecondi:
    eseguirlo nel thread grafico produrrebbe uno scatto visibile ogni
    volta, proprio mentre scorrono i sottotitoli.
    """

    result = Signal(object)

    def cancel(self) -> None:
        """
        Presente perche' chi chiude la finestra tratta tutti i worker
        allo stesso modo: senza, questo finiva nel ramo di attesa senza
        limite di tempo.
        """
        self.requestInterruption()

    def run(self) -> None:  # noqa: D401 - metodo di QThread
        try:
            from app import platform_detect

            risultato = platform_detect.detect()
            if not self.isInterruptionRequested():
                self.result.emit(risultato)
        except Exception:
            log.debug("Rilevamento piattaforma fallito", exc_info=True)
