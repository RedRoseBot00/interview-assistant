"""
Registro eventi (log) e cattura dei crash.

Obiettivo: se l'app si chiude in modo anomalo, l'utente deve poter
recuperare un file di testo che spieghi cosa e' successo, invece di
trovarsi davanti a una finestra sparita senza spiegazioni.

Copriamo tre livelli:
  1. logging standard di Python -> file di testo giornaliero;
  2. eccezioni non gestite nel thread principale e nei thread secondari;
  3. crash "duri" a livello di CPU (es. istruzione non supportata),
     tramite faulthandler, che scrive lo stack anche quando
     l'interprete Python viene terminato dal sistema operativo.
"""
from __future__ import annotations

import atexit
import datetime as _dt
import faulthandler
import logging
import sys
import threading
from pathlib import Path

# Il numero di processo e' indispensabile: l'applicazione avvia processi
# figli che scrivono nello stesso file, e senza questo dato le righe si
# mescolerebbero in modo illeggibile.
_LOG_FORMAT = "%(asctime)s | %(process)5d | %(levelname)-7s | %(name)-26s | %(message)s"
_fault_file = None


def _log_dir() -> Path:
    from app import config

    return config.LOG_DIR


def current_log_file() -> Path:
    today = _dt.date.today().isoformat()
    return _log_dir() / f"interview-assistant-{today}.log"


def _process_role() -> str:
    """Ruolo del processo corrente, usato per non sovrascrivere file altrui."""
    argv = sys.argv[1:]
    if argv[:1] == ["--self-test"]:
        return "test-motore"
    if argv[:1] == ["--generate-report"]:
        return "report"
    if argv[:1] == ["--smoke-test"]:
        return "verifica-pacchetto"
    return "principale"


def crash_file() -> Path:
    # Un file per ruolo: altrimenti l'avvio di un processo figlio
    # cancellerebbe il resoconto del crash del processo principale,
    # cioe' proprio l'informazione che stiamo cercando di conservare.
    return _log_dir() / f"ultimo-crash-{_process_role()}.txt"


def setup_logging(verbose: bool = False) -> Path:
    """Configura il logging su file (e su console se disponibile)."""
    log_path = current_log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = []

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handlers.append(file_handler)

    # In una app GUI compilata sys.stderr puo' essere None: va gestito.
    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handlers.append(stream_handler)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=handlers,
        force=True,
    )

    # Le librerie di terze parti sono molto verbose a livello DEBUG.
    for noisy in ("urllib3", "huggingface_hub", "filelock", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _install_fault_handler()
    _install_exception_hooks()

    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("Avvio applicazione")
    try:
        from app import compat, config

        log.info("Versione: %s", config.APP_VERSION)
        log.info("Hardware: %s", compat.describe_cpu())
        log.info("Cartella dati: %s", config.APP_DATA_DIR)
    except Exception:
        log.exception("Impossibile registrare le informazioni di sistema")

    return log_path


def _install_fault_handler() -> None:
    """
    Attiva faulthandler: intercetta i crash a basso livello (segmentation
    fault, istruzione illegale) e ne scrive lo stack su file.
    """
    global _fault_file
    try:
        path = crash_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        _fault_file = open(path, "w", encoding="utf-8")
        _fault_file.write(
            "Questo file viene sovrascritto a ogni avvio.\n"
            "Se l'applicazione si chiude di colpo, qui sotto trovi il punto "
            "esatto in cui e' avvenuto il crash.\n\n"
        )
        _fault_file.flush()
        faulthandler.enable(file=_fault_file, all_threads=True)
        atexit.register(_close_fault_file)
    except Exception:
        logging.getLogger(__name__).warning(
            "faulthandler non attivabile", exc_info=True
        )


def _close_fault_file() -> None:
    global _fault_file
    try:
        if _fault_file is not None:
            faulthandler.disable()
            _fault_file.close()
            _fault_file = None
    except Exception:
        pass


def _install_exception_hooks() -> None:
    """Registra le eccezioni non gestite invece di perderle."""
    log = logging.getLogger("crash")

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.critical(
            "Eccezione non gestita nel thread principale",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _hook

    def _thread_hook(args):
        log.critical(
            "Eccezione non gestita nel thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    # threading.excepthook esiste da Python 3.8
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook


def install_qt_message_handler() -> None:
    """
    Convoglia anche i messaggi interni di Qt nel nostro file di log:
    spesso contengono la vera causa di finestre che non si aprono.
    """
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:
        return

    log = logging.getLogger("qt")
    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(mode, context, message):
        log.log(levels.get(mode, logging.INFO), "%s", message)

    qInstallMessageHandler(handler)
