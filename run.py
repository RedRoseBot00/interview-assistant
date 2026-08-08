"""
Punto di ingresso dell'applicazione.

Lo stesso eseguibile svolge quattro ruoli, distinti dagli argomenti
della riga di comando:

  (nessun argomento)         avvia l'interfaccia grafica
  --self-test transcription  verifica che il motore di trascrizione sia
                             eseguibile su questo processore
  --generate-report IN OUT   genera il report leggendo e scrivendo due
                             file JSON
  --smoke-test               controlla che tutte le librerie native
                             siano state incluse nel pacchetto compilato
                             (usato dalla compilazione automatica)

I comandi di servizio vengono lanciati dal programma stesso in un
processo separato: se le librerie di calcolo incontrano una istruzione
non supportata dalla CPU, a terminare e' il processo figlio e non
l'applicazione, che puo' cosi' reagire con un messaggio chiaro.

Nota: la configurazione della CPU va applicata prima di importare
qualunque libreria di calcolo, quindi qui gli import "pesanti"
avvengono dentro le funzioni e non in cima al file.
"""
from __future__ import annotations

import os
import sys


# --------------------------------------------------------------------------
# Rete di sicurezza per l'avvio
# --------------------------------------------------------------------------
def _ensure_standard_streams() -> None:
    """
    Garantisce che stdout e stderr esistano.

    In un'applicazione compilata senza finestra di console, Python puo'
    trovarsi con sys.stdout e sys.stderr a None: in quel caso una
    semplice print() solleverebbe un'eccezione, facendo fallire il
    processo per un motivo del tutto secondario.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            try:
                stream = open(os.devnull, "w", encoding="utf-8")
                setattr(sys, name, stream)
                setattr(sys, f"__{name}__", stream)
            except Exception:
                pass


def _silence_windows_error_dialogs() -> None:
    """
    Nei processi di servizio un errore grave deve produrre un codice di
    uscita, non una finestra di sistema invisibile che blocca il
    processo principale in attesa di un clic che nessuno puo' dare.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        )
    except Exception:
        pass


def _fatal(message: str) -> None:
    """
    Ultima rete di sicurezza: se qualcosa fallisce cosi' presto da non
    permettere nemmeno l'apertura di una finestra Qt, l'utente deve
    comunque vedere un messaggio e trovare un file da inviarci.
    """
    import datetime
    import tempfile
    import traceback

    detail = traceback.format_exc()
    path = "(non scrivibile)"
    try:
        base = os.environ.get("APPDATA") or tempfile.gettempdir()
        path = os.path.join(base, "InterviewAssistant-avvio-fallito.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                f"{datetime.datetime.now().isoformat()}\n{message}\n\n{detail}"
            )
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                f"{message}\n\nDettagli tecnici salvati in:\n{path}",
                "Interview Assistant — errore all'avvio",
                0x10,  # MB_ICONERROR
            )
        except Exception:
            pass


# --------------------------------------------------------------------------
# Comandi di servizio
# --------------------------------------------------------------------------
def _run_self_test(mode: str) -> int:
    _silence_windows_error_dialogs()
    from app import logging_setup

    logging_setup.setup_logging()
    if mode == "transcription":
        from app.diagnostics import selftest_transcription_child

        return selftest_transcription_child()

    import logging

    logging.getLogger(__name__).error("Modalita' di test sconosciuta: %s", mode)
    return 2


def _run_report_generation(input_path: str, output_path: str) -> int:
    _silence_windows_error_dialogs()
    from app import logging_setup
    from app.summarization.llm import generate_report_child

    logging_setup.setup_logging()
    return generate_report_child(input_path, output_path)


def _run_smoke_test() -> int:
    """
    Verifica che l'eseguibile compilato riesca a importare tutte le
    librerie native. Viene eseguito dalla compilazione automatica: se
    una libreria manca dal pacchetto, il problema emerge qui e non sul
    computer del cliente.
    """
    import importlib
    import logging

    from app import logging_setup

    logging_setup.setup_logging()
    log = logging.getLogger("smoke")

    moduli = (
        "numpy",
        "requests",
        "psutil",
        "docx",
        "PySide6.QtWidgets",
        "faster_whisper",
        "ctranslate2",
        "onnxruntime",
        "llama_cpp",
        "av",
    )
    if sys.platform == "win32":
        moduli += ("pyaudiowpatch",)

    errori = []
    for nome in moduli:
        try:
            importlib.import_module(nome)
            log.info("modulo '%s': presente", nome)
        except Exception as exc:
            errori.append(f"{nome}: {type(exc).__name__}: {exc}")
            log.error("modulo '%s' NON disponibile: %s", nome, exc)

    if errori:
        log.error("Verifica del pacchetto fallita: %d moduli mancanti", len(errori))
        return 1

    log.info("Verifica del pacchetto superata: %d moduli", len(moduli))
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    _ensure_standard_streams()
    argv = sys.argv[1:]

    if argv and argv[0] == "--self-test":
        mode = argv[1] if len(argv) > 1 else "transcription"
        return _run_self_test(mode)

    if argv and argv[0] == "--generate-report":
        if len(argv) < 3:
            return 2
        return _run_report_generation(argv[1], argv[2])

    if argv and argv[0] == "--smoke-test":
        return _run_smoke_test()

    from app.main import run_gui

    return run_gui()


if __name__ == "__main__":
    try:
        _code = main()
    except SystemExit:
        raise
    except BaseException:
        _fatal("L'applicazione non e' riuscita ad avviarsi.")
        _code = 1
    sys.exit(_code)
