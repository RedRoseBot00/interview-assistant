"""
Avvio dell'interfaccia grafica.

L'ordine delle operazioni non e' casuale:
  1. configurazione del registro eventi, cosi' anche un errore
     immediato lascia una traccia su file;
  2. configurazione della CPU, che deve avvenire prima di importare le
     librerie di calcolo;
  3. creazione della finestra.
"""
from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def _configure_cpu() -> bool:
    """Applica la modalita' di calcolo scelta o rilevata."""
    from app import compat, settings

    mode = settings.get("cpu_mode", "auto")
    if mode == "compatible":
        force = True
    elif mode == "fast":
        force = False
    else:
        # In automatico rispettiamo l'esito dell'ultimo test di avvio.
        force = None
        if settings.get("engine_selftest", "") == "ok-compatible":
            force = True

    return compat.apply_cpu_compat(force)


def run_gui() -> int:
    from app import logging_setup

    logging_setup.setup_logging()
    compatible_mode = _configure_cpu()
    log.info("Modalita' compatibilita' CPU: %s", compatible_mode)

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    logging_setup.install_qt_message_handler()

    # Su schermi ad alta densita' evita testo e icone sfocati.
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("InterviewAssistant")
    app.setOrganizationName("InterviewAssistant")

    from app import config
    from app.ui import theme

    app.setApplicationVersion(config.APP_VERSION)
    app.setStyleSheet(theme.STYLESHEET)

    try:
        from app.ui.main_window import MainWindow

        window = MainWindow()
        window.show()
    except Exception as exc:
        log.exception("Impossibile aprire la finestra principale")
        QMessageBox.critical(
            None,
            "Errore all'avvio",
            "L'applicazione non e' riuscita ad aprirsi.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"Dettagli nel file di log:\n{config.LOG_DIR}",
        )
        return 1

    return app.exec()


def main() -> int:
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
