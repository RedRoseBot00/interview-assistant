"""
Istanza singola dell'applicazione.

Due copie aperte contemporaneamente si contendono gli stessi file:
scaricano lo stesso modello nella stessa cartella, scrivono nello
stesso archivio dei colloqui e tentano di aprire lo stesso microfono.
Il sintomo tipico su Windows e' l'errore "il file e' utilizzato da un
altro processo".

Usiamo un mutex di sistema, il meccanismo previsto da Windows proprio
per questo: e' associato alla sessione dell'utente e viene rilasciato
automaticamente dal sistema operativo se il programma termina in modo
anomalo, quindi non lascia mai blocchi permanenti come farebbe un file
di lock.

Attenzione: il controllo va eseguito SOLO per la finestra principale.
L'applicazione riavvia se stessa come processo di servizio (verifica
del motore, generazione del report) e quei processi devono poter
partire liberamente.
"""
from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

# Nome univoco del programma. "Local\\" limita il mutex alla sessione
# dell'utente corrente: su un computer condiviso due utenti diversi
# possono usare l'applicazione nello stesso momento.
_MUTEX_NAME = "Local\\InterviewAssistant-FinestraPrincipale"
_ERROR_ALREADY_EXISTS = 183

_handle = None


def acquire() -> bool:
    """
    Restituisce True se questa e' l'unica istanza in esecuzione.

    Il riferimento al mutex viene conservato in una variabile del
    modulo: se venisse rilasciato subito, il blocco non avrebbe alcun
    effetto.
    """
    global _handle

    if sys.platform != "win32":
        return True

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateMutexW.restype = wintypes.HANDLE

        _handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
        last_error = ctypes.get_last_error()

        if not _handle:
            log.warning("Controllo istanza singola non riuscito (nessun riferimento)")
            return True  # nel dubbio lasciamo partire il programma

        if last_error == _ERROR_ALREADY_EXISTS:
            log.warning("Un'altra copia dell'applicazione e' gia' in esecuzione")
            return False

        return True
    except Exception:
        log.warning("Controllo istanza singola non disponibile", exc_info=True)
        return True


def release() -> None:
    global _handle
    if _handle and sys.platform == "win32":
        try:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(_handle)
        except Exception:
            pass
        _handle = None
