"""
Impostazioni utente persistenti (file JSON in %APPDATA%).

Tenute separate da config.py: li' stanno le costanti del programma, qui
le scelte dell'utente, che sopravvivono alla chiusura dell'app e agli
aggiornamenti.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

from app import config

log = logging.getLogger(__name__)

def _default_model_size() -> str:
    """
    Modello proposto alla prima installazione, in base al processore.

    Su un computer con pochi core il modello 'piccolo' non riesce a
    stare al passo del parlato e la trascrizione accumula ritardo fino
    a perdere pezzi: meglio partire da uno piu' rapido, l'utente potra'
    sempre alzarlo dalle impostazioni.
    """
    import os

    return "base" if (os.cpu_count() or 2) <= 2 else "small"


DEFAULTS: dict[str, Any] = {
    # Trascrizione
    "whisper_model_size": _default_model_size(),   # tiny | base | small | medium
    "capture_microphone": True,
    "capture_system_audio": True,
    "transcription_language": "auto",  # "auto" oppure codice ISO (it, en, ...)
    # Etichette interlocutori mostrate nella trascrizione
    "label_recruiter": "Tu",
    "label_candidate": "Candidato",
    # Gestione dell'eco quando non si usano le cuffie:
    #   "off"    -> nessun intervento
    #   "auto"   -> riconosce e scarta i blocchi di eco
    #   "cancel" -> sottrae l'eco dal segnale del microfono
    "echo_mode": "auto",
    # Report
    "report_language": "auto",
    "auto_generate_report": True,
    # Compatibilita' hardware:
    #   "auto"       -> decide il programma in base alla CPU
    #   "compatible" -> forza i kernel generici (piu' lento, sempre sicuro)
    #   "fast"       -> forza i kernel ottimizzati (piu' veloce, puo' non partire)
    "cpu_mode": "auto",
    # Esito dell'ultimo test di avvio del motore di trascrizione
    "engine_selftest": "",  # "" | "ok" | "ok-compatible" | "failed"
    "engine_selftest_size": "",  # modello su cui il test e' stato eseguito
    # Interfaccia
    "always_on_top": False,
}

# Valori ammessi per le impostazioni a scelta chiusa: un file
# modificato a mano non deve poter mettere il programma in uno stato
# incoerente e difficile da diagnosticare.
_ALLOWED: dict[str, tuple] = {
    "whisper_model_size": config.WHISPER_MODEL_SIZES,
    "cpu_mode": ("auto", "compatible", "fast"),
    "echo_mode": ("off", "auto", "cancel"),
    "engine_selftest": ("", "ok", "ok-compatible", "failed"),
}

_lock = threading.RLock()
_cache: dict[str, Any] | None = None


def _path():
    return config.APP_DATA_DIR / "settings.json"


def _valid(key: str, value: Any) -> bool:
    default = DEFAULTS.get(key)
    if isinstance(default, bool) and not isinstance(value, bool):
        return False
    if isinstance(default, str) and not isinstance(value, str):
        return False
    allowed = _ALLOWED.get(key)
    if allowed is not None and value not in allowed:
        return False
    return True


def _quarantine_corrupted_file() -> None:
    """
    Mette da parte un file illeggibile invece di lasciarlo al suo posto:
    altrimenti l'errore si ripeterebbe a ogni avvio, in silenzio.
    """
    try:
        broken = _path().with_suffix(".json.danneggiato")
        broken.unlink(missing_ok=True)
        _path().replace(broken)
        log.warning("File impostazioni danneggiato spostato in %s", broken)
    except Exception:
        log.debug("Impossibile mettere da parte il file danneggiato", exc_info=True)


def load() -> dict[str, Any]:
    global _cache
    with _lock:
        if _cache is not None:
            return dict(_cache)

        data = dict(DEFAULTS)
        path = _path()
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("Impostazioni illeggibili: uso i valori predefiniti")
                _quarantine_corrupted_file()
                stored = {}
            if isinstance(stored, dict):
                for key in DEFAULTS:
                    if key in stored and _valid(key, stored[key]):
                        data[key] = stored[key]
        _cache = data
        return dict(data)


def get(key: str, default: Any = None) -> Any:
    return load().get(key, DEFAULTS.get(key, default))


def set_many(values: dict[str, Any]) -> None:
    global _cache
    # Il caricamento va fatto prima di acquisire il lock in scrittura:
    # senza questa riga, una scrittura eseguita prima di qualunque
    # lettura ripartirebbe dai valori predefiniti, cancellando le
    # preferenze salvate dall'utente.
    load()
    with _lock:
        data = dict(_cache or DEFAULTS)
        for key, value in values.items():
            if key not in DEFAULTS:
                log.debug("Impostazione sconosciuta ignorata: %s", key)
                continue
            data[key] = value
        _cache = data
        _write(data)


def set(key: str, value: Any) -> None:
    set_many({key: value})


def _write(data: dict[str, Any]) -> None:
    """
    Scrittura atomica: prima su file temporaneo, poi rinomina. Una
    interruzione a meta' lascerebbe altrimenti un JSON troncato, e al
    riavvio tutte le impostazioni tornerebbero ai valori di fabbrica.
    """
    path = _path()
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(path)
    except Exception:
        log.warning("Impossibile salvare le impostazioni", exc_info=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def reset() -> None:
    global _cache
    with _lock:
        _cache = dict(DEFAULTS)
        try:
            _path().unlink(missing_ok=True)
        except Exception:
            pass
