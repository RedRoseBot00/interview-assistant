"""
Impostazioni utente persistenti (file JSON in %APPDATA%).

Tenute separate da config.py: li' stanno le costanti del programma, qui
le scelte dell'utente, che sopravvivono alla chiusura dell'app e agli
aggiornamenti.
"""
from __future__ import annotations

import copy
import json
import logging
import threading
from typing import Any

from app import config

log = logging.getLogger(__name__)

# Numero di revisione della taratura automatica. Quando lo alziamo, le
# scelte che il programma aveva deciso da solo in passato (dimensione
# del modello, esito del test del motore) vengono ricalcolate una volta
# sola al primo avvio. Serve perche' un'installazione esistente si
# porta dietro il valore salvato dalla versione precedente, e senza
# questo meccanismo continuerebbe a usare per sempre una taratura che
# nel frattempo abbiamo scoperto essere sbagliata.
TUNING_REVISION = 5

# Impostazioni scelte dal programma, non dall'utente: sono quelle che
# il cambio di revisione azzera.
#
# whisper_model_size sta in questo elenco ma e' un caso speciale: e'
# scritto sia dalla taratura automatica sia dalla scheda Impostazioni.
# Azzerandolo senza distinguere, a chi aveva scelto "medium" a mano il
# programma lo riportava a "base" al primo avvio dopo un aggiornamento,
# in silenzio. Per questo teniamo traccia delle scelte esplicite in
# "user_choices" e quelle non si toccano.
_AUTO_TUNED = ("whisper_model_size", "engine_selftest", "engine_selftest_size",
               "engine_selftest_fingerprint")


def _default_model_size() -> str:
    """
    Modello proposto alla prima installazione, in base al processore.

    La scelta e' cambiata dopo aver constatato sul campo che 'base'
    produce spesso un testo poco comprensibile. Nei modelli piu'
    piccoli la precisione varia molto da lingua a lingua, e un
    programma che deve funzionare in qualunque lingua non puo' partire
    dal modello che in diverse di esse non capisce.

    Resta pero' il fatto che Whisper elabora sempre una finestra di
    trenta secondi: il costo e' PER CHIAMATA, e in un colloquio le
    chiamate sono una per battuta, circa venti al minuto. Su un computer
    a due core misurati sul campo, 'small' costa dieci secondi a frase:
    il ritardo non si stabilizza mai, cresce per tutta la durata del
    colloquio, e alla fine mancano dal report le frasi non elaborate.
    Non e' un modello impreciso, e' un modello che non arriva.

    Quindi: 'small' da quattro core in su, dove sta comodamente al
    passo; 'base' sui due o tre core; 'tiny' dove c'e' un core solo.
    Chi vuole di piu' puo' sempre alzarlo dalle impostazioni, e quella
    scelta non viene mai scavalcata.

    Il conteggio va fatto sui core FISICI: un portatile a due core con
    SMT ne dichiara quattro.
    """
    fisici = config._physical_cores()
    if fisici >= 4:
        return "small"
    return "base" if fisici >= 2 else "tiny"


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
    # Versione del programma, processore e modello su cui vale l'esito
    # qui sopra. Prima si confrontava solo il modello: aggiornando il
    # programma — e con lui le librerie di calcolo — si ereditava un
    # verdetto preso su un'altra libreria.
    "engine_selftest_fingerprint": "",
    # Secondi misurati dal controllo preventivo per una chiamata di
    # trascrizione su questo computer. Zero = mai misurato.
    "engine_speed_seconds": 0.0,
    # Interfaccia
    "always_on_top": False,
    # Uso interno: vedi TUNING_REVISION.
    "tuning_revision": 0,
    # Impostazioni che l'utente ha cambiato di persona dalla scheda
    # Impostazioni: non vanno mai sovrascritte da una taratura automatica.
    "user_choices": [],
}

# Lingue offerte nell'interfaccia: un valore fuori da questo elenco non
# comparirebbe nel menu a tendina, che mostrerebbe "Rilevamento
# automatico" mentre il programma userebbe di nascosto un'altra lingua.
_LANGUAGE_CHOICES = ("auto", "it", "en", "es", "fr", "de", "pt")

# Valori ammessi per le impostazioni a scelta chiusa: un file
# modificato a mano non deve poter mettere il programma in uno stato
# incoerente e difficile da diagnosticare.
_ALLOWED: dict[str, tuple] = {
    "whisper_model_size": config.WHISPER_MODEL_SIZES,
    "cpu_mode": ("auto", "compatible", "fast"),
    "echo_mode": ("off", "auto", "cancel"),
    "engine_selftest": ("", "ok", "ok-compatible", "failed"),
    "transcription_language": _LANGUAGE_CHOICES,
    "report_language": _LANGUAGE_CHOICES,
}

_lock = threading.RLock()
_cache: dict[str, Any] | None = None


def _path():
    return config.APP_DATA_DIR / "settings.json"


def _valid(key: str, value: Any) -> bool:
    default = DEFAULTS.get(key)
    if isinstance(default, list):
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if isinstance(default, bool) and not isinstance(value, bool):
        return False
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool) or not isinstance(value, int):
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
            # Copia PROFONDA anche qui: dict() lascerebbe la lista
            # user_choices condivisa con la cache, e chi la mutasse
            # sporcherebbe le impostazioni per tutto il processo.
            return copy.deepcopy(_cache)

        # Copia PROFONDA: dict() e' superficiale, quindi la lista
        # user_choices restava lo stesso identico oggetto dei valori
        # predefiniti e finiva in mano ai chiamanti. Bastava che uno di
        # loro ci aggiungesse una voce per contaminare i predefiniti per
        # tutta la vita del processo, reset() compreso.
        data = copy.deepcopy(DEFAULTS)
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
                    if key not in stored:
                        continue
                    if _valid(key, stored[key]):
                        data[key] = stored[key]
                    else:
                        log.warning(
                            "Impostazione '%s' con valore non ammesso (%r): "
                            "uso il predefinito %r",
                            key, stored[key], DEFAULTS[key],
                        )

        # Taratura automatica obsoleta: la ricalcoliamo una volta sola,
        # rispettando pero' cio' che l'utente ha scelto di persona.
        if int(data.get("tuning_revision", 0) or 0) < TUNING_REVISION:
            # Attenzione: questo modulo definisce una funzione set(), che
            # oscura il tipo predefinito di Python. Qui si usa una lista.
            scelte = list(data.get("user_choices") or ())
            for key in _AUTO_TUNED:
                if key in scelte:
                    log.info("Scelta dell'utente su '%s' conservata", key)
                    continue
                data[key] = DEFAULTS[key]
            # L'esito del test va comunque rifatto: se il modello e'
            # cambiato, il test precedente non dice piu' nulla.
            data["engine_selftest"] = DEFAULTS["engine_selftest"]
            data["engine_selftest_size"] = DEFAULTS["engine_selftest_size"]
            data["engine_selftest_fingerprint"] = DEFAULTS[
                "engine_selftest_fingerprint"
            ]
            data["tuning_revision"] = TUNING_REVISION
            log.info(
                "Taratura automatica aggiornata alla revisione %s: "
                "modello di trascrizione riportato a '%s'",
                TUNING_REVISION, data["whisper_model_size"],
            )
            _cache = data
            _write(data)
            return copy.deepcopy(data)

        _cache = data
        return copy.deepcopy(data)


def get(key: str, default: Any = None) -> Any:
    return load().get(key, DEFAULTS.get(key, default))


def set_many(values: dict[str, Any], explicit: bool = False) -> None:
    """
    Salva piu' impostazioni insieme.

    explicit=True va usato SOLO quando e' l'utente a premere "Salva"
    nella scheda Impostazioni: quelle chiavi vengono marcate come scelte
    consapevoli e nessuna taratura automatica potra' piu' toccarle.
    """
    global _cache
    # Il caricamento va fatto prima di acquisire il lock in scrittura:
    # senza questa riga, una scrittura eseguita prima di qualunque
    # lettura ripartirebbe dai valori predefiniti, cancellando le
    # preferenze salvate dall'utente.
    load()
    with _lock:
        data = dict(_cache or DEFAULTS)
        if explicit:
            scelte = list(data.get("user_choices") or ())
            for chiave in values:
                if chiave in _AUTO_TUNED and chiave not in scelte:
                    scelte.append(chiave)
            data["user_choices"] = sorted(scelte)
        for key, value in values.items():
            if key not in DEFAULTS:
                log.debug("Impostazione sconosciuta ignorata: %s", key)
                continue
            # Senza questo controllo un valore fuori dominio finiva sul
            # disco e veniva scartato al riavvio: l'utente vedeva
            # l'impostazione "non ricordarsi", senza alcuna traccia.
            if not _valid(key, value):
                log.warning("Valore non ammesso per '%s': %r", key, value)
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
        _cache = copy.deepcopy(DEFAULTS)
        try:
            _path().unlink(missing_ok=True)
        except Exception:
            pass
