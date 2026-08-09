"""
Scaricamento dei modelli AI al primo avvio.

L'installer resta leggero (poche decine di MB): i modelli veri e propri
(alcuni GB) vengono scaricati una sola volta, al primo avvio, dentro
%APPDATA%. Dalla seconda volta in poi l'applicazione funziona
completamente offline.

Vengono scaricati esplicitamente sia il modello di trascrizione
(Whisper in formato CTranslate2) sia il modello linguistico per i
report: scaricarli qui, con una barra di avanzamento, e' molto piu'
chiaro per l'utente rispetto a un download "a sorpresa" nel momento in
cui preme "Avvia colloquio".
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import requests

from app import config

log = logging.getLogger(__name__)

# Repository ufficiali dei modelli Whisper gia' convertiti per
# CTranslate2, quelli usati internamente da faster-whisper.
WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
}

# File che compongono un modello di trascrizione.
_WHISPER_FILES_REQUIRED = ("config.json", "model.bin", "tokenizer.json")

# Il vocabolario e' indispensabile: senza, il motore di calcolo rifiuta
# di caricare il modello con l'errore "Cannot load the vocabulary from
# the model directory". Il nome del file cambia a seconda di quando il
# modello e' stato pubblicato, quindi proviamo entrambe le varianti e
# ne pretendiamo almeno una.
_WHISPER_FILES_VOCABULARY = ("vocabulary.json", "vocabulary.txt")

# Utili ma non indispensabili: se mancano, il modello funziona lo stesso.
_WHISPER_FILES_EXTRA = ("preprocessor_config.json",)

ProgressFn = Optional[Callable[[str, int, int], None]]


class DownloadError(Exception):
    """Errore recuperabile: mostrato all'utente con un messaggio chiaro."""


# --------------------------------------------------------------------------
# Utilita' comuni
# --------------------------------------------------------------------------
def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


FILE_ASSENTE = -1        # il server risponde 404: il file non esiste
DIMENSIONE_IGNOTA = -2   # non siamo riusciti a saperlo (rete, proxy, CDN)


def _remote_size(url: str, timeout: int = 20) -> int:
    """
    Dimensione dichiarata dal server.

    Distinguere "file assente" da "non sono riuscito a chiederlo" e'
    importante: confonderli farebbe fallire l'intero download per un
    file facoltativo, con un messaggio d'errore fuorviante.
    """
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        if resp.status_code == 404:
            return FILE_ASSENTE
        resp.raise_for_status()
        declared = resp.headers.get("content-length")
        return int(declared) if declared else DIMENSIONE_IGNOTA
    except requests.RequestException:
        return DIMENSIONE_IGNOTA


def _download_file(
    url: str,
    destination: Path,
    on_chunk: Callable[[int], None] | None = None,
    chunk_size: int = 1024 * 512,
    expected_size: int = 0,
) -> int:
    """
    Scarica su file temporaneo e rinomina solo a download completato e
    verificato.

    La verifica della dimensione e' indispensabile: se la connessione
    cade a meta' e la risposta non dichiarava una lunghezza, la libreria
    non solleva alcun errore e un file troncato verrebbe promosso a
    modello valido — per poi far fallire ogni colloquio successivo con
    un errore incomprensibile.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    written = 0
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            declared = resp.headers.get("content-length")
            declared_size = int(declared) if declared else expected_size
            with open(tmp, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if on_chunk:
                        on_chunk(len(chunk))

        if declared_size > 0 and written != declared_size:
            raise DownloadError(
                f"Download incompleto di {destination.name}: ricevuti "
                f"{written} byte su {declared_size} attesi."
            )
        tmp.replace(destination)
        return written
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# Modello di trascrizione (Whisper)
# --------------------------------------------------------------------------
def whisper_model_dir(size: str) -> Path:
    return config.WHISPER_CACHE_DIR / f"faster-whisper-{size}"


def _file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def whisper_model_present(size: str) -> bool:
    """
    Il modello e' completo e utilizzabile.

    Il controllo comprende il vocabolario: un'installazione priva di
    quel file supera un controllo superficiale ma poi fa fallire ogni
    colloquio, ed e' quindi peggio di un modello assente, perche' non
    verrebbe mai riparata da sola.
    """
    directory = whisper_model_dir(size)
    if not directory.is_dir():
        return False
    if not all(_file_ok(directory / name) for name in _WHISPER_FILES_REQUIRED):
        return False
    return any(_file_ok(directory / name) for name in _WHISPER_FILES_VOCABULARY)


def download_whisper_model(size: str, on_progress: ProgressFn = None) -> Path:
    if size not in WHISPER_REPOS:
        raise DownloadError(f"Modello di trascrizione sconosciuto: {size}")

    repo = WHISPER_REPOS[size]
    target = whisper_model_dir(size)
    target.mkdir(parents=True, exist_ok=True)

    # Le dimensioni servono solo per rendere veritiera la barra di
    # avanzamento. Se il server non le comunica, procediamo comunque:
    # decidere di NON scaricare un file solo perche' non se ne conosce
    # la dimensione significherebbe consegnare un modello incompleto.
    plan: list[tuple[str, int]] = [
        (name, max(0, _remote_size(_hf_url(repo, name))))
        for name in _WHISPER_FILES_REQUIRED
    ]
    total = sum(size for _, size in plan)
    done = 0

    def _tick(delta: int) -> None:
        nonlocal done
        done += delta
        if on_progress:
            on_progress("trascrizione", done, total)

    def _fetch(name: str, expected: int = 0) -> bool:
        """Scarica un file. Restituisce False se il server non ce l'ha."""
        destination = target / name
        if destination.exists():
            actual = destination.stat().st_size
            if actual > 0 and (expected == 0 or actual == expected):
                _tick(expected)
                return True
            log.warning(
                "File '%s' incompleto (%d byte su %d attesi): lo riscarico",
                name, actual, expected,
            )
            destination.unlink(missing_ok=True)
        try:
            _download_file(
                _hf_url(repo, name),
                destination,
                on_chunk=_tick,
                expected_size=expected,
            )
            return True
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                return False
            raise

    try:
        for name, size_bytes in plan:
            if not _fetch(name, size_bytes):
                raise DownloadError(
                    f"Il file '{name}' del modello di trascrizione non e' "
                    f"disponibile nel repository {repo}."
                )

        # Vocabolario: proviamo le varianti finche' una risponde.
        if not any(_file_ok(target / name) for name in _WHISPER_FILES_VOCABULARY):
            for name in _WHISPER_FILES_VOCABULARY:
                if _fetch(name):
                    log.info("Vocabolario scaricato: %s", name)
                    break
            else:
                raise DownloadError(
                    "Il vocabolario del modello di trascrizione non e' stato "
                    "trovato. Senza di esso il motore non puo' avviarsi."
                )

        for name in _WHISPER_FILES_EXTRA:
            try:
                _fetch(name)
            except Exception:
                log.debug("File accessorio '%s' non scaricato", name, exc_info=True)

    except requests.RequestException as exc:
        raise DownloadError(
            "Download del modello di trascrizione non riuscito: "
            f"{exc}. Verifica la connessione a internet e riprova."
        ) from exc

    if not whisper_model_present(size):
        raise DownloadError(
            "Il modello di trascrizione risulta incompleto dopo il download. "
            "Riprova: i file parziali verranno riscaricati."
        )
    return target


def remove_whisper_model(size: str) -> None:
    shutil.rmtree(whisper_model_dir(size), ignore_errors=True)


# --------------------------------------------------------------------------
# Modello linguistico per i report (GGUF, eseguito da llama.cpp)
# --------------------------------------------------------------------------
def llm_model_present() -> bool:
    """
    Il modello e' presente e plausibilmente integro.

    Il file completo pesa circa 2 GB: una soglia molto piu' bassa
    lascerebbe passare un download interrotto, che poi farebbe fallire
    ogni singolo report senza spiegare perche'.
    """
    path = config.LLM_MODEL_PATH
    return path.exists() and path.stat().st_size >= config.LLM_MODEL_MIN_BYTES


def download_llm_model(on_progress: ProgressFn = None) -> Path:
    total = _remote_size(config.LLM_MODEL_URL)
    if total == FILE_ASSENTE:
        raise DownloadError(
            "Il modello per i report non e' piu' disponibile all'indirizzo previsto."
        )
    if total < 0:
        total = 0
    done = 0

    def _tick(delta: int) -> None:
        nonlocal done
        done += delta
        if on_progress:
            on_progress("report", done, total)

    # Un residuo di un tentativo precedente va rimosso: riprendere da un
    # file parziale produrrebbe un modello illeggibile.
    if config.LLM_MODEL_PATH.exists() and not llm_model_present():
        log.warning("Modello per i report incompleto: lo riscarico da capo")
        config.LLM_MODEL_PATH.unlink(missing_ok=True)

    try:
        _download_file(
            config.LLM_MODEL_URL,
            config.LLM_MODEL_PATH,
            on_chunk=_tick,
            expected_size=total,
        )
    except requests.RequestException as exc:
        raise DownloadError(
            f"Download del modello per i report non riuscito: {exc}. "
            "Verifica la connessione a internet e riprova."
        ) from exc

    if not llm_model_present():
        config.LLM_MODEL_PATH.unlink(missing_ok=True)
        raise DownloadError(
            "Il modello per i report risulta incompleto dopo il download. Riprova."
        )
    return config.LLM_MODEL_PATH


# --------------------------------------------------------------------------
# Orchestrazione
# --------------------------------------------------------------------------
def missing_models(whisper_size: str) -> list[str]:
    missing = []
    if not whisper_model_present(whisper_size):
        missing.append("trascrizione")
    if not llm_model_present():
        missing.append("report")
    return missing


def ensure_models_ready(whisper_size: str, on_progress: ProgressFn = None) -> None:
    """Scarica i modelli mancanti. Da eseguire fuori dal thread grafico."""
    if not whisper_model_present(whisper_size):
        log.info("Scarico il modello di trascrizione '%s'", whisper_size)
        download_whisper_model(whisper_size, on_progress)
    if not llm_model_present():
        log.info("Scarico il modello per i report")
        download_llm_model(on_progress)
