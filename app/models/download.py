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
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from app import config

log = logging.getLogger(__name__)

# Tentativi e pause fra un tentativo e l'altro. Una linea domestica che
# cade a nove decimi di un file da due gigabyte non deve costringere a
# ricominciare da capo, e HuggingFace risponde "troppe richieste" sotto
# carico: senza questi tentativi un solo intoppo rendeva impossibile
# completare il primo avvio.
DOWNLOAD_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = (3, 8, 20)

# Spazio richiesto prima di cominciare, per modello. Prima era un'unica
# cifra da quattro gigabyte pretesa appena mancava qualcosa: chi aveva
# gia' il modello di trascrizione e doveva scaricare solo quello del
# report — due gigabyte —
# veniva bloccato pur avendone tre liberi, cosa ordinaria su un disco
# virtuale. Il file parziale viene rinominato, non copiato, quindi non
# serve il doppio dello spazio.
LLM_REQUIRED_BYTES = 2_300_000_000
WHISPER_REQUIRED_BYTES = {
    "tiny": 100_000_000,
    "base": 200_000_000,
    "small": 600_000_000,
    "medium": 1_800_000_000,
}

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


class DownloadCancelled(DownloadError):
    """L'utente ha chiuso la finestra durante lo scaricamento."""


StopFn = Optional[Callable[[], bool]]


def _check_stop(should_stop: StopFn) -> None:
    if should_stop is not None and should_stop():
        raise DownloadCancelled("Scaricamento interrotto su richiesta.")


# --------------------------------------------------------------------------
# Utilita' comuni
# --------------------------------------------------------------------------
def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _part_path(destination: Path) -> Path:
    """
    Nome del file parziale, stabile fra un avvio e l'altro.

    Le versioni precedenti ci mettevano dentro il numero del processo,
    quindi ogni riavvio ne creava uno nuovo e nessuno cancellava i
    vecchi: tre tentativi falliti sul modello da due gigabyte
    lasciavano sei gigabyte di spazzatura sul disco dell'utente, che
    sopravvivevano perfino alla disinstallazione. Un nome fisso serve
    anche a poter riprendere da dove ci si era interrotti.
    """
    return destination.with_suffix(destination.suffix + ".part")


def _remove_orphan_parts(directory: Path, keep: tuple[str, ...] = ()) -> None:
    """
    Elimina i frammenti che non appartengono a nessun download in corso,
    compresi quelli con il vecchio nome contenente il numero di processo.
    """
    try:
        for leftover in directory.glob("*.part"):
            if leftover.name in keep:
                continue
            try:
                leftover.unlink()
                log.info("Rimosso frammento inutilizzabile: %s", leftover.name)
            except OSError:
                log.debug("Frammento '%s' in uso, lo lascio", leftover.name)
    except Exception:
        log.debug("Pulizia dei frammenti non riuscita", exc_info=True)


FILE_ASSENTE = -1        # il server risponde 404: il file non esiste
DIMENSIONE_IGNOTA = -2   # non siamo riusciti a saperlo (rete, proxy, CDN)


def _remote_size(url: str, timeout: int = 20) -> int:
    """
    Dimensione dichiarata dal server.

    Distinguere "file assente" da "non sono riuscito a chiederlo" e'
    importante: confonderli farebbe fallire l'intero download per un
    file facoltativo, con un messaggio d'errore fuorviante.
    """
    for tentativo in range(3):
        try:
            resp = requests.head(url, allow_redirects=True, timeout=timeout)
            if resp.status_code == 404:
                return FILE_ASSENTE
            resp.raise_for_status()
            declared = resp.headers.get("content-length")
            if declared:
                return int(declared)
            break        # risposta buona ma muta: la lunghezza la chiedo alla GET
        except requests.RequestException:
            if tentativo == 2:
                break
            time.sleep(2 * (tentativo + 1))

    # Ripiego. Molte reti di distribuzione e molti proxy aziendali non
    # rispondono alla richiesta di sola intestazione, o non dichiarano la
    # lunghezza. Prima bastava questo per far fallire tutta
    # l'installazione con un messaggio che dava la colpa al server. Qui
    # si chiede la stessa cosa a una richiesta normale: il corpo non
    # viene mai letto, quindi costa una connessione, non un download.
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
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
    should_stop: StopFn = None,
) -> int:
    """
    Scarica su file temporaneo e rinomina solo a download completato e
    verificato, riprendendo da dove si era interrotto.

    La verifica della dimensione e' indispensabile: se la connessione
    cade a meta' e la risposta non dichiarava una lunghezza, la libreria
    non solleva alcun errore e un file troncato verrebbe promosso a
    modello valido — per poi far fallire ogni colloquio successivo con
    un errore incomprensibile.

    La ripresa conta soprattutto per il modello dei report, che pesa due
    gigabyte: senza, ogni singolo intoppo di rete costringeva a
    ricominciare da zero, e su una linea instabile il primo avvio non
    arrivava mai in fondo.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = _part_path(destination)

    scaricati = tmp.stat().st_size if tmp.exists() else 0
    if expected_size and scaricati > expected_size:
        # Frammento piu' grande del file atteso: non e' nostro.
        tmp.unlink(missing_ok=True)
        scaricati = 0
    if scaricati:
        log.info(
            "Riprendo lo scaricamento di %s da %.0f MB",
            destination.name, scaricati / 1e6,
        )
        if on_chunk:
            on_chunk(scaricati)

    ultimo: Exception | None = None
    for tentativo in range(DOWNLOAD_ATTEMPTS):
        _check_stop(should_stop)
        totale = 0
        try:
            headers = {"Range": f"bytes={scaricati}-"} if scaricati else {}
            with requests.get(
                url, stream=True, timeout=60, headers=headers
            ) as resp:
                if scaricati and resp.status_code == 200:
                    # Il server non sa riprendere: si ricomincia da capo.
                    log.info("Ripresa non supportata dal server: riparto da zero")
                    if on_chunk:
                        on_chunk(-scaricati)
                    scaricati = 0
                    tmp.unlink(missing_ok=True)
                resp.raise_for_status()

                dichiarata = resp.headers.get("content-length")
                totale = (
                    int(dichiarata) + scaricati
                    if dichiarata
                    else (expected_size or 0)
                )
                with open(tmp, "ab" if scaricati else "wb") as handle:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        _check_stop(should_stop)
                        if not chunk:
                            continue
                        handle.write(chunk)
                        scaricati += len(chunk)
                        if on_chunk:
                            on_chunk(len(chunk))

            if totale > 0 and scaricati != totale:
                raise DownloadError(
                    f"Download incompleto di {destination.name}: ricevuti "
                    f"{scaricati} byte su {totale} attesi."
                )
            tmp.replace(destination)
            return scaricati

        except DownloadCancelled:
            raise
        except (requests.RequestException, DownloadError, OSError) as exc:
            ultimo = exc
            # Alcune risposte non miglioreranno mai aspettando: il file
            # non c'e' (404), oppure il proxy aziendale lo nega (403).
            # Ritentarle costava trentun secondi di attesa a vuoto per
            # ciascun file, e capita al primo avvio ogni volta, perche'
            # dei file facoltativi si prova prima una forma e poi l'altra.
            stato = getattr(getattr(exc, "response", None), "status_code", None)
            if stato in (400, 401, 403, 404, 405, 410):
                raise
            scaricati = tmp.stat().st_size if tmp.exists() else 0
            if tentativo + 1 >= DOWNLOAD_ATTEMPTS:
                break
            attesa = RETRY_BACKOFF_SECONDS[
                min(tentativo, len(RETRY_BACKOFF_SECONDS) - 1)
            ]
            log.warning(
                "Scaricamento di %s interrotto (%s): riprovo fra %d s "
                "da %.0f MB",
                destination.name, exc, attesa, scaricati / 1e6,
            )
            for _ in range(attesa * 4):
                _check_stop(should_stop)
                time.sleep(0.25)

    raise ultimo if ultimo is not None else DownloadError(
        f"Scaricamento di {destination.name} non riuscito."
    )


# --------------------------------------------------------------------------
# Modello di trascrizione (Whisper)
# --------------------------------------------------------------------------
def whisper_model_dir(size: str) -> Path:
    return config.WHISPER_CACHE_DIR / f"faster-whisper-{size}"


def _file_ok(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


# Dimensione minima plausibile di model.bin per ciascun modello: circa
# l'ottanta per cento di quella vera. "Esiste ed e' piu' grande di
# zero" non bastava: un download troncato dall'antivirus o da un
# riavvio superava il controllo, il programma dichiarava il modello
# presente, e il caricamento esplodeva a colloquio gia' avviato — con
# la registrazione che proseguiva senza che nessuno trascrivesse.
_MODEL_BIN_MIN_BYTES = {
    "tiny": 60_000_000,
    "base": 110_000_000,
    "small": 380_000_000,
    "medium": 1_200_000_000,
}


def whisper_model_present(size: str) -> bool:
    """
    Il modello e' completo e utilizzabile.

    Il controllo comprende il vocabolario e la DIMENSIONE del file dei
    pesi: un'installazione troncata supera un controllo superficiale ma
    poi fa fallire ogni colloquio, ed e' quindi peggio di un modello
    assente, perche' non verrebbe mai riparata da sola.
    """
    directory = whisper_model_dir(size)
    if not directory.is_dir():
        return False
    if not all(_file_ok(directory / name) for name in _WHISPER_FILES_REQUIRED):
        return False
    if not _file_ok(directory / "model.bin",
                    _MODEL_BIN_MIN_BYTES.get(size, 1)):
        return False
    return any(_file_ok(directory / name) for name in _WHISPER_FILES_VOCABULARY)


def download_whisper_model(
    size: str, on_progress: ProgressFn = None, should_stop: StopFn = None
) -> Path:
    if size not in WHISPER_REPOS:
        raise DownloadError(f"Modello di trascrizione sconosciuto: {size}")

    repo = WHISPER_REPOS[size]
    target = whisper_model_dir(size)
    target.mkdir(parents=True, exist_ok=True)
    _remove_orphan_parts(
        target,
        keep=tuple(
            _part_path(target / name).name
            for name in _WHISPER_FILES_REQUIRED
            + _WHISPER_FILES_VOCABULARY
            + _WHISPER_FILES_EXTRA
        ),
    )

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
                should_stop=should_stop,
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


def download_llm_model(
    on_progress: ProgressFn = None, should_stop: StopFn = None
) -> Path:
    total = _remote_size(config.LLM_MODEL_URL)
    if total == FILE_ASSENTE:
        raise DownloadError(
            "Il modello per i report non e' piu' disponibile all'indirizzo previsto."
        )
    if total <= 0:
        # Senza lunghezza dichiarata non possiamo accorgerci di un file
        # troncato: verrebbe promosso a modello valido e poi ogni report
        # fallirebbe per sempre con un errore incomprensibile, senza che
        # nulla lo riscarichi. Meglio fermarsi con un messaggio chiaro.
        raise DownloadError(
            "Il server non dichiara la dimensione del modello per i report, "
            "quindi non e' possibile verificarne l'integrita'. Se sei dietro "
            "a un proxy aziendale, riprova da un'altra connessione."
        )

    # Frammenti di versioni precedenti (avevano il numero di processo nel
    # nome e non erano riprendibili): vanno tolti di mezzo, altrimenti
    # restano sul disco per sempre a occupare gigabyte.
    _remove_orphan_parts(
        config.MODELS_DIR, keep=(_part_path(config.LLM_MODEL_PATH).name,)
    )

    done = 0

    def _tick(delta: int) -> None:
        nonlocal done
        done += delta
        if on_progress:
            on_progress("report", done, total)

    # Un file gia' presente ma incompleto va rimosso: la ripresa avviene
    # dal frammento .part, non dal file finale.
    if config.LLM_MODEL_PATH.exists() and not llm_model_present():
        log.warning("Modello per i report incompleto: lo riscarico")
        config.LLM_MODEL_PATH.unlink(missing_ok=True)

    try:
        _download_file(
            config.LLM_MODEL_URL,
            config.LLM_MODEL_PATH,
            on_chunk=_tick,
            expected_size=total,
            should_stop=should_stop,
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


def _check_disk_space(whisper_size: str) -> None:
    """
    Verifica lo spazio prima di cominciare, per quello che manca davvero.

    Senza questo controllo l'utente aspettava venti minuti e riceveva
    poi un messaggio grezzo del sistema ("No space left on device")
    proveniente dal punto piu' improbabile del programma. Chiedere pero'
    lo spazio di TUTTI i modelli anche quando ne manca uno solo bloccava
    installazioni perfettamente possibili.
    """
    richiesto = 0
    if not whisper_model_present(whisper_size):
        richiesto += WHISPER_REQUIRED_BYTES.get(whisper_size, 600_000_000)
    if not llm_model_present():
        richiesto += LLM_REQUIRED_BYTES
    if richiesto <= 0:
        return
    richiesto = int(richiesto * 1.1)      # margine per il file in corso
    try:
        libero = shutil.disk_usage(config.MODELS_DIR).free
    except Exception:
        log.debug("Spazio su disco non verificabile", exc_info=True)
        return
    if libero >= richiesto:
        return
    raise DownloadError(
        "Spazio su disco insufficiente per i modelli da scaricare: servono "
        f"almeno {richiesto / 1e9:.1f} GB liberi, ne risultano "
        f"{libero / 1e9:.1f} GB. Libera spazio e riavvia il programma."
    )


def ensure_models_ready(
    whisper_size: str,
    on_progress: ProgressFn = None,
    should_stop: StopFn = None,
) -> None:
    """Scarica i modelli mancanti. Da eseguire fuori dal thread grafico."""
    if missing_models(whisper_size):
        _check_disk_space(whisper_size)
    if not whisper_model_present(whisper_size):
        log.info("Scarico il modello di trascrizione '%s'", whisper_size)
        download_whisper_model(whisper_size, on_progress, should_stop=should_stop)
    _check_stop(should_stop)
    # Il paracadute. Quando il computer non sta al passo, il programma
    # scende automaticamente verso un modello piu' leggero — ma puo'
    # atterrare solo su un modello gia' scaricato, e prima si scaricava
    # soltanto quello selezionato: sulla macchina lenta, cioe' proprio
    # quella che ne aveva bisogno, il meccanismo non aveva mai nulla su
    # cui atterrare. 'tiny' pesa meno di cento megabyte: e' l'unica
    # assicurazione che il colloquio possa comunque essere trascritto.
    if whisper_size != "tiny" and not whisper_model_present("tiny"):
        try:
            log.info("Scarico anche il modello 'tiny' come riserva")
            download_whisper_model("tiny", on_progress, should_stop=should_stop)
        except DownloadCancelled:
            raise
        except Exception:
            # La riserva non deve mai bloccare l'avvio: senza, il
            # programma funziona esattamente come prima.
            log.warning("Modello di riserva non scaricato", exc_info=True)
    _check_stop(should_stop)
    if not llm_model_present():
        log.info("Scarico il modello per i report")
        download_llm_model(on_progress, should_stop=should_stop)
