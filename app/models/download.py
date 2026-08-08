"""
Download dei modelli AI al primo avvio.

Per mantenere l'installer .exe leggero (poche decine di MB) i modelli
pesanti (Whisper + LLM, alcuni GB) vengono scaricati automaticamente al
primo avvio dell'app, una sola volta, e salvati in %APPDATA%.

Whisper (faster-whisper) gestisce da solo il proprio download/cache
tramite huggingface_hub la prima volta che viene istanziato: non serve
codice dedicato per lui, viene semplicemente scaricato "on demand"
all'interno di TranscriptionEngine.load_model().

Qui gestiamo invece il download esplicito del modello LLM (file .gguf
singolo), con una barra di progresso da mostrare nella UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import requests

from app import config


class DownloadError(Exception):
    pass


def llm_model_present() -> bool:
    return config.LLM_MODEL_PATH.exists() and config.LLM_MODEL_PATH.stat().st_size > 0


def download_llm_model(
    on_progress: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Scarica il file GGUF del modello LLM locale.

    on_progress(bytes_scaricati, bytes_totali) viene chiamato periodicamente
    per aggiornare una eventuale barra di progresso nella UI.
    """
    tmp_path = config.LLM_MODEL_PATH.with_suffix(".part")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(config.LLM_MODEL_URL, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total)
        tmp_path.replace(config.LLM_MODEL_PATH)
        return config.LLM_MODEL_PATH
    except requests.RequestException as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise DownloadError(
            f"Impossibile scaricare il modello AI: {exc}. "
            "Verifica la connessione internet e riprova."
        ) from exc


def ensure_models_ready(on_progress: Optional[Callable[[str, int, int], None]] = None):
    """
    Da chiamare all'avvio dell'app (in un thread separato dalla UI).
    Scarica il modello LLM se manca. Il modello Whisper viene scaricato
    automaticamente al primo utilizzo da faster-whisper stesso.
    """
    if not llm_model_present():
        def _progress(done, total):
            if on_progress:
                on_progress("llm", done, total)

        download_llm_model(on_progress=_progress)
