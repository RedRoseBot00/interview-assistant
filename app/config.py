"""
Configurazione centrale dell'applicazione.

Qui vivono percorsi, nomi dei modelli e costanti tecniche. Le scelte
modificabili dall'utente stanno invece in app/settings.py.
"""
import os
import sys
import tempfile
from pathlib import Path

APP_NAME = "InterviewAssistant"
APP_DISPLAY_NAME = "Interview Assistant"
APP_VERSION = "0.2.0"

# --------------------------------------------------------------------------
# Percorsi applicazione
# --------------------------------------------------------------------------
# I dati utente non vanno mai scritti nella cartella di installazione
# (puo' essere Program Files, non scrivibile senza privilegi da
# amministratore): usiamo %APPDATA%.
#
# Proviamo piu' destinazioni in ordine: su alcuni computer aziendali
# %APPDATA% e' reindirizzato su un disco di rete o bloccato da criteri
# di sicurezza, e senza un ripiego l'applicazione non partirebbe
# nemmeno, per giunta senza poter scrivere il motivo da nessuna parte.
def _candidate_dirs() -> list[Path]:
    candidates: list[Path] = []
    if sys.platform == "win32":
        for variable in ("APPDATA", "LOCALAPPDATA"):
            value = os.environ.get(variable)
            if value:
                candidates.append(Path(value))
        candidates.append(Path.home() / "AppData" / "Roaming")
    else:
        value = os.environ.get("XDG_DATA_HOME")
        if value:
            candidates.append(Path(value))
        candidates.append(Path.home() / ".local" / "share")
    candidates.append(Path(tempfile.gettempdir()))
    return candidates


def _app_data_dir() -> Path:
    last_error: Exception | None = None
    for base in _candidate_dirs():
        try:
            path = base / APP_NAME
            path.mkdir(parents=True, exist_ok=True)
            # Verifica di scrittura effettiva: l'esistenza della
            # cartella non garantisce il permesso di scriverci.
            probe = path / ".verifica-scrittura"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return path
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Nessuna cartella dati scrivibile sul computer. "
        f"Ultimo errore: {last_error}"
    )


APP_DATA_DIR = _app_data_dir()
MODELS_DIR = APP_DATA_DIR / "models"
DB_PATH = APP_DATA_DIR / "interviews.db"
EXPORTS_DIR = APP_DATA_DIR / "exports"
LOG_DIR = APP_DATA_DIR / "logs"
WHISPER_CACHE_DIR = MODELS_DIR / "whisper"

for _d in (MODELS_DIR, EXPORTS_DIR, LOG_DIR, WHISPER_CACHE_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Una sottocartella non creabile non deve impedire l'avvio:
        # l'errore verra' segnalato quando quella funzione servira'.
        pass

# --------------------------------------------------------------------------
# Modello di trascrizione (Whisper locale via faster-whisper)
# --------------------------------------------------------------------------
# Whisper riconosce automaticamente circa 99 lingue: nessuna
# configurazione necessaria per il supporto multilingua.
WHISPER_MODEL_SIZES = ("tiny", "base", "small", "medium")
WHISPER_MODEL_SIZE_DEFAULT = "small"
WHISPER_COMPUTE_TYPE = "int8"  # quantizzato: gira su CPU senza GPU dedicata

# --------------------------------------------------------------------------
# Modello LLM locale per la generazione del report (no API, no costi)
# --------------------------------------------------------------------------
# Qwen2.5-3B-Instruct: licenza Apache 2.0 (uso commerciale libero),
# fortemente multilingue, ~2 GB in quantizzazione Q4_K_M.
LLM_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)
LLM_MODEL_PATH = MODELS_DIR / LLM_MODEL_FILENAME
# Il file completo pesa circa 2,0 GB: sotto questa soglia si tratta
# certamente di un download interrotto, non di un modello utilizzabile.
LLM_MODEL_MIN_BYTES = 1_700_000_000
LLM_CONTEXT_SIZE = 8192
# Lasciamo circa 1800 token liberi per la risposta: il resto e' il tetto
# massimo di trascrizione che possiamo inserire nel prompt.
LLM_MAX_TRANSCRIPT_CHARS = 14000

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000  # richiesto da Whisper
TRANSCRIBE_CHUNK_SECONDS = 5.0     # durata della finestra trascritta
TRANSCRIBE_OVERLAP_SECONDS = 0.6   # sovrapposizione: evita parole troncate
SILENCE_RMS_THRESHOLD = 0.004      # sotto questa soglia il blocco e' silenzio

# Etichette interne dei due canali audio
SPEAKER_RECRUITER = "recruiter"   # microfono locale: chi conduce il colloquio
SPEAKER_CANDIDATE = "candidate"   # audio di sistema: l'altra persona in call
