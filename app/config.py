"""
Configurazione centrale dell'applicazione.

Tutti i percorsi, i nomi dei modelli e le impostazioni di default
vivono qui, cosi' il resto del codice non deve conoscere dettagli
sul filesystem o sui modelli usati.
"""
import os
import sys
from pathlib import Path

APP_NAME = "InterviewAssistant"
APP_DISPLAY_NAME = "AI Interview Assistant"
APP_VERSION = "0.1.0-mvp"

# --------------------------------------------------------------------------
# Percorsi applicazione
# --------------------------------------------------------------------------
# In produzione (eseguibile PyInstaller) i dati utente NON vanno mai scritti
# dentro la cartella di installazione (che puo' essere Program Files, non
# scrivibile senza privilegi admin). Usiamo sempre %APPDATA%.
def _app_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        # Utile per sviluppo/test su Linux/Mac durante la fase di build.
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


APP_DATA_DIR = _app_data_dir()
MODELS_DIR = APP_DATA_DIR / "models"
DB_PATH = APP_DATA_DIR / "interviews.db"
EXPORTS_DIR = APP_DATA_DIR / "exports"
LOG_DIR = APP_DATA_DIR / "logs"

for d in (MODELS_DIR, EXPORTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Modello di trascrizione (Whisper locale via faster-whisper)
# --------------------------------------------------------------------------
# "small" è un buon compromesso velocità/qualità su CPU e supporta il
# riconoscimento automatico di ~99 lingue. Configurabile dall'utente.
WHISPER_MODEL_SIZE_DEFAULT = "small"
WHISPER_COMPUTE_TYPE = "int8"  # più leggero, va bene su CPU senza GPU dedicata
WHISPER_CACHE_DIR = MODELS_DIR / "whisper"
WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Modello LLM locale per la generazione del report (no API, no costi)
# --------------------------------------------------------------------------
# Qwen2.5-3B-Instruct: licenza Apache 2.0 (uso commerciale libero),
# multilingue, gira su CPU con quantizzazione GGUF Q4_K_M (~2 GB).
LLM_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)
LLM_MODEL_PATH = MODELS_DIR / LLM_MODEL_FILENAME
LLM_CONTEXT_SIZE = 8192

# --------------------------------------------------------------------------
# Impostazioni audio
# --------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000  # richiesto da Whisper
AUDIO_CHANNELS = 1
TRANSCRIBE_CHUNK_SECONDS = 6  # ogni quanti secondi si esegue una trascrizione
TRANSCRIBE_OVERLAP_SECONDS = 1  # overlap per non tagliare le parole a metà

# --------------------------------------------------------------------------
# Impostazioni utente modificabili da UI (valori di default)
# --------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "whisper_model_size": WHISPER_MODEL_SIZE_DEFAULT,
    "capture_system_audio": True,   # cattura anche l'audio in uscita (es. Teams/Zoom)
    "capture_microphone": True,
    "report_language": "auto",      # "auto" = stessa lingua del colloquio, oppure "it", "en", ...
}
