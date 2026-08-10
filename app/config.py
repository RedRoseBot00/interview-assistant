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
APP_VERSION = "3.4.0"

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
# Qualita' della decodifica
# --------------------------------------------------------------------------
# Whisper prova a trascrivere e poi controlla il risultato: se il testo
# e' troppo ripetitivo o poco sicuro, RIPROVA con una temperatura piu'
# alta. Passandogli una temperatura singola quel meccanismo si spegne:
# il primo tentativo viene tenuto anche quando fallisce i controlli,
# cioe' proprio quando e' farfugliato.
#
# Il costo e' asimmetrico e conviene: sui blocchi riusciti al primo colpo
# non si paga nulla, perche' i tentativi successivi non vengono
# nemmeno eseguiti. Si paga solo dove il risultato sarebbe stato da
# buttare — ed e' li' che serve. E' la differenza fra una trascrizione
# leggibile e una inutile, tanto piu' marcata quanto piu' il modello e'
# piccolo e la lingua parlata e' poco rappresentata nel suo addestramento.
DECODE_TEMPERATURES_FULL = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DECODE_TEMPERATURES_FAST = (0.0, 0.4)

# La ricerca a fascio esamina piu' trascrizioni possibili invece di
# prendere sempre la parola piu' probabile. Costa tempo a OGNI blocco,
# quindi viene alzata solo quando il computer sta comodamente al passo.
DECODE_BEAM_MIN = 1
DECODE_BEAM_MID = 3
DECODE_BEAM_MAX = 5
# Quante trascrizioni alternative generare quando il primo tentativo
# viene male e si riprova a temperatura piu' alta. Sul computer che non
# sta al passo ne basta una: il tentativo di recupero resta, ma costa
# una decodifica invece di cinque.
DECODE_BEST_OF_MIN = 1
DECODE_BEST_OF_MAX = 5
# Soglie sul rapporto fra durata dell'audio e tempo impiegato, con
# isteresi per non oscillare fra una configurazione e l'altra.
SPEED_RAISE_QUALITY = 2.5
SPEED_LOWER_QUALITY = 1.3

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
# Il contesto non e' piu' fisso: viene calcolato sulla lunghezza reale
# del testo in ingresso (vedi summarization/llm.py). Ogni token di
# contesto in piu' costa memoria per la cache delle chiavi e rallenta
# tutto su un computer da ufficio, dove la memoria e' la vera strozzatura.
LLM_CONTEXT_MIN = 1024
LLM_CONTEXT_MAX = 4096
# Su un computer da ufficio il tempo di generazione cresce quasi in
# proporzione alla lunghezza del testo, sia in ingresso sia in uscita:
# un report piu' asciutto e una trascrizione piu' compatta riducono
# l'attesa da diversi minuti a poco piu' di uno, senza perdere le
# informazioni che servono davvero al selezionatore.
#
# La trascrizione viene prima compattata (battute unite, intercalari
# tolti): 4500 caratteri di testo compattato contengono piu' sostanza
# di 7000 caratteri grezzi, e costano un terzo del tempo di lettura.
LLM_MAX_TRANSCRIPT_CHARS = 4500
# Il formato richiesto al modello sta in 300-380 token: con un tetto
# generoso il modello riempie lo spazio disponibile con ripetizioni,
# facendo aspettare l'utente per testo che non aggiunge nulla.
LLM_MAX_TOKENS = 450
# Da non alzare sopra 512 senza prima verificarlo: la versione di
# llama-cpp-python che usiamo non espone n_ubatch, che resta fisso a
# 512. Con n_batch piu' grande i due valori divergono e la lettura del
# prompt peggiora invece di migliorare.
LLM_BATCH_SIZE = 512


def _physical_cores() -> int:
    """
    Numero di core FISICI, non di processori logici.

    os.cpu_count() conta i thread hardware: su un portatile AMD a due
    core con SMT restituisce 4. Dimensionare il calcolo su quel numero
    significa lanciare piu' thread di quanti core esistano davvero, e su
    carichi che saturano le unita' di calcolo (trascrizione e LLM sono
    esattamente cosi') il risultato non e' piu' veloce: e' piu' lento,
    perche' i thread si contendono la stessa unita' e la stessa cache.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            return max(1, int(physical))
    except Exception:
        pass
    logical = os.cpu_count() or 2
    # Senza psutil ipotizziamo SMT attivo: sbagliare per difetto costa
    # poco, sbagliare per eccesso dimezza le prestazioni.
    return max(1, logical // 2) if logical > 2 else max(1, logical)


def llm_threads() -> int:
    # Stessa regola della trascrizione: mai tutta la macchina dove c'e'
    # margine, tutta dove non ce n'e'. Il report si genera a colloquio
    # finito, ma l'utente sta comunque guardando lo schermo e la
    # finestra deve restare viva.
    return transcription_threads()

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000  # richiesto da Whisper
# Soglia di silenzio applicata al decimo di secondo piu' sonoro della
# frase. Il rilevatore di voce ha gia' scartato i silenzi: questa e'
# solo una rete di sicurezza, e va tenuta bassa. Con un valore alto un
# microfono integrato a guadagno modesto vedeva sparire intere risposte
# prima ancora di arrivare al riconoscimento vocale, senza che nulla lo
# segnalasse all'utente.
SILENCE_RMS_THRESHOLD = 0.0018

# Il taglio dell'audio non avviene piu' a intervalli fissi ma sulle
# pause del parlato: i parametri stanno in app/audio/vad.py. Queste
# durate restano come riferimento per le finestre di confronto dell'eco.
TRANSCRIBE_CHUNK_SECONDS = 5.0
TRANSCRIBE_OVERLAP_SECONDS = 0.6

# Suggerimento dato al riconoscimento vocale: orienta il modello sul
# lessico di un colloquio di lavoro e migliora la punteggiatura.
#
# Va dato NELLA LINGUA del colloquio, e solo quando quella lingua e'
# stata riconosciuta con certezza. Un suggerimento scritto in una lingua
# diversa da quella parlata spinge il modello verso la lingua sbagliata;
# e su frasi brevi o disturbate Whisper tende a restituire il
# suggerimento stesso al posto di cio' che ha sentito.
#
# Il programma non privilegia alcuna lingua: ogni lingua supportata ha
# il proprio suggerimento, con lo stesso identico significato. Per le
# lingue non presenti in questa tabella non viene dato alcun
# suggerimento, che e' la scelta neutra.
TRANSCRIPTION_PROMPTS = {
    "it": "Colloquio di lavoro tra un selezionatore e un candidato.",
    "en": "Job interview between a recruiter and a candidate.",
    "es": "Entrevista de trabajo entre un reclutador y un candidato.",
    "fr": "Entretien d'embauche entre un recruteur et un candidat.",
    "de": "Vorstellungsgesprach zwischen einem Personalverantwortlichen "
          "und einem Bewerber.",
    "pt": "Entrevista de emprego entre um recrutador e um candidato.",
    "nl": "Sollicitatiegesprek tussen een recruiter en een kandidaat.",
    "ro": "Interviu de angajare intre un recrutor si un candidat.",
    "pl": "Rozmowa kwalifikacyjna miedzy rekruterem a kandydatem.",
}

# Quante frasi concordi servono per fissare la lingua del colloquio.
#
# Le prime battute di un colloquio sono le peggiori su cui decidere:
# "Buongiorno", "Mi sente?", "Perfetto". Due voti bastavano a fissare
# per sempre la lingua sbagliata e a rendere illeggibile tutto il
# resto. Ora servono piu' voti, presi solo su frasi lunghe e sicure.
LANGUAGE_LOCK_VOTES = 4
LANGUAGE_VOTE_MIN_SECONDS = 2.0    # frasi troppo brevi non votano
LANGUAGE_VOTE_MIN_WORDS = 5
LANGUAGE_VOTE_MIN_PROBABILITY = 0.80
LANGUAGE_LOCK_MARGIN = 2           # scarto minimo sulla seconda classificata
LANGUAGE_UNLOCK_VOTES = 3          # frasi sicure e concordi per cambiare idea
LANGUAGE_UNLOCK_PROBABILITY = 0.90


def transcription_threads() -> int:
    """
    Thread da assegnare al riconoscimento vocale.

    Su un computer senza scheda grafica dedicata — e su una macchina
    virtuale sempre — l'anteprima della videochiamata non e' disegnata
    dalla GPU ma composta dal processore, dal servizio grafico di
    Windows. Saturando tutti i core quel servizio resta senza tempo di
    calcolo: l'anteprima va a scatti, e con lei tutto il desktop.
    Lasciarne uno libero e' quindi giusto — ma solo dove ce n'e' da
    lasciare.

    Su due core, togliere e' peggio del male: la trascrizione si ritrova
    con un thread solo, cioe' meta' della macchina, e il calcolo del
    riconoscimento vocale scala quasi linearmente con i core. Su una
    macchina virtuale la cosa peggiorava ancora, perche' l'ipervisore
    quasi mai dichiara i thread hardware: due processori virtuali
    risultano due core "fisici", e la sottrazione lasciava un thread
    solo dove ne servivano due. Da quattro core in su si lascia
    respirare il sistema; sotto, si usa quello che c'e'.
    """
    fisici = _physical_cores()
    if fisici >= 4:
        return fisici - 1
    logici = os.cpu_count() or 2
    return max(1, min(logici, max(2, fisici)))

# Etichette interne dei due canali audio
SPEAKER_RECRUITER = "recruiter"   # microfono locale: chi conduce il colloquio
SPEAKER_CANDIDATE = "candidate"   # audio di sistema: l'altra persona in call
