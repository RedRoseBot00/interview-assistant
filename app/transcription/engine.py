"""
Motore di trascrizione in tempo reale.

Basato su faster-whisper (Whisper eseguito da CTranslate2): gira in
locale, senza connessione e senza costi, e riconosce automaticamente
circa 99 lingue: il colloquio puo' svolgersi in qualunque lingua
senza configurare nulla, e nessuna lingua e' privilegiata.

Il motore consuma i blocchi audio gia' etichettati per interlocutore
prodotti da app.audio.capture e restituisce frasi attribuite a "Tu"
(microfono) o al "Candidato" (audio della videochiamata).

Qui avviene anche la gestione dell'eco: quando il selezionatore non usa
le cuffie, il microfono ricattura la voce che esce dagli altoparlanti.
I blocchi del microfono vengono quindi confrontati con l'audio che il
computer stava riproducendo nello stesso istante e, se risultano una
copia di quello, scartati o ripuliti (vedi app/audio/echo.py).
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from app import config
from app.audio import echo as echo_module
from app.audio.capture import AudioChunk, rms_level

log = logging.getLogger(__name__)

# Quanto una frase del microfono puo' restare in attesa dell'audio di
# riferimento prima di essere trascritta comunque. Ora che il canale del
# candidato segnala anche i propri silenzi, l'attesa e' quasi sempre
# nulla: questo valore serve solo come rete di sicurezza.
REFERENCE_WAIT_SECONDS = 1.2

# Accorpamento delle frasi vicine.
#
# Il riconoscimento vocale lavora sempre su una finestra di trenta
# secondi, che riempie di silenzio quando l'audio e' piu' corto: il
# costo di una chiamata e' quindi quasi lo stesso per una frase di
# mezzo secondo e per una di otto. Unire le frasi gia' in attesa nella
# coda riduce il numero di chiamate senza aggiungere alcun ritardo,
# perche' si accorpa soltanto cio' che e' gia' arrivato.
# La finestra di Whisper e' di trenta secondi: restando sotto i
# ventisei si sfrutta quasi tutta la chiamata invece di sprecarne il
# sessanta per cento in silenzio aggiunto. Accorpare non ritarda nulla,
# perche' si uniscono solo frasi gia' presenti in coda.
MERGE_GAP_SECONDS = 1.5
MERGE_MAX_SECONDS = 26.0
MERGE_LOOKAHEAD = 32
# Sovrapposizione massima ammessa quando si riattacca il seguito di una
# frase troncata: e' il pre-roll che il rilevatore di voce conserva
# davanti a ogni frase (0,30 s), piu' un margine.
MERGE_CONTINUATION_OVERLAP = 0.45
# Quanto parlato si puo' tenere da parte per l'accorpamento. Serve solo
# a riempire una finestra da ventisei secondi: tutto cio' che eccede e'
# tempo sottratto al conteggio dell'arretrato senza alcun vantaggio, e
# allarga il tetto reale dell'attesa oltre quello dichiarato.
LOOKAHEAD_MAX_SECONDS = MERGE_MAX_SECONDS

# Quando la coda cresce, l'unica cosa che conta e' tornare al presente:
# si accorpa fino al limite di durata ignorando la lunghezza delle
# pause, dimezzando il numero di chiamate al riconoscimento vocale.
# L'arretrato va misurato in SECONDI di parlato, non in numero di
# elementi: gli avvisi di silenzio dell'altro canale sono elementi che
# valgono zero secondi, e ne arrivano due al secondo. Contandoli, la
# condizione "sono in ritardo" risultava vera quasi sempre, anche a coda
# vuota, e il programma restava perennemente in modalita' di emergenza.
BACKLOG_AGGRESSIVE_SECONDS = 8.0
BACKLOG_GAP_SECONDS = 6.0

# --------------------------------------------------------------------------
# Adattamento automatico del modello
# --------------------------------------------------------------------------
# Whisper elabora sempre una finestra di trenta secondi: il costo dipende
# dal NUMERO di chiamate, non da quanto parlato contengono. In un
# colloquio le battute si alternano fra due persone circa venti volte al
# minuto, e ogni battuta e' una chiamata. Se una chiamata costa piu' del
# tempo che passa fra una battuta e l'altra, il ritardo non si stabilizza
# mai: cresce per tutta la durata del colloquio.
#
# Non c'e' accorpamento che possa rimediare — le battute consecutive sono
# di persone diverse e vanno trascritte separatamente. L'unica leva vera
# e' il costo della singola chiamata, cioe' la dimensione del modello.
#
# Il programma se ne accorge da solo e scende di un gradino, spiegando
# perche'. Meglio un modello un po' meno preciso che sta al passo, che
# uno migliore i cui sottotitoli arrivano un minuto dopo la voce.
OVERLOAD_RATIO = 1.15          # oltre questo carico non si sta al passo
OVERLOAD_MIN_CALLS = 8         # non si decide su due misure
MODEL_LADDER = ("medium", "small", "base", "tiny")
# Quanti gradini si possono scendere in un solo colloquio. Due bastano
# per arrivare da 'small' a 'tiny', che e' il caso delle macchine
# virtuali a due core; di piu' sarebbe solo altalena.
MAX_DOWNGRADES = 2
# Corsia d'emergenza: quando una chiamata costa piu' di questo multiplo
# dell'audio che contiene, si scende dopo appena due conferme invece di
# aspettarne otto — che sul computer misurato sul campo significava
# decidere dopo la fine del colloquio.
FAST_OVERLOAD_CALLS = 2
FAST_OVERLOAD_FACTOR = 2.5

# Finestra di analisi del riconoscimento, in secondi. Trenta e' il
# valore su cui il modello e' addestrato; sotto i dieci la resa degrada
# sensibilmente e non si scende mai.
ENCODER_WINDOW_FULL = 30
ENCODER_WINDOW_MIN = 10

# Secondi di parlato perso oltre i quali si tenta l'alleggerimento
# d'emergenza, qualunque cosa dica il carico stimato.
DROP_ALARM_SECONDS = 3.0

# Ogni quante frasi lunghe si lascia il modello libero di ridire la sua
# sulla lingua, per potersi ricredere su un blocco iniziale sbagliato.
LANGUAGE_PROBE_EVERY = 8
# Finche' la lingua non e' confermata si sonda molto piu' spesso: la
# prima ipotesi guida gia' la decodifica, quindi se e' sbagliata deve
# poter essere corretta in fretta e non dopo otto frasi illeggibili.
LANGUAGE_PROBE_UNCONFIRMED = 2
# Durata minima perche' valga la pena rimettere in discussione la
# lingua. Sotto, il riconoscimento non e' affidabile e la frase verrebbe
# soltanto rovinata.
LANGUAGE_PROBE_MIN_SECONDS = 4.0
# Fiducia minima per ADOTTARE una lingua come ipotesi provvisoria. E'
# volutamente bassa: un'ipotesi anche mediocre e' molto meglio di
# nessuna, perche' senza lingua ogni chiamata decodifica in quella
# indovinata a caso su due secondi di audio. L'ipotesi resta comunque
# in discussione a ogni sonda.
LANGUAGE_ADOPT_PROBABILITY = 0.5

# Quanto tempo si concede alla trascrizione per recuperare l'arretrato
# dopo che l'utente ha premuto "Termina colloquio". Oltre, si abbandona
# quello che resta: e' preferibile a un thread che continua a occupare
# il processore per minuti mentre il programma scrive il report.
STOP_GRACE_SECONDS = 40.0

# Finestra temporale entro cui due frasi uguali su canali diversi sono
# considerate la stessa frase (eco), non due interventi distinti.
DUPLICATE_WINDOW_SECONDS = 12.0


@dataclass
class TranscriptSegment:
    speaker: str                     # config.SPEAKER_RECRUITER | SPEAKER_CANDIDATE
    text: str
    language: str = ""
    offset_seconds: float = 0.0      # secondi dall'inizio del colloquio
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "language": self.language,
            "offset_seconds": round(self.offset_seconds, 2),
        }


def _loudest_rms(samples: np.ndarray, window: int = 1600) -> float:
    """
    Livello del decimo di secondo piu' sonoro della frase.

    La media sull'intera frase e' fuorviante, perche' comprende il
    silenzio conservato prima e dopo il parlato per non troncare le
    parole: una risposta breve ma nitida risulterebbe sotto soglia.
    """
    if samples.size == 0:
        return 0.0
    if samples.size <= window:
        return rms_level(samples)
    usable = (samples.size // window) * window
    blocks = samples[:usable].reshape(-1, window).astype(np.float64)
    return float(np.sqrt(np.max(np.mean(blocks * blocks, axis=1))))


def _tame_peaks(samples: np.ndarray, ratio: float = 4.0) -> np.ndarray:
    """
    Ammorbidisce i picchi isolati molto piu' forti del parlato.

    Whisper normalizza il proprio spettrogramma rispetto al valore
    massimo del segmento e scarta tutto cio' che sta piu' di 80 dB sotto
    di esso. Un clic del mouse, un colpo di tosse o una porta che sbatte
    alzano quel massimo e schiacciano la voce verso il fondo della
    dinamica: la frase risulta improvvisamente incomprensibile per un
    motivo che non ha nulla a che vedere con chi parla.

    La curva usata e' continua anche nella derivata, quindi non
    introduce le armoniche stridule di un taglio netto.
    """
    if samples.size == 0:
        return samples
    # Il livello di riferimento e' quello del tratto piu' sonoro, non la
    # media dell'intera frase. Con la media, una frase nata da un
    # accorpamento — che contiene per costruzione il silenzio delle
    # pause — abbassava il riferimento al punto che il compressore
    # entrava sulle sillabe vere: la resa del riconoscimento finiva per
    # dipendere da quanto arretrato aveva la coda in quel momento.
    livello = _loudest_rms(samples)
    if livello <= 0.0:
        return samples
    limite = ratio * livello
    modulo = np.abs(samples)
    if float(modulo.max()) <= limite:
        return samples
    compresso = np.sign(samples) * limite * (2.0 - limite / np.maximum(modulo, 1e-9))
    return np.where(modulo <= limite, samples, compresso).astype(np.float32)


# Livello a cui portiamo le frasi troppo deboli prima del
# riconoscimento: un microfono integrato a guadagno basso produce un
# segnale che, pur essendo parlato nitido, resta vicino al fondo scala.
TARGET_PEAK = 0.25
MAX_GAIN = 8.0


def _normalise_gain(samples: np.ndarray) -> np.ndarray:
    picco = float(np.max(np.abs(samples))) if samples.size else 0.0
    if picco <= 0.0 or picco >= TARGET_PEAK:
        return samples
    return (samples * min(MAX_GAIN, TARGET_PEAK / picco)).astype(np.float32)


def _normalise(word: str) -> str:
    return word.strip(".,;:!?'\"()").lower()


def _echoes_prompt(text: str) -> bool:
    """La trascrizione e' solo una copia del suggerimento di contesto?"""
    pulito = " ".join(_normalise(w) for w in text.split())
    if not pulito:
        return False
    for frase in config.TRANSCRIPTION_PROMPTS.values():
        modello = " ".join(_normalise(w) for w in frase.split())
        if pulito == modello or (len(pulito) > 20 and pulito in modello):
            return True
    return False


def _strip_overlap(previous: str, current: str, max_words: int = 8) -> str:
    """
    Rimuove la ripetizione dovuta alla sovrapposizione tra blocchi.

    I blocchi audio si sovrappongono di qualche decimo di secondo per
    non troncare le parole a meta', quindi l'inizio di una trascrizione
    puo' ripetere la fine della precedente.

    Due accorgimenti: la sovrapposizione tipica e' di una o due parole,
    quindi va considerato anche il caso di una sola parola (purche' non
    sia una parolina breve, che si ripete spesso in modo legittimo); e
    non si scarta mai l'intera frase, altrimenti una risposta ripetuta
    ("Perfetto." detto due volte) sparirebbe dalla trascrizione.
    """
    if not previous or not current:
        return current

    prev_words = previous.split()
    curr_words = current.split()
    limit = min(max_words, len(prev_words), len(curr_words))

    for size in range(limit, 0, -1):
        if size >= len(curr_words):
            continue
        tail = [_normalise(w) for w in prev_words[-size:]]
        head = [_normalise(w) for w in curr_words[:size]]
        if tail != head:
            continue
        if size == 1 and len(tail[0]) < 4:
            continue
        return " ".join(curr_words[size:]).strip()
    return current


class TranscriptionEngine:
    """Trascrive i blocchi audio in un thread dedicato."""

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL_SIZE_DEFAULT,
        language: str = "auto",
        echo_mode: str = echo_module.MODE_AUTO,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.model_size = model_size
        self.language = language
        self.on_segment = on_segment
        self.on_status = on_status
        self.on_error = on_error

        self._model = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Lingua: si riconosce nei primi interventi e poi si fissa. Il
        # riconoscimento automatico ripetuto a ogni frase costa tempo e,
        # sulle frasi brevi, sbaglia spesso attribuendo il parlato
        # alla lingua sbagliata.
        self._locked_language: Optional[str] = (
            None if language == "auto" else language
        )
        # Quando la lingua e' stata scelta dall'utente non deve mai
        # cambiare; quando l'abbiamo dedotta noi, invece, restiamo
        # disposti a ricrederci.
        self._language_forced = language != "auto"
        # La lingua attraversa tre stati: ignota, ipotizzata, confermata.
        # Lo stato intermedio e' quello che mancava: la prima ipotesi
        # guida gia' la decodifica — perche' decodificare nella lingua
        # sbagliata rende una frase incomprensibile, non solo imprecisa —
        # ma resta in discussione finche' non arrivano abbastanza voti
        # concordi, e nel frattempo viene riverificata a frasi alterne.
        self._language_confirmed = self._language_forced
        self._language_votes: Counter[str] = Counter()
        # Sblocco: servono voti CONSECUTIVI sulla stessa lingua nuova.
        self._unlock_candidate: Optional[str] = None
        self._unlock_streak = 0
        self._probe_countdown = 0

        # Rapporto fra durata dell'audio e tempo impiegato a trascriverlo.
        self.realtime_factor = 0.0
        self._speed_samples = 0
        # Secondi di orologio spesi in media per una chiamata al
        # riconoscimento vocale, qualunque sia la durata dell'audio.
        self._call_cost = 0.0
        # Ritmo con cui il rilevatore di voce produce le frasi.
        self._arrival_gap = 0.0
        self._last_arrival: Optional[float] = None
        # Alleggerimento automatico del modello: una volta sola.
        # Quante volte si e' gia' sceso di un gradino nella scala dei
        # modelli. Era un si'/no: su un computer molto lento un solo
        # gradino non basta.
        self._downgraded = 0
        # Modello consigliato all'utente quando e' lui ad aver scelto
        # quello attuale e questo non regge.
        self.suggested_model = ""
        self.model_changed_to = ""
        # 0 = minimo indispensabile, 1 = ricerca media, 2 = ricerca ampia.
        # Si parte dal MINIMO e si sale solo dopo aver misurato che il
        # computer regge. Partendo dal livello medio, le prime frasi di
        # ogni colloquio erano lente proprio sulle macchine modeste, e
        # bastavano a creare un arretrato da cui non si rientrava piu'.
        self._quality_level = 0

        self.segments: list[TranscriptSegment] = []
        self.backlog = 0
        self.backlog_seconds = 0.0

        self._languages: Counter[str] = Counter()
        self._last_text: dict[str, str] = {}
        self._lock = threading.Lock()

        # Gestione dell'eco
        self.echo = echo_module.EchoProcessor(echo_mode, config.AUDIO_SAMPLE_RATE)
        self._reference = echo_module.ReferenceBuffer(config.AUDIO_SAMPLE_RATE)
        self._waiting: deque[tuple[AudioChunk, float]] = deque()
        self._local: deque[AudioChunk] = deque()
        self.echo_dropped = 0
        self.duplicates_dropped = 0
        # Blocchi abbandonati perche' l'arretrato era troppo grande al
        # momento dell'arresto: vanno detti all'utente, non nascosti.
        self.dropped_on_stop = 0
        # Secondi di parlato gia' visti scartare dalla coda: serve a
        # reagire solo agli scarti NUOVI.
        self._dropped_seen = 0.0
        self.merged_utterances = 0

        # Conteggi aggiornati man mano. Ricalcolarli ogni secondo
        # scorrendo tutti i segmenti significava tenere il lock del
        # motore e rifare quattro passate sul testo dal thread grafico,
        # con l'interfaccia che si faceva sempre piu' pesante col
        # passare dei minuti di colloquio.
        self._stats = {
            "questions": 0,
            "recruiter_words": 0,
            "candidate_words": 0,
            "segments": 0,
        }

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Carica il modello in memoria. Operazione lenta: mai dal thread grafico."""
        from faster_whisper import WhisperModel

        from app.models.download import whisper_model_dir, whisper_model_present

        if not whisper_model_present(self.model_size):
            raise RuntimeError(
                f"Il modello di trascrizione '{self.model_size}' non e' installato "
                "o e' incompleto. Riavvia l'applicazione per completarne il download."
            )

        self._notify_status("Caricamento del modello di trascrizione...")
        started = time.monotonic()
        self._model = WhisperModel(
            str(whisper_model_dir(self.model_size)),
            device="cpu",
            compute_type=config.WHISPER_COMPUTE_TYPE,
            # Senza questa indicazione la libreria usa un solo thread e
            # su un portatile la trascrizione resta indietro rispetto al
            # parlato. Lasciamo un core libero solo se ce ne sono almeno
            # quattro: su un computer con due core servono entrambi.
            cpu_threads=config.transcription_threads(),
            num_workers=1,
        )
        # Queste righe sono la prima cosa da leggere quando un utente
        # segnala lentezza: dicono con quanti thread si sta lavorando e
        # con quale precisione. Senza, un computer che gira con un solo
        # thread o con i kernel generici e' indistinguibile da uno che
        # gira come dovrebbe.
        interno = getattr(self._model, "model", None)
        log.info(
            "Modello '%s' caricato in %.1f s — %d thread "
            "(core fisici %s, logici %s), precisione richiesta '%s', "
            "effettiva '%s'",
            self.model_size,
            time.monotonic() - started,
            config.transcription_threads(),
            config._physical_cores(),
            os.cpu_count(),
            config.WHISPER_COMPUTE_TYPE,
            getattr(interno, "compute_type", "sconosciuta"),
        )

    # ------------------------------------------------------------------
    def start(self, audio_queue: "queue.Queue[AudioChunk]") -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(audio_queue,), name="transcription", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 25.0) -> bool:
        """
        Ferma il motore lasciandogli il tempo di smaltire la coda: le
        ultime frasi del colloquio devono comparire nel report.

        Restituisce True solo se il thread e' davvero terminato: chi
        chiama deve saperlo, perche' un thread ancora vivo continua a
        scrivere nella lista dei segmenti.
        """
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True

        thread.join(timeout=timeout)
        if thread.is_alive():
            log.warning(
                "Trascrizione ancora in corso dopo %.0f s (%d blocchi arretrati)",
                timeout,
                self.backlog,
            )
            self._notify_status(
                f"Completamento della trascrizione: {self.backlog} blocchi rimasti..."
            )
            thread.join(timeout=timeout)

        if thread.is_alive():
            log.error("Il thread di trascrizione non si e' fermato")
            # Il thread continua a girare da solo, e questo non lo si puo'
            # impedire: e' fermo dentro una chiamata al riconoscimento
            # vocale, che non e' interrompibile. Cio' che si PUO' impedire
            # e' che continui a parlare con l'interfaccia. Chi ha chiamato
            # stop() sta per chiudere la sessione, e l'oggetto grafico
            # verra' distrutto: il primo segmento che il thread produceva
            # dopo quel momento faceva morire tutto con "Internal C++
            # object already deleted", e con esso spariva anche il
            # tentativo di segnalare l'errore.
            self.detach_callbacks()
            return False

        self._thread = None
        return True

    def detach_callbacks(self) -> None:
        """
        Taglia ogni collegamento con l'interfaccia.

        Da chiamare quando la sessione sta per essere distrutta ma il
        thread di trascrizione e' ancora vivo. Da quel momento il thread
        lavora nel vuoto: consuma quello che ha in mano e finisce, senza
        poter toccare oggetti che non esistono piu'.
        """
        self.on_segment = None
        self.on_status = None
        self.on_error = None

    def release_model(self) -> None:
        """
        Restituisce alla macchina la memoria del modello.

        Va chiamata a colloquio finito. Subito dopo il programma carica
        il modello che scrive il report, che pesa due gigabyte: tenere in
        piedi anche questo, che non serve piu' a nessuno, e' quanto basta
        perche' un computer con quattro gigabyte di memoria cominci a
        lavorare sul disco invece che in memoria — ed e' la situazione
        normale su una macchina virtuale.
        """
        if self._model is None:
            return
        self._model = None
        import gc

        gc.collect()
        log.info("Modello di trascrizione rilasciato")

    # ------------------------------------------------------------------
    def _notify_status(self, message: str) -> None:
        avviso = self.on_status
        if avviso is None:
            return
        try:
            avviso(message)
        except RuntimeError:
            self.detach_callbacks()
        except Exception:
            log.debug("Notifica di stato fallita", exc_info=True)

    def _notify_error(self, exc: Exception) -> None:
        """
        Segnala un errore all'interfaccia senza poterne generare un altro.

        Era l'ultimo anello della catena che faceva morire il thread di
        trascrizione: il primo errore veniva raccolto, ma il tentativo di
        raccontarlo all'interfaccia — nel frattempo distrutta — ne
        sollevava un secondo, e quello non lo prendeva piu' nessuno.
        """
        segnala = self.on_error
        if segnala is None:
            return
        try:
            segnala(exc)
        except RuntimeError:
            self.detach_callbacks()
        except Exception:
            log.debug("Segnalazione dell'errore non riuscita", exc_info=True)

    @staticmethod
    def _lower_own_priority() -> None:
        """
        Abbassa la priorita' del thread che esegue il riconoscimento.

        La trascrizione e' un lavoro pesante ma NON urgente: puo'
        arrivare mezzo secondo dopo senza che nessuno se ne accorga.
        L'interfaccia, la cattura audio e il servizio grafico di Windows
        sono invece urgenti: un loro ritardo si vede subito, sotto forma
        di finestra che non risponde e anteprima a scatti.

        A parita' di priorita' Windows alterna i thread a rotazione, e un
        calcolo che tiene il processore occupato senza pause vince quasi
        sempre il confronto. Bastano due gradini in meno perche' il
        sistema restituisca la precedenza a chi disegna lo schermo, senza
        che la trascrizione ci perda in pratica nulla: il processore
        resta comunque suo per quasi tutto il tempo.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes

            THREAD_PRIORITY_BELOW_NORMAL = -1
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel32.SetThreadPriority(
                kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL
            ):
                log.debug("Priorita' del thread non modificata")
        except Exception:
            log.debug("Priorita' del thread non modificabile", exc_info=True)

    def _run(self, audio_queue: "queue.Queue[AudioChunk]") -> None:
        self._lower_own_priority()
        try:
            if self._model is None:
                self.load_model()
            self._notify_status("In ascolto")
        except Exception as exc:
            log.exception("Caricamento del modello non riuscito")
            self._notify_error(exc)
            # Svuotiamo comunque la coda: senza consumatore i thread
            # audio la riempirebbero fino a scartare blocchi a vuoto.
            self._drain(audio_queue)
            return

        scadenza_arresto: Optional[float] = None
        while True:
            # Dopo "Termina colloquio" si concede un tempo limitato per
            # recuperare l'arretrato, poi si smette. Prima non c'era
            # alcun limite: con cinquanta frasi in coda e dieci secondi
            # per frase, il thread restava a macinare piu' di otto minuti
            # dopo la fine del colloquio, invisibile, tenendo occupato
            # tutto il processore mentre il programma cercava di scrivere
            # il report — e nel frattempo la finestra era gia' stata
            # buttata via, il che lo faceva morire con un errore.
            if self._stop.is_set():
                if scadenza_arresto is None:
                    scadenza_arresto = time.monotonic() + STOP_GRACE_SECONDS
                elif time.monotonic() > scadenza_arresto:
                    rimasti = (
                        audio_queue.qsize() + len(self._local) + len(self._waiting)
                    )
                    if rimasti:
                        log.warning(
                            "Arresto: abbandono %d blocchi non trascritti dopo %.0f s",
                            rimasti, STOP_GRACE_SECONDS,
                        )
                        self.dropped_on_stop = rimasti
                    break
            try:
                chunk = self._next_chunk(audio_queue, timeout=0.4)
            except queue.Empty:
                self.backlog = len(self._waiting) + len(self._local)
                # Va aggiornato anche qui: quando non arriva nulla di
                # nuovo restava fermo al valore dell'ultimo blocco, e
                # l'avviso "non sta al passo" poteva restare acceso
                # mentre l'arretrato si stava smaltendo.
                in_attesa = getattr(audio_queue, "pending_seconds", None)
                if in_attesa is not None:
                    self.backlog_seconds = float(in_attesa)
                try:
                    self._flush_waiting(force=self._stop.is_set())
                except Exception as exc:
                    log.exception("Errore nello smaltimento delle frasi in attesa")
                    self._notify_error(exc)
                    self._waiting.clear()
                self._update_parked(audio_queue)
                if self._stop.is_set() and not self._waiting:
                    break
                continue

            self.backlog = audio_queue.qsize() + len(self._local) + len(self._waiting)
            # Secondi di parlato ancora da trascrivere. E' questo, non il
            # numero di elementi, a dire se si e' indietro: il canale del
            # candidato manda due avvisi di silenzio al secondo, che non
            # valgono nulla ma facevano salire il conteggio a quattro nel
            # giro di due secondi. L'avviso "non sta al passo" compariva
            # cosi' quasi sempre, anche a trascrizione perfettamente in
            # pari, e falsava la diagnosi di chi cercava il problema.
            # pending_seconds comprende gia' quello che abbiamo in mano,
            # _local e _waiting inclusi: e' _update_parked a dichiararlo
            # alla coda. Sommarci di nuovo _waiting lo contava due volte.
            in_coda = getattr(audio_queue, "pending_seconds", None)
            self.backlog_seconds = (
                float(self.backlog) if in_coda is None else float(in_coda)
            )
            try:
                self._accept(self._merge_following(audio_queue, chunk))
            except Exception as exc:
                log.exception("Errore durante l'elaborazione di un blocco")
                self._notify_error(exc)
            self._update_parked(audio_queue)

            # Parlato buttato via dalla coda: prova DEFINITIVA di
            # sovraccarico. Il carico stimato qui si inganna da solo —
            # quando si scarta, il ritmo misurato e' quello dei blocchi
            # sopravvissuti e sembra tutto in regola — quindi il
            # regolatore ordinario non scattava mai proprio mentre si
            # perdeva parlato. Il parlato perso invece non mente.
            scarti = float(getattr(audio_queue, "dropped_seconds", 0.0) or 0.0)
            if scarti - self._dropped_seen >= DROP_ALARM_SECONDS:
                self._dropped_seen = scarti
                log.warning(
                    "Persi %.0f s di parlato per sovraccarico: provo ad "
                    "alleggerire", scarti,
                )
                self._consider_lighter_model(emergenza=True)

        # Lo svuotamento finale va protetto quanto il resto: e' il
        # momento in cui escono le ultime frasi del colloquio, e un
        # errore qui le faceva sparire tutte in silenzio, per giunta
        # senza mai segnalare la fine della trascrizione.
        try:
            self._flush_waiting(force=True)
        except Exception as exc:
            log.exception("Errore nello svuotamento finale")
            self._notify_error(exc)
        if self.echo_dropped or self.duplicates_dropped:
            log.info(
                "Eco gestita: %d blocchi scartati sul segnale, %d frasi duplicate",
                self.echo_dropped,
                self.duplicates_dropped,
            )
        self._notify_status("Trascrizione conclusa")

    @staticmethod
    def _drain(audio_queue: "queue.Queue[AudioChunk]") -> None:
        try:
            while True:
                audio_queue.get_nowait()
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # Accorpamento delle frasi contigue
    # ------------------------------------------------------------------
    def _next_chunk(
        self, audio_queue: "queue.Queue[AudioChunk]", timeout: float
    ) -> AudioChunk:
        if self._local:
            blocco = self._local.popleft()
            self._update_parked(audio_queue)
            return blocco
        blocco = audio_queue.get(timeout=timeout)
        self._note_arrival(blocco)
        return blocco

    def _fill_local(self, audio_queue: "queue.Queue[AudioChunk]") -> None:
        """
        Sbircia le frasi successive per capire se si possono accorpare.

        Il limite e' doppio, per numero e per durata. Contare solo gli
        elementi non bastava: una frase puo' durare dieci secondi, quindi
        trentadue elementi potevano valere piu' di cinque minuti di
        parlato tolti dalla coda e tenuti da parte. Quel tempo spariva
        dalla vista del limite sull'arretrato e faceva scattare a vuoto
        lo scarto dell'audio piu' vecchio. Le frasi che restano in coda
        non si perdono: vengono prelevate al giro successivo.
        """
        raccolti = self._durata_locale()
        try:
            while (
                len(self._local) < MERGE_LOOKAHEAD
                and raccolti < LOOKAHEAD_MAX_SECONDS
            ):
                nuovo = audio_queue.get_nowait()
                self._local.append(nuovo)
                self._note_arrival(nuovo)
                raccolti += nuovo.samples.size / float(config.AUDIO_SAMPLE_RATE)
        except queue.Empty:
            pass
        self._update_parked(audio_queue)

    def _note_arrival(self, chunk: AudioChunk) -> None:
        """
        Ogni quanto vengono prodotte le frasi, in secondi di orologio.

        Si misura sull'istante di NASCITA del blocco, non su quando lo
        preleviamo noi: cosi' il valore descrive il ritmo di chi parla e
        non quanto siamo indietro. E' il termine di paragone con cui
        decidere se conviene aspettare.
        """
        if chunk.samples.size == 0:
            return
        precedente = self._last_arrival
        self._last_arrival = chunk.wall_time
        if precedente is None:
            return
        distanza = chunk.wall_time - precedente
        if not (0.0 < distanza < 60.0):
            return
        if self._arrival_gap <= 0:
            self._arrival_gap = distanza
        else:
            self._arrival_gap = 0.7 * self._arrival_gap + 0.3 * distanza

    def _mergeable_seconds(self, speaker: str) -> float:
        """
        Parlato in attesa che finira' DAVVERO nella prossima chiamata.

        L'accorpamento si ferma al primo blocco di un altro
        interlocutore: conta quindi solo la testa della coda locale,
        esattamente come fara' _merge_following.
        """
        totale = 0.0
        for blocco in self._local:
            if blocco.samples.size == 0:
                continue      # avviso di silenzio: non ferma la fusione
            if blocco.speaker != speaker or blocco.continues_previous:
                break
            totale += blocco.samples.size / float(config.AUDIO_SAMPLE_RATE)
        return totale

    def _durata_locale(self) -> float:
        return sum(c.samples.size for c in self._local) / float(
            config.AUDIO_SAMPLE_RATE
        )

    def _update_parked(self, audio_queue) -> None:
        """
        Dichiara alla coda quanti secondi di audio abbiamo in mano.

        Il limite sull'arretrato vive nel thread di cattura e guarda il
        contatore della coda. Estraendo fino a trentadue blocchi per
        poterli accorpare, quel tempo spariva dal contatore: il limite
        vedeva pochi secondi mentre il ritardo reale superava i cinque
        minuti, e non scartava mai nulla.
        """
        if not hasattr(audio_queue, "parked_seconds"):
            return
        campioni = sum(c.samples.size for c in self._local)
        campioni += sum(c.samples.size for c, _ in self._waiting)
        audio_queue.parked_seconds = campioni / float(config.AUDIO_SAMPLE_RATE)

    def _merge_following(
        self, audio_queue: "queue.Queue[AudioChunk]", chunk: AudioChunk
    ) -> AudioChunk:
        """
        Unisce alla frase corrente quelle immediatamente successive dello
        stesso interlocutore, se sono gia' in coda e molto ravvicinate.

        Il silenzio fra una frase e l'altra viene reinserito: cosi' la
        durata complessiva resta fedele al tempo reale e il confronto
        con l'audio della videochiamata, usato per riconoscere l'eco,
        continua a combaciare.
        """
        if chunk.samples.size == 0:
            return chunk

        rate = config.AUDIO_SAMPLE_RATE
        self._fill_local(audio_queue)
        if not self._local:
            return chunk

        parti = [chunk.samples]
        fine = chunk.offset + chunk.samples.size / rate
        uniti = 0
        # Silenzio aggiunto per ricostruire le pause. Va tenuto da parte:
        # e' audio che non contiene parlato, e conteggiarlo come tale
        # faceva passare per "frase lunga e sicura" un accorpamento di
        # due monosillabi separati da un secondo di pausa — proprio le
        # battute su cui non ci si deve fidare per decidere la lingua.
        silenzio_aggiunto = 0.0

        # In arretrato conta solo smaltire: si accorpa anche attraverso
        # pause piu' lunghe, perche' una chiamata da venti secondi costa
        # quanto quattro da cinque.
        in_ritardo = self._durata_locale() >= BACKLOG_AGGRESSIVE_SECONDS
        pausa_massima = BACKLOG_GAP_SECONDS if in_ritardo else MERGE_GAP_SECONDS

        while self._local:
            successiva = self._local[0]
            if successiva.samples.size == 0:
                # Avviso di silenzio dell'altro canale: non deve fermare
                # l'accorpamento e non va rimandato. Elaborarlo subito
                # tiene aggiornato il riferimento per il riconoscimento
                # dell'eco, che altrimenti scadeva proprio mentre noi
                # accorpavamo.
                self._local.popleft()
                try:
                    self._accept(successiva)
                except Exception:
                    log.debug("Avviso di silenzio non elaborato", exc_info=True)
                continue
            if successiva.speaker != chunk.speaker:
                break

            pausa = successiva.offset - fine
            durata_totale = (
                successiva.offset + successiva.samples.size / rate - chunk.offset
            )
            # Il SEGUITO di una frase troncata per durata massima e' lo
            # stesso parlato che continua: e' il candidato ad accorpare
            # per eccellenza. Prima veniva escluso — e comunque la sua
            # sovrapposizione di pre-roll faceva scattare il limite sulla
            # pausa negativa — cosi' una risposta di quaranta secondi
            # restava spezzata in quattro chiamate da dieci: proprio sul
            # caso piu' comune di un colloquio, il candidato che
            # racconta, l'accorpamento era spento. La sovrapposizione
            # viene tolta dal ramo dei campioni doppi qui sotto.
            if successiva.continues_previous:
                if pausa < -(MERGE_CONTINUATION_OVERLAP + 0.1):
                    break
            elif pausa < -0.05 or pausa > pausa_massima:
                break
            if durata_totale > MERGE_MAX_SECONDS:
                break

            self._local.popleft()
            campioni = successiva.samples
            if pausa > 0:
                riempimento = int(round(pausa * rate))
                parti.append(np.zeros(riempimento, dtype=np.float32))
                silenzio_aggiunto += riempimento / rate
            elif pausa < 0:
                # Le due frasi si sovrappongono: i campioni comuni vanno
                # tolti, non incollati due volte. Concatenandoli si
                # allungava la linea temporale a ogni accorpamento, e
                # dopo venti giunzioni la finestra usata per cercare
                # l'eco slittava di piu' del ritardo massimo cercabile.
                doppi = min(int(round(-pausa * rate)), campioni.size)
                campioni = campioni[doppi:]
                if campioni.size == 0:
                    fine = successiva.offset + successiva.samples.size / rate
                    uniti += 1
                    continue
            parti.append(campioni)
            fine = successiva.offset + successiva.samples.size / rate
            uniti += 1

        self._update_parked(audio_queue)
        if not uniti:
            return chunk

        self.merged_utterances += uniti
        return AudioChunk(
            chunk.speaker,
            np.concatenate(parti),
            chunk.offset,
            chunk.wall_time,
            chunk.continues_previous,
            chunk.padding_seconds + silenzio_aggiunto,
        )

    # ------------------------------------------------------------------
    # Smistamento dei blocchi
    # ------------------------------------------------------------------
    def _accept(self, chunk: AudioChunk) -> None:
        if chunk.speaker == config.SPEAKER_CANDIDATE:
            # Blocco vuoto: non e' audio, e' l'avviso che il canale del
            # candidato e' stato analizzato fino a quell'istante e taceva.
            if chunk.samples.size == 0:
                if self.echo.enabled:
                    self._reference.note_silence(chunk.offset)
                    self._flush_waiting()
                return

            # L'audio della videochiamata e' anche il riferimento con cui
            # riconoscere l'eco nel microfono.
            if self.echo.enabled:
                self._reference.append(chunk.offset, chunk.samples)
            self._transcribe(chunk)
            self._flush_waiting()
            return

        if not self.echo.enabled or not self._reference.active:
            self._transcribe(chunk)
            return

        # Il blocco corrispondente dell'altro canale puo' non essere
        # ancora arrivato: mettiamo il microfono in attesa per un istante
        # invece di rinunciare al confronto.
        self._waiting.append((chunk, time.monotonic()))
        self._flush_waiting()

    def _flush_waiting(self, force: bool = False) -> None:
        while self._waiting:
            chunk, arrived = self._waiting[0]
            duration = chunk.samples.size / config.AUDIO_SAMPLE_RATE
            # Serve il riferimento fino a poco oltre la fine della frase:
            # il ritardo dell'eco sposta indietro il riferimento, non in
            # avanti, quindi non occorre attendere altri mezzi secondi.
            needed = chunk.offset + duration + echo_module.FORWARD_MARGIN_SECONDS
            waited = time.monotonic() - arrived

            if not force and not self._reference.covers(needed) and waited < REFERENCE_WAIT_SECONDS:
                return

            self._waiting.popleft()
            reference = self._reference.segment(chunk.offset, duration)
            samples = chunk.samples
            if reference is not None:
                reference_samples, pre_samples = reference
                result = self.echo.process(
                    chunk.samples, reference_samples, pre_samples
                )
                if result.is_echo:
                    self.echo_dropped += 1
                    # La frase scartata interrompe la catena: senza
                    # questo azzeramento, la frase successiva marcata
                    # come "seguito" veniva confrontata con un testo
                    # vecchio di parecchi secondi e ne perdeva l'inizio.
                    self._forget_last(chunk.speaker)
                    log.debug(
                        "Blocco microfono scartato come eco "
                        "(somiglianza %.2f, ritardo %.0f ms)",
                        result.correlation,
                        result.delay_seconds * 1000,
                    )
                    continue
                samples = result.samples

            self._transcribe(
                AudioChunk(
                    chunk.speaker,
                    samples,
                    chunk.offset,
                    chunk.wall_time,
                    chunk.continues_previous,
                    chunk.padding_seconds,
                )
            )

    # ------------------------------------------------------------------
    # Trascrizione vera e propria
    # ------------------------------------------------------------------
    def _transcribe(self, chunk: AudioChunk) -> None:
        if chunk.samples.size == 0 or self._model is None:
            return

        # Il trattamento del segnale viene PRIMA del controllo di
        # silenzio. Nell'ordine inverso, il guadagno pensato per i
        # microfoni deboli non entrava mai in gioco: quelle frasi
        # venivano scartate dal controllo un istante prima, pur essendo
        # parlato perfettamente comprensibile una volta alzate.
        audio = _normalise_gain(_tame_peaks(np.asarray(chunk.samples, dtype=np.float32)))

        if _loudest_rms(audio) < config.SILENCE_RMS_THRESHOLD:
            # Va dimenticato anche l'ultimo testo di questo interlocutore,
            # come fa ogni altro scarto. Era l'unico ramo che se ne
            # dimenticava: la frase seguente, marcata come continuazione,
            # veniva confrontata con un testo vecchio di secondi e
            # perdeva le proprie parole iniziali.
            self._forget_last(chunk.speaker)
            return

        duration = audio.size / config.AUDIO_SAMPLE_RATE
        # Tempo di parlato vero, senza il silenzio reinserito dagli
        # accorpamenti: e' questo che deve decidere se la frase e'
        # abbastanza lunga da poter votare sulla lingua, e su questo si
        # misura la velocita' dichiarata all'utente.
        parlato = max(0.0, duration - chunk.padding_seconds)

        # Ogni tanto lasciamo che il modello ridica la sua sulla lingua:
        # se il colloquio prosegue davvero in un'altra lingua dobbiamo
        # accorgercene, invece di insistere per un'ora su quella
        # riconosciuta male nelle prime due battute.
        sonda = self._should_probe_language(parlato)
        lingua_chiamata = None if sonda else self._locked_language

        fascio, temperature, ipotesi = self._decoding_quality()

        started = time.monotonic()
        segments, info = self._model.transcribe(
            audio,
            # La finestra di analisi. Il riconoscimento riempie SEMPRE la
            # finestra di silenzio: con quella intera da trenta secondi,
            # una frase di quattro paga l'analisi di ventisei secondi di
            # nulla — ed e' l'analisi, non la scrittura del testo, a
            # dominare il costo su un computer lento. Quando il computer
            # e' in affanno conclamato la finestra viene ritagliata
            # sull'audio vero: due-tre volte meno lavoro a chiamata.
            # ATTENZIONE a chi tocca questo codice: la libreria RICORDA
            # l'ultimo valore ricevuto, quindi va passato SEMPRE, anche
            # quando e' quello normale — ometterlo non significa "usa il
            # valore normale" ma "usa l'ultimo che ti ho detto".
            chunk_length=self._encoder_window(duration),
            language=lingua_chiamata,
            # Ampiezza della ricerca e scala dei tentativi decise in base
            # a quanto il computer sta al passo: vedi _decoding_quality.
            beam_size=fascio,
            best_of=ipotesi,
            temperature=temperature,
            # Il silenzio l'abbiamo gia' tolto noi con il rilevatore di
            # voce: rifarlo qui costerebbe tempo per nulla.
            vad_filter=False,
            # Ogni frase e' indipendente: incatenarle fa propagare gli
            # errori di riconoscimento da una frase alla successiva.
            condition_on_previous_text=False,
            # Difese contro le frasi inventate sul rumore di fondo.
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            # I marcatori temporali non vengono usati da nessuna parte
            # in questo programma: farli produrre al decoder costa token
            # a ogni frase ed e' una fonte nota di righe inventate sul
            # rumore, dove il modello entra in un ciclo di soli tempi.
            without_timestamps=True,
            # Il suggerimento dipende dalla lingua del COLLOQUIO, non da
            # quella passata alla chiamata. Durante una sonda passiamo
            # None per lasciar decidere il modello, ma il riconoscimento
            # della lingua usa solo l'encoder: togliere anche il
            # suggerimento peggiorava la trascrizione senza influenzare
            # in alcun modo la scelta della lingua.
            initial_prompt=self._prompt_for_call(self._locked_language),
        )
        parts = [seg.text.strip() for seg in segments]

        elapsed = time.monotonic() - started
        self._record_speed(parlato, elapsed)

        text = " ".join(part for part in parts if part).strip()
        if not text:
            self._forget_last(chunk.speaker)
            return

        # Difesa contro un guasto noto: su una frase breve o disturbata
        # il modello, invece di trascrivere, restituisce il suggerimento
        # di contesto che gli abbiamo dato. Il risultato e' una riga che
        # nessuno ha pronunciato, ripetuta a ogni pausa.
        if _echoes_prompt(text):
            log.debug("Scartata una ripetizione del suggerimento di contesto")
            self._forget_last(chunk.speaker)
            return

        detected = getattr(info, "language", "") or ""
        self._consider_language(
            detected,
            float(getattr(info, "language_probability", 0.0) or 0.0),
            parlato,
            len(text.split()),
            # Solo i verdetti espressi liberamente contano: se la lingua
            # gliel'abbiamo passata noi, il riconoscimento ce la ripete.
            libera=lingua_chiamata is None,
        )

        with self._lock:
            # La ripetizione di parole esiste solo quando una frase e'
            # stata troncata per durata massima e prosegue nella
            # successiva. Fuori da quel caso, due frasi sono separate da
            # una pausa reale e una parola ripetuta e' voluta: toglierla
            # cancellerebbe testo legittimo.
            previous = (
                self._last_text.get(chunk.speaker, "")
                if chunk.continues_previous
                else ""
            )
            text = _strip_overlap(previous, text)
            if not text:
                self._last_text.pop(chunk.speaker, None)
                return

            # Ultima rete di sicurezza contro l'eco: la stessa frase
            # comparsa poco fa sull'altro canale e' una ripetizione, non
            # un intervento nuovo.
            if self.echo.enabled and chunk.speaker == config.SPEAKER_RECRUITER:
                if self._is_echo_of_candidate(text, chunk.offset):
                    self.duplicates_dropped += 1
                    self.echo.detections += 1
                    # La catena si interrompe qui: il testo appena
                    # scartato non deve restare come termine di
                    # confronto per la frase successiva.
                    self._last_text.pop(chunk.speaker, None)
                    log.debug("Frase scartata perche' duplicata dall'altro canale")
                    return

            self._last_text[chunk.speaker] = text

            if detected:
                self._languages[detected] += 1

            parole = len(text.split())
            self._stats["segments"] += 1
            if chunk.speaker == config.SPEAKER_RECRUITER:
                self._stats["recruiter_words"] += parole
                if "?" in text:
                    self._stats["questions"] += 1
            else:
                self._stats["candidate_words"] += parole

            segment = TranscriptSegment(
                speaker=chunk.speaker,
                text=text,
                language=detected,
                offset_seconds=max(0.0, chunk.offset),
                timestamp=chunk.wall_time,
            )
            self.segments.append(segment)

        # La consegna all'interfaccia non deve mai poter uccidere il
        # thread. Fra il controllo e la chiamata la sessione puo' essere
        # stata distrutta — e' proprio quello che succedeva chiudendo il
        # colloquio mentre la trascrizione era ancora indietro: partiva
        # un "Internal C++ object already deleted" che faceva morire il
        # thread, e con lui tutte le frasi ancora da trascrivere.
        consegna = self.on_segment
        if consegna is not None:
            try:
                consegna(segment)
            except RuntimeError:
                log.info(
                    "Interfaccia non piu' disponibile: proseguo senza mostrare "
                    "le frasi a schermo"
                )
                self.detach_callbacks()
            except Exception:
                log.exception("Consegna di una frase all'interfaccia non riuscita")

    # ------------------------------------------------------------------
    def _forget_last(self, speaker: str) -> None:
        """
        Dimentica l'ultima frase di un interlocutore.

        Da chiamare ogni volta che un blocco viene scartato: la
        rimozione delle parole ripetute confronta la frase nuova con la
        precedente, e se quella precedente non e' mai stata pubblicata
        il confronto avviene con un testo vecchio di secondi. Il
        risultato era la sparizione dell'inizio della frase successiva.
        """
        with self._lock:
            self._last_text.pop(speaker, None)

    def _encoder_window(self, duration: float) -> int:
        """
        Secondi di finestra da chiedere all'analisi per questa chiamata.

        Il valore addestrato e' trenta, e trenta resta finche' il
        computer regge: accorciare la finestra e' una rinuncia di
        qualita' piccola ma non nulla, e non va fatta gratis. Quando
        pero' il computer e' gia' dovuto scendere di modello, o il costo
        di una chiamata supera comunque il ritmo del dialogo, la
        priorita' e' una sola: stare al passo. La finestra viene allora
        ritagliata sull'audio vero, con un margine, mai sotto i dieci
        secondi.
        """
        in_affanno = self._downgraded > 0 or (
            self._call_cost > 0
            and self._arrival_gap > 0
            and self._call_cost > self._arrival_gap
        )
        if not in_affanno:
            return ENCODER_WINDOW_FULL
        finestra = int(np.ceil(duration)) + 2
        return int(min(ENCODER_WINDOW_FULL, max(ENCODER_WINDOW_MIN, finestra)))

    def _decoding_quality(self) -> tuple[int, tuple[float, ...], int]:
        """
        Quanta cura mettere nella trascrizione del prossimo blocco.

        Il tempo risparmiato con le correzioni sull'arretrato va speso
        in precisione, non lasciato sul tavolo — ma solo finche' il
        computer regge. La scala dei tentativi a temperatura crescente
        resta SEMPRE attiva, perche' costa qualcosa unicamente quando il
        primo tentativo e' venuto male; a variare e' solo quanto in
        profondita' si cerca.
        """
        # Il margine NON si misura piu' con realtime_factor. Da quando
        # l'attesa regola quanto audio entra in ogni chiamata, quel
        # rapporto tende a fissarsi sul fattore di margine scelto — circa
        # 1,6 su qualunque computer — e finisce nella fascia morta fra le
        # due soglie: la qualita' non sarebbe mai piu' salita, nemmeno
        # sulle macchine che potevano permetterselo. Confrontiamo invece
        # due grandezze che il programma non controlla: quanto costa una
        # chiamata e ogni quanto vengono prodotte le frasi.
        arretrato = self._durata_locale()
        if self._call_cost > 0 and self._arrival_gap > 0:
            margine = self._arrival_gap / self._call_cost
        else:
            margine = 0.0

        if arretrato >= BACKLOG_AGGRESSIVE_SECONDS or (
            margine and margine < config.SPEED_LOWER_QUALITY
        ):
            self._quality_level = 0
        elif self._speed_samples >= 3 and margine >= config.SPEED_RAISE_QUALITY:
            self._quality_level = min(2, self._quality_level + 1)

        # Il terzo valore e' il numero di ipotesi generate quando scatta
        # un tentativo a temperatura piu' alta. Restava fisso a cinque
        # anche al livello piu' basso, cioe' proprio sul computer che non
        # sta al passo: in quel caso il ripiego, che sulle frasi brevi e
        # disturbate di un colloquio scatta spesso, costava cinque
        # decodifiche invece di una. Al livello minimo se ne genera una
        # sola: il tentativo di recupero resta, il conto no.
        if self._quality_level >= 2:
            return (config.DECODE_BEAM_MAX, config.DECODE_TEMPERATURES_FULL,
                    config.DECODE_BEST_OF_MAX)
        if self._quality_level == 1:
            return (config.DECODE_BEAM_MID, config.DECODE_TEMPERATURES_FULL,
                    config.DECODE_BEST_OF_MAX)
        return (config.DECODE_BEAM_MIN, config.DECODE_TEMPERATURES_FAST,
                config.DECODE_BEST_OF_MIN)

    def _prompt_for_call(self, language: Optional[str]) -> Optional[str]:
        """
        Suggerimento di contesto nella lingua del colloquio.

        Va dato solo quando la lingua e' nota e solo nella lingua
        parlata: un suggerimento scritto in un'altra lingua spinge il
        modello verso quella sbagliata. Nessuna lingua e' privilegiata —
        chi non ha un suggerimento in tabella semplicemente non lo
        riceve, che e' la scelta neutra e senza effetti collaterali.
        """
        if not language:
            return None
        return config.TRANSCRIPTION_PROMPTS.get(language)

    def _should_probe_language(self, duration: float) -> bool:
        """Vero quando conviene rifare il riconoscimento della lingua."""
        if self._locked_language is None or self._language_forced:
            return False
        # Una sonda si paga: quella frase viene decodificata nella lingua
        # che il modello indovina, e su un frammento di due secondi
        # indovina malissimo — nel log di un colloquio italiano sono
        # uscite frasi in giapponese proprio cosi'. Si sonda quindi solo
        # su frasi abbastanza lunghe da meritare fiducia; sulle altre si
        # resta sulla lingua gia' nota, che e' sempre la scelta migliore.
        if duration < LANGUAGE_PROBE_MIN_SECONDS:
            return False
        if self._probe_countdown > 0:
            self._probe_countdown -= 1
            return False
        # Finche' la lingua e' solo un'ipotesi si controlla molto piu'
        # spesso: e' quell'ipotesi a guidare la decodifica, quindi se e'
        # sbagliata va corretta subito.
        self._probe_countdown = (
            LANGUAGE_PROBE_EVERY if self._language_confirmed
            else LANGUAGE_PROBE_UNCONFIRMED
        )
        return True

    def _vote_is_trustworthy(
        self, detected: str, probability: float, duration: float, words: int
    ) -> bool:
        """
        Una frase puo' decidere la lingua solo se e' lunga e sicura.

        Le prime battute di un colloquio sono "Buongiorno", "Mi sente?",
        "Perfetto": proprio quelle su cui il riconoscimento sbaglia piu'
        spesso. Bastavano due errori concordi per fissare la lingua
        sbagliata e rendere illeggibile tutto il resto del colloquio.
        """
        return bool(
            detected
            and duration >= config.LANGUAGE_VOTE_MIN_SECONDS
            and words >= config.LANGUAGE_VOTE_MIN_WORDS
            and probability >= config.LANGUAGE_VOTE_MIN_PROBABILITY
        )

    def _consider_language(
        self,
        detected: str,
        probability: float,
        duration: float,
        words: int,
        libera: bool = False,
    ) -> None:
        """
        `libera` = alla chiamata NON era stata imposta alcuna lingua.

        E' la condizione senza la quale nulla di tutto questo ha senso.
        Quando si impone una lingua, il riconoscimento la restituisce
        identica con probabilita' 1: contarla come un voto significava
        farsi dare ragione da se stessi, e una prima ipotesi sbagliata si
        sarebbe autoconfermata in quattro frasi senza scampo.
        """
        if self._language_forced or not detected or not libera:
            return

        # ADOTTARE un'ipotesi provvisoria costa poco e rende molto: senza
        # una lingua, ogni chiamata decodifica in quella indovinata a
        # caso sul momento. Le soglie severe qui sotto pretendono cinque
        # parole di testo sensato — ma il testo sensato arriva solo SE la
        # lingua e' giusta: un circolo vizioso in cui l'ipotesi non si
        # agganciava mai e tutto il colloquio usciva in lingue a caso.
        # Per l'adozione basta quindi un verdetto appena decente: viene
        # comunque rimesso in discussione ogni due frasi lunghe.
        if (
            self._locked_language is None
            and duration >= config.LANGUAGE_VOTE_MIN_SECONDS
            and probability >= LANGUAGE_ADOPT_PROBABILITY
        ):
            self._locked_language = detected
            self._probe_countdown = 0
            log.info("Lingua provvisoria del colloquio: '%s'", detected)

        if not self._vote_is_trustworthy(detected, probability, duration, words):
            return

        if not self._language_confirmed:
            self._language_votes[detected] += 1
            classifica = self._language_votes.most_common(2)
            lingua, voti = classifica[0]
            seconda = classifica[1][1] if len(classifica) > 1 else 0

            # Ipotesi provvisoria, presa gia' al primo verdetto
            # affidabile. E' la correzione piu' importante di tutte:
            # finche' non c'era alcuna lingua, OGNI chiamata lasciava il
            # modello libero di indovinarla da capo — su frasi di due
            # secondi. Nei log di un colloquio italiano si leggevano di
            # seguito francese, francese, russo, inglese e giapponese: e
            # una frase decodificata nella lingua sbagliata non e'
            # imprecisa, e' incomprensibile. Da qui in avanti si decodifica
            # sempre nella lingua piu' probabile finora, e la si continua
            # a mettere in discussione con le sonde.
            if lingua != self._locked_language:
                if self._locked_language is None:
                    log.info("Lingua provvisoria del colloquio: '%s'", lingua)
                else:
                    log.info(
                        "Lingua provvisoria corretta da '%s' a '%s'",
                        self._locked_language, lingua,
                    )
                self._locked_language = lingua
                self._probe_countdown = 0     # riverifica alla prossima frase

            if (
                voti >= config.LANGUAGE_LOCK_VOTES
                and voti - seconda >= config.LANGUAGE_LOCK_MARGIN
            ):
                self._language_confirmed = True
                log.info("Lingua del colloquio fissata su '%s'", lingua)
                self._notify_status(f"Lingua riconosciuta: {lingua}")
            return

        # I voti di sblocco devono essere CONSECUTIVI. Accumulandoli
        # senza mai farli decadere, tre sonde sbagliate anche distanti
        # mezz'ora l'una dall'altra ribaltavano la lingua di un
        # colloquio che nel frattempo era stato confermato decine di
        # volte: da quel momento tutto il resto usciva illeggibile.
        if detected == self._locked_language or probability < (
            config.LANGUAGE_UNLOCK_PROBABILITY
        ):
            self._unlock_candidate = None
            self._unlock_streak = 0
            return

        if detected != self._unlock_candidate:
            self._unlock_candidate = detected
            self._unlock_streak = 0
        self._unlock_streak += 1

        if self._unlock_streak >= config.LANGUAGE_UNLOCK_VOTES:
            precedente = self._locked_language
            self._locked_language = detected
            self._unlock_candidate = None
            self._unlock_streak = 0
            self._language_votes.clear()
            self._language_votes[detected] = config.LANGUAGE_LOCK_VOTES
            log.info("Lingua del colloquio corretta da '%s' a '%s'", precedente, detected)
            self._notify_status(f"Lingua riconosciuta: {detected}")

    def _record_speed(self, audio_seconds: float, elapsed: float) -> None:
        if audio_seconds <= 0 or elapsed <= 0:
            return
        # Costo medio di UNA chiamata, indipendente da quanto audio
        # conteneva: e' il numero su cui si decide quanto accorpare.
        #
        # La primissima chiamata non fa testo: comprende il riscaldamento
        # delle librerie di calcolo e puo' costare il doppio delle
        # successive. Prendendola come valore iniziale della media, le
        # prime attese risultavano sovrastimate per diversi minuti.
        if self._speed_samples == 0:
            pass
        elif self._call_cost <= 0:
            self._call_cost = elapsed
        else:
            self._call_cost = 0.75 * self._call_cost + 0.25 * elapsed
        factor = audio_seconds / elapsed
        # Media mobile: un singolo intervento anomalo non deve far
        # sembrare il programma piu' lento o piu' veloce di quanto sia.
        if self._speed_samples == 0:
            self.realtime_factor = factor
        else:
            self.realtime_factor = 0.8 * self.realtime_factor + 0.2 * factor
        self._speed_samples += 1

        # Corsia d'emergenza. La valutazione ordinaria aspetta otto
        # chiamate: sul computer misurato sul campo — dieci secondi a
        # chiamata — significava decidere DOPO la fine di un colloquio di
        # novanta secondi, a parlato ormai buttato. Ma quando una singola
        # chiamata costa due volte e mezzo l'audio che contiene non c'e'
        # niente da aspettare: la seconda conferma basta, e si scende
        # subito.
        if (
            self._downgraded < MAX_DOWNGRADES
            and self._speed_samples >= FAST_OVERLOAD_CALLS
            and elapsed > FAST_OVERLOAD_FACTOR * max(audio_seconds, 1.0)
        ):
            self._consider_lighter_model(emergenza=True)
            return

        self._consider_lighter_model()

    # ------------------------------------------------------------------
    @property
    def load(self) -> float:
        """
        Quanto e' occupato il processore, da 0 a oltre 1.

        E' il rapporto fra il costo di una chiamata e l'intervallo con
        cui arrivano le frasi. Sopra 1 si consuma piu' tempo di quanto ne
        passi: il ritardo dei sottotitoli cresce e non si riassorbe piu'.
        """
        if self._call_cost <= 0 or self._arrival_gap <= 0:
            return 0.0
        return self._call_cost / self._arrival_gap

    def _consider_lighter_model(self, emergenza: bool = False) -> None:
        """
        Passa a un modello piu' leggero se questo non sta al passo.

        La via ordinaria decide su misure abbondanti, per non reagire a
        una raffica passeggera. La via d'emergenza — chiamate che costano
        piu' del doppio dell'audio che contengono — decide su due
        conferme, perche' aspettarne otto significava decidere a
        colloquio finito. Se e' stato l'utente a scegliere il modello,
        non si tocca nulla: si avvisa, una volta sola e con un consiglio
        concreto.
        """
        if self._downgraded >= MAX_DOWNGRADES:
            return
        if not emergenza:
            if self._speed_samples < OVERLOAD_MIN_CALLS:
                return
            if self.load <= OVERLOAD_RATIO:
                return

        try:
            posizione = MODEL_LADDER.index(self.model_size)
        except ValueError:
            return

        from app.models.download import whisper_model_present

        # Primo modello PRESENTE SUL DISCO scendendo la scala. Prima ci
        # si fermava al gradino immediatamente sotto: se non era stato
        # scaricato — e il programma scarica solo il modello selezionato —
        # si rinunciava, dopo aver gia' speso il budget dei cambi. Il
        # risultato era un meccanismo che sulla macchina che ne aveva
        # piu' bisogno non poteva mai funzionare.
        piu_leggero = next(
            (m for m in MODEL_LADDER[posizione + 1:] if whisper_model_present(m)),
            None,
        )
        consiglio = piu_leggero or (
            MODEL_LADDER[posizione + 1] if posizione + 1 < len(MODEL_LADDER) else ""
        )
        if not consiglio:
            return

        from app import settings

        # In "user_choices" stanno i NOMI delle impostazioni scelte a
        # mano, non i loro valori: confrontarci la dimensione del
        # modello dava sempre falso, e la scelta dell'utente sarebbe
        # stata scavalcata in silenzio.
        if "whisper_model_size" in (settings.get("user_choices") or ()):
            self._downgraded = MAX_DOWNGRADES     # una volta sola, e basta
            self.suggested_model = consiglio
            # Il tempo si cita solo se e' un numero che significa
            # qualcosa detto ad alta voce: "circa 0 secondi" farebbe
            # sembrare rotto l'avviso invece del computer.
            quanto = (
                f"impiega circa {self._call_cost:.0f} secondi per ogni frase"
                if self._call_cost >= 1.5
                else "non riesce a stare al passo del parlato"
            )
            self._notify_status(
                f"Il modello '{self.model_size}' {quanto} su questo computer: "
                f"i sottotitoli restano sempre piu' indietro. Lo hai scelto "
                f"tu, quindi non lo cambio — ma dalle impostazioni conviene "
                f"passare a '{consiglio}'."
            )
            return

        if piu_leggero is None:
            # Nessun modello di ripiego sul disco: si avvisa senza
            # bruciare il budget dei cambi — se un modello leggero
            # comparira', il meccanismo deve poter ancora agire.
            self.suggested_model = consiglio
            self._notify_status(
                f"Il modello '{self.model_size}' non sta al passo su questo "
                f"computer e il modello '{consiglio}' non e' ancora "
                "scaricato. Al termine del colloquio selezionalo dalle "
                "impostazioni: verra' scaricato e usato dal prossimo "
                "colloquio."
            )
            self._downgraded = MAX_DOWNGRADES     # inutile riprovare ora
            return

        log.warning(
            "%s con il modello '%s': passo a '%s'",
            "Emergenza" if emergenza else f"Carico {self.load:.2f}",
            self.model_size, piu_leggero,
        )
        self._notify_status(
            f"Il modello '{self.model_size}' non riesce a stare al passo: "
            f"passo a '{piu_leggero}' per non far accumulare ritardo ai "
            "sottotitoli."
        )
        try:
            self._swap_model(piu_leggero)
        except Exception:
            log.exception("Cambio del modello non riuscito: proseguo con quello attuale")
            return
        # Il budget si spende solo quando il cambio avviene davvero.
        self._downgraded += 1

    def _swap_model(self, nuovo: str) -> None:
        """
        Sostituisce il modello in corsa.

        Avviene fra una chiamata e l'altra, nello stesso thread che le
        esegue: nessuno sta usando il modello in questo istante. Il
        vecchio viene lasciato andare PRIMA di caricare il nuovo, per non
        tenere in memoria entrambi su un computer che e' gia' in
        difficolta'.
        """
        from faster_whisper import WhisperModel

        from app.models.download import whisper_model_dir

        vecchio = self.model_size
        self._model = None
        import gc

        gc.collect()

        def _carica(size: str):
            return WhisperModel(
                str(whisper_model_dir(size)),
                device="cpu",
                compute_type=config.WHISPER_COMPUTE_TYPE,
                cpu_threads=config.transcription_threads(),
                num_workers=1,
            )

        try:
            self._model = _carica(nuovo)
            self.model_size = nuovo
        except Exception:
            # Il vecchio modello e' gia' stato lasciato andare: senza
            # questo ripristino il motore restava SENZA ALCUN modello e
            # scartava in silenzio ogni frase fino a fine colloquio, con
            # i log che per giunta dichiaravano il nome del modello
            # nuovo. Meglio lento che muto.
            log.exception(
                "Caricamento di '%s' fallito: ricarico '%s'", nuovo, vecchio
            )
            self._model = _carica(vecchio)
            self.model_size = vecchio
            raise

        self.model_changed_to = nuovo
        # Le misure precedenti riguardavano un altro modello.
        self._call_cost = 0.0
        self._speed_samples = 0
        self.realtime_factor = 0.0
        log.info("Modello di trascrizione sostituito con '%s'", nuovo)

    def _is_echo_of_candidate(self, text: str, offset: float) -> bool:
        """Da chiamare tenendo il lock."""
        for segment in reversed(self.segments):
            if offset - segment.offset_seconds > DUPLICATE_WINDOW_SECONDS:
                break
            if segment.speaker != config.SPEAKER_CANDIDATE:
                continue
            if echo_module.texts_are_duplicate(segment.text, text):
                return True
        return False

    # ------------------------------------------------------------------
    @property
    def detected_language(self) -> str:
        """
        Lingua prevalente del colloquio.

        Non ci si puo' basare sul primo blocco riconosciuto: spesso e'
        un "mmm" o un "ok" che Whisper attribuisce a una lingua a caso.
        """
        with self._lock:
            if not self._languages:
                return ""
            return self._languages.most_common(1)[0][0]

    @property
    def speakers_detected(self) -> bool:
        """True quando i dati indicano l'uso degli altoparlanti, non delle cuffie."""
        return self.echo.speakers_detected

    def transcript_lines(self, labels: dict[str, str]) -> list[str]:
        with self._lock:
            return [
                f"{labels.get(s.speaker, s.speaker)}: {s.text}" for s in self.segments
            ]

    def full_transcript(self, labels: dict[str, str]) -> str:
        return "\n".join(self.transcript_lines(labels))

    def segments_as_dicts(self) -> list[dict]:
        with self._lock:
            return [s.to_dict() for s in self.segments]

    def statistics(self) -> dict:
        """
        Numeri mostrati accanto alle note, aggiornati man mano.

        Vengono letti una volta al secondo dal thread grafico: e' quindi
        essenziale che il costo non cresca con la durata del colloquio.
        """
        with self._lock:
            recruiter = self._stats["recruiter_words"]
            candidate = self._stats["candidate_words"]
            totale = recruiter + candidate
            return {
                "questions": self._stats["questions"],
                "recruiter_words": recruiter,
                "candidate_words": candidate,
                "candidate_share": round(100 * candidate / totale) if totale else 0,
                "segments": self._stats["segments"],
            }
