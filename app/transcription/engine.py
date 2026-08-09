"""
Motore di trascrizione in tempo reale.

Basato su faster-whisper (Whisper eseguito da CTranslate2): gira in
locale, senza connessione e senza costi, e riconosce automaticamente
circa 99 lingue, quindi il colloquio puo' svolgersi in italiano,
inglese o qualunque altra lingua senza configurare nulla.

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
import queue
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

# Quando la coda cresce, l'unica cosa che conta e' tornare al presente:
# si accorpa fino al limite di durata ignorando la lunghezza delle
# pause, dimezzando il numero di chiamate al riconoscimento vocale.
BACKLOG_AGGRESSIVE_MERGE = 3
BACKLOG_GAP_SECONDS = 6.0

# Ogni quante frasi lunghe si lascia il modello libero di ridire la sua
# sulla lingua, per potersi ricredere su un blocco iniziale sbagliato.
LANGUAGE_PROBE_EVERY = 8

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
    modello = " ".join(_normalise(w) for w in config.TRANSCRIPTION_PROMPT.split())
    if not pulito:
        return False
    return pulito == modello or (len(pulito) > 20 and pulito in modello)


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
        # sulle frasi brevi, sbaglia spesso attribuendo parole italiane
        # a un'altra lingua.
        self._locked_language: Optional[str] = (
            None if language == "auto" else language
        )
        # Quando la lingua e' stata scelta dall'utente non deve mai
        # cambiare; quando l'abbiamo dedotta noi, invece, restiamo
        # disposti a ricrederci.
        self._language_forced = language != "auto"
        self._language_votes: Counter[str] = Counter()
        # Sblocco: servono voti CONSECUTIVI sulla stessa lingua nuova.
        self._unlock_candidate: Optional[str] = None
        self._unlock_streak = 0
        self._probe_countdown = 0

        # Rapporto fra durata dell'audio e tempo impiegato a trascriverlo.
        self.realtime_factor = 0.0
        self._speed_samples = 0
        # 0 = minimo indispensabile, 1 = ricerca media, 2 = ricerca ampia.
        # Si parte dal livello medio: sui computer che reggono resta li'
        # o sale, su quelli lenti scende dopo le prime frasi.
        self._quality_level = 1

        self.segments: list[TranscriptSegment] = []
        self.backlog = 0

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
        log.info(
            "Modello '%s' caricato in %.1f s",
            self.model_size,
            time.monotonic() - started,
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
            return False

        self._thread = None
        return True

    # ------------------------------------------------------------------
    def _notify_status(self, message: str) -> None:
        if self.on_status:
            try:
                self.on_status(message)
            except Exception:
                log.debug("Notifica di stato fallita", exc_info=True)

    def _run(self, audio_queue: "queue.Queue[AudioChunk]") -> None:
        try:
            if self._model is None:
                self.load_model()
            self._notify_status("In ascolto")
        except Exception as exc:
            log.exception("Caricamento del modello non riuscito")
            if self.on_error:
                self.on_error(exc)
            # Svuotiamo comunque la coda: senza consumatore i thread
            # audio la riempirebbero fino a scartare blocchi a vuoto.
            self._drain(audio_queue)
            return

        while True:
            try:
                chunk = self._next_chunk(audio_queue, timeout=0.4)
            except queue.Empty:
                self.backlog = len(self._waiting) + len(self._local)
                try:
                    self._flush_waiting(force=self._stop.is_set())
                except Exception as exc:
                    log.exception("Errore nello smaltimento delle frasi in attesa")
                    if self.on_error:
                        self.on_error(exc)
                    self._waiting.clear()
                self._update_parked(audio_queue)
                if self._stop.is_set() and not self._waiting:
                    break
                continue

            self.backlog = audio_queue.qsize() + len(self._local) + len(self._waiting)
            try:
                self._accept(self._merge_following(audio_queue, chunk))
            except Exception as exc:
                log.exception("Errore durante l'elaborazione di un blocco")
                if self.on_error:
                    self.on_error(exc)
            self._update_parked(audio_queue)

        # Lo svuotamento finale va protetto quanto il resto: e' il
        # momento in cui escono le ultime frasi del colloquio, e un
        # errore qui le faceva sparire tutte in silenzio, per giunta
        # senza mai segnalare la fine della trascrizione.
        try:
            self._flush_waiting(force=True)
        except Exception as exc:
            log.exception("Errore nello svuotamento finale")
            if self.on_error:
                self.on_error(exc)
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
        return audio_queue.get(timeout=timeout)

    def _fill_local(self, audio_queue: "queue.Queue[AudioChunk]") -> None:
        try:
            while len(self._local) < MERGE_LOOKAHEAD:
                self._local.append(audio_queue.get_nowait())
        except queue.Empty:
            pass
        self._update_parked(audio_queue)

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

        self._fill_local(audio_queue)
        if not self._local:
            return chunk

        rate = config.AUDIO_SAMPLE_RATE
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
        in_ritardo = (
            len(self._local) + len(self._waiting) >= BACKLOG_AGGRESSIVE_MERGE
        )
        pausa_massima = BACKLOG_GAP_SECONDS if in_ritardo else MERGE_GAP_SECONDS

        while self._local:
            successiva = self._local[0]
            if (
                successiva.speaker != chunk.speaker
                or successiva.samples.size == 0
                or successiva.continues_previous
            ):
                break

            pausa = successiva.offset - fine
            durata_totale = (
                successiva.offset + successiva.samples.size / rate - chunk.offset
            )
            if pausa < -0.05 or pausa > pausa_massima:
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

        fascio, temperature = self._decoding_quality()

        started = time.monotonic()
        segments, info = self._model.transcribe(
            audio,
            language=lingua_chiamata,
            # Ampiezza della ricerca e scala dei tentativi decise in base
            # a quanto il computer sta al passo: vedi _decoding_quality.
            beam_size=fascio,
            best_of=5,
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
            probe=sonda,
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

        if self.on_segment:
            self.on_segment(segment)

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

    def _decoding_quality(self) -> tuple[int, tuple[float, ...]]:
        """
        Quanta cura mettere nella trascrizione del prossimo blocco.

        Il tempo risparmiato con le correzioni sull'arretrato va speso
        in precisione, non lasciato sul tavolo — ma solo finche' il
        computer regge. La scala dei tentativi a temperatura crescente
        resta SEMPRE attiva, perche' costa qualcosa unicamente quando il
        primo tentativo e' venuto male; a variare e' solo quanto in
        profondita' si cerca.
        """
        arretrato = len(self._local) + len(self._waiting)
        veloce = self.realtime_factor

        if arretrato >= BACKLOG_AGGRESSIVE_MERGE or (
            self._speed_samples and veloce < config.SPEED_LOWER_QUALITY
        ):
            self._quality_level = 0
        elif self._speed_samples >= 3 and veloce >= config.SPEED_RAISE_QUALITY:
            self._quality_level = min(2, self._quality_level + 1)

        if self._quality_level >= 2:
            return config.DECODE_BEAM_MAX, config.DECODE_TEMPERATURES_FULL
        if self._quality_level == 1:
            return config.DECODE_BEAM_MID, config.DECODE_TEMPERATURES_FULL
        return config.DECODE_BEAM_MIN, config.DECODE_TEMPERATURES_FAST

    def _prompt_for_call(self, language: Optional[str]) -> Optional[str]:
        """
        Il suggerimento di contesto va dato solo nella sua lingua.

        Passarlo sempre significava spingere verso l'italiano anche un
        colloquio in inglese, e — su frasi brevi o disturbate — indurre
        il modello a restituire il suggerimento stesso al posto di cio'
        che aveva sentito, riempiendo la trascrizione di righe che
        nessuno aveva pronunciato.
        """
        if language != config.TRANSCRIPTION_PROMPT_LANGUAGE:
            return None
        return config.TRANSCRIPTION_PROMPT

    def _should_probe_language(self, duration: float) -> bool:
        """Vero quando conviene rifare il riconoscimento della lingua."""
        if self._locked_language is None or self._language_forced:
            return False
        if duration < config.LANGUAGE_VOTE_MIN_SECONDS:
            return False
        if self._probe_countdown > 0:
            self._probe_countdown -= 1
            return False
        self._probe_countdown = LANGUAGE_PROBE_EVERY
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
        probe: bool = False,
    ) -> None:
        if self._language_forced or not detected:
            return
        if not self._vote_is_trustworthy(detected, probability, duration, words):
            return

        if self._locked_language is None:
            self._language_votes[detected] += 1
            classifica = self._language_votes.most_common(2)
            lingua, voti = classifica[0]
            seconda = classifica[1][1] if len(classifica) > 1 else 0
            if (
                voti >= config.LANGUAGE_LOCK_VOTES
                and voti - seconda >= config.LANGUAGE_LOCK_MARGIN
            ):
                self._locked_language = lingua
                log.info("Lingua del colloquio fissata su '%s'", lingua)
                self._notify_status(f"Lingua riconosciuta: {lingua}")
            return

        # Lingua gia' fissata: il verdetto vale solo se e' arrivato da
        # una sonda, cioe' da una chiamata in cui il modello era libero
        # di scegliere. Altrimenti ci confermerebbe soltanto la lingua
        # che gli abbiamo imposto noi.
        if not probe:
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
        factor = audio_seconds / elapsed
        # Media mobile: un singolo intervento anomalo non deve far
        # sembrare il programma piu' lento o piu' veloce di quanto sia.
        if self._speed_samples == 0:
            self.realtime_factor = factor
        else:
            self.realtime_factor = 0.8 * self.realtime_factor + 0.2 * factor
        self._speed_samples += 1

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
