"""
Cattura audio da microfono e da audio di sistema.

Due sorgenti, tenute separate:

  * microfono            -> chi conduce il colloquio ("Tu")
  * audio di sistema     -> l'altra persona in videochiamata ("Candidato")

Tenerle separate, invece di mixarle, permette di attribuire ogni frase
alla persona giusta: e' il motivo per cui la trascrizione si legge come
un dialogo.

L'audio di sistema viene letto tramite i dispositivi WASAPI "loopback"
esposti da PyAudioWPatch, che registrano cio' che esce dagli
altoparlanti: funziona quindi con Teams, Zoom, Google Meet e qualunque
altro programma, senza installare cavi audio virtuali.

Due scelte implementative importanti:

1. La lettura avviene con chiamate bloccanti dentro normali thread
   Python, non con le callback in C di PortAudio. Le callback vengono
   invocate da thread creati in C e, nelle applicazioni compilate, sono
   una causa nota di chiusure improvvise difficili da diagnosticare.
   Con la lettura bloccante ogni errore diventa una normale eccezione
   Python, che possiamo registrare e mostrare.

2. Il ricampionamento a 16 kHz avviene una sola volta per finestra da
   alcuni secondi, non a ogni lettura da 1024 campioni. Convertire
   blocchi minuscoli in modo indipendente introdurrebbe una piccola
   discontinuita' a ogni giunzione (decine di volte al secondo) e un
   errore di arrotondamento che, accumulato, sfaserebbe le due sorgenti
   di diversi secondi nell'arco di un colloquio.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from app import config
from app.audio.vad import VoiceSegmenter

log = logging.getLogger(__name__)

try:  # pragma: no cover - dipende dalla piattaforma
    import pyaudiowpatch as pyaudio

    HAS_LOOPBACK = True
except ImportError:  # pragma: no cover - utile in sviluppo su Linux/Mac
    try:
        import pyaudio  # type: ignore

        HAS_LOOPBACK = False
    except ImportError:
        pyaudio = None  # type: ignore
        HAS_LOOPBACK = False


READ_FRAMES = 1024
MAX_CONSECUTIVE_READ_ERRORS = 60

# L'arretrato si misura in SECONDI DI PARLATO, non in numero di blocchi.
# La versione precedente teneva in coda fino a 96 blocchi: essendo ogni
# blocco una frase intera (fino a diversi secondi), significava
# accumulare cinque-otto minuti di ritardo prima di scartare qualcosa.
# Su un computer lento la coda restava stabilmente piena e i sottotitoli
# arrivavano minuti dopo la voce: era questa, e non il riconoscimento
# vocale in se', la causa principale della lentezza percepita.
MAX_BACKLOG_SECONDS = 25.0
# Spazio che resta comunque alla coda anche quando la trascrizione tiene
# gia' da parte molto audio. Senza questo minimo il limite qui sopra
# poteva risultare irraggiungibile e la coda veniva svuotata per intero.
QUEUE_FLOOR_SECONDS = 10.0
# Limite di sicurezza sul numero di elementi, per non far crescere la
# memoria senza fine se il motore si ferma del tutto.
QUEUE_HARD_LIMIT = 512
PROGRESS_INTERVAL_SECONDS = 0.5  # frequenza degli avvisi di avanzamento
# Ogni quanto si controlla che i dispositivi audio predefiniti siano
# ancora quelli su cui stiamo registrando.
DEVICE_WATCH_SECONDS = 4.0
# Oltre questo numero di elementi in coda gli avvisi di avanzamento non
# servono piu' a nulla e occuperebbero soltanto spazio.
PROGRESS_QUEUE_LIMIT = 48


class AudioError(Exception):
    """Errore di configurazione audio, mostrato all'utente."""


@dataclass
class DeviceInfo:
    index: int
    name: str
    sample_rate: int
    channels: int
    is_loopback: bool = False


@dataclass
class AudioChunk:
    """Frase completa, mono a 16 kHz, gia' attribuita a un interlocutore."""

    speaker: str
    samples: np.ndarray
    # Istante di inizio, in secondi dall'avvio della registrazione, con
    # la stessa origine per entrambi i canali. Usiamo un orologio
    # monotono: quello di sistema puo' essere corretto dalla rete
    # durante il colloquio e produrrebbe durate sbagliate.
    offset: float = 0.0
    wall_time: float = field(default_factory=time.time)
    # Vero quando la frase e' il seguito di una troncata per durata
    # massima: solo in quel caso ha senso togliere le parole ripetute.
    continues_previous: bool = False
    # Silenzio reinserito dall'accorpamento per ricostruire le pause.
    # Va sottratto ogni volta che si ragiona su "quanto parlato c'e'":
    # e' tempo, non voce.
    padding_seconds: float = 0.0


class AudioQueue(queue.Queue):
    """
    Coda che sa quanti secondi di audio contiene.

    Serve a limitare l'arretrato per durata invece che per numero di
    elementi: un blocco puo' valere mezzo secondo o quindici, quindi
    contare gli elementi non dice nulla su quanto ritardo abbia
    accumulato la trascrizione. Il conteggio viene aggiornato dentro
    _put e _get, che la classe base chiama gia' con il proprio lock
    acquisito: non serve alcuna sincronizzazione aggiuntiva.
    """

    def __init__(self, maxsize: int = 0, sample_rate: int = config.AUDIO_SAMPLE_RATE):
        super().__init__(maxsize=maxsize)
        self._sample_rate = float(max(1, sample_rate))
        self._queued_seconds = 0.0
        # Secondi gia' estratti dalla coda ma non ancora trascritti:
        # il motore ne tiene da parte fino a trentadue per poterli
        # accorpare. Senza contarli, il limite sull'arretrato guardava
        # solo la punta dell'iceberg e non scattava mai: l'attesa reale
        # poteva superare i cinque minuti con il contatore quasi a zero.
        self.parked_seconds = 0.0
        # Secondi di parlato buttati via perche' la trascrizione non
        # stava al passo. E' il sensore di sovraccarico piu' onesto che
        # esista: il carico stimato puo' ingannarsi — quando si scarta,
        # il ritmo misurato degli arrivi e' quello dei sopravvissuti e
        # sembra tutto in regola — ma il parlato perso non mente mai.
        self.dropped_seconds = 0.0

    @property
    def pending_seconds(self) -> float:
        """Arretrato complessivo, in coda e in mano al motore."""
        return self._queued_seconds + self.parked_seconds

    @staticmethod
    def _samples(item) -> int:
        data = getattr(item, "samples", None)
        return int(getattr(data, "size", 0) or 0)

    def _put(self, item):
        super()._put(item)
        self._queued_seconds += self._samples(item) / self._sample_rate

    def _get(self):
        item = super()._get()
        self._queued_seconds = max(
            0.0, self._queued_seconds - self._samples(item) / self._sample_rate
        )
        return item

    def trim_to(self, max_seconds: float) -> int:
        """
        Scarta il parlato piu' vecchio finche' l'attesa rientra nel limite.

        Due accortezze, entrambe imparate da guasti veri.

        La prima: gli avvisi di silenzio durano zero secondi per
        definizione, quindi buttarli via non abbrevia l'attesa di un solo
        millisecondo, mentre toglie all'altro canale il riferimento con
        cui riconosce l'eco e alla trascrizione il segno di dove sia
        arrivata l'analisi. Qui restano al loro posto, in testa e
        nell'ordine originale.

        La seconda: si scarta solo finche' il conteggio scende davvero.
        Il ciclo si ferma quando non c'e' piu' parlato da togliere,
        quindi non puo' svuotare la coda a vuoto.
        """
        scartati = 0
        with self.mutex:
            trattenuti = []
            while self.queue and self._queued_seconds > max_seconds:
                elemento = self.queue.popleft()
                durata = self._samples(elemento) / self._sample_rate
                if durata <= 0.0:
                    trattenuti.append(elemento)
                    continue
                self._queued_seconds = max(0.0, self._queued_seconds - durata)
                self.dropped_seconds += durata
                scartati += 1
            for elemento in reversed(trattenuti):
                self.queue.appendleft(elemento)
            if scartati:
                # Chi fosse in attesa di spazio puo' procedere.
                self.not_full.notify(scartati)
                # Gli elementi tolti qui non passeranno mai da get(), e
                # quindi nessuno chiamera' task_done() per loro: senza
                # questo aggiustamento un eventuale join() sulla coda non
                # tornerebbe mai piu'.
                self.unfinished_tasks -= scartati
                if self.unfinished_tasks <= 0:
                    self.unfinished_tasks = 0
                    self.all_tasks_done.notify_all()
        return scartati


# --------------------------------------------------------------------------
# Conversione del formato
# --------------------------------------------------------------------------
class ChannelMixer:
    """
    Riduce a un canale solo, con una decisione STABILE nel tempo.

    La media aritmetica sembra la scelta ovvia, ma e' sbagliata nel caso
    piu' comune sui portatili: un microfono dichiarato stereo che porta
    il segnale su un canale solo e silenzio sull'altro. Mediandoli si
    perdono 6 dB, e il parlato finisce sotto le soglie di silenzio senza
    che nulla lo segnali.

    Decidere blocco per blocco quali canali siano attivi era pero'
    peggio del male: con un secondo canale che oscilla attorno alla
    soglia (diafonia, respiro, una ventola) la configurazione cambiava
    anche venti volte dentro la stessa frase, e ogni cambio produceva un
    salto di volume di quasi 6 dB — un clic secco in mezzo a una parola,
    che manda fuori giri sia la soglia del rilevatore di voce sia il
    confronto con l'eco.

    Qui la decisione si basa su una media che si muove lentamente e ha
    due soglie diverse per entrare e per uscire, cosi' un canale al
    limite non fa avanti e indietro.
    """

    ENTER_RATIO = 0.020    # sopra il 2% del canale piu' forte: e' attivo
    LEAVE_RATIO = 0.008    # sotto lo 0,8%: e' spento
    SMOOTHING = 0.15       # quanto pesa il blocco corrente sulla media

    def __init__(self, channels: int):
        self.channels = max(1, channels)
        self._energy: np.ndarray | None = None
        self._active: np.ndarray | None = None

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        channels = self.channels
        if channels <= 1:
            # I dati provengono da un buffer di sola lettura: la copia
            # evita che una futura elaborazione sul posto fallisca in
            # modo oscuro.
            return np.array(samples, dtype=np.float32, copy=True)

        usable = (samples.size // channels) * channels
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        frame = samples[:usable].reshape(-1, channels)

        livello = np.sqrt(np.mean(np.square(frame, dtype=np.float64), axis=0))
        if self._energy is None:
            self._energy = livello
        else:
            self._energy = (
                (1.0 - self.SMOOTHING) * self._energy + self.SMOOTHING * livello
            )

        forte = float(self._energy.max())
        if forte > 0.0:
            if self._active is None:
                self._active = self._energy >= forte * self.ENTER_RATIO
            else:
                accesi = self._energy >= forte * self.ENTER_RATIO
                spenti = self._energy < forte * self.LEAVE_RATIO
                self._active = np.where(
                    accesi, True, np.where(spenti, False, self._active)
                )
            if not self._active.any():
                self._active = None

        if self._active is not None and not self._active.all():
            return frame[:, self._active].mean(axis=1).astype(np.float32)
        return frame.mean(axis=1).astype(np.float32)


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Versione senza memoria, usata solo dalle prove automatiche."""
    return ChannelMixer(channels)(samples)


_FIR_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _antialias_kernel(source_rate: int, target_rate: int) -> np.ndarray:
    """
    Filtro passa-basso a fase lineare, calcolato una volta sola.

    La versione precedente usava una media mobile su tre campioni. A
    8 kHz — cioe' proprio al limite della banda che Whisper analizza —
    quel filtro attenua di appena 3,5 dB: tutto il contenuto fra 8 e
    16 kHz si ripiegava dentro la banda vocale quasi intatto, sporcando
    il segnale. E' la ragione principale per cui la trascrizione
    "capiva male", del tutto indipendente dalla potenza del computer.
    """
    key = (source_rate, target_rate)
    cached = _FIR_CACHE.get(key)
    if cached is not None:
        return cached

    taps = 127                      # dispari: ritardo intero, facile da togliere
    # Taglio poco sotto la nuova frequenza di Nyquist, per lasciare
    # spazio alla transizione del filtro senza toccare il parlato.
    cutoff = 0.45 * target_rate / source_rate      # in cicli/campione
    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    kernel *= np.hamming(taps)
    total = kernel.sum()
    if total:
        kernel /= total             # guadagno unitario in continua
    kernel = kernel.astype(np.float32)
    _FIR_CACHE[key] = kernel
    return kernel


def _convolve_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convoluzione centrata, passando per la FFT sui segnali lunghi."""
    if signal.size < 4096:
        if signal.size >= kernel.size:
            return np.convolve(signal, kernel, mode="same").astype(np.float32)
        # Con un segnale piu' corto del filtro, la modalita' "same" di
        # np.convolve centra il risultato sul FILTRO e non sul segnale:
        # i campioni restituiti sono quelli sbagliati. Oggi non ci si
        # arriva mai (le frasi durano almeno un decimo di secondo), ma e'
        # una trappola per chiunque riusi questa funzione altrove.
        inizio = (kernel.size - 1) // 2
        pieno = np.convolve(signal, kernel, mode="full")
        return pieno[inizio : inizio + signal.size].astype(np.float32)

    size = 1
    needed = signal.size + kernel.size - 1
    while size < needed:
        size *= 2
    spectrum = np.fft.rfft(signal, size) * np.fft.rfft(kernel, size)
    full = np.fft.irfft(spectrum, size)
    start = (kernel.size - 1) // 2
    return full[start : start + signal.size].astype(np.float32)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Riduce la frequenza di campionamento applicando prima un vero
    filtro anti-aliasing.

    Da applicare su finestre lunghe (secondi), non su blocchi minuscoli:
    vedi la nota in cima al modulo.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0 or source_rate == target_rate:
        return samples

    # La lunghezza va decisa PRIMA di filtrare: np.convolve in modalita'
    # "same" restituisce max(segnale, filtro) campioni, quindi su un
    # segnale piu' corto del filtro il risultato veniva gonfiato a 127
    # campioni e la durata dichiarata triplicava.
    target_length = int(round(samples.size * target_rate / source_rate))

    if source_rate > target_rate and samples.size > 8:
        filtrato = _convolve_same(
            samples, _antialias_kernel(source_rate, target_rate)
        )
        samples = filtrato[: samples.size]

    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size == 1:
        return np.repeat(samples, target_length).astype(np.float32)

    return np.interp(
        np.linspace(0, samples.size - 1, target_length, dtype=np.float64),
        np.arange(samples.size, dtype=np.float64),
        samples,
    ).astype(np.float32)


def rms_level(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


# --------------------------------------------------------------------------
# Enumerazione dispositivi
# --------------------------------------------------------------------------
class DeviceCatalog:
    """Elenca i dispositivi audio disponibili sul computer."""

    def __init__(self):
        if pyaudio is None:
            raise AudioError(
                "Il supporto audio non e' installato correttamente "
                "(libreria PyAudio mancante)."
            )
        self._pa = pyaudio.PyAudio()

    @property
    def handle(self):
        return self._pa

    def close(self) -> None:
        try:
            self._pa.terminate()
        except Exception:
            log.debug("Chiusura di PortAudio non riuscita", exc_info=True)

    # -- microfono ---------------------------------------------------
    def default_microphone(self) -> Optional[DeviceInfo]:
        try:
            info = self._pa.get_default_input_device_info()
        except Exception:
            log.warning("Nessun microfono predefinito disponibile", exc_info=True)
            return None
        channels = int(info.get("maxInputChannels") or 0)
        if channels < 1:
            return None
        return DeviceInfo(
            index=int(info["index"]),
            name=str(info.get("name", "Microfono")),
            sample_rate=int(info.get("defaultSampleRate") or 44100),
            channels=min(2, channels),
        )

    def input_devices(self) -> list[DeviceInfo]:
        devices: list[DeviceInfo] = []
        try:
            for i in range(self._pa.get_device_count()):
                info = self._pa.get_device_info_by_index(i)
                channels = int(info.get("maxInputChannels") or 0)
                if channels < 1:
                    continue
                devices.append(
                    DeviceInfo(
                        index=i,
                        name=str(info.get("name", f"Dispositivo {i}")),
                        sample_rate=int(info.get("defaultSampleRate") or 44100),
                        channels=min(2, channels),
                        is_loopback=bool(info.get("isLoopbackDevice", False)),
                    )
                )
        except Exception:
            log.warning("Enumerazione dispositivi non riuscita", exc_info=True)
        return devices

    # -- audio di sistema --------------------------------------------
    def default_loopback(self) -> Optional[DeviceInfo]:
        """
        Dispositivo che registra l'audio in uscita dagli altoparlanti.

        Disponibile solo su Windows tramite PyAudioWPatch; altrove
        restituiamo None e l'app continua con il solo microfono.
        """
        if not HAS_LOOPBACK:
            return None

        try:
            wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except Exception:
            log.info("Interfaccia audio WASAPI non disponibile", exc_info=True)
            return None

        speakers = None
        try:
            speakers = self._pa.get_device_info_by_index(
                int(wasapi["defaultOutputDevice"])
            )
        except Exception:
            log.warning("Altoparlanti predefiniti non individuati", exc_info=True)

        candidate = None
        if speakers is not None and speakers.get("isLoopbackDevice", False):
            candidate = speakers
        else:
            speaker_name = str(speakers.get("name", "")) if speakers else ""
            try:
                for loopback in self._pa.get_loopback_device_info_generator():
                    if speaker_name and speaker_name in str(loopback.get("name", "")):
                        candidate = loopback  # gemello degli altoparlanti attivi
                        break
                    if candidate is None:
                        candidate = loopback  # ripiego: il primo disponibile
            except Exception:
                log.warning("Ricerca dispositivi loopback fallita", exc_info=True)

        if candidate is None:
            return None

        channels = int(candidate.get("maxInputChannels") or 0)
        if channels < 1:
            # Un dispositivo di sola uscita non e' registrabile: aprirlo
            # con zero canali farebbe terminare il programma.
            log.warning(
                "Il dispositivo loopback '%s' non espone canali di ingresso",
                candidate.get("name"),
            )
            return None

        return DeviceInfo(
            index=int(candidate["index"]),
            name=str(candidate.get("name", "Audio di sistema")),
            sample_rate=int(candidate.get("defaultSampleRate") or 48000),
            # Il loopback WASAPI in modalita' condivisa va aperto con
            # ESATTAMENTE il formato del mixer di Windows. Limitarlo a
            # due canali faceva fallire l'apertura su chi ha un impianto
            # surround o un monitor HDMI multicanale, e in quel caso la
            # voce del candidato non veniva trascritta affatto.
            channels=max(1, channels),
            is_loopback=True,
        )


# --------------------------------------------------------------------------
# Registrazione
# --------------------------------------------------------------------------
class _SourceReader(threading.Thread):
    """Legge da un dispositivo e accoda finestre audio da alcuni secondi."""

    def __init__(
        self,
        pa,
        pa_lock: threading.Lock,
        device: DeviceInfo,
        speaker: str,
        output: "queue.Queue[AudioChunk]",
        stop_event: threading.Event,
        started_at: float,
        on_level: Optional[Callable[[str, float], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        super().__init__(name=f"audio-{speaker}", daemon=True)
        self._pa = pa
        self._pa_lock = pa_lock
        self.device = device
        self.speaker = speaker
        self._output = output
        # Il nome NON puo' essere "_stop": threading.Thread usa gia' un
        # metodo interno con quel nome, chiamato da join() e is_alive()
        # non appena il thread e' finito. Coprendolo con un oggetto
        # evento, ogni arresto della registrazione moriva con un
        # "'Event' object is not callable": il secondo dispositivo non
        # veniva mai chiuso, PortAudio restava aperto a ogni colloquio e
        # gli avvisi veri venivano sostituiti da quell'errore assurdo.
        self._stop_event = stop_event
        self._started_at = started_at
        self._on_level = on_level
        self._on_error = on_error

        self.started_ok = threading.Event()
        self.dropped_chunks = 0

        # Il taglio in frasi avviene alla frequenza nativa del
        # dispositivo; la conversione a 16 kHz viene fatta una sola
        # volta, sulla frase completa.
        self._segmenter = VoiceSegmenter(device.sample_rate)
        # Il riduttore a un canale conserva memoria fra un blocco e
        # l'altro: la decisione su quali canali siano attivi deve
        # restare la stessa per tutta la frase.
        self._mixer = ChannelMixer(device.channels)

        # Scarto fra l'avvio della registrazione e il primo campione di
        # QUESTA sorgente. I due dispositivi non si aprono nello stesso
        # istante e senza questa correzione ciascun canale conterebbe i
        # secondi da un momento diverso: differenze anche di pochi
        # decimi mandano l'eco fuori dalla finestra di ricerca.
        self._epoch = 0.0
        self._last_progress = -1.0

    # ------------------------------------------------------------------
    def _enqueue(
        self, samples: np.ndarray, offset: float, continues: bool = False
    ) -> None:
        """
        Accoda senza mai bloccare il thread audio.

        Se la trascrizione non tiene il passo scartiamo l'audio piu'
        vecchio, finche' l'attesa torna sotto il limite: e' preferibile
        perdere qualche secondo lontano nel tempo piuttosto che mostrare
        i sottotitoli con minuti di ritardo. Scartare il piu' vecchio,
        e non il piu' recente, riporta il motore al presente.
        """
        chunk = AudioChunk(
            self.speaker, samples, offset=offset, continues_previous=continues
        )
        duration = samples.size / float(config.AUDIO_SAMPLE_RATE)

        # L'audio che la trascrizione tiene gia' in mano restringe lo
        # spazio concesso alla coda, ma non oltre un minimo garantito.
        # Contarlo senza quel minimo era un guasto silenzioso e grave:
        # quando il motore ne teneva da parte piu' del limite, nessuna
        # quantita' di scarti poteva far scendere il totale, e a ogni
        # frase la coda veniva svuotata per intero — parlato compreso —
        # senza che il limite risultasse mai rispettato.
        # Il minimo va applicato DOPO aver sottratto la frase in arrivo,
        # altrimenti si perde di nuovo. Una risposta lunga viene tagliata
        # dal rilevatore di voce a dieci secondi esatti, cioe' quanto il
        # minimo: sottraendola per ultima il limite scendeva a zero e
        # l'intera coda spariva — proprio sulle risposte piu' lunghe, che
        # sono quelle che contano.
        parcheggiati = getattr(self._output, "parked_seconds", 0.0)
        limite = max(
            QUEUE_FLOOR_SECONDS,
            MAX_BACKLOG_SECONDS - parcheggiati - duration,
        )
        trim = getattr(self._output, "trim_to", None)
        if trim is not None:
            self.dropped_chunks += trim(limite)

        try:
            self._output.put_nowait(chunk)
        except queue.Full:
            self.dropped_chunks += 1

    def _emit(self, utterance) -> None:
        samples = resample(
            utterance.samples, self.device.sample_rate, config.AUDIO_SAMPLE_RATE
        )
        if samples.size:
            self._enqueue(
                samples,
                self._epoch + utterance.start_offset,
                utterance.continues_previous,
            )

    def _publish_progress(self) -> None:
        """
        Segnala all'altro canale fin dove questo e' stato analizzato.

        Serve al riconoscimento dell'eco: se il candidato tace, senza
        questo avviso il microfono resterebbe in attesa di un
        riferimento che non arrivera' mai, e ogni domanda comparirebbe
        nei sottotitoli con secondi di ritardo.
        """
        if self.speaker != config.SPEAKER_CANDIDATE:
            return
        settled = self._epoch + self._segmenter.settled_seconds
        if settled - self._last_progress < PROGRESS_INTERVAL_SECONDS:
            return
        # Se la coda e' gia' lunga questi avvisi non aiutano piu'
        # nessuno: toglierebbero soltanto spazio all'audio vero.
        if self._output.qsize() >= PROGRESS_QUEUE_LIMIT:
            return
        self._last_progress = settled
        try:
            self._output.put_nowait(
                AudioChunk(self.speaker, np.zeros(0, dtype=np.float32), offset=settled)
            )
        except queue.Full:
            pass

    def run(self) -> None:  # noqa: C901 - flusso lineare, leggibile
        stream = None
        consecutive_errors = 0
        try:
            with self._pa_lock:
                stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=self.device.channels,
                    rate=self.device.sample_rate,
                    input=True,
                    input_device_index=self.device.index,
                    frames_per_buffer=READ_FRAMES,
                )
            self.started_ok.set()
            # Origine comune ai due canali, misurata quando il
            # dispositivo comincia davvero a produrre campioni.
            self._epoch = max(0.0, time.monotonic() - self._started_at)
            log.info(
                "Sorgente '%s' avviata: %s (%d Hz, %d canali)",
                self.speaker,
                self.device.name,
                self.device.sample_rate,
                self.device.channels,
            )

            while not self._stop_event.is_set():
                try:
                    raw = stream.read(READ_FRAMES, exception_on_overflow=False)
                    consecutive_errors = 0
                except OSError as exc:
                    consecutive_errors += 1
                    if consecutive_errors > MAX_CONSECUTIVE_READ_ERRORS:
                        raise AudioError(
                            f"Il dispositivo '{self.device.name}' ha smesso di "
                            "rispondere: potrebbe essere stato scollegato."
                        ) from exc
                    # Senza questa pausa un dispositivo scollegato
                    # farebbe girare il ciclo a vuoto saturando la CPU.
                    time.sleep(0.05)
                    continue

                samples = np.frombuffer(raw, dtype=np.float32)
                if samples.size == 0:
                    continue

                mono = self._mixer(samples)
                if self._on_level is not None:
                    self._on_level(self.speaker, rms_level(mono))

                # Il rilevatore restituisce solo le frasi concluse: il
                # silenzio non arriva mai al riconoscimento vocale.
                for utterance in self._segmenter.feed(mono):
                    self._emit(utterance)
                self._publish_progress()

        except Exception as exc:
            log.exception("Sorgente audio '%s' interrotta", self.speaker)
            if self._on_error:
                self._on_error(self.speaker, exc)
        finally:
            # Ultima frase: non perdiamo la coda del discorso.
            try:
                for utterance in self._segmenter.flush():
                    self._emit(utterance)
            except Exception as exc:
                # Qui esce l'ultima risposta del candidato: se si perde,
                # sparisce dal report senza lasciare traccia. A livello
                # "debug" non compariva nemmeno nei log.
                log.exception("Ultima frase della sorgente '%s' non recuperata",
                              self.speaker)
                if self._on_error:
                    self._on_error(self.speaker, exc)

            log.info(
                "Sorgente '%s': parlato rilevato sul %.0f%% del tempo",
                self.speaker,
                self._segmenter.speech_ratio * 100,
            )
            if stream is not None:
                with self._pa_lock:
                    for action in (stream.stop_stream, stream.close):
                        try:
                            action()
                        except Exception:
                            # Se la chiusura fallisce il dispositivo
                            # resta occupato, e al colloquio successivo
                            # l'utente legge "in uso da un altro
                            # programma" senza il minimo indizio nei log.
                            log.warning(
                                "Chiusura dello stream '%s' non riuscita",
                                self.speaker, exc_info=True,
                            )
            log.info(
                "Sorgente '%s' terminata (blocchi scartati: %d)",
                self.speaker,
                self.dropped_chunks,
            )


class AudioRecorder:
    """Coordina le sorgenti audio e produce blocchi etichettati."""

    def __init__(
        self,
        capture_microphone: bool = True,
        capture_system_audio: bool = True,
        on_level: Optional[Callable[[str, float], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        self.capture_microphone = capture_microphone
        self.capture_system_audio = capture_system_audio
        self._on_level = on_level
        self._on_error = on_error

        self.audio_queue: AudioQueue = AudioQueue(
            maxsize=QUEUE_HARD_LIMIT, sample_rate=config.AUDIO_SAMPLE_RATE
        )
        self._stop = threading.Event()
        self._readers: list[_SourceReader] = []
        self._catalog: Optional[DeviceCatalog] = None
        self._pa_lock = threading.Lock()
        self._watcher: Optional[threading.Thread] = None
        # Avvio e arresto non possono sovrapporsi. Senza questo blocco,
        # chi chiudeva la finestra nei primi secondi di un colloquio
        # faceva chiamare terminate() su PortAudio mentre l'altro thread
        # stava ancora aprendo gli stream: il programma spariva senza
        # messaggio, ed era il guasto piu' difficile da riprodurre.
        self._lifecycle = threading.Lock()

        self.active_sources: dict[str, str] = {}   # etichetta -> dispositivo
        self.warnings: list[str] = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lifecycle:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._readers:
            return
        # Un arresto gia' richiesto ha la precedenza: aprire adesso i
        # dispositivi significherebbe lasciarli aperti per sempre,
        # perche' chi doveva chiuderli ha gia' fatto il suo giro.
        if self._stop.is_set():
            raise AudioError("Registrazione annullata prima dell'avvio.")

        self.warnings.clear()
        self.active_sources.clear()

        catalog = DeviceCatalog()
        try:
            self._start_with_catalog(catalog)
        except Exception:
            # Senza questa chiusura ogni tentativo fallito lascerebbe
            # viva un'istanza di PortAudio, e dopo qualche tentativo
            # l'enumerazione dei dispositivi comincerebbe a fallire.
            self._stop.set()
            for reader in self._readers:
                reader.join(timeout=2)
            ancora_vivi = [r for r in self._readers if r.is_alive()]
            self._readers = ancora_vivi
            if ancora_vivi:
                # Chiudere PortAudio con uno stream ancora aperto in un
                # altro thread fa terminare di colpo il programma: e'
                # preferibile lasciarlo aperto.
                log.error(
                    "PortAudio non chiuso: sorgenti ancora attive %s",
                    [r.speaker for r in ancora_vivi],
                )
            else:
                catalog.close()
                self._catalog = None
            raise

    def _start_with_catalog(self, catalog: DeviceCatalog) -> None:
        self._catalog = catalog
        planned: list[tuple[DeviceInfo, str]] = []

        if self.capture_microphone:
            mic = catalog.default_microphone()
            if mic is not None:
                planned.append((mic, config.SPEAKER_RECRUITER))
            else:
                self.warnings.append(
                    "Nessun microfono rilevato: la tua voce non verra' trascritta."
                )

        if self.capture_system_audio:
            loopback = catalog.default_loopback()
            if loopback is not None:
                planned.append((loopback, config.SPEAKER_CANDIDATE))
            else:
                self.warnings.append(
                    "Audio di sistema non disponibile: la voce del candidato in "
                    "videochiamata non verra' trascritta. Verifica che gli "
                    "altoparlanti del computer siano attivi."
                )

        if not planned:
            raise AudioError(
                "Nessuna sorgente audio utilizzabile. Collega un microfono "
                "oppure attiva gli altoparlanti, poi riprova."
            )

        started_at = time.monotonic()
        for device, speaker in planned:
            reader = _SourceReader(
                catalog.handle,
                self._pa_lock,
                device,
                speaker,
                self.audio_queue,
                self._stop,
                started_at,
                on_level=self._on_level,
                on_error=self._handle_source_error,
            )
            reader.start()
            self._readers.append(reader)

        # Attendiamo l'apertura effettiva di ciascuno stream, con
        # un'attesa indipendente per sorgente: cosi' un dispositivo lento
        # non fa dichiarare guasto anche l'altro.
        for reader in self._readers:
            if reader.started_ok.wait(timeout=4.0):
                self.active_sources[reader.speaker] = reader.device.name
            else:
                self.warnings.append(
                    f"Il dispositivo '{reader.device.name}' non ha risposto: "
                    "potrebbe essere in uso da un altro programma."
                )

        if not self.active_sources:
            raise AudioError(
                "Non e' stato possibile aprire nessun dispositivo audio. "
                "Chiudi i programmi che stanno usando il microfono e riprova."
            )

        self._start_device_watch(catalog)

    # ------------------------------------------------------------------
    # Sorveglianza dei dispositivi durante la registrazione
    # ------------------------------------------------------------------
    def _start_device_watch(self, catalog: DeviceCatalog) -> None:
        """
        Controlla che i dispositivi predefiniti restino quelli in uso.

        Windows cambia il dispositivo predefinito da solo quando si
        collega un auricolare Bluetooth, si inserisce un jack o si
        attacca un monitor con altoparlanti. La registrazione in corso
        NON segue quel cambio: continua a leggere dal vecchio
        dispositivo, che di colpo non riceve piu' nulla. Il colloquio
        prosegue in perfetto silenzio e nessuno se ne accorge finche'
        non si apre il report e lo si trova vuoto.

        Non possiamo cambiare dispositivo a caldo senza interrompere e
        risincronizzare i due canali, cosa che perderebbe comunque
        dell'audio: la cosa piu' utile e' avvisare subito, mentre c'e'
        ancora tempo per rimediare.
        """
        atteso: dict[str, str] = {}
        for reader in self._readers:
            atteso[reader.speaker] = reader.device.name

        def _controlla() -> None:
            gia_avvisati: set[str] = set()
            while not self._stop.wait(DEVICE_WATCH_SECONDS):
                try:
                    with self._pa_lock:
                        corrente = {
                            config.SPEAKER_RECRUITER: catalog.default_microphone(),
                            config.SPEAKER_CANDIDATE: catalog.default_loopback(),
                        }
                except Exception:
                    log.debug("Controllo dei dispositivi non riuscito", exc_info=True)
                    continue

                for speaker, nome_atteso in atteso.items():
                    if speaker in gia_avvisati:
                        continue
                    info = corrente.get(speaker)
                    if info is None or info.name == nome_atteso:
                        continue
                    gia_avvisati.add(speaker)
                    quale = (
                        "il microfono"
                        if speaker == config.SPEAKER_RECRUITER
                        else "l'audio di sistema"
                    )
                    messaggio = (
                        f"ATTENZIONE: {quale} predefinito di Windows e' cambiato "
                        f"durante il colloquio (da '{nome_atteso}' a "
                        f"'{info.name}'). La registrazione continua sul "
                        "dispositivo di partenza: se non senti piu' nulla, "
                        "ferma il colloquio e riavvialo per usare il nuovo "
                        "dispositivo."
                    )
                    log.warning(messaggio)
                    self.warnings.append(messaggio)
                    if self._on_error:
                        self._on_error(speaker, RuntimeError(messaggio))

        self._watcher = threading.Thread(
            target=_controlla, name="audio-watch", daemon=True
        )
        self._watcher.start()

    # ------------------------------------------------------------------
    def _handle_source_error(self, speaker: str, exc: Exception) -> None:
        self.warnings.append(f"Sorgente audio '{speaker}' interrotta: {exc}")
        if self._on_error:
            self._on_error(speaker, exc)

    def stop(self, timeout: float = 6.0) -> bool:
        """
        Ferma le sorgenti. Restituisce True solo se tutti i thread sono
        davvero terminati.

        Se un thread resta bloccato nel driver audio NON terminiamo
        PortAudio: chiuderlo mentre uno stream e' ancora aperto in un
        altro thread provoca la chiusura immediata del programma.
        """
        # Il segnale di arresto va dato PRIMA di prendere il blocco: se
        # un avvio e' in corso deve accorgersene subito, e noi dobbiamo
        # aspettare che finisca invece di chiudere sotto i suoi piedi.
        self._stop.set()
        with self._lifecycle:
            return self._stop_locked(timeout)

    def _stop_locked(self, timeout: float) -> bool:
        # La sorveglianza usa PortAudio: va fermata PRIMA di chiuderlo,
        # altrimenti puo' interrogare dispositivi mentre l'istanza viene
        # distrutta, e quello fa terminare il programma all'istante.
        if self._watcher is not None:
            self._watcher.join(timeout=DEVICE_WATCH_SECONDS + 2)
            if self._watcher.is_alive():
                log.error("Sorveglianza dei dispositivi ancora attiva")
                return False
            self._watcher = None

        # Ogni sorgente ha il proprio tempo di attesa. Con una scadenza
        # unica, una sorgente lenta consumava tutto il budget e la
        # seconda riceveva join(0): veniva dichiarata "ancora attiva"
        # anche se stava per finire, PortAudio non veniva chiuso e il
        # colloquio successivo trovava i dispositivi occupati.
        still_alive: list[_SourceReader] = []
        for reader in self._readers:
            reader.join(timeout=timeout)
            if reader.is_alive():
                still_alive.append(reader)

        dropped = sum(reader.dropped_chunks for reader in self._readers)
        if dropped:
            self.warnings.append(
                "Il computer non e' riuscito a trascrivere in tempo reale: "
                f"{dropped} frasi non sono state elaborate per non far "
                "accumulare ritardo ai sottotitoli. Nelle impostazioni "
                "puoi scegliere un modello di trascrizione piu' leggero."
            )

        if still_alive:
            log.error(
                "Sorgenti ancora attive dopo %.1f s: %s. PortAudio non viene "
                "chiuso per evitare una chiusura anomala.",
                timeout,
                [reader.speaker for reader in still_alive],
            )
            self._readers = still_alive
            return False

        self._readers.clear()
        if self._catalog is not None:
            self._catalog.close()
            self._catalog = None
        return True

    @property
    def is_running(self) -> bool:
        return any(reader.is_alive() for reader in self._readers)
