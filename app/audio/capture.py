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
QUEUE_MAX_CHUNKS = 96  # ~7 minuti di arretrato per sorgente: oltre, si scarta


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
    """Blocco audio mono a 16 kHz, gia' attribuito a un interlocutore."""

    speaker: str
    samples: np.ndarray
    # Istante di inizio del blocco, in secondi dall'avvio della
    # registrazione. Usiamo un orologio monotono: quello di sistema puo'
    # essere corretto dalla rete durante il colloquio e produrrebbe
    # durate sbagliate, perfino negative.
    offset: float = 0.0
    wall_time: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Conversione del formato
# --------------------------------------------------------------------------
def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return samples
    usable = (samples.size // channels) * channels
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return samples[:usable].reshape(-1, channels).mean(axis=1).astype(np.float32)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Riduce la frequenza di campionamento applicando prima un filtro
    anti-aliasing elementare.

    Senza il filtro le frequenze alte si ripiegherebbero sulle basse,
    producendo un sibilo che peggiora sensibilmente il riconoscimento
    vocale. Da applicare su finestre lunghe (secondi), non su blocchi
    minuscoli: vedi la nota in cima al modulo.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0 or source_rate == target_rate:
        return samples

    if source_rate > target_rate:
        window = max(1, int(round(source_rate / target_rate)))
        if window > 1 and samples.size > window:
            kernel = np.ones(window, dtype=np.float32) / window
            samples = np.convolve(samples, kernel, mode="same").astype(np.float32)

    target_length = int(round(samples.size * target_rate / source_rate))
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
            channels=min(2, channels),
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
        self._stop = stop_event
        self._started_at = started_at
        self._on_level = on_level
        self._on_error = on_error

        self.started_ok = threading.Event()
        self.dropped_chunks = 0

        # Le soglie sono calcolate nella frequenza nativa del
        # dispositivo: il ricampionamento avviene una volta sola, sulla
        # finestra completa.
        rate = device.sample_rate
        self._chunk_samples = int(config.TRANSCRIBE_CHUNK_SECONDS * rate)
        overlap = int(config.TRANSCRIBE_OVERLAP_SECONDS * rate)
        # Difesa contro una configurazione incoerente: con una
        # sovrapposizione maggiore della finestra il buffer non
        # calerebbe mai e il ciclo girerebbe all'infinito.
        self._overlap_samples = max(0, min(overlap, self._chunk_samples // 2))
        self._buffer = np.zeros(0, dtype=np.float32)
        self._consumed = 0  # campioni gia' emessi, per calcolare l'offset

    # ------------------------------------------------------------------
    def _enqueue(self, samples: np.ndarray, offset: float) -> None:
        """
        Accoda senza mai bloccare il thread audio.

        Se la trascrizione non tiene il passo, scartiamo il blocco piu'
        vecchio: e' preferibile perdere qualche secondo di audio
        piuttosto che bloccare la registrazione, congelare gli
        indicatori di livello e impedire la chiusura del programma.
        """
        chunk = AudioChunk(self.speaker, samples, offset=offset)
        try:
            self._output.put_nowait(chunk)
            return
        except queue.Full:
            pass

        try:
            self._output.get_nowait()
            self.dropped_chunks += 1
        except queue.Empty:
            pass
        try:
            self._output.put_nowait(chunk)
        except queue.Full:
            self.dropped_chunks += 1

    def _emit_window(self) -> None:
        window_native = self._buffer[: self._chunk_samples]
        advance = self._chunk_samples - self._overlap_samples
        offset = self._consumed / self.device.sample_rate
        self._consumed += advance
        self._buffer = self._buffer[advance:]

        window = resample(
            window_native, self.device.sample_rate, config.AUDIO_SAMPLE_RATE
        )
        if window.size:
            self._enqueue(window, offset)

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
            log.info(
                "Sorgente '%s' avviata: %s (%d Hz, %d canali)",
                self.speaker,
                self.device.name,
                self.device.sample_rate,
                self.device.channels,
            )

            while not self._stop.is_set():
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

                mono = to_mono(samples, self.device.channels)
                if self._on_level is not None:
                    self._on_level(self.speaker, rms_level(mono))

                self._buffer = np.concatenate([self._buffer, mono])
                while self._buffer.size >= self._chunk_samples:
                    self._emit_window()

        except Exception as exc:
            log.exception("Sorgente audio '%s' interrotta", self.speaker)
            if self._on_error:
                self._on_error(self.speaker, exc)
        finally:
            # Ultimo tratto: non perdiamo la coda del discorso. Il
            # confronto tiene conto della sovrapposizione, che e' gia'
            # stata trascritta con la finestra precedente.
            try:
                fresh = self._buffer.size - self._overlap_samples
                if fresh > 0.25 * self.device.sample_rate:
                    tail = resample(
                        self._buffer, self.device.sample_rate, config.AUDIO_SAMPLE_RATE
                    )
                    if tail.size:
                        self._enqueue(tail, self._consumed / self.device.sample_rate)
            except Exception:
                log.debug("Ultimo blocco audio non recuperato", exc_info=True)

            self._buffer = np.zeros(0, dtype=np.float32)
            if stream is not None:
                with self._pa_lock:
                    for action in (stream.stop_stream, stream.close):
                        try:
                            action()
                        except Exception:
                            pass
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

        self.audio_queue: "queue.Queue[AudioChunk]" = queue.Queue(
            maxsize=QUEUE_MAX_CHUNKS
        )
        self._stop = threading.Event()
        self._readers: list[_SourceReader] = []
        self._catalog: Optional[DeviceCatalog] = None
        self._pa_lock = threading.Lock()

        self.active_sources: dict[str, str] = {}   # etichetta -> dispositivo
        self.warnings: list[str] = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._readers:
            return

        self.warnings.clear()
        self.active_sources.clear()
        self._stop.clear()

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
            self._readers.clear()
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
        self._stop.set()

        deadline = time.monotonic() + timeout
        still_alive: list[_SourceReader] = []
        for reader in self._readers:
            reader.join(timeout=max(0.0, deadline - time.monotonic()))
            if reader.is_alive():
                still_alive.append(reader)

        dropped = sum(reader.dropped_chunks for reader in self._readers)
        if dropped:
            self.warnings.append(
                "Il computer non e' riuscito a trascrivere in tempo reale: "
                f"{dropped} blocchi audio non sono stati elaborati. "
                "Nelle impostazioni puoi scegliere un modello piu' leggero."
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
