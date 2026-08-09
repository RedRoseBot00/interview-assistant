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

# Oltre questo intervallo fra due blocchi della stessa persona il
# contesto precedente non serve piu' a togliere le ripetizioni: due
# frasi identiche pronunciate a distanza sono due frasi diverse.
CONTEXT_EXPIRY_SECONDS = config.TRANSCRIBE_CHUNK_SECONDS * 2.5

# Quanto un blocco del microfono puo' restare in attesa dell'audio di
# riferimento prima di essere trascritto comunque.
REFERENCE_WAIT_SECONDS = 2.5

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


def _normalise(word: str) -> str:
    return word.strip(".,;:!?'\"()").lower()


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
        self._use_vad = True

        self.segments: list[TranscriptSegment] = []
        self.backlog = 0

        self._languages: Counter[str] = Counter()
        self._last_text: dict[str, str] = {}
        self._last_offset: dict[str, float] = {}
        self._lock = threading.Lock()

        # Gestione dell'eco
        self.echo = echo_module.EchoProcessor(echo_mode, config.AUDIO_SAMPLE_RATE)
        self._reference = echo_module.ReferenceBuffer(config.AUDIO_SAMPLE_RATE)
        self._waiting: deque[tuple[AudioChunk, float]] = deque()
        self.echo_dropped = 0
        self.duplicates_dropped = 0

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
                chunk = audio_queue.get(timeout=0.4)
            except queue.Empty:
                self._flush_waiting(force=self._stop.is_set())
                if self._stop.is_set() and not self._waiting:
                    break
                continue

            self.backlog = audio_queue.qsize() + len(self._waiting)
            try:
                self._accept(chunk)
            except Exception as exc:
                log.exception("Errore durante l'elaborazione di un blocco")
                if self.on_error:
                    self.on_error(exc)

        self._flush_waiting(force=True)
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
    # Smistamento dei blocchi
    # ------------------------------------------------------------------
    def _accept(self, chunk: AudioChunk) -> None:
        if chunk.speaker == config.SPEAKER_CANDIDATE:
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
            needed = chunk.offset + duration + echo_module.MAX_DELAY_SECONDS
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
                    log.debug(
                        "Blocco microfono scartato come eco "
                        "(somiglianza %.2f, ritardo %.0f ms)",
                        result.correlation,
                        result.delay_seconds * 1000,
                    )
                    continue
                samples = result.samples

            self._transcribe(AudioChunk(chunk.speaker, samples, chunk.offset, chunk.wall_time))

    # ------------------------------------------------------------------
    # Trascrizione vera e propria
    # ------------------------------------------------------------------
    def _transcribe(self, chunk: AudioChunk) -> None:
        if chunk.samples.size == 0 or self._model is None:
            return

        # Il silenzio costituisce gran parte di un colloquio: scartarlo
        # subito risparmia CPU e riduce le "allucinazioni" del modello,
        # che sull'audio muto tende a inventare frasi.
        if rms_level(chunk.samples) < config.SILENCE_RMS_THRESHOLD:
            return

        audio = np.asarray(chunk.samples, dtype=np.float32)
        language = None if self.language == "auto" else self.language

        try:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=self._use_vad,
                condition_on_previous_text=False,
            )
            parts = [seg.text.strip() for seg in segments]
        except Exception as exc:
            # Il filtro di rilevamento voce richiede una libreria
            # aggiuntiva: se manca, proseguiamo senza, invece di
            # interrompere un colloquio in corso.
            if not self._use_vad:
                raise
            log.warning("Filtro voce non disponibile, proseguo senza: %s", exc)
            self._use_vad = False
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            parts = [seg.text.strip() for seg in segments]

        text = " ".join(part for part in parts if part).strip()
        if not text:
            return

        detected = getattr(info, "language", "") or ""

        with self._lock:
            previous = self._last_text.get(chunk.speaker, "")
            if chunk.offset - self._last_offset.get(chunk.speaker, -999.0) > CONTEXT_EXPIRY_SECONDS:
                previous = ""
            text = _strip_overlap(previous, text)
            if not text:
                return

            # Ultima rete di sicurezza contro l'eco: la stessa frase
            # comparsa poco fa sull'altro canale e' una ripetizione, non
            # un intervento nuovo.
            if self.echo.enabled and chunk.speaker == config.SPEAKER_RECRUITER:
                if self._is_echo_of_candidate(text, chunk.offset):
                    self.duplicates_dropped += 1
                    self.echo.detections += 1
                    log.debug("Frase scartata perche' duplicata dall'altro canale")
                    return

            self._last_text[chunk.speaker] = text
            self._last_offset[chunk.speaker] = chunk.offset
            if detected:
                self._languages[detected] += 1

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
