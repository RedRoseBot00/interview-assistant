"""
Motore di trascrizione live basato su faster-whisper (Whisper locale,
CTranslate2), completamente offline e gratuito.

Whisper riconosce automaticamente ~99 lingue: non serve configurare nulla
per supportare colloqui in italiano, inglese o qualunque altra lingua.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from app import config


@dataclass
class TranscriptSegment:
    text: str
    language: str
    start_offset: float  # secondi dall'inizio del colloquio
    timestamp: float = field(default_factory=time.time)


class TranscriptionEngine:
    """
    Consuma i chunk audio prodotti da AudioRecorder e produce testo via
    faster-whisper, in un thread dedicato in modo da non bloccare la UI.
    """

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL_SIZE_DEFAULT,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        on_ready: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.model_size = model_size
        self.on_segment = on_segment
        self.on_ready = on_ready
        self.on_error = on_error

        self._model = None
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._elapsed = 0.0
        self.segments: List[TranscriptSegment] = []

        # Buffer di accumulo per finestre di trascrizione con overlap,
        # cosi' non tagliamo le parole a meta' frase.
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_lock = threading.Lock()

    # ------------------------------------------------------------------
    def load_model(self):
        """Carica il modello Whisper in memoria (operazione lenta, va
        eseguita in un thread separato dalla UI al primo avvio)."""
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=str(config.WHISPER_CACHE_DIR),
        )
        if self.on_ready:
            self.on_ready()

    # ------------------------------------------------------------------
    def feed(self, samples: np.ndarray):
        """Chiamato dal thread di cattura audio per ogni chunk mixato."""
        with self._pending_lock:
            self._pending = np.concatenate([self._pending, samples])

    def start(self, audio_queue):
        """Avvia il loop di trascrizione, leggendo chunk da audio_queue
        (una queue.Queue popolata da AudioRecorder)."""
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_loop, args=(audio_queue,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    def _run_loop(self, audio_queue):
        if self._model is None:
            try:
                self.load_model()
            except Exception as exc:
                if self.on_error:
                    self.on_error(exc)
                return

        chunk_samples = int(
            config.TRANSCRIBE_CHUNK_SECONDS * config.AUDIO_SAMPLE_RATE
        )
        overlap_samples = int(
            config.TRANSCRIBE_OVERLAP_SECONDS * config.AUDIO_SAMPLE_RATE
        )

        while self._running.is_set():
            try:
                samples = audio_queue.get(timeout=0.5)
            except Exception:
                continue

            with self._pending_lock:
                self._pending = np.concatenate([self._pending, samples])
                ready = self._pending.size >= chunk_samples

            if not ready:
                continue

            with self._pending_lock:
                window = self._pending[:chunk_samples]
                # manteniamo l'overlap per la finestra successiva
                self._pending = self._pending[chunk_samples - overlap_samples:]

            self._transcribe_window(window)

    def _transcribe_window(self, window: np.ndarray):
        if window.size == 0 or self._model is None:
            return
        try:
            segments, info = self._model.transcribe(
                window,
                language=None,  # autodetect: supporto multilingua
                vad_filter=True,  # filtra i silenzi, riduce le allucinazioni
                beam_size=1,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if not text:
                return
            segment = TranscriptSegment(
                text=text,
                language=info.language,
                start_offset=self._elapsed,
            )
            self.segments.append(segment)
            if self.on_segment:
                self.on_segment(segment)
        except Exception as exc:
            if self.on_error:
                self.on_error(exc)
        finally:
            self._elapsed += config.TRANSCRIBE_CHUNK_SECONDS

    # ------------------------------------------------------------------
    def full_transcript(self) -> str:
        return "\n".join(s.text for s in self.segments)
