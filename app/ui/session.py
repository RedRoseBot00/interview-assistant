"""
Ponte tra i moduli "core" (audio + trascrizione, che girano su thread
Python puri) e la UI Qt, che deve ricevere gli aggiornamenti in modo
thread-safe tramite i signal di Qt.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QObject, Signal

from app.audio.capture import AudioRecorder
from app.transcription.engine import TranscriptionEngine, TranscriptSegment


class InterviewSession(QObject):
    segment_received = Signal(str)       # testo del nuovo segmento trascritto
    model_loading = Signal()
    model_ready = Signal()
    error_occurred = Signal(str)

    def __init__(self, whisper_model_size: str, capture_mic: bool,
                 capture_system: bool, parent=None):
        super().__init__(parent)
        self._start_time = None
        self.last_language = "it"

        self.recorder = AudioRecorder(
            capture_microphone=capture_mic,
            capture_system_audio=capture_system,
            on_error=lambda exc: self.error_occurred.emit(str(exc)),
        )
        self.engine = TranscriptionEngine(
            model_size=whisper_model_size,
            on_segment=self._handle_segment,
            on_ready=lambda: self.model_ready.emit(),
            on_error=lambda exc: self.error_occurred.emit(str(exc)),
        )

    def _handle_segment(self, segment: TranscriptSegment):
        self.last_language = segment.language
        self.segment_received.emit(segment.text)

    def start(self):
        self._start_time = time.time()
        self.model_loading.emit()
        self.recorder.start()
        self.engine.start(self.recorder.audio_queue)

    def stop(self):
        self.engine.stop()
        self.recorder.stop()

    def close(self):
        self.recorder.close()

    @property
    def elapsed_seconds(self) -> float:
        if not self._start_time:
            return 0.0
        return time.time() - self._start_time

    @property
    def full_transcript(self) -> str:
        return self.engine.full_transcript()
