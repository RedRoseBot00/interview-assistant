"""
Cattura audio da microfono e/o audio di sistema (loopback).

Su Windows usiamo PyAudioWPatch, un fork di PyAudio che espone i device
WASAPI "loopback": permette di registrare l'audio che sta uscendo dagli
altoparlanti (es. la voce dell'interlocutore in una chiamata Teams/Zoom)
senza bisogno di cavi virtuali o software di terze parti.

L'obiettivo e' produrre un flusso continuo di campioni mono a 16kHz,
pronto per essere passato al motore di trascrizione (Whisper).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from app import config

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover - fallback per sviluppo su Linux/Mac
    import pyaudio  # type: ignore


@dataclass
class AudioSourceInfo:
    index: int
    name: str
    sample_rate: int
    channels: int


class AudioRecorder:
    """
    Registra audio da microfono e/o da un device di loopback di sistema,
    li mixa in un unico stream mono a 16kHz e mette i chunk risultanti
    in una coda thread-safe, consumata dal motore di trascrizione.
    """

    def __init__(
        self,
        capture_microphone: bool = True,
        capture_system_audio: bool = True,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.capture_microphone = capture_microphone
        self.capture_system_audio = capture_system_audio
        self.on_error = on_error

        self._pa = pyaudio.PyAudio()
        self._streams = []
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()

        # Coda di chunk audio grezzi (numpy float32 mono 16kHz), consumata
        # dal transcription engine.
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        # Buffer separati per mic e system audio, mixati periodicamente.
        self._mic_buffer = np.zeros(0, dtype=np.float32)
        self._sys_buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Enumerazione dispositivi
    # ------------------------------------------------------------------
    def find_default_loopback_device(self) -> Optional[AudioSourceInfo]:
        """Trova il device WASAPI loopback associato all'output di default."""
        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            return None  # WASAPI non disponibile (non siamo su Windows)

        default_speakers = self._pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers.get("isLoopbackDevice", False):
            # Su pyaudiowpatch bisogna cercare esplicitamente il device
            # "loopback" gemello del device di output di default.
            for loopback in self._pa.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break

        return AudioSourceInfo(
            index=default_speakers["index"],
            name=default_speakers["name"],
            sample_rate=int(default_speakers["defaultSampleRate"]),
            channels=default_speakers["maxInputChannels"],
        )

    def find_default_microphone(self) -> Optional[AudioSourceInfo]:
        try:
            info = self._pa.get_default_input_device_info()
        except OSError:
            return None
        return AudioSourceInfo(
            index=info["index"],
            name=info["name"],
            sample_rate=int(info["defaultSampleRate"]),
            channels=1,
        )

    # ------------------------------------------------------------------
    # Avvio / arresto registrazione
    # ------------------------------------------------------------------
    def start(self):
        if self._running.is_set():
            return
        self._running.set()

        if self.capture_microphone:
            mic = self.find_default_microphone()
            if mic:
                self._open_stream(mic, self._on_mic_frame)

        if self.capture_system_audio:
            loopback = self.find_default_loopback_device()
            if loopback:
                self._open_stream(loopback, self._on_sys_frame)

        # Thread che periodicamente mixa i buffer e li spedisce in coda.
        mixer_thread = threading.Thread(target=self._mixer_loop, daemon=True)
        mixer_thread.start()
        self._threads.append(mixer_thread)

    def stop(self):
        self._running.clear()
        for stream in self._streams:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        self._streams.clear()

    def close(self):
        self.stop()
        self._pa.terminate()

    # ------------------------------------------------------------------
    # Interni
    # ------------------------------------------------------------------
    def _open_stream(self, source: AudioSourceInfo, callback):
        def _cb(in_data, frame_count, time_info, status):
            try:
                samples = np.frombuffer(in_data, dtype=np.float32)
                if source.channels > 1:
                    samples = samples.reshape(-1, source.channels).mean(axis=1)
                callback(samples, source.sample_rate)
            except Exception as exc:  # pragma: no cover
                if self.on_error:
                    self.on_error(exc)
            return (None, pyaudio.paContinue)

        stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=source.channels,
            rate=source.sample_rate,
            input=True,
            input_device_index=source.index,
            frames_per_buffer=1024,
            stream_callback=_cb,
        )
        stream.start_stream()
        self._streams.append(stream)

    def _resample(self, samples: np.ndarray, orig_rate: int) -> np.ndarray:
        if orig_rate == config.AUDIO_SAMPLE_RATE or samples.size == 0:
            return samples
        duration = samples.size / orig_rate
        target_len = int(duration * config.AUDIO_SAMPLE_RATE)
        if target_len <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.interp(
            np.linspace(0, samples.size, target_len, endpoint=False),
            np.arange(samples.size),
            samples,
        ).astype(np.float32)

    def _on_mic_frame(self, samples: np.ndarray, rate: int):
        samples = self._resample(samples, rate)
        with self._buffer_lock:
            self._mic_buffer = np.concatenate([self._mic_buffer, samples])

    def _on_sys_frame(self, samples: np.ndarray, rate: int):
        samples = self._resample(samples, rate)
        with self._buffer_lock:
            self._sys_buffer = np.concatenate([self._sys_buffer, samples])

    def _mixer_loop(self):
        """Ogni ~0.5s preleva i buffer accumulati, li mixa e li accoda."""
        while self._running.is_set():
            time.sleep(0.5)
            with self._buffer_lock:
                mic = self._mic_buffer
                sys_audio = self._sys_buffer
                self._mic_buffer = np.zeros(0, dtype=np.float32)
                self._sys_buffer = np.zeros(0, dtype=np.float32)

            if mic.size == 0 and sys_audio.size == 0:
                continue

            length = max(mic.size, sys_audio.size)
            if mic.size < length:
                mic = np.pad(mic, (0, length - mic.size))
            if sys_audio.size < length:
                sys_audio = np.pad(sys_audio, (0, length - sys_audio.size))

            mixed = np.clip(mic + sys_audio, -1.0, 1.0)
            if mixed.size:
                self.audio_queue.put(mixed)
