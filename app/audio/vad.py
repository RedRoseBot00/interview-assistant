"""
Rilevamento del parlato e taglio in frasi.

Perche' serve
-------------
Nella versione precedente l'audio veniva spezzato in blocchi di durata
fissa e ogni blocco finiva al riconoscimento vocale, silenzi compresi.
Due conseguenze pesanti: si sprecava tempo di calcolo su lunghi tratti
muti, e una frase compariva solo al termine del blocco che la
conteneva, spesso tagliata a meta'.

Qui l'audio viene osservato in continuo e diviso dove parla la persona:
si accumula finche' c'e' voce, si chiude la frase quando arriva una
pausa. Il riconoscimento riceve cosi' frasi intere.

Il rilevatore usa l'energia del segnale con una soglia che si adatta al
rumore di fondo dell'ambiente, quindi non richiede alcuna regolazione
manuale ne' in un ufficio silenzioso ne' con un microfono dal guadagno
elevato.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

FRAME_MS = 30                  # granularita' dell'analisi
PRE_ROLL_SECONDS = 0.30        # audio conservato prima dell'inizio del parlato
# Una pausa di riflessione a meta' frase e' frequentissima in un
# colloquio. Con mezzo secondo scarso la frase veniva chiusa li': il
# riconoscimento riceveva mezza proposizione, senza contesto per
# completarla, e il risultato era testo spezzato e mal punteggiato.
SILENCE_HANG_SECONDS = 0.75
# "Si'", "no", "ok", "tre anni": in un colloquio le risposte secche
# contano. Con due decimi di secondo venivano buttate via prima di
# arrivare al riconoscimento vocale.
MIN_VOICED_SECONDS = 0.12
# Taglio di sicurezza per chi parla a lungo senza pause nette.
#
# E' un compromesso fra due esigenze opposte. Tagliare di rado spezza
# meno parole a meta' e fa costare meno il riconoscimento (che lavora
# comunque su una finestra di trenta secondi, quindi una frase lunga
# non costa piu' di una corta). Tagliare troppo di rado pero' ritarda
# la comparsa dei sottotitoli: finche' la frase non si chiude, a
# schermo non compare nulla. Dieci secondi tengono la reazione entro
# un tempo accettabile; l'accorpamento a valle rimette insieme i pezzi
# quando c'e' arretrato da smaltire.
MAX_UTTERANCE_SECONDS = 10.0
ABSOLUTE_FLOOR = 0.0025        # soglia minima: sotto e' silenzio in ogni caso
NOISE_MULTIPLIER = 2.8         # quanto la voce deve superare il rumore di fondo
NOISE_WINDOW_SECONDS = 3.0     # finestra su cui si cerca il livello di fondo


@dataclass
class Utterance:
    """Un tratto di parlato, pronto per il riconoscimento vocale."""

    samples: np.ndarray
    start_offset: float           # secondi dall'inizio della registrazione
    duration: float
    continues_previous: bool = False  # nata dal taglio di sicurezza


class VoiceSegmenter:
    """
    Accumula i campioni in arrivo e restituisce frasi complete.

    Lo stato e' volutamente semplice: o stiamo raccogliendo parlato, o
    stiamo aspettando che qualcuno cominci. Il tratto di pre-roll
    conservato durante l'attesa evita di tagliare la prima sillaba, che
    e' quasi sempre quella che il rilevatore nota in ritardo.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.frame_size = max(1, int(sample_rate * FRAME_MS / 1000))
        self.pre_roll_frames = max(1, int(PRE_ROLL_SECONDS * 1000 / FRAME_MS))
        self.hang_frames = max(1, int(SILENCE_HANG_SECONDS * 1000 / FRAME_MS))
        self.max_frames = max(1, int(MAX_UTTERANCE_SECONDS * 1000 / FRAME_MS))
        self.min_voiced_frames = max(1, int(MIN_VOICED_SECONDS * 1000 / FRAME_MS))
        self.noise_window_frames = max(1, int(NOISE_WINDOW_SECONDS * 1000 / FRAME_MS))

        self._pending = np.zeros(0, dtype=np.float32)   # campioni da analizzare
        self._consumed_frames = 0                        # posizione nel flusso

        self._pre_roll: list[np.ndarray] = []
        self._speech: list[np.ndarray] = []
        self._speech_start_frame = 0
        self._silence_run = 0
        self._voiced_frames = 0
        self._in_speech = False
        # Contrassegno "questa frase prosegue la precedente", valido per
        # la durata del pre-roll dopo un taglio per durata massima.
        self._opened_as_continuation = False
        self._continuation_credit = 0

        self._noise_floor = ABSOLUTE_FLOOR
        self._window_min = float("inf")
        self._window_frames = 0

        self.frames_seen = 0
        self.speech_frames = 0

    # ------------------------------------------------------------------
    @property
    def threshold(self) -> float:
        return max(ABSOLUTE_FLOOR, self._noise_floor * NOISE_MULTIPLIER)

    def _update_noise(self, level: float) -> None:
        """
        Stima del rumore di fondo come minimo su una finestra mobile.

        Il minimo osservato in alcuni secondi e' rumore per costruzione:
        nessuno parla cosi' a lungo senza una pausa fra le sillabe.
        Questa misura va aggiornata SEMPRE, anche sui frame giudicati
        voce: aggiornarla solo durante il silenzio significherebbe non
        accorgersi mai di un rumore comparso all'improvviso (una ventola
        che parte, un condizionatore), perche' quel rumore verrebbe
        classificato come voce e la soglia resterebbe bloccata bassa per
        tutto il resto del colloquio.
        """
        self._window_min = min(self._window_min, level)
        self._window_frames += 1
        if self._window_frames < self.noise_window_frames:
            return

        target = self._window_min
        self._window_min = float("inf")
        self._window_frames = 0

        if target < self._noise_floor:
            self._noise_floor = 0.7 * self._noise_floor + 0.3 * target
        else:
            self._noise_floor = 0.9 * self._noise_floor + 0.1 * target

    # ------------------------------------------------------------------
    def feed(self, samples: np.ndarray) -> list[Utterance]:
        """Aggiunge campioni e restituisce le frasi eventualmente concluse."""
        if samples.size:
            self._pending = np.concatenate([self._pending, samples])

        finished: list[Utterance] = []

        while self._pending.size >= self.frame_size:
            frame = self._pending[: self.frame_size]
            self._pending = self._pending[self.frame_size :]
            self.frames_seen += 1

            level = float(np.sqrt(np.dot(frame, frame) / frame.size))
            is_speech = level > self.threshold
            self._update_noise(level)
            if is_speech:
                self.speech_frames += 1

            if self._in_speech:
                self._speech.append(frame)
                if is_speech:
                    self._voiced_frames += 1
                    self._silence_run = 0
                else:
                    self._silence_run += 1

                too_long = len(self._speech) >= self.max_frames
                if self._silence_run >= self.hang_frames or too_long:
                    utterance = self._close(force=too_long)
                    if utterance is not None:
                        finished.append(utterance)
            else:
                self._pre_roll.append(frame)
                if len(self._pre_roll) > self.pre_roll_frames:
                    self._pre_roll.pop(0)
                # Il legame con la frase troncata vale solo per il tempo
                # del pre-roll: passato quello, e' una frase nuova.
                if self._continuation_credit > 0:
                    self._continuation_credit -= 1

                if is_speech:
                    self._in_speech = True
                    self._silence_run = 0
                    self._voiced_frames = 1
                    self._opened_as_continuation = self._continuation_credit > 0
                    self._continuation_credit = 0
                    self._speech = list(self._pre_roll)
                    self._speech_start_frame = (
                        self._consumed_frames + 1 - len(self._speech)
                    )
                    self._pre_roll = []

            self._consumed_frames += 1

        return finished

    # ------------------------------------------------------------------
    def _close(self, force: bool = False) -> Utterance | None:
        frames = self._speech
        voiced = self._voiced_frames
        continues = self._opened_as_continuation

        self._speech = []
        self._voiced_frames = 0
        self._in_speech = False
        self._silence_run = 0
        self._opened_as_continuation = False
        # Se il taglio e' avvenuto per durata massima il parlato
        # prosegue: la frase successiva, se comincia subito, va
        # segnalata come suo seguito.
        self._continuation_credit = self.pre_roll_frames if force else 0

        if not frames:
            return None

        if force:
            # Conserviamo la coda come pre-roll della frase seguente,
            # per non perdere le parole a cavallo del taglio.
            self._pre_roll = frames[-self.pre_roll_frames :]
        else:
            self._pre_roll = []

        # Conta solo la voce effettiva: il pre-roll e la pausa finale
        # fanno parte di ogni frase e da soli supererebbero qualunque
        # durata minima, rendendo il controllo inutile. Senza questo,
        # un clic del mouse o un colpo di tosse diventa una chiamata al
        # riconoscimento vocale.
        if voiced < self.min_voiced_frames:
            return None

        samples = np.concatenate(frames)
        return Utterance(
            samples=samples.astype(np.float32, copy=False),
            start_offset=self._speech_start_frame * self.frame_size / self.sample_rate,
            duration=samples.size / self.sample_rate,
            continues_previous=continues,
        )

    def flush(self) -> list[Utterance]:
        """
        Chiude quanto resta in sospeso: da usare al termine del
        colloquio, per non perdere l'ultima frase.
        """
        finished: list[Utterance] = []

        if self._pending.size:
            padding = self.frame_size - (self._pending.size % self.frame_size)
            # Completiamo l'ultimo frame con silenzio, altrimenti
            # resterebbe inanalizzato.
            finished.extend(self.feed(np.zeros(padding, dtype=np.float32)))

        if self._in_speech and self._speech:
            last = self._close()
            if last is not None:
                finished.append(last)
        return finished

    # ------------------------------------------------------------------
    @property
    def settled_seconds(self) -> float:
        """
        Istante fino al quale l'analisi e' definitiva.

        Se una frase e' in corso, tutto cio' che la segue e' ancora da
        emettere; altrimenti abbiamo esaminato tutto il flusso. Serve
        all'altro canale per sapere che qui, nel frattempo, c'era
        silenzio: anche l'assenza di suono e' un'informazione utile.
        """
        frame = self._speech_start_frame if self._in_speech else self._consumed_frames
        return frame * self.frame_size / self.sample_rate

    @property
    def speech_ratio(self) -> float:
        """Quota di tempo in cui e' stata rilevata voce: utile nei log."""
        if not self.frames_seen:
            return 0.0
        return self.speech_frames / self.frames_seen
