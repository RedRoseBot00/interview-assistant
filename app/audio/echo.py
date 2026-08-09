"""
Gestione dell'eco acustica quando il selezionatore non indossa le cuffie.

Il problema
-----------
Senza cuffie la voce del candidato esce dagli altoparlanti, viaggia
nell'aria e rientra nel microfono con un ritardo di qualche centesimo
di secondo. Il risultato, nella trascrizione, e' la stessa frase
attribuita a tutti e due gli interlocutori.

Perche' qui e' piu' facile del solito
-------------------------------------
Un normale cancellatore d'eco deve indovinare quale parte del segnale
sia eco. Noi invece possediamo il segnale di riferimento esatto: e' il
flusso audio che stiamo gia' registrando dal canale della
videochiamata, cioe' precisamente cio' che gli altoparlanti stanno
riproducendo. Il confronto e' quindi diretto.

Due livelli di intervento
-------------------------
1. RILEVAMENTO (sempre attivo quando la funzione e' abilitata)
   Si misura quanto il segnale del microfono somiglia a una copia
   ritardata dell'audio riprodotto. Se la somiglianza e' netta, il
   blocco viene scartato prima di arrivare al riconoscimento vocale:
   niente frasi doppie e, in piu', processore risparmiato.

2. CANCELLAZIONE (facoltativa, piu' esigente in termini di calcolo)
   Invece di scartare il blocco, si stima il filtro che trasforma
   l'audio riprodotto nell'eco registrata e lo si sottrae dal
   microfono. Cio' che resta e' la voce del selezionatore ripulita:
   serve nei momenti in cui i due parlano insieme, dove scartare il
   blocco significherebbe perdere l'interruzione.

   Il filtro viene stimato una volta per blocco risolvendo le equazioni
   normali di Wiener (autocorrelazione e correlazione incrociata
   calcolate con la trasformata di Fourier). E' piu' stabile di un
   filtro adattivo campione per campione e abbastanza rapido da girare
   comodamente in tempo reale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Modalita' selezionabili dall'utente
MODE_OFF = "off"          # nessun intervento (corretto quando si usano le cuffie)
MODE_AUTO = "auto"        # rilevamento e scarto dei blocchi di eco
MODE_CANCEL = "cancel"    # cancellazione vera e propria con filtro

# Ritardo massimo atteso fra riproduzione e ripresa dal microfono:
# comprende la latenza delle schede audio, il tempo di volo del suono
# nella stanza e i buffer del sistema operativo.
MAX_DELAY_SECONDS = 0.45
# Piccolo margine in avanti: i due flussi audio sono catturati da thread
# distinti e possono risultare disallineati di qualche millisecondo nel
# verso opposto a quello fisico.
FORWARD_MARGIN_SECONDS = 0.10

# Lunghezza del filtro di cancellazione: 32 ms coprono la coda di
# riverbero di una stanza normale. Allungarlo migliora poco e costa
# molto: la soluzione del sistema cresce con il cubo del numero di
# coefficienti.
FILTER_TAPS = 512

# Riconoscimento dell'uso delle cuffie.
#
# Con le cuffie la voce del candidato non passa dall'aria e non torna
# mai nel microfono: l'analisi dell'eco gira allora a vuoto su ogni
# singola frase, per tutta la durata del colloquio, spendendo tempo di
# calcolo che sul computer del cliente serve alla trascrizione. Dopo un
# certo numero di frasi senza la minima somiglianza smettiamo, e ci
# limitiamo a un controllo saltuario per accorgerci se le cuffie
# vengono sfilate a meta' colloquio.
HEADPHONES_QUIET_UTTERANCES = 15
# Fattore di riduzione usato per il controllo rapido durante la
# sospensione: a un quarto della frequenza di campionamento la ricerca
# costa circa un quinto e resta piu' che sufficiente per rispondere
# alla sola domanda "c'e' o non c'e' eco".
PROBE_DECIMATION = 4
# Sospetto: sopra questo valore nel controllo rapido si torna
# all'analisi completa. Volutamente piu' basso della soglia vera, per
# non rischiare di restare sospesi mentre l'eco e' tornata.
PROBE_CORRELATION = 0.15
# Sotto questo livello il riferimento e' silenzio: da un confronto con
# il silenzio non si impara nulla, quindi non fa testo.
REFERENCE_ACTIVE_RMS = 0.005

# Soglie della modalita' "bilanciata": si scarta solo quando la
# somiglianza e' netta, per non perdere gli interventi reali.
CORRELATION_ECHO = 0.50          # oltre questa soglia il blocco e' considerato eco
CORRELATION_MIN_INTEREST = 0.22  # sotto questa soglia non vale la pena elaborare
CORRELATION_CANCEL_FLOOR = 0.06  # in modalita' avanzata si tenta anche con poca eco
RESIDUAL_ENERGY_RATIO = 0.28     # dopo la cancellazione: quanto deve restare per essere voce vera
RESIDUAL_RMS_FLOOR = 0.004       # residuo cosi' debole da essere silenzio
PROMINENCE_MIN = 4.0             # quanto il massimo deve spiccare per fidarsi del ritardo
NO_IMPROVEMENT_RATIO = 0.97      # oltre questa soglia la sottrazione non ha giovato

# Regolarizzazione del filtro di cancellazione. Il valore effettivo
# cresce quando l'eco e' debole rispetto alla voce, per non rovinare
# quest'ultima: i valori sono stati scelti misurando il danno prodotto
# su segnali privi di eco (vedi commento in estimate_echo_filter).
BASE_REGULARISATION = 1e-3
ADAPTIVE_REGULARISATION = 0.10


@dataclass
class EchoResult:
    """Esito dell'analisi di un blocco del microfono."""

    samples: np.ndarray      # segnale da trascrivere (eventualmente ripulito)
    is_echo: bool            # True se il blocco va scartato
    correlation: float       # somiglianza con l'audio riprodotto (0-1)
    delay_seconds: float     # ritardo stimato dell'eco
    cancelled: bool = False  # True se il filtro di cancellazione e' stato applicato
    attenuation_db: float = 0.0  # quanta eco e' stata tolta


class ReferenceBuffer:
    """
    Conserva l'audio riprodotto dal computer, indicizzato per istante.

    Serve a ritrovare, per un dato blocco del microfono, il tratto di
    audio che gli altoparlanti stavano riproducendo nello stesso
    momento. I blocchi in arrivo si sovrappongono leggermente fra loro,
    quindi vengono scritti nella loro posizione assoluta invece di
    essere semplicemente accodati.
    """

    def __init__(self, sample_rate: int, max_seconds: float = 40.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self._data = np.zeros(0, dtype=np.float32)
        self._base_offset = 0.0
        self._known_until = 0.0
        self.active = False

    def note_silence(self, until: float) -> None:
        """
        Registra che il canale e' stato analizzato fino a un certo
        istante senza che nessuno parlasse.

        Anche l'assenza di suono e' un riferimento valido: senza questa
        informazione, ogni frase del selezionatore pronunciata mentre il
        candidato tace resterebbe in attesa di un riferimento che non
        arrivera' mai.
        """
        if until > self._known_until:
            self._known_until = until
            self.active = True

    def append(self, offset: float, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self.active = True

        if self._data.size == 0:
            self._base_offset = offset
            self._data = np.asarray(samples, dtype=np.float32).copy()
            return

        start = int(round((offset - self._base_offset) * self.sample_rate))
        if start < 0:
            # Blocco piu' vecchio dell'inizio del buffer: lo anteponiamo.
            pad = np.zeros(-start, dtype=np.float32)
            self._data = np.concatenate([pad, self._data])
            self._base_offset = offset
            start = 0

        end = start + samples.size
        if end > self._data.size:
            self._data = np.concatenate(
                [self._data, np.zeros(end - self._data.size, dtype=np.float32)]
            )
        # La scrittura per posizione assoluta gestisce da sola la
        # sovrapposizione fra blocchi consecutivi.
        self._data[start:end] = samples

        if self._data.size > self.max_samples:
            excess = self._data.size - self.max_samples
            self._data = self._data[excess:]
            self._base_offset += excess / self.sample_rate

    @property
    def end_offset(self) -> float:
        if self._data.size == 0:
            return self._base_offset
        return self._base_offset + self._data.size / self.sample_rate

    def covers(self, until: float) -> bool:
        fine = self.end_offset if self._data.size else 0.0
        return max(fine, self._known_until) >= until

    def segment(self, start: float, duration: float) -> tuple[np.ndarray, int] | None:
        """
        Tratto di riferimento che puo' aver generato l'eco presente nel
        blocco del microfono.

        L'eco arriva in ritardo, quindi l'inizio del blocco contiene
        l'eco di suoni riprodotti PRIMA del blocco stesso: il tratto
        restituito parte percio' un po' indietro nel tempo.

        Restituisce il segnale e il numero di campioni che precedono
        l'inizio del blocco del microfono: serve a ritrovare
        l'allineamento esatto.
        """
        if self._data.size == 0:
            return None

        begin = start - MAX_DELAY_SECONDS
        finish = start + duration + FORWARD_MARGIN_SECONDS

        first = int(round((begin - self._base_offset) * self.sample_rate))
        last = int(round((finish - self._base_offset) * self.sample_rate))
        if first < 0:
            first = 0
        if last <= first:
            return None

        chunk = self._data[first : min(last, self._data.size)]
        if chunk.size < self.sample_rate // 10:  # meno di 100 ms: inutilizzabile
            return None

        start_index = int(round((start - self._base_offset) * self.sample_rate))
        pre_samples = max(0, start_index - first)
        return chunk.copy(), pre_samples


# --------------------------------------------------------------------------
# Analisi del segnale
# --------------------------------------------------------------------------
def _slice_padded(array: np.ndarray, start: int, length: int) -> np.ndarray:
    """Estrae un tratto completando con zeri cio' che cade fuori dai bordi."""
    result = np.zeros(length, dtype=np.float32)
    source_start = max(0, start)
    source_end = min(array.size, start + length)
    if source_end > source_start:
        result[source_start - start : source_end - start] = array[
            source_start:source_end
        ]
    return result


def _normalised_cross_correlation(
    mic: np.ndarray,
    reference: np.ndarray,
    pre_samples: int,
    max_lag_samples: int | None = None,
    max_lead_samples: int = 0,
) -> tuple[float, int, float]:
    """
    Somiglianza massima fra microfono e riferimento al variare del
    ritardo, e ritardo corrispondente in campioni.

    Il microfono viene fatto precedere da tanti zeri quanti sono i
    campioni di riferimento anteriori al blocco: cosi' i due segnali
    partono dallo stesso istante e il ritardo cercato risulta sempre
    positivo, il che rende la ricerca del massimo semplice e sicura.

    Il calcolo passa dalla trasformata di Fourier: farlo direttamente
    costerebbe centinaia di milioni di operazioni per ogni blocco.
    """
    if mic.size == 0 or reference.size == 0:
        return 0.0, 0, 0.0

    mic = mic - float(np.mean(mic))
    reference = reference - float(np.mean(reference))

    mic_energy = float(np.sqrt(np.sum(mic**2)))
    ref_energy = float(np.sqrt(np.sum(reference**2)))
    if mic_energy < 1e-8 or ref_energy < 1e-8:
        return 0.0, 0, 0.0

    padded = np.concatenate([np.zeros(pre_samples, dtype=np.float32), mic])

    size = 1
    while size < padded.size + reference.size:
        size *= 2

    spectrum_ref = np.fft.rfft(reference, size)
    spectrum_mic = np.fft.rfft(padded, size)
    correlation = np.fft.irfft(spectrum_mic * np.conj(spectrum_ref), size)

    max_lag = min(correlation.size - 1, reference.size - 1)
    if max_lag_samples is not None:
        max_lag = min(max_lag, max(1, max_lag_samples))

    # Ritardi POSITIVI: l'eco segue la riproduzione, e' il caso fisico.
    finestra = [correlation[: max_lag + 1]]
    # Ritardi NEGATIVI: fisicamente impossibili, ma i due flussi audio
    # sono catturati da thread distinti e un blocco di lettura perso su
    # un canale sposta indietro la sua base tempi. Bastano tre blocchi
    # (settanta millisecondi) perche' il ritardo apparente diventi
    # negativo: cercando solo in avanti, da quel momento l'eco non
    # veniva piu' riconosciuta per tutto il resto del colloquio, e ogni
    # domanda compariva due volte nella trascrizione.
    lead = min(int(max_lead_samples), correlation.size - max_lag - 1)
    if lead > 0:
        finestra.append(correlation[-lead:])
    window = np.concatenate(finestra)
    if window.size == 0:
        return 0.0, 0, 0.0

    magnitudes = np.abs(window)
    best = int(np.argmax(magnitudes))
    if best > max_lag:
        # Indice nella coda dell'array: corrisponde a un ritardo negativo.
        best = best - window.size
    value = float(magnitudes[np.argmax(magnitudes)]) / (mic_energy * ref_energy)

    # Quanto il massimo spicca sul resto della curva. Serve a capire se
    # il ritardo trovato e' un'informazione reale o solo il punto piu'
    # alto di una curva piatta: in quest'ultimo caso allinearsi su di
    # esso peggiorerebbe il segnale invece di ripulirlo.
    background = float(np.median(magnitudes)) if magnitudes.size else 0.0
    prominence = float(magnitudes[best] / background) if background > 1e-12 else 99.0

    return min(1.0, value), best, prominence


def _align(
    reference: np.ndarray, lag: int, pre_samples: int, length: int
) -> np.ndarray:
    """
    Tratto di riferimento che ha generato l'eco presente nel blocco.

    Il ritardo misurato indica di quanto l'eco segue la riproduzione:
    per sovrapporre i due segnali bisogna quindi arretrare nel
    riferimento, non avanzare.
    """
    return _slice_padded(reference, pre_samples - lag, length)


def estimate_echo_filter(
    mic: np.ndarray,
    reference: np.ndarray,
    taps: int | None = None,
    regularisation: float = BASE_REGULARISATION,
) -> np.ndarray:
    """
    Filtro che trasforma l'audio riprodotto nell'eco registrata.

    Si risolvono le equazioni normali di Wiener: l'autocorrelazione del
    riferimento forma una matrice di Toeplitz, il termine noto e' la
    correlazione incrociata con il microfono.

    Il termine di regolarizzazione sulla diagonale merita una nota. Non
    serve solo alla stabilita' numerica: e' cio' che impedisce al filtro
    di "spiegare" con l'eco anche pezzi della voce del selezionatore,
    quando le due voci si somigliano. Piu' l'eco e' debole rispetto al
    parlato, piu' il valore va alzato, cosi' il filtro si avvicina a
    zero e nel dubbio lascia il segnale com'e'.
    """
    # Il valore va letto qui e non nella firma della funzione: un
    # parametro predefinito verrebbe fissato al momento dell'importazione
    # e non rispecchierebbe piu' eventuali regolazioni successive.
    taps = min(taps or FILTER_TAPS, max(16, mic.size // 4))

    size = 1
    while size < mic.size + taps:
        size *= 2

    spectrum_ref = np.fft.rfft(reference, size)
    spectrum_mic = np.fft.rfft(mic, size)

    autocorr = np.fft.irfft(spectrum_ref * np.conj(spectrum_ref), size)[:taps]
    crosscorr = np.fft.irfft(spectrum_mic * np.conj(spectrum_ref), size)[:taps]

    if autocorr[0] <= 1e-12:
        return np.zeros(taps, dtype=np.float32)

    indices = np.abs(np.subtract.outer(np.arange(taps), np.arange(taps)))
    matrix = autocorr[indices]
    matrix[np.diag_indices(taps)] += regularisation * autocorr[0] + 1e-12

    try:
        weights = np.linalg.solve(matrix, crosscorr)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(matrix, crosscorr, rcond=None)[0]
    return weights.astype(np.float32)


def _apply_filter(reference: np.ndarray, weights: np.ndarray, length: int) -> np.ndarray:
    size = 1
    while size < reference.size + weights.size:
        size *= 2
    convolved = np.fft.irfft(
        np.fft.rfft(reference, size) * np.fft.rfft(weights, size), size
    )
    return convolved[:length].astype(np.float32)


class EchoProcessor:
    """Applica rilevamento e, se richiesto, cancellazione dell'eco."""

    def __init__(self, mode: str = MODE_AUTO, sample_rate: int = 16000):
        self.mode = mode
        self.sample_rate = sample_rate
        self.detections = 0
        self.processed = 0
        self.last_correlation = 0.0
        # Ritardo misurato nei momenti in cui l'eco era chiaramente
        # riconoscibile. Il percorso del suono nella stanza non cambia
        # durante un colloquio, quindi questo valore resta valido e ci
        # permette di ripulire anche i blocchi in cui l'eco e' coperta
        # dalla voce del selezionatore e la misura sarebbe inaffidabile.
        self._stable_lag: int | None = None
        # Stato del riconoscimento "cuffie": vedi le costanti in cima.
        self._quiet_streak = 0
        self._sleeping = False

    @property
    def enabled(self) -> bool:
        return self.mode in (MODE_AUTO, MODE_CANCEL)

    @property
    def dormant(self) -> bool:
        """True quando l'analisi e' sospesa perche' non si rileva eco."""
        return self._sleeping

    def _note_quiet(self, reference_active: bool) -> None:
        if not reference_active:
            return
        self._quiet_streak += 1
        if not self._sleeping and self._quiet_streak >= HEADPHONES_QUIET_UTTERANCES:
            self._sleeping = True
            log.info(
                "Nessuna eco rilevata in %d frasi: sembra che si stiano usando "
                "le cuffie. Passo al controllo rapido.",
                self._quiet_streak,
            )

    def _note_active(self) -> None:
        self._quiet_streak = 0
        if self._sleeping:
            self._sleeping = False
            log.info("Eco nuovamente presente: riprendo l'analisi completa.")

    def _probably_echoing(
        self, mic: np.ndarray, reference: np.ndarray, pre_samples: int
    ) -> bool:
        """
        Controllo rapido: c'e' motivo di fare l'analisi completa?

        Un semplice contatore di frasi da saltare sarebbe stato piu'
        economico, ma avrebbe lasciato passare decine di frasi di eco se
        l'utente si sfila le cuffie a meta' colloquio. Qui invece ogni
        frase viene comunque esaminata, solo a risoluzione ridotta.
        """
        passo = PROBE_DECIMATION
        if mic.size < passo * 8 or reference.size < passo * 8:
            return True

        def riduci(x: np.ndarray) -> np.ndarray:
            usable = (x.size // passo) * passo
            # La media sui campioni scartati fa anche da filtro
            # anti-aliasing: senza, la riduzione introdurrebbe un
            # rumore che falserebbe il confronto.
            return x[:usable].reshape(-1, passo).mean(axis=1).astype(np.float32)

        correlation, _lag, _prom = _normalised_cross_correlation(
            riduci(mic), riduci(reference), pre_samples // passo, None
        )
        return correlation >= PROBE_CORRELATION

    @property
    def speakers_detected(self) -> bool:
        """
        True quando i dati raccolti indicano che si sta usando gli
        altoparlanti invece delle cuffie.

        Serve a mostrare un'indicazione onesta nell'interfaccia: con le
        cuffie questa condizione non si verifica mai, perche' la voce
        del candidato non passa dall'aria.
        """
        return self.detections >= 2

    def process(
        self, mic: np.ndarray, reference: np.ndarray | None, pre_samples: int = 0
    ) -> EchoResult:
        if not self.enabled or reference is None or reference.size == 0:
            return EchoResult(mic, False, 0.0, 0.0)

        # Analisi ridotta (cuffie): l'esame completo su una frase lunga
        # costa quanto un decimo della trascrizione, e ripeterlo per
        # tutto il colloquio senza mai trovare nulla e' tempo tolto ai
        # sottotitoli.
        if self._sleeping and not self._probably_echoing(mic, reference, pre_samples):
            return EchoResult(mic, False, 0.0, 0.0)

        reference_active = (
            float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
            >= REFERENCE_ACTIVE_RMS
        )

        self.processed += 1
        # Il ritardo dell'eco non puo' superare i limiti fisici: cercare
        # oltre non serve e apre la porta a somiglianze casuali che
        # farebbero scartare frasi vere del selezionatore. All'indietro
        # invece un margine serve davvero, per assorbire lo scarto fra
        # le basi tempi dei due canali (vedi il commento nella ricerca
        # del massimo): prima quel margine veniva sommato per errore al
        # limite in avanti, dove non serviva a nulla.
        correlation, lag, prominence = _normalised_cross_correlation(
            mic,
            reference,
            pre_samples,
            max_lag_samples=pre_samples,
            max_lead_samples=int(FORWARD_MARGIN_SECONDS * self.sample_rate),
        )
        self.last_correlation = correlation

        reliable = correlation >= CORRELATION_ECHO and prominence >= PROMINENCE_MIN
        if reliable:
            self._stable_lag = lag
        elif self._stable_lag is not None:
            # Misura poco attendibile perche' la voce del selezionatore
            # copre l'eco: usiamo il ritardo appreso nei momenti in cui
            # l'eco era chiara, invece di inseguire un massimo casuale.
            lag = self._stable_lag

        delay = lag / self.sample_rate

        # In modalita' avanzata conviene tentare la sottrazione anche
        # quando la somiglianza e' modesta: e' esattamente il caso in cui
        # i due parlano insieme e la voce del selezionatore "diluisce"
        # l'eco. Se eco non ce n'e', il filtro stimato risulta prossimo a
        # zero e il segnale resta intatto.
        floor = (
            CORRELATION_CANCEL_FLOOR
            if self.mode == MODE_CANCEL
            else CORRELATION_MIN_INTEREST
        )
        if correlation < CORRELATION_MIN_INTEREST:
            self._note_quiet(reference_active)
        else:
            self._note_active()

        if correlation < floor:
            # Nessuna somiglianza: il microfono sta registrando altro,
            # quasi certamente la voce del selezionatore.
            return EchoResult(mic, False, correlation, delay)

        if self.mode == MODE_AUTO:
            # Non basta che la somiglianza sia alta: deve anche esserci
            # un massimo netto a un ritardo preciso. Su una curva piatta
            # un valore alto e' una coincidenza, e scartare il blocco
            # farebbe sparire dalla trascrizione una domanda vera.
            is_echo = correlation >= CORRELATION_ECHO and prominence >= PROMINENCE_MIN
            if is_echo:
                self.detections += 1
            return EchoResult(mic, is_echo, correlation, delay)

        # --- modalita' cancellazione ------------------------------------
        aligned = _align(reference, lag, pre_samples, mic.size)
        # Quanta parte del microfono e' spiegabile con l'eco: e' la
        # somiglianza al quadrato. Il resto e' voce vera, e va protetta.
        echo_share = max(correlation * correlation, 0.01)
        regularisation = BASE_REGULARISATION + ADAPTIVE_REGULARISATION * (
            1.0 - echo_share
        ) / echo_share
        weights = estimate_echo_filter(mic, aligned, regularisation=regularisation)
        estimated_echo = _apply_filter(aligned, weights, mic.size)
        residual = mic - estimated_echo

        original_energy = float(np.sum(mic**2))
        residual_energy = float(np.sum(residual**2))
        if original_energy <= 1e-12:
            return EchoResult(mic, True, correlation, delay)

        ratio = residual_energy / original_energy

        # Garanzia di non peggiorare: se dopo la sottrazione il segnale
        # non e' piu' pulito di prima, la stima era sbagliata e teniamo
        # l'originale. Meglio un po' di eco che una voce rovinata.
        if ratio >= NO_IMPROVEMENT_RATIO:
            log.debug(
                "Cancellazione senza beneficio (residuo %.0f%%): tengo l'originale",
                ratio * 100,
            )
            return EchoResult(mic, False, correlation, delay)

        attenuation = -10.0 * np.log10(max(ratio, 1e-6))
        residual_rms = float(np.sqrt(residual_energy / max(1, residual.size)))
        # Se dopo la sottrazione resta pochissimo, il blocco conteneva
        # solo eco; altrimenti cio' che resta e' voce vera e va
        # trascritta, ora ripulita dall'eco.
        #
        # Scartare l'intero blocco e' pero' lecito solo se l'eco era
        # davvero riconoscibile: quando i due parlano insieme la
        # sottrazione toglie inevitabilmente troppo, e senza questa
        # condizione la domanda del selezionatore sparirebbe proprio
        # nei momenti piu' vivaci del colloquio.
        eco_riconoscibile = correlation >= CORRELATION_ECHO
        is_echo = eco_riconoscibile and (
            ratio < RESIDUAL_ENERGY_RATIO or residual_rms < RESIDUAL_RMS_FLOOR
        )
        if is_echo:
            self.detections += 1
            return EchoResult(mic, True, correlation, delay, True, attenuation)

        if correlation >= CORRELATION_ECHO:
            self.detections += 1

        return EchoResult(
            residual.astype(np.float32), False, correlation, delay, True, attenuation
        )


# --------------------------------------------------------------------------
# Rete di sicurezza sul testo
# --------------------------------------------------------------------------
def texts_are_duplicate(first: str, second: str, threshold: float = 0.78) -> bool:
    """
    Verifica se due trascrizioni dicono sostanzialmente la stessa cosa.

    Ultimo controllo dopo l'elaborazione del segnale: se una frase
    compare su entrambi i canali a pochi secondi di distanza, quella
    del microfono e' quasi certamente l'eco di quella della
    videochiamata.
    """
    import difflib

    a = " ".join(first.lower().split())
    b = " ".join(second.lower().split())
    if not a or not b:
        return False
    if a == b:
        return True
    # Frasi molto brevi ("si", "certo") si somigliano per caso: meglio
    # non considerarle duplicati.
    if min(len(a), len(b)) < 12:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold
