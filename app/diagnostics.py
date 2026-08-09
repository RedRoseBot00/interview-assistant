"""
Verifica di avvio del motore di trascrizione, eseguita in un processo
separato.

Perche' un processo separato: se le librerie di calcolo incontrano una
istruzione non supportata dalla CPU, il processo viene terminato dal
sistema operativo senza possibilita' di intercettare l'errore. Se questo
accadesse nel processo principale, l'utente vedrebbe la finestra sparire
di colpo. Facendolo in un processo figlio, invece, il programma
principale sopravvive, legge il codice di uscita e reagisce: attiva la
modalita' compatibilita' e riprova.

Il test viene eseguito una sola volta e l'esito e' memorizzato nelle
impostazioni, quindi non rallenta gli avvii successivi.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from app import compat, settings

StopFn = Optional[Callable[[], bool]]

log = logging.getLogger(__name__)

# Tempo massimo del test, per dimensione del modello.
#
# Un valore unico era sbagliato: 'medium' pesa dieci volte 'base' e,
# su un portatile con disco lento, caricarlo e farlo girare una volta
# puo' superare abbondantemente i tre minuti. Il test scadeva, e
# all'utente veniva detto che il motore "non riesce ad avviarsi su
# questo computer" — con le funzioni di registrazione disattivate — per
# il solo fatto di aver scelto un modello piu' grande.
SELFTEST_TIMEOUT_SECONDS = {
    "tiny": 180,
    "base": 240,
    "small": 420,
    "medium": 900,
}
SELFTEST_TIMEOUT_DEFAULT = 300

# Codici usati internamente per distinguere i modi di fallire.
EXIT_MODEL_MISSING = 3
EXIT_CANCELLED = -97
EXIT_LAUNCH_FAILED = -98
EXIT_TIMEOUT = -99

_MOTIVI = {
    EXIT_TIMEOUT: (
        "il controllo preventivo ha impiegato troppo tempo su questo "
        "computer (il modello scelto e' pesante)"
    ),
    EXIT_MODEL_MISSING: (
        "il modello di trascrizione scelto non risulta ancora scaricato "
        "per intero"
    ),
    EXIT_LAUNCH_FAILED: "non e' stato possibile avviare il controllo preventivo",
}

# Codici di uscita che indicano una terminazione da parte del sistema
# operativo, non un errore del programma. Solo questi giustificano il
# passaggio alla modalita' compatibilita': un modello mancante o una
# eccezione Python non hanno nulla a che vedere con le istruzioni della
# CPU, e ripiegare su kernel generici renderebbe l'app cinque volte
# piu' lenta per sempre, senza risolvere nulla.
_STATUS_ILLEGAL_INSTRUCTION = 0xC000001D
_STATUS_ACCESS_VIOLATION = 0xC0000005
_STATUS_PRIVILEGED_INSTRUCTION = 0xC0000096


def _is_native_crash(code: int) -> bool:
    """True se il processo figlio e' stato ucciso dal sistema operativo."""
    if code == 0:
        return False
    if sys.platform == "win32":
        # Windows restituisce il codice come intero con segno: lo
        # riportiamo alla forma a 32 bit senza segno prima di confrontarlo.
        unsigned = code & 0xFFFFFFFF
        return unsigned in (
            _STATUS_ILLEGAL_INSTRUCTION,
            _STATUS_ACCESS_VIOLATION,
            _STATUS_PRIVILEGED_INSTRUCTION,
        )
    # Su Unix un codice negativo e' il numero del segnale ricevuto
    # (SIGILL = 4, SIGSEGV = 11).
    return code in (-4, -11)


@dataclass
class SelfTestResult:
    ok: bool
    compatible_mode: bool     # True se serve la modalita' compatibilita'
    detail: str
    # Vero quando il test non ha potuto concludersi (tempo scaduto,
    # modello non ancora scaricato, impossibile avviare il processo).
    # NON e' la stessa cosa di "il motore non funziona": in questi casi
    # la registrazione va lasciata disponibile, perche' il piu' delle
    # volte funziona benissimo — e' solo il controllo preventivo a non
    # aver dato una risposta.
    inconclusive: bool = False
    reason: str = ""

    @property
    def state(self) -> str:
        if self.inconclusive:
            return ""
        if not self.ok:
            return "failed"
        return "ok-compatible" if self.compatible_mode else "ok"


def _child_command(mode: str) -> list[str]:
    """
    Comando per lanciare una copia di questo programma in modalita' test.

    Funziona sia da sorgente (python run.py ...) sia dall'eseguibile
    compilato con PyInstaller, dove sys.executable e' l'app stessa.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--self-test", mode]
    entry = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run.py")
    return [sys.executable, entry, "--self-test", mode]


def _run_child(
    force_generic: bool, whisper_size: str, should_stop: StopFn = None
) -> tuple[int, str]:
    env = dict(os.environ)
    if force_generic:
        env["CT2_FORCE_CPU_ISA"] = "GENERIC"
    else:
        env.pop("CT2_FORCE_CPU_ISA", None)
    env["INTERVIEW_ASSISTANT_SELFTEST_MODEL"] = whisper_size

    creationflags = 0
    if sys.platform == "win32":
        # Evita che compaia una finestra nera del prompt dei comandi.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # Il test dura fino a tre minuti, e va ripetuto due volte. Con una
    # attesa non interrompibile, chi chiudeva la finestra durante il
    # primo avvio restava bloccato per tutto quel tempo; il programma
    # arrivava a uccidere il proprio thread, e uccidere un thread Python
    # mentre tiene il blocco globale lascia la finestra a schermo per
    # sempre. Qui invece si controlla ogni decimo di secondo.
    try:
        proc = subprocess.Popen(
            _child_command("transcription"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Senza errors="replace", un output non decodificabile con la
            # codifica locale di Windows farebbe perdere proprio il
            # messaggio diagnostico che stiamo cercando.
            errors="replace",
            creationflags=creationflags,
        )
    except Exception as exc:  # pragma: no cover
        return EXIT_LAUNCH_FAILED, f"Impossibile eseguire il test di avvio: {exc}"

    # L'uscita del figlio va SVUOTATA di continuo, in un thread a parte.
    # Sorvegliare il processo senza leggere il tubo sembra innocuo, ma
    # su Windows quel tubo tiene solo poche decine di kilobyte: quando
    # si riempie il figlio si blocca in scrittura, non termina mai, e il
    # test scade — facendo dichiarare guasto un motore perfettamente
    # funzionante, solo perche' era stato loquace.
    raccolto: list[str] = []

    def _svuota() -> None:
        try:
            if proc.stdout is not None:
                for riga in proc.stdout:
                    raccolto.append(riga)
                    if len(raccolto) > 400:
                        del raccolto[:200]
        except Exception:
            log.debug("Lettura dell'uscita del test non riuscita", exc_info=True)

    lettore = threading.Thread(target=_svuota, name="selftest-out", daemon=True)
    lettore.start()

    massimo = SELFTEST_TIMEOUT_SECONDS.get(whisper_size, SELFTEST_TIMEOUT_DEFAULT)
    scadenza = time.monotonic() + massimo
    while proc.poll() is None:
        if should_stop is not None and should_stop():
            proc.kill()
            lettore.join(timeout=5)
            return EXIT_CANCELLED, "Test di avvio interrotto su richiesta."
        if time.monotonic() > scadenza:
            proc.kill()
            lettore.join(timeout=5)
            return EXIT_TIMEOUT, (
                f"Il test non si e' concluso entro {massimo} secondi. "
                + "".join(raccolto[-40:]).strip()
            )
        time.sleep(0.1)

    lettore.join(timeout=15)
    return proc.returncode, "".join(raccolto).strip()


def run_transcription_selftest(
    whisper_size: str, should_stop: StopFn = None
) -> SelfTestResult:
    """
    Prova a caricare il motore di trascrizione, prima in modalita'
    veloce e — se il processo figlio muore — in modalita' compatibilita'.
    """
    mode = settings.get("cpu_mode", "auto")

    # L'utente puo' imporre una modalita' specifica dalle impostazioni.
    if mode == "compatible":
        attempts = [True]
    elif mode == "fast":
        attempts = [False]
    else:
        # In automatico: sulle macchine emulate partiamo direttamente
        # dalla modalita' sicura, altrove proviamo prima la piu' veloce.
        attempts = [True] if compat.is_emulated() else [False, True]

    last_detail = ""
    last_code = 0
    for index, force_generic in enumerate(attempts):
        log.info(
            "Test di avvio del motore di trascrizione (compatibilita'=%s)",
            force_generic,
        )
        code, detail = _run_child(force_generic, whisper_size, should_stop)
        last_detail, last_code = detail, code
        if code == 0:
            log.info("Test superato (compatibilita'=%s)", force_generic)
            return SelfTestResult(True, force_generic, detail)
        if code == EXIT_CANCELLED:
            return SelfTestResult(
                False, False, detail, inconclusive=True, reason="annullato"
            )
        log.warning("Test fallito con codice %s: %s", code, detail[:2000])

        # Tempo scaduto, modello non ancora presente, processo non
        # avviabile: il test non ha dato una risposta. Non e' un verdetto
        # di incompatibilita', e riprovare in modalita' compatibilita'
        # non cambierebbe nulla.
        if code in (EXIT_TIMEOUT, EXIT_MODEL_MISSING, EXIT_LAUNCH_FAILED):
            return SelfTestResult(
                False, False, detail, inconclusive=True,
                reason=_MOTIVI.get(code, "non concluso"),
            )

        # Il ripiego sui kernel generici ha senso solo se il processo
        # figlio e' stato ucciso dal sistema operativo. Se invece si e'
        # fermato da solo (modello mancante, eccezione Python, tempo
        # scaduto) riprovare piu' lentamente fallirebbe allo stesso
        # modo, con l'aggravante di lasciare in memoria l'esito
        # "serve la modalita' compatibilita'" per sempre.
        remaining = index + 1 < len(attempts)
        if remaining and not _is_native_crash(code):
            log.info(
                "Il test non e' fallito per istruzioni non supportate: "
                "non attivo la modalita' compatibilita'."
            )
            break

    # Un'eccezione Python nel figlio (codice 1) non e' un verdetto sulla
    # CPU: il piu' delle volte e' un file del modello incompleto o un
    # problema momentaneo. Lo trattiamo come "non concluso" e lasciamo
    # la registrazione disponibile, invece di bloccare il programma.
    if not _is_native_crash(last_code):
        return SelfTestResult(
            False, False, last_detail, inconclusive=True,
            reason="il controllo preventivo si e' interrotto con un errore",
        )
    return SelfTestResult(False, False, last_detail)


# --------------------------------------------------------------------------
# Codice eseguito NEL processo figlio
# --------------------------------------------------------------------------
def selftest_transcription_child() -> int:
    """
    Carica davvero il modello e trascrive un secondo di audio sintetico.

    Se la CPU non supporta le istruzioni richieste, il processo viene
    terminato qui: e' esattamente cio' che vogliamo scoprire.
    """
    try:
        import numpy as np

        from app import config
        from app.models.download import whisper_model_dir, whisper_model_present

        size = os.environ.get("INTERVIEW_ASSISTANT_SELFTEST_MODEL", "small")
        if not whisper_model_present(size):
            log.error("Modello '%s' non presente: test non eseguibile", size)
            return 3

        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(whisper_model_dir(size)),
            device="cpu",
            compute_type=config.WHISPER_COMPUTE_TYPE,
            # Stessi thread dell'uso reale: il test deve esercitare la
            # stessa configurazione, altrimenti verifica qualcos'altro.
            cpu_threads=config.transcription_threads(),
            num_workers=1,
        )

        # Un secondo di rumore molto debole: sufficiente a far girare
        # tutti i kernel di calcolo senza dipendere da un file audio.
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(config.AUDIO_SAMPLE_RATE) * 0.001).astype("float32")

        segments, _info = model.transcribe(audio, language="en", beam_size=1)
        list(segments)  # forza l'esecuzione effettiva del decoder

        log.info("Motore di trascrizione funzionante")
        return 0
    except Exception:
        # Il dettaglio finisce nel file di log: nelle applicazioni senza
        # console la stampa a video non e' affidabile.
        log.exception("Test del motore di trascrizione non riuscito")
        return 1
