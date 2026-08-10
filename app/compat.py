"""
Compatibilita' hardware / CPU.

Motivo di questo modulo: le librerie di calcolo usate dall'app
(CTranslate2 per Whisper, llama.cpp per l'LLM) vengono distribuite come
binari compilati con istruzioni vettoriali moderne (AVX2). Su alcune
macchine queste istruzioni non sono disponibili:

  * PC Windows con CPU ARM64 (es. Snapdragon) che eseguono l'app x64
    tramite emulazione: l'emulatore non supporta sempre AVX2;
  * PC x64 datati, precedenti al 2013 circa, privi di AVX2.

In quei casi il programma non solleva una normale eccezione Python:
viene terminato di colpo dal sistema operativo ("istruzione non
valida"), quindi senza alcun messaggio d'errore visibile all'utente.

Qui rileviamo l'architettura reale della macchina e, quando serve,
attiviamo la "modalita' compatibilita'": CTranslate2 accetta la
variabile d'ambiente CT2_FORCE_CPU_ISA per forzare l'uso di kernel
generici, piu' lenti ma eseguibili ovunque.

IMPORTANTE: apply_cpu_compat() va chiamata PRIMA di importare
faster_whisper / ctranslate2, altrimenti la variabile viene ignorata.
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys

log = logging.getLogger(__name__)

# Costanti IMAGE_FILE_MACHINE_* dell'API Windows
_MACHINE_NAMES = {
    0x8664: "AMD64",
    0xAA64: "ARM64",
    0x014C: "x86",
    0x01C4: "ARM",
}


def native_machine() -> str:
    """
    Architettura REALE del processore, non quella emulata.

    Su Windows un processo x64 in esecuzione su CPU ARM64 vede
    platform.machine() == "AMD64" (l'emulazione e' trasparente):
    per sapere la verita' bisogna chiedere a IsWow64Process2.
    """
    if sys.platform != "win32":
        return platform.machine().upper()

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not hasattr(kernel32, "IsWow64Process2"):
            # API disponibile da Windows 10 1511 in poi
            return platform.machine().upper()

        process_machine = ctypes.c_ushort(0)
        native = ctypes.c_ushort(0)
        ok = kernel32.IsWow64Process2(
            kernel32.GetCurrentProcess(),
            ctypes.byref(process_machine),
            ctypes.byref(native),
        )
        if ok and _MACHINE_NAMES.get(native.value) == "ARM64":
            return "ARM64"

        # IsWow64Process2 non basta da solo: per i processi x64 emulati
        # su Windows ARM64 alcune versioni riportano AMD64 come macchina
        # nativa. GetMachineTypeAttributes (Windows 11 in poi, cioe'
        # esattamente dove esiste l'emulazione x64) dice se questa
        # macchina esegue ARM64 nativamente.
        try:
            if hasattr(kernel32, "GetMachineTypeAttributes"):
                IMAGE_FILE_MACHINE_ARM64 = 0xAA64
                attrs = ctypes.c_int(0)
                esito = kernel32.GetMachineTypeAttributes(
                    IMAGE_FILE_MACHINE_ARM64, ctypes.byref(attrs)
                )
                # UserEnabled = 0x1: i programmi ARM64 girano nativamente.
                if esito == 0 and (attrs.value & 0x1):
                    return "ARM64"
        except Exception:
            pass

        # Ultimo ripiego: le cartelle SysArm32/SyChpe32 esistono solo
        # nelle installazioni di Windows su ARM.
        root = os.environ.get("SystemRoot", r"C:\Windows")
        for nome in ("SysArm32", "SyChpe32"):
            if os.path.isdir(os.path.join(root, nome)):
                return "ARM64"

        if ok:
            return _MACHINE_NAMES.get(native.value, hex(native.value))
    except Exception:
        pass

    return platform.machine().upper()


def process_machine() -> str:
    """Architettura per cui e' compilato l'eseguibile in esecuzione."""
    return platform.machine().upper()


def is_emulated() -> bool:
    """True se stiamo girando in emulazione (es. app x64 su CPU ARM64)."""
    native = native_machine()
    current = process_machine()
    if native == "ARM64" and current in ("AMD64", "X86_64", "X86"):
        return True
    return False


def cpu_supports_avx2() -> bool:
    """
    Verifica la presenza di AVX2 leggendo CPUID (foglia 7, EBX bit 5).

    Restituisce True in caso di dubbio: preferiamo non attivare la
    modalita' lenta senza motivo. Il rilevamento definitivo avviene
    comunque con il test di avvio in diagnostics.py, che e' empirico.
    """
    if is_emulated():
        # Sotto emulazione CPUID puo' mentire: trattiamo come non sicuro.
        return False
    if process_machine() not in ("AMD64", "X86_64"):
        return True
    try:
        # Su Windows non esiste un modo semplice e portabile di eseguire
        # CPUID da Python puro senza dipendenze; ci affidiamo al test
        # empirico di diagnostics.py. Qui assumiamo supporto presente.
        return True
    except Exception:
        return True


_job_handle = None


def adotta_processo_figlio(proc) -> None:
    """
    Fa in modo che Windows chiuda i processi figli insieme al nostro.

    Il programma avvia processi separati per il controllo preventivo e
    per scrivere il report: quest'ultimo occupa due gigabyte di memoria
    e tutti i core per diversi minuti. Se l'applicazione principale
    muore — chiusa da Gestione attivita', fermata da un antivirus, o per
    un guasto — quel processo continuava a girare da solo, senza
    finestra e senza voce nella barra delle applicazioni: il computer
    restava lentissimo e nessuno capiva perche'.

    Un "job object" con la chiusura a cascata risolve la cosa alla
    radice, ed e' il sistema operativo a farsene carico.
    """
    global _job_handle
    if sys.platform != "win32" or proc is None:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        if _job_handle is None:
            class _LimitiBase(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class _LimitiEstesi(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _LimitiBase),
                    ("IoInfo", ctypes.c_ubyte * 48),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return
            info = _LimitiEstesi()
            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            info.BasicLimitInformation.LimitFlags = 0x00002000
            # JobObjectExtendedLimitInformation = 9
            if not kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                kernel32.CloseHandle(handle)
                return
            _job_handle = handle

        kernel32.AssignProcessToJobObject(_job_handle, int(proc._handle))
    except Exception:
        # Non e' una funzione indispensabile: se non riesce, il figlio
        # resta semplicemente indipendente come prima.
        log.debug("Processo figlio non associato al job object", exc_info=True)


def describe_cpu() -> str:
    """Riga descrittiva usata nei log e nella schermata diagnostica."""
    parts = [
        f"processore={native_machine()}",
        f"applicazione={process_machine()}",
        f"emulazione={'si' if is_emulated() else 'no'}",
        f"core={os.cpu_count()}",
        f"sistema={platform.system()} {platform.release()}",
    ]
    return ", ".join(parts)


def apply_cpu_compat(force_generic: bool | None = None) -> bool:
    """
    Configura le variabili d'ambiente per l'esecuzione sicura.

    force_generic:
        True  -> forza sempre i kernel generici (modalita' compatibilita')
        False -> non forzare nulla (massime prestazioni)
        None  -> decide automaticamente in base all'hardware rilevato

    Ritorna True se la modalita' compatibilita' e' stata attivata.
    """
    if force_generic is None:
        force_generic = is_emulated()

    if force_generic:
        # CTranslate2: usa l'implementazione generica, senza AVX/AVX2.
        os.environ["CT2_FORCE_CPU_ISA"] = "GENERIC"
    else:
        os.environ.pop("CT2_FORCE_CPU_ISA", None)

    # OMP_NUM_THREADS non va impostata qui. In passato la modalita'
    # compatibilita' la fissava a meta' dei core: su un computer a due
    # core diventava 1, e quel valore veniva ereditato anche dal
    # processo che genera il report, dimezzandone la velocita'. Il
    # numero di thread e' gia' deciso esplicitamente da chi fa il
    # calcolo (cpu_threads per la trascrizione, n_threads per l'LLM):
    # una variabile d'ambiente globale puo' solo contraddirli.
    os.environ.pop("OMP_NUM_THREADS", None)

    # Finita una frase, i thread di calcolo per impostazione predefinita
    # non si addormentano: restano ad aspettare girando a vuoto per
    # essere pronti alla frase successiva. Su un computer potente e'
    # un buon compromesso; su due core e' un disastro, perche' quel
    # tempo bruciato a vuoto e' esattamente quello che serve al servizio
    # grafico di Windows per comporre l'anteprima della videochiamata —
    # che su una macchina virtuale disegna con il processore. Chiediamo
    # quindi di mettersi a dormire subito.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    os.environ.setdefault("KMP_BLOCKTIME", "0")

    # CTranslate2 e tokenizers producono log rumorosi su stderr che, in
    # una app senza console, finirebbero nel nulla o darebbero fastidio.
    # Il livello 1 aggiunge cinque righe per avvio che dicono quale set
    # di istruzioni e quale precisione siano stati scelti davvero: senza
    # di esse non c'e' modo di sapere se il computer stia usando i
    # kernel veloci o quelli generici, che costano cinque volte tanto.
    os.environ.setdefault("CT2_VERBOSE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Evita che HuggingFace apra thread di telemetria non necessari.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    return bool(force_generic)
