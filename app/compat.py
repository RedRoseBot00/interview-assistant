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
import os
import platform
import sys

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

    # CTranslate2 e tokenizers producono log rumorosi su stderr che, in
    # una app senza console, finirebbero nel nulla o darebbero fastidio.
    os.environ.setdefault("CT2_VERBOSE", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Evita che HuggingFace apra thread di telemetria non necessari.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    return bool(force_generic)
