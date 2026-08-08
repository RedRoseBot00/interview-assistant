"""
Generazione del report di fine colloquio con un modello linguistico
locale (nessuna API, nessun costo ricorrente, nessun dato in rete).

Il modello (Qwen2.5-3B-Instruct, licenza Apache 2.0) viene eseguito da
llama.cpp sulla CPU del computer.

Due accorgimenti importanti:

1. L'elaborazione avviene in un PROCESSO SEPARATO. Le librerie di
   calcolo possono usare istruzioni non supportate da alcune CPU e in
   quel caso il processo viene terminato dal sistema operativo, senza
   che sia possibile intercettare l'errore. Isolandolo, il programma
   principale sopravvive e ripiega sul riepilogo alternativo.

2. Se il modello non e' utilizzabile, il colloquio non va perso: viene
   comunque prodotto un resoconto strutturato ricavato dalla
   trascrizione, dichiarando apertamente che e' stato generato senza
   modello linguistico.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from app import config

log = logging.getLogger(__name__)

GENERATION_TIMEOUT_SECONDS = 900

LANGUAGE_NAMES = {
    "it": "italiano",
    "en": "inglese",
    "es": "spagnolo",
    "fr": "francese",
    "de": "tedesco",
    "pt": "portoghese",
    "nl": "olandese",
    "ro": "rumeno",
    "pl": "polacco",
    "ru": "russo",
    "ar": "arabo",
    "zh": "cinese",
    "ja": "giapponese",
}

SYSTEM_PROMPT = (
    "Sei un assistente professionale per selezionatori del personale. "
    "Sei preciso, sintetico e non inventi mai informazioni: se un dato non "
    "compare nella trascrizione, scrivi che non e' emerso."
)

PROMPT_TEMPLATE = """Di seguito la trascrizione automatica di un colloquio di \
lavoro appena concluso. Le battute sono attribuite a chi le ha pronunciate; \
la trascrizione puo' contenere piccoli errori di riconoscimento vocale.

Candidato: {candidate_name}
Posizione: {role}
Durata: {duration}

--- TRASCRIZIONE ---
{transcript}
--- FINE TRASCRIZIONE ---

Scrivi un resoconto strutturato in {output_language}, con queste sezioni \
numerate:

1. Sintesi del colloquio (3-5 frasi)
2. Punti di forza emersi
3. Aree di attenzione o da approfondire
4. Competenze ed esperienze citate
5. Domande di follow-up suggerite per un eventuale secondo colloquio
6. Valutazione complessiva: scegli tra Positiva, Da approfondire, Negativa, \
con una breve motivazione

Basati esclusivamente su quanto detto nella trascrizione."""


@dataclass
class ReportResult:
    text: str
    model_name: str
    used_llm: bool
    warning: str = ""


# --------------------------------------------------------------------------
# Preparazione del testo
# --------------------------------------------------------------------------
_TRUNCATION_MARKER = (
    "\n\n[...parte centrale del colloquio omessa per limiti di lunghezza...]\n\n"
)


def truncate_transcript(
    transcript: str, max_chars: int | None = None
) -> tuple[str, bool]:
    """
    Limita la lunghezza della trascrizione alla finestra di contesto.

    Di un colloquio molto lungo conserviamo l'inizio (presentazione e
    impostazione) e la fine (conclusioni e domande finali), che sono le
    parti piu' informative, segnalando il taglio nel mezzo.

    Il testo del segnalatore va sottratto dal budget, altrimenti il
    risultato supererebbe sempre il limite dichiarato.
    """
    if max_chars is None:
        max_chars = config.LLM_MAX_TRANSCRIPT_CHARS
    text = transcript.strip()
    if len(text) <= max_chars:
        return text, False

    budget = max_chars - len(_TRUNCATION_MARKER)
    if budget <= 0:
        return text[:max_chars], True

    head_size = int(budget * 0.6)
    tail_size = budget - head_size

    # Tagliamo a fine riga solo se cio' non ci costa troppo testo: senza
    # questa condizione, una singola battuta molto lunga ridurrebbe la
    # porzione conservata a poche parole.
    raw_head = text[:head_size]
    cut = raw_head.rfind("\n")
    head = raw_head[:cut] if cut > head_size * 0.5 else raw_head

    raw_tail = text[-tail_size:]
    cut = raw_tail.find("\n")
    tail = raw_tail[cut + 1 :] if 0 <= cut < tail_size * 0.5 else raw_tail

    return head + _TRUNCATION_MARKER + tail, True


def _format_duration(seconds: float | None) -> str:
    try:
        seconds = max(0, int(seconds or 0))
    except (TypeError, ValueError):
        seconds = 0
    minutes, secs = divmod(seconds, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"
    return f"{minutes} min {secs:02d} s"


def build_prompt(
    transcript: str,
    candidate_name: str,
    role: str,
    duration_seconds: float,
    language_code: str,
) -> str:
    output_language = LANGUAGE_NAMES.get(language_code, "italiano")
    body, truncated = truncate_transcript(transcript)
    if not body:
        body = "(nessuna trascrizione disponibile)"
    prompt = PROMPT_TEMPLATE.format(
        candidate_name=candidate_name or "non indicato",
        role=role or "non indicata",
        duration=_format_duration(duration_seconds),
        transcript=body,
        output_language=output_language,
    )
    if truncated:
        log.info("Trascrizione troncata per rientrare nella finestra di contesto")
    return prompt


# --------------------------------------------------------------------------
# Riepilogo di riserva (senza modello linguistico)
# --------------------------------------------------------------------------
_STOPWORDS = {
    "che", "con", "per", "non", "una", "uno", "del", "della", "dei", "delle",
    "sono", "come", "piu", "più", "anche", "quindi", "poi", "cosa", "molto",
    "the", "and", "for", "that", "with", "have", "this", "from", "was", "are",
    "you", "your", "but", "not", "all", "can", "has", "our", "about", "would",
}


def _lines_of(segments: list, speaker: str) -> list[str]:
    """
    Estrae le battute di un interlocutore tollerando dati imperfetti.

    I segmenti possono provenire dal database, dove il contenuto JSON
    non e' garantito: una voce malformata non deve far fallire la
    generazione del resoconto, che spesso e' l'ultima possibilita' di
    salvare il lavoro del colloquio.
    """
    result: list[str] = []
    for item in segments or []:
        if not isinstance(item, dict) or item.get("speaker") != speaker:
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            result.append(text.strip())
    return result


def fallback_report(
    segments: list[dict],
    labels: dict[str, str],
    candidate_name: str,
    role: str,
    duration_seconds: float,
    reason: str = "",
) -> str:
    """Resoconto ricavato dalla sola trascrizione, senza modello AI."""
    recruiter_lines = _lines_of(segments, config.SPEAKER_RECRUITER)
    candidate_lines = _lines_of(segments, config.SPEAKER_CANDIDATE)

    questions = [line for line in recruiter_lines if "?" in line][:10]
    longest = sorted(candidate_lines, key=len, reverse=True)[:5]

    words: dict[str, int] = {}
    for line in candidate_lines:
        for raw in line.lower().split():
            word = raw.strip(".,;:!?()[]\"'")
            if len(word) < 4 or word in _STOPWORDS or not word.isalpha():
                continue
            words[word] = words.get(word, 0) + 1
    keywords = [w for w, _ in sorted(words.items(), key=lambda kv: -kv[1])[:12]]

    total_words = sum(len(line.split()) for line in candidate_lines)
    recruiter_words = sum(len(line.split()) for line in recruiter_lines)
    share = (
        f"{round(100 * total_words / max(1, total_words + recruiter_words))}%"
        if (total_words + recruiter_words)
        else "non calcolabile"
    )

    parts: list[str] = []
    parts.append("RESOCONTO AUTOMATICO DEL COLLOQUIO")
    if reason:
        parts.append(
            "Nota: questo resoconto e' stato generato senza il modello "
            f"linguistico locale. Motivo: {reason} "
            "Il contenuto qui sotto e' ricavato direttamente dalla trascrizione."
        )
    parts.append("")
    parts.append(f"Candidato: {candidate_name or 'non indicato'}")
    parts.append(f"Posizione: {role or 'non indicata'}")
    parts.append(f"Durata: {_format_duration(duration_seconds)}")
    parts.append(
        f"Interventi: {len(recruiter_lines)} di {labels.get(config.SPEAKER_RECRUITER, 'intervistatore')}, "
        f"{len(candidate_lines)} di {labels.get(config.SPEAKER_CANDIDATE, 'candidato')}"
    )
    parts.append(f"Quota di parlato del candidato: {share}")
    parts.append("")

    parts.append("1. DOMANDE POSTE")
    if questions:
        parts.extend(f"   - {q}" for q in questions)
    else:
        parts.append("   - Nessuna domanda riconosciuta automaticamente.")
    parts.append("")

    parts.append("2. RISPOSTE PIU' ARTICOLATE DEL CANDIDATO")
    if longest:
        for answer in longest:
            text = answer if len(answer) <= 400 else answer[:400].rstrip() + "..."
            parts.append(f"   - {text}")
    else:
        parts.append("   - Nessuna risposta registrata.")
    parts.append("")

    parts.append("3. TERMINI RICORRENTI NELLE RISPOSTE")
    parts.append(
        "   " + (", ".join(keywords) if keywords else "nessun termine significativo")
    )
    parts.append("")

    parts.append("4. VALUTAZIONE")
    parts.append(
        "   Da compilare manualmente: senza modello linguistico il programma "
        "non esprime giudizi sul candidato."
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Esecuzione isolata in un processo separato
# --------------------------------------------------------------------------
def _child_command(input_path: Path, output_path: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--generate-report", str(input_path), str(output_path)]
    entry = Path(__file__).resolve().parents[2] / "run.py"
    return [
        sys.executable,
        str(entry),
        "--generate-report",
        str(input_path),
        str(output_path),
    ]


# Riferimento al processo di generazione in corso, per poterlo
# interrompere quando l'utente chiude la finestra: altrimenti resterebbe
# a occupare tutti i core del computer per minuti, invisibile.
_active_child: subprocess.Popen | None = None
_child_lock = threading.Lock()


def cancel_running_generation() -> None:
    """Interrompe la generazione del report, se ne e' in corso una."""
    with _child_lock:
        proc = _active_child
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            log.info("Generazione del report interrotta su richiesta")
        except Exception:
            log.debug("Interruzione del processo non riuscita", exc_info=True)


def _safe_fallback(
    segments: list[dict],
    labels: dict[str, str],
    candidate_name: str,
    role: str,
    duration_seconds: float,
    reason: str,
    transcript: str,
) -> str:
    """
    Il resoconto di riserva non deve mai fallire: e' l'ultima rete di
    sicurezza per non perdere il lavoro di un intero colloquio.
    """
    try:
        return fallback_report(
            segments, labels, candidate_name, role, duration_seconds, reason=reason
        )
    except Exception:
        log.exception("Anche il resoconto di riserva e' fallito")
        return (
            "RESOCONTO NON DISPONIBILE\n\n"
            f"Motivo: {reason}\n\n"
            "Di seguito la trascrizione integrale del colloquio.\n\n"
            + (transcript or "(nessuna trascrizione registrata)")
        )


def generate_report(
    transcript: str,
    segments: list[dict],
    labels: dict[str, str],
    candidate_name: str = "",
    role: str = "",
    duration_seconds: float = 0.0,
    detected_language: str = "it",
    report_language: str = "auto",
) -> ReportResult:
    """
    Produce il report. Non solleva eccezioni: in caso di problemi
    restituisce il resoconto di riserva con una nota esplicativa.
    """
    language_code = (
        detected_language if report_language == "auto" else report_language
    ) or "it"

    from app.models.download import llm_model_present

    if not llm_model_present():
        return ReportResult(
            text=_safe_fallback(
                segments, labels, candidate_name, role, duration_seconds,
                "il modello per i report non risulta installato.", transcript,
            ),
            model_name="",
            used_llm=False,
            warning="Modello per i report non installato.",
        )

    prompt = build_prompt(
        transcript, candidate_name, role, duration_seconds, language_code
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="interview-report-"))
    input_path = tmp_dir / "input.json"
    output_path = tmp_dir / "output.json"

    try:
        input_path.write_text(
            json.dumps(
                {
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "model_path": str(config.LLM_MODEL_PATH),
                    "context_size": config.LLM_CONTEXT_SIZE,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        global _active_child
        proc = subprocess.Popen(
            _child_command(input_path, output_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Senza errors="replace" un output non decodificabile
            # solleverebbe un'eccezione proprio quando serve leggere il
            # messaggio d'errore del processo figlio.
            errors="replace",
            env=dict(os.environ),
            creationflags=creationflags,
        )
        with _child_lock:
            _active_child = proc
        try:
            child_out, child_err = proc.communicate(
                timeout=GENERATION_TIMEOUT_SECONDS
            )
        finally:
            with _child_lock:
                _active_child = None

        if proc.returncode == 0 and output_path.exists():
            data = json.loads(output_path.read_text(encoding="utf-8"))
            text = (data.get("text") or "").strip()
            if text:
                return ReportResult(
                    text=text,
                    model_name=config.LLM_MODEL_FILENAME,
                    used_llm=True,
                )
            reason = "il modello non ha prodotto alcun testo."
        elif proc.returncode == 0:
            reason = "il modello non ha restituito alcun risultato."
        else:
            detail = ((child_err or "") + (child_out or "")).strip()
            log.error(
                "Generazione del report fallita (codice %s): %s",
                proc.returncode,
                detail[:2000],
            )
            reason = (
                "il modello linguistico non e' compatibile con questo "
                "processore oppure la memoria disponibile non e' sufficiente."
            )

    except subprocess.TimeoutExpired:
        log.error("Generazione del report interrotta per timeout")
        try:
            proc.kill()
            proc.communicate(timeout=10)
        except Exception:
            pass
        reason = "la generazione ha superato il tempo massimo consentito."
    except Exception as exc:
        log.exception("Errore imprevisto nella generazione del report")
        reason = f"errore imprevisto ({type(exc).__name__})."
    finally:
        for path in (input_path, output_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass

    return ReportResult(
        text=_safe_fallback(
            segments, labels, candidate_name, role, duration_seconds, reason, transcript
        ),
        model_name="",
        used_llm=False,
        warning=reason,
    )


# --------------------------------------------------------------------------
# Codice eseguito NEL processo figlio
# --------------------------------------------------------------------------
def generate_report_child(input_path: str, output_path: str) -> int:
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))

        from llama_cpp import Llama

        threads = max(1, (os.cpu_count() or 4) - 1)
        llm = Llama(
            model_path=payload["model_path"],
            n_ctx=int(payload.get("context_size", 8192)),
            n_threads=threads,
            verbose=False,
        )
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": payload["system"]},
                {"role": "user", "content": payload["prompt"]},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        text = result["choices"][0]["message"]["content"].strip()
        Path(output_path).write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"
        )
        return 0
    except Exception:
        log.exception("Generazione del report nel processo figlio non riuscita")
        return 1
