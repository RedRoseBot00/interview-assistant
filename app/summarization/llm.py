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
from typing import Callable

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

Scrivi un resoconto in {output_language} con queste sezioni numerate:

1. SINTESI — 3 frasi
2. PUNTI DI FORZA — massimo 4 punti elenco, una riga ciascuno
3. AREE DI ATTENZIONE — massimo 3 punti elenco, una riga ciascuno
4. COMPETENZE CITATE — solo un elenco di parole chiave
5. DOMANDE DI APPROFONDIMENTO — massimo 3
6. VALUTAZIONE — Positiva, Da approfondire oppure Negativa, con una riga \
di motivazione

Sii asciutto e concreto: niente premesse, niente ripetizioni, nessun \
commento sul tuo lavoro. Basati esclusivamente sulla trascrizione."""


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

# Intercalari che da soli non aggiungono nulla a un resoconto. Vengono
# tolti solo quando costituiscono l'intera battuta.
_INTERCALARI = {
    "si", "sì", "no", "ok", "okay", "certo", "esatto", "perfetto", "bene",
    "mmm", "mhm", "ah", "eh", "beh", "allora", "ecco", "va bene", "d'accordo",
    "yes", "yeah", "no", "right", "okay", "sure", "exactly", "perfect",
}


def compact_transcript(transcript: str) -> str:
    """
    Compatta la trascrizione senza perdere informazione.

    Il tempo di lettura del modello cresce in proporzione alla lunghezza
    del testo, e su un computer da ufficio quella lettura e' la meta'
    dell'attesa totale. Una trascrizione automatica pero' e' piena di
    ridondanza strutturale: la stessa persona compare su dieci righe di
    seguito perche' il rilevatore di voce ha tagliato alle pause, e fra
    una risposta e l'altra si accumulano battute di puro assenso.

    Unendo le righe consecutive dello stesso interlocutore e togliendo
    gli intercalari isolati si risparmia in genere un terzo del testo
    senza togliere un solo contenuto: il modello legge meno, quindi
    risponde prima, e non riceve nulla di meno su cui ragionare.
    """
    righe = [r.strip() for r in (transcript or "").splitlines()]
    unite: list[tuple[str, list[str]]] = []

    for riga in righe:
        if not riga:
            continue
        etichetta, separatore, testo = riga.partition(":")
        if not separatore:
            etichetta, testo = "", riga
        testo = testo.strip()
        if not testo:
            continue

        nudo = testo.lower().strip(".,;:!? ")
        if len(testo.split()) <= 2 and nudo in _INTERCALARI:
            continue

        if unite and unite[-1][0] == etichetta:
            unite[-1][1].append(testo)
        else:
            unite.append((etichetta, [testo]))

    fuse: list[str] = []
    for etichetta, pezzi in unite:
        corpo = " ".join(pezzi)
        fuse.append(f"{etichetta}: {corpo}" if etichetta else corpo)
    return "\n".join(fuse)


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


def context_size_for(prompt: str, system: str, max_tokens: int) -> int:
    """
    Finestra di contesto adatta a questo preciso report.

    Tenerla fissa a quattromila token significa allocare la memoria per
    la cache delle chiavi e i buffer di calcolo sempre al massimo, anche
    per un colloquio di dieci minuti. Su un portatile con quattro
    gigabyte di memoria, con l'interfaccia e il modello di trascrizione
    gia' residenti, e' proprio quella memoria in piu' a far cominciare
    lo scambio su disco: da li' in avanti non conta piu' nient'altro.

    La stima e' volutamente prudente (due caratteri per token contro i
    tre e mezzo tipici dell'italiano): sbagliare per eccesso costa un
    po' di memoria, sbagliare per difetto farebbe fallire il report.
    """
    caratteri = len(prompt) + len(system)
    stima = int(caratteri / 2.0) + max_tokens + 256
    arrotondato = ((stima + 255) // 256) * 256
    return max(config.LLM_CONTEXT_MIN, min(config.LLM_CONTEXT_MAX, arrotondato))


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
    originale = len(transcript or "")
    compatto = compact_transcript(transcript)
    if originale and compatto:
        log.info(
            "Trascrizione compattata da %d a %d caratteri (-%.0f%%)",
            originale, len(compatto), 100 * (1 - len(compatto) / originale),
        )
    body, truncated = truncate_transcript(compatto)
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


def _read_stream(
    proc: subprocess.Popen, on_partial: "Callable[[str], None] | None"
) -> str:
    """
    Legge il testo prodotto dal processo figlio man mano che arriva.

    Ogni riga e' un frammento in formato JSON: viene consegnato subito
    all'interfaccia, cosi' il report compare parola per parola invece di
    apparire tutto insieme alla fine.
    """
    raccolto: list[str] = []
    try:
        if proc.stdout is None:
            return ""
        for riga in proc.stdout:
            riga = riga.strip()
            if not riga:
                continue
            try:
                pezzo = json.loads(riga).get("t", "")
            except Exception:
                # Riga non nostra (avviso di una libreria): la teniamo
                # solo per l'eventuale diagnosi dell'errore.
                raccolto.append(riga)
                continue
            if pezzo and on_partial:
                try:
                    on_partial(pezzo)
                except Exception:
                    log.debug("Aggiornamento parziale non riuscito", exc_info=True)
    except Exception:
        log.debug("Lettura del flusso interrotta", exc_info=True)
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
    return "\n".join(raccolto[-40:])


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
    on_partial: "Callable[[str], None] | None" = None,
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
                    "context_size": context_size_for(
                        prompt, SYSTEM_PROMPT, config.LLM_MAX_TOKENS
                    ),
                    "max_tokens": config.LLM_MAX_TOKENS,
                    "batch_size": config.LLM_BATCH_SIZE,
                    "threads": config.llm_threads(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        # Il numero di thread lo decide il figlio, che sa quanti core
        # fisici ci sono. Una variabile d'ambiente ereditata dal padre
        # potrebbe solo contraddirlo: in passato la modalita'
        # compatibilita' la fissava a 1 e dimezzava la generazione.
        ambiente = dict(os.environ)
        ambiente.pop("OMP_NUM_THREADS", None)

        global _active_child
        proc = subprocess.Popen(
            _child_command(input_path, output_path),
            stdout=subprocess.PIPE,
            # Gli errori del processo figlio finiscono nel file di log
            # condiviso: leggere due flussi contemporaneamente rischia di
            # bloccarsi quando uno dei due si riempie.
            stderr=subprocess.DEVNULL,
            text=True,
            # Senza errors="replace" un carattere non decodificabile
            # interromperebbe la lettura a meta' generazione.
            errors="replace",
            env=ambiente,
            creationflags=creationflags,
        )
        with _child_lock:
            _active_child = proc

        child_out = ""
        try:
            # La lettura del flusso va sorvegliata: se il processo figlio
            # si blocca senza chiudere l'uscita (memoria esaurita, disco
            # che comincia a scambiare) un semplice ciclo di lettura non
            # tornerebbe mai, e l'utente vedrebbe girare la barra di
            # attesa all'infinito senza alcun messaggio.
            esito: dict[str, str] = {}
            lettore = threading.Thread(
                target=lambda: esito.__setitem__(
                    "out", _read_stream(proc, on_partial)
                ),
                name="report-stream",
                daemon=True,
            )
            lettore.start()
            lettore.join(GENERATION_TIMEOUT_SECONDS)
            if lettore.is_alive():
                proc.kill()          # chiude l'uscita e sblocca il lettore
                lettore.join(10)
                raise subprocess.TimeoutExpired(
                    proc.args, GENERATION_TIMEOUT_SECONDS
                )
            child_out = esito.get("out", "")
            proc.wait(timeout=60)
        finally:
            with _child_lock:
                _active_child = None

        child_err = ""
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
def _fit_prompt(llm, prompt: str, system: str, context: int, max_tokens: int) -> str:
    """
    Accorcia il testo, se serve, contando i token veri.

    La lunghezza in caratteri e' solo una stima: nomi propri, sigle e
    numeri si spezzano in molti piu' token del testo corrente. Qui il
    conto lo fa il modello stesso, quindi il report non puo' piu'
    fallire per un prompt fuori misura — che era il modo peggiore di
    perdere il lavoro di un colloquio intero.
    """
    disponibili = context - max_tokens - 96      # margine per il formato chat
    if disponibili <= 256:
        return prompt

    def conta(testo: str) -> int:
        try:
            return len(llm.tokenize(testo.encode("utf-8"), add_bos=False))
        except Exception:
            return len(testo) // 3

    fissi = conta(system)
    for _ in range(6):
        if fissi + conta(prompt) <= disponibili:
            return prompt
        # Togliamo dal centro: l'inizio del colloquio (presentazione) e
        # la fine (conclusioni) sono le parti piu' informative.
        taglio = int(len(prompt) * 0.15)
        meta = len(prompt) // 2
        prompt = (
            prompt[: meta - taglio // 2]
            + _TRUNCATION_MARKER
            + prompt[meta + taglio // 2 :]
        )
    log.warning("Prompt ancora lungo dopo sei riduzioni: procedo comunque")
    return prompt


def generate_report_child(input_path: str, output_path: str) -> int:
    """
    Genera il report e lo trasmette man mano che viene scritto.

    Ogni pezzo di testo prodotto viene stampato subito su stdout, dove
    il processo principale lo legge e lo mostra nella finestra: cosi'
    l'utente vede il report formarsi invece di fissare una barra di
    attesa per minuti. Il testo completo viene comunque salvato nel
    file di uscita, che resta la fonte definitiva.
    """
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))

        from llama_cpp import Llama

        threads = int(payload.get("threads") or max(1, (os.cpu_count() or 2)))
        contesto = int(payload.get("context_size", 4096))
        batch = int(payload.get("batch_size", 512))
        max_tokens = int(payload.get("max_tokens", 450))
        llm = Llama(
            model_path=payload["model_path"],
            n_ctx=contesto,
            n_threads=threads,
            n_threads_batch=threads,
            n_batch=batch,
            # Micro-lotto pari al lotto: su CPU la lettura del prompt
            # procede a blocchi piu' grandi, con meno passaggi sui pesi.
            n_ubatch=batch,
            verbose=False,
        )

        prompt = _fit_prompt(llm, payload["prompt"], payload["system"],
                             contesto, max_tokens)

        pezzi: list[str] = []
        flusso = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": payload["system"]},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            # Senza un freno alle ripetizioni il modello, arrivato in
            # fondo alle sezioni richieste, riempie lo spazio rimasto
            # ripetendosi: minuti di attesa per testo inutile.
            repeat_penalty=1.05,
            stream=True,
        )
        for blocco in flusso:
            delta = blocco["choices"][0].get("delta", {}).get("content")
            if not delta:
                continue
            pezzi.append(delta)
            try:
                sys.stdout.write(json.dumps({"t": delta}, ensure_ascii=False) + "\n")
                sys.stdout.flush()
            except Exception:
                # Se il processo principale ha chiuso la lettura non e'
                # un motivo per interrompere la generazione.
                pass

        text = "".join(pezzi).strip()
        Path(output_path).write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"
        )
        return 0
    except Exception:
        log.exception("Generazione del report nel processo figlio non riuscita")
        return 1
