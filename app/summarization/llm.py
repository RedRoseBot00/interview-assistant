"""
Generazione del report di fine colloquio con un LLM locale (no cloud,
no API a pagamento, nessun costo ricorrente).

Usiamo llama-cpp-python per eseguire un modello quantizzato in formato
GGUF (Qwen2.5-3B-Instruct, licenza Apache 2.0, fortemente multilingue)
direttamente sulla CPU del PC del cliente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app import config


PROMPT_TEMPLATE = """Sei un assistente che aiuta un selezionatore (recruiter) \
a redigere il resoconto di un colloquio di lavoro appena concluso.

Di seguito trovi la trascrizione grezza del colloquio (puo' contenere \
piccoli errori di trascrizione automatica).

Candidato: {candidate_name}
Posizione: {role}

--- TRASCRIZIONE ---
{transcript}
--- FINE TRASCRIZIONE ---

Scrivi un resoconto strutturato in {output_language}, con queste sezioni:
1. Sintesi generale del colloquio (3-5 frasi)
2. Punti di forza del candidato emersi
3. Aree di attenzione o dubbi da approfondire
4. Competenze/esperienze rilevanti citate
5. Domande di follow-up suggerite per un secondo colloquio
6. Valutazione complessiva (Positiva / Da approfondire / Negativa) con una \
breve motivazione

Sii oggettivo, basati solo su quanto detto nella trascrizione, non inventare \
informazioni non presenti nel testo.
"""

LANGUAGE_NAMES = {
    "it": "italiano",
    "en": "inglese",
    "es": "spagnolo",
    "fr": "francese",
    "de": "tedesco",
    "pt": "portoghese",
}


@dataclass
class ReportResult:
    text: str
    model_name: str


class LocalLLM:
    """Wrapper leggero attorno a llama-cpp-python."""

    def __init__(self, model_path=None):
        self.model_path = str(model_path or config.LLM_MODEL_PATH)
        self._llm = None

    def load(self):
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=config.LLM_CONTEXT_SIZE,
            n_threads=self._default_threads(),
            verbose=False,
        )

    @staticmethod
    def _default_threads() -> int:
        import os

        cpu_count = os.cpu_count() or 4
        # lasciamo un core libero per UI/audio, evitiamo di saturare la CPU
        return max(1, cpu_count - 1)

    def generate_report(
        self,
        transcript: str,
        candidate_name: str = "N/D",
        role: str = "N/D",
        detected_language: str = "it",
        report_language: str = "auto",
    ) -> ReportResult:
        if self._llm is None:
            self.load()

        lang_code = detected_language if report_language == "auto" else report_language
        output_language = LANGUAGE_NAMES.get(lang_code, "italiano")

        prompt = PROMPT_TEMPLATE.format(
            candidate_name=candidate_name or "N/D",
            role=role or "N/D",
            transcript=transcript.strip() or "(nessuna trascrizione disponibile)",
            output_language=output_language,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Sei un assistente professionale per recruiter, preciso "
                    "e sintetico, che non inventa informazioni."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        result = self._llm.create_chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        text = result["choices"][0]["message"]["content"].strip()
        return ReportResult(text=text, model_name=config.LLM_MODEL_FILENAME)

    def unload(self):
        self._llm = None
