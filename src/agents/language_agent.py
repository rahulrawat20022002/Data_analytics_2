"""Language detection and language-aware helpers for the multilingual pipeline.

Every other agent asks this one "what language is this, and what should I do
about it" so that detection rules live in exactly one place.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from langdetect import DetectorFactory, LangDetectException, detect_langs

import config_loader

# langdetect is randomised by default and will return different answers for the
# same short string across runs. Seeding makes eval numbers reproducible.
DetectorFactory.seed = 0

# Unicode-aware word tokenizer. The previous BM25 path used `text.split(" ")`,
# which leaves punctuation glued to words and mishandles German compounds and
# umlauts once casing varies.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


class LanguageAgent:
    """Detects query/document language and exposes language-aware utilities."""

    # Below this probability the detection is treated as unreliable (very short
    # queries in particular) and we fall back to the configured default.
    MIN_CONFIDENCE = 0.60

    def __init__(self) -> None:
        cfg = config_loader.get_config()["languages"]
        self.default = cfg.get("default", "en")
        self.supported: List[str] = list(cfg.get("supported", ["en"]))
        self.names: Dict[str, str] = dict(cfg.get("names", {}))
        self.spacy_models: Dict[str, str] = dict(cfg.get("spacy_models", {}))
        print(
            "✅ LanguageAgent initialized. "
            f"First-class languages: {', '.join(self.supported)} "
            f"(default: {self.default})"
        )

    # --- detection ---------------------------------------------------------

    def detect(self, text: str) -> str:
        """Returns a language code, falling back to the default when unsure."""
        code, _ = self.detect_with_confidence(text)
        return code

    def detect_with_confidence(self, text: str) -> Tuple[str, float]:
        """Returns ``(language_code, probability)``.

        Falls back to the configured default when the text is empty, when
        detection throws, or when the top candidate is below MIN_CONFIDENCE.
        """
        if not text or not text.strip():
            return self.default, 0.0

        try:
            candidates = detect_langs(text)
        except LangDetectException:
            print(f"  - Language detection failed. Falling back to '{self.default}'.")
            return self.default, 0.0

        if not candidates:
            return self.default, 0.0

        best = candidates[0]
        code, probability = best.lang, float(best.prob)

        if probability < self.MIN_CONFIDENCE:
            print(
                f"  - Low-confidence detection ({code} @ {probability:.2f}). "
                f"Falling back to '{self.default}'."
            )
            return self.default, probability

        return code, probability

    def is_supported(self, code: str) -> bool:
        """True when the language is first-class (tuned prompts, eval coverage)."""
        return code in self.supported

    # --- naming / prompting ------------------------------------------------

    def name_of(self, code: str) -> str:
        """Human-readable language name for use inside prompts."""
        return self.names.get(code, code)

    def instruction_for(self, code: str) -> str:
        """The sentence appended to generation prompts to pin the output language."""
        name = self.name_of(code)
        return (
            f"Write your entire response in {name}. "
            f"Do not translate quoted source text or the citation tags "
            f"(`[Source X: file=..., page=...]`) -- reproduce those verbatim."
        )

    # --- tokenization ------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        """Unicode-aware, case-folded tokenization used for BM25 on any language.

        Note: this is whitespace/word-boundary based, so it is appropriate for
        the Latin-script languages configured here (en, de) but would need a
        segmenter for unspaced scripts such as Chinese or Japanese.
        """
        if not text:
            return []
        return _WORD_RE.findall(text.lower())

    # --- spaCy routing -----------------------------------------------------

    def spacy_model_for(self, code: str) -> Optional[str]:
        """Returns the spaCy pipeline name, or None to use the blank ``xx`` one."""
        return self.spacy_models.get(code)

    # --- evaluation support -----------------------------------------------

    def response_matches_language(self, response: str, expected: str) -> Dict[str, Any]:
        """Checks that a generated answer came back in the requested language.

        Used by the eval harness: a multilingual RAG that silently answers a
        German question in English is failing even when the content is right.
        """
        detected, probability = self.detect_with_confidence(response)
        matches = detected == expected
        return {
            "check": "language_match",
            "expected": expected,
            "detected": detected,
            "confidence": round(probability, 4),
            "match": matches,
            "pass": "✅" if matches else "❌",
        }


if __name__ == "__main__":
    print("--- Starting Language Agent (Test Mode) ---")
    agent = LanguageAgent()

    samples = [
        "How do EU and US policies on artificial intelligence differ?",
        "Wie unterscheiden sich die KI-Richtlinien der EU und der USA?",
        "Welche Maßnahmen sieht der Europäische Grüne Deal zur CO2-Bepreisung vor?",
        "ok",
    ]

    for sample in samples:
        code, probability = agent.detect_with_confidence(sample)
        print(f"\nText: {sample[:70]}")
        print(f"  -> {code} ({agent.name_of(code)}) confidence={probability:.2f}")
        print(f"  -> supported={agent.is_supported(code)}")
        print(f"  -> tokens={agent.tokenize(sample)[:8]}")

    print("\n--- Language match check ---")
    print(agent.response_matches_language("Die Richtlinie regelt den Handel.", "de"))
    print(agent.response_matches_language("The directive regulates trade.", "de"))

    print("\n--- Language Agent Finished ---")
