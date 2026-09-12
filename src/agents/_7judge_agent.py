"""LLM-as-Judge: scores a generated answer against the context it was built from.

This complements the two evaluation pieces that already existed:
  - VerifierAgent   -> per-sentence NLI + citation presence (mechanical checks)
  - EvaluatorAgent  -> judges which *retrieval* strategy won (A/B on documents)
  - JudgeAgent      -> judges the *final answer* on a rubric (this file)

The judge runs on Ollama like the rest of the pipeline, so evaluation stays
fully local and no answer text leaves the machine.
"""

import json
import re
from typing import Any, Dict, List, Optional

import ollama

import config_loader
from _1language_agent import LanguageAgent

# The rubric. Each dimension is scored on 1..judge_scale with a short rationale.
RUBRIC: Dict[str, str] = {
    "groundedness": (
        "Is every claim in the answer directly supported by the provided sources? "
        "Penalise anything asserted that the sources do not state."
    ),
    "relevance": (
        "Does the answer actually address the user's question, without drifting "
        "into adjacent but unasked topics?"
    ),
    "completeness": (
        "Does the answer use the relevant material available in the sources, or "
        "does it leave important supporting detail unused?"
    ),
    "citation_quality": (
        "Are [Source X: file=..., page=...] tags present, attached to the specific "
        "claims they support, and pointing at sources that actually contain them?"
    ),
    "language_quality": (
        "Is the answer written fluently and entirely in the requested language, "
        "with correct terminology and no untranslated leftovers?"
    ),
}


class JudgeAgent:
    """Scores generated answers on a fixed rubric using a local LLM."""

    def __init__(self, model_name: str = None):
        self.MODEL_NAME = model_name or config_loader.get("llm.judge_model", "mistral:7b")
        self.scale = config_loader.get("evaluation.judge_scale", 5)
        self.pass_threshold = config_loader.get("evaluation.pass_threshold", 4)

        generator_model = config_loader.get("llm.model")
        self.self_judging = self.MODEL_NAME == generator_model
        if self.self_judging:
            # Not fatal -- it is a legitimate fallback when only one model is
            # available -- but the scores are optimistically biased and every
            # report must say so rather than presenting them as neutral.
            print(
                f"Judge model '{self.MODEL_NAME}' is the SAME as the generation "
                f"model. Scores will be inflated by self-preference bias. "
                f"Set a different llm.judge_model in config.yaml."
            )

        try:
            ollama.show(self.MODEL_NAME)
        except Exception:
            # Deliberately fatal, and deliberately NOT falling back to the
            # generation model: a silent fallback would reintroduce self-judging
            # without appearing anywhere in the report.
            print(f"Error: Judge model '{self.MODEL_NAME}' is not available in Ollama.")
            print(f"Please run 'ollama pull {self.MODEL_NAME}' in your terminal.")
            print(
                "  (This is a second local model, separate from the generation "
                "model. Both run locally via Ollama.)"
            )
            raise

        self.language_agent = LanguageAgent()
        print(f"JudgeAgent initialized. Judge model: {self.MODEL_NAME}")

    # --- prompt construction ----------------------------------------------

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(context):
            source = doc.get("metadata", {}).get("source", "unknown")
            page = doc.get("metadata", {}).get("page", "N/A")
            parts.append(
                f"[Source {i+1}: file={source}, page={page}]\n"
                f"{doc.get('text', '')}"
            )
        return "\n\n".join(parts).strip()

    def _build_prompt(
        self, query: str, context: List[Dict[str, Any]], answer: str, language: str
    ) -> str:
        rubric_lines = "\n".join(
            f'  - "{name}": {description}' for name, description in RUBRIC.items()
        )
        schema_fields = ",\n".join(
            f'    "{name}": {{"score": <1-{self.scale}>, "reason": "<one sentence>"}}'
            for name in RUBRIC
        )
        language_name = self.language_agent.name_of(language)

        return f"""
        [INST]
        You are a strict, impartial evaluator of a Retrieval-Augmented Generation
        system. Score the ANSWER against the SOURCES and the QUERY.

        Score each dimension on an integer scale from 1 (very poor) to {self.scale}
        (excellent):
{rubric_lines}

        The answer was required to be written in {language_name}.

        Be critical. An answer that reads well but states things the sources do
        not support must score low on groundedness. Do not reward length.

        Respond with ONLY a valid JSON object, no markdown fences, no preamble,
        in exactly this shape:
        {{
{schema_fields},
          "overall_comment": "<two sentences summarising the main weakness>"
        }}

        ---
        QUERY:
        "{query}"
        ---
        SOURCES:
        {self._format_context(context)}
        ---
        ANSWER TO EVALUATE:
        {answer}
        ---
        JSON evaluation:
        [/INST]
        """

    # --- response parsing --------------------------------------------------

    def _parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parses the judge's JSON, tolerating fences and surrounding prose."""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Small models often wrap JSON in ```json fences or add a sentence
        # before it; salvage the outermost object rather than failing the run.
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _coerce_score(self, value: Any) -> Optional[int]:
        """Normalises a score to an int within 1..scale, or None if unusable."""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            score = int(round(value))
        elif isinstance(value, str):
            match = re.search(r"\d+", value)
            if not match:
                return None
            score = int(match.group(0))
        else:
            return None
        return max(1, min(self.scale, score))

    # --- public API --------------------------------------------------------

    def run(
        self,
        query: str,
        context: List[Dict[str, Any]],
        answer: str,
        language: str = None,
    ) -> Dict[str, Any]:
        """Judges one answer and returns structured scores.

        Always returns a dict. On failure, `judge_ok` is False and the scores
        are None -- so a flaky judge degrades the report rather than crashing
        the pipeline.
        """
        if language is None:
            language = self.language_agent.detect(query)

        if not answer or not answer.strip():
            return self._empty_result(language, "Answer was empty.")
        if not context:
            return self._empty_result(language, "No context supplied to judge against.")

        print(f"Judging answer with {self.MODEL_NAME}...")
        prompt = self._build_prompt(query, context, answer, language)

        try:
            response = ollama.chat(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                # Deterministic scoring: the same answer should get the same
                # score across runs, otherwise the eval report is not comparable.
                options={"temperature": 0.0},
            )
            raw = response["message"]["content"]
        except Exception as e:
            print(f"Judge LLM call failed: {e}")
            return self._empty_result(language, f"Judge call failed: {e}")

        parsed = self._parse_response(raw)
        if not parsed:
            print("Judge returned unparseable output.")
            return self._empty_result(language, "Judge returned unparseable output.")

        scores: Dict[str, Any] = {}
        valid_scores: List[int] = []
        for dimension in RUBRIC:
            entry = parsed.get(dimension) or {}
            if not isinstance(entry, dict):
                # Some models emit `"relevance": 4` instead of an object.
                entry = {"score": entry, "reason": ""}
            score = self._coerce_score(entry.get("score"))
            scores[dimension] = {
                "score": score,
                "reason": str(entry.get("reason", "")).strip(),
            }
            if score is not None:
                valid_scores.append(score)

        overall = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None

        # An independent, non-LLM check that the output language is right. The
        # judge's own opinion on this is unreliable, so we measure it directly.
        language_check = self.language_agent.response_matches_language(answer, language)

        result = {
            "judge_ok": True,
            "judge_model": self.MODEL_NAME,
            # Recorded per result so a self-judged score is never mistaken for
            # a neutral one further down the line.
            "self_judged": self.self_judging,
            "scale": self.scale,
            "language": language,
            "scores": scores,
            "overall_score": overall,
            "passed": overall is not None and overall >= self.pass_threshold,
            "language_check": language_check,
            "overall_comment": str(parsed.get("overall_comment", "")).strip(),
        }

        if overall is not None:
            print(f"Judge overall score: {overall}/{self.scale}")
        return result

    def _empty_result(self, language: str, reason: str) -> Dict[str, Any]:
        return {
            "judge_ok": False,
            "judge_model": self.MODEL_NAME,
            "self_judged": self.self_judging,
            "scale": self.scale,
            "language": language,
            "scores": {name: {"score": None, "reason": ""} for name in RUBRIC},
            "overall_score": None,
            "passed": False,
            "language_check": None,
            "overall_comment": reason,
        }

    def compare(
        self,
        query: str,
        context: List[Dict[str, Any]],
        answer_a: str,
        answer_b: str,
        label_a: str = "A",
        label_b: str = "B",
        language: str = None,
    ) -> Dict[str, Any]:
        """Pairwise judgement of two answers to the same query.

        Scores both independently on the rubric, then picks a winner. Judging
        each answer on its own avoids the position bias that pairwise prompts
        suffer from, where the first option listed tends to win.
        """
        if language is None:
            language = self.language_agent.detect(query)

        result_a = self.run(query, context, answer_a, language)
        result_b = self.run(query, context, answer_b, language)

        score_a = result_a.get("overall_score")
        score_b = result_b.get("overall_score")

        if score_a is None and score_b is None:
            winner = "undecided"
        elif score_b is None or (score_a is not None and score_a > score_b):
            winner = label_a
        elif score_a is None or score_b > score_a:
            winner = label_b
        else:
            winner = "tie"

        return {
            label_a: result_a,
            label_b: result_b,
            "winner": winner,
            "margin": (
                round(abs(score_a - score_b), 2)
                if score_a is not None and score_b is not None
                else None
            ),
        }


if __name__ == "__main__":
    print("--- Starting Judge Agent (Test Mode) ---")

    demo_context = [
        {
            "text": (
                "The European Green Deal commits the EU to cutting net greenhouse gas "
                "emissions by at least 55% by 2030, relative to 1990 levels."
            ),
            "metadata": {"source": "eu_green_deal.pdf", "page": 4},
        }
    ]

    grounded_answer = (
        "The European Green Deal targets at least a 55% cut in net greenhouse gas "
        "emissions by 2030 against a 1990 baseline "
        "[Source 1: file=eu_green_deal.pdf, page=4]."
    )
    hallucinated_answer = (
        "The European Green Deal requires a 90% emissions cut by 2025 and imposes "
        "a global carbon tax of 200 EUR per tonne on all member states."
    )

    try:
        agent = JudgeAgent()

        print("\n--- Judging a grounded answer ---")
        print(json.dumps(agent.run(
            "What is the EU's 2030 emissions target?", demo_context, grounded_answer, "en"
        ), indent=2))

        print("\n--- Judging a hallucinated answer ---")
        print(json.dumps(agent.run(
            "What is the EU's 2030 emissions target?", demo_context, hallucinated_answer, "en"
        ), indent=2))

        print("\n--- Pairwise comparison ---")
        comparison = agent.compare(
            "What is the EU's 2030 emissions target?",
            demo_context,
            grounded_answer,
            hallucinated_answer,
            label_a="grounded",
            label_b="hallucinated",
        )
        print(f"Winner: {comparison['winner']} (margin: {comparison['margin']})")

    except Exception as e:
        print(f"\n--- Judge Agent Failed ---\nError: {e}")

    print("--- Judge Agent Finished ---")
