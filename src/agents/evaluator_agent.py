import json
import re
from pathlib import Path
from typing import Any, Dict, List

# Runs on Ollama like the rest of the pipeline (previously the HuggingFace
# Inference API), so evaluation needs no network access or API key.
import ollama

import config_loader


class EvaluatorAgent:
    """
    "LLM-as-Judge" evaluation for *retrieval strategies*.

    It loads a comparison file and uses an LLM to judge which retrieval method
    produced better results. For judging the *final generated answer*, see
    judge_agent.JudgeAgent.
    """

    def __init__(self, model_name: str = None):
        self.MODEL_NAME = model_name or config_loader.get("llm.judge_model", "mistral:7b")
        try:
            ollama.show(self.MODEL_NAME)
        except Exception:
            print(f"Error: Ollama model '{self.MODEL_NAME}' not found.")
            print(f"Please run 'ollama pull {self.MODEL_NAME}' in your terminal.")
            raise
        print(f"EvaluatorAgent initialized. Using Ollama model: {self.MODEL_NAME}")

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Formats a list of retrieved docs for the prompt."""
        formatted_str = ""
        for i, doc in enumerate(results):
            score = doc.get('score', doc.get('rerank_score', 0)) or 0
            formatted_str += f"  Result {i+1} (Score: {float(score):.4f}):\n"
            formatted_str += f"  Text: {doc['text'][:200]}...\n" # Show snippet
            formatted_str += f"  Source: {doc['metadata']['source']}\n"
            # Surfaced so the judge can spot cross-lingual retrieval behaviour.
            formatted_str += f"  Language: {doc.get('language', 'unknown')}\n\n"
        return formatted_str.strip()

    def run(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the LLM-as-Judge evaluation over a retrieval comparison.

        Args:
            comparison_data: The dictionary loaded from 'retrieval_comparison.json'.

        Returns:
            A dict with the winner, the justification, and the raw evaluation text.
        """
        query = comparison_data['query']
        baseline_results = self._format_results(comparison_data['baseline_results (hybrid)'])
        advanced_results = self._format_results(comparison_data['advanced_results (reranked)'])

        print(f"Evaluating retrieval quality for query: '{query}'")

        prompt = f"""
        [INST]
        You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.
        Your task is to compare two sets of retrieval results for a given query and
        determine which set is better.

        - "Baseline (Hybrid)" uses a standard BM25 + dense vector search.
        - "Advanced (Reranked)" uses a Cross-Encoder to rerank the baseline results.

        Evaluate based on **relevance**, **clarity**, and **factuality**.
        The query and the documents may be in different languages; a document in
        another language is still relevant if it answers the query.

        Respond with ONLY a valid JSON object, no markdown fences and no preamble:
        {{
          "winner": "baseline" | "advanced" | "tie",
          "justification": "<two to three sentences explaining the decision>",
          "relevance_comment": "<one sentence>",
          "factuality_comment": "<one sentence>"
        }}

        ---
        QUERY:
        "{query}"
        ---

        --- SET 1: Baseline (Hybrid) Results ---
        {baseline_results}
        ---

        --- SET 2: Advanced (Reranked) Results ---
        {advanced_results}
        ---

        JSON evaluation:
        [/INST]
        """

        try:
            print(f"Calling Ollama model: {self.MODEL_NAME} for evaluation...")

            response = ollama.chat(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                # Deterministic so repeated runs are comparable.
                options={"temperature": 0.0},
            )

            raw = response['message']['content']
            parsed = self._parse_json(raw)

            if not parsed:
                print("Evaluator returned unparseable output; keeping raw text.")
                return {
                    "evaluator_ok": False,
                    "winner": "undecided",
                    "justification": "",
                    "raw": raw,
                }

            winner = str(parsed.get("winner", "")).strip().lower()
            if winner not in {"baseline", "advanced", "tie"}:
                winner = "undecided"

            print(f"Evaluation complete. Winner: {winner}")
            return {
                "evaluator_ok": True,
                "winner": winner,
                "justification": str(parsed.get("justification", "")).strip(),
                "relevance_comment": str(parsed.get("relevance_comment", "")).strip(),
                "factuality_comment": str(parsed.get("factuality_comment", "")).strip(),
                "raw": raw,
            }

        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            return {
                "evaluator_ok": False,
                "winner": "undecided",
                "justification": f"Error: Could not generate evaluation. {e}",
                "raw": "",
            }

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """Parses JSON, tolerating markdown fences or surrounding prose."""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {}
            return {}


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # Input file (from RetrieverExperimentAgent)
    INPUT_FILE = PROJECT_ROOT / "results" / "retrieval_comparison.json"
    OUTPUT_FILE = PROJECT_ROOT / "results" / "retrieval_evaluation.json"

    print("--- Starting Evaluator Agent ---")

    try:
        agent = EvaluatorAgent()

        # 1. Load the comparison data
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                comparison_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: {INPUT_FILE} not found. Run 'retriever_experiment_agent.py' first.")
            exit()

        # 2. Run evaluation
        evaluation = agent.run(comparison_data)

        print("\n--- LLM-AS-JUDGE EVALUATION (RETRIEVAL) ---")
        print(f"Winner: {evaluation['winner']}")
        print(f"Justification: {evaluation.get('justification', '')}")

        # 3. Persist it alongside the other results
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation to {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n--- Evaluator Agent Failed ---")
        print(f"Error: {e}")

    print("--- Evaluator Agent Finished ---")
