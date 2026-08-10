"""LLM evaluation harness for the multilingual RAG pipeline.

Runs a labelled query set end-to-end and produces a report combining three
families of metric:

  1. Retrieval (needs ground-truth relevant doc ids in the eval set)
       hit@k, precision@k, recall@k, MRR, nDCG@k
  2. Generation (no ground truth needed)
       answer relevancy, context utilisation ("faithfulness" proxy),
       citation density, and the LLM-as-Judge rubric scores
  3. Multilingual
       whether the answer came back in the language that was asked

Everything runs locally: SBERT for the embedding-based metrics, Ollama for the
judge. Results are aggregated overall and broken down per language, since the
whole point of the multilingual work is knowing whether German queries are
served as well as English ones.
"""

import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import config_loader
from judge_agent import JudgeAgent
from language_agent import LanguageAgent

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Sentence splitter shared by the context-utilisation metric.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------
# Pure metric functions (no model calls) -- kept separate so they are testable.
# --------------------------------------------------------------------------

def hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """1.0 if at least one relevant document appears in the top k."""
    return 1.0 if set(retrieved_ids[:k]) & set(relevant_ids) else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in set(relevant_ids))
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of all relevant documents captured in the top k."""
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(set(relevant_ids))


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """1/rank of the first relevant document, 0 if none retrieved."""
    relevant = set(relevant_ids)
    for position, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    dcg = 0.0
    for position, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            # log2(position + 1); position is 1-indexed.
            dcg += 1.0 / np.log2(position + 1)

    # Ideal DCG: every relevant document packed into the top positions.
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

    return float(dcg / idcg) if idcg > 0 else 0.0


def citation_density(answer: str) -> Dict[str, Any]:
    """Counts citation tags and how many sentences carry one."""
    citations = re.findall(r"\[Source \d+", answer or "")
    sentences = [s for s in _SENTENCE_RE.split(answer or "") if s.strip()]
    cited_sentences = sum(1 for s in sentences if "[Source" in s)
    return {
        "citations": len(citations),
        "sentences": len(sentences),
        "cited_sentence_ratio": (
            round(cited_sentences / len(sentences), 4) if sentences else 0.0
        ),
    }


def _mean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------

class EvalAgent:
    """Runs the eval set through a pipeline callable and scores the results."""

    def __init__(self, use_judge: bool = True):
        cfg = config_loader.get_config()["evaluation"]
        self.k_values: List[int] = list(cfg.get("k_values", [1, 3, 5]))
        self.report_json = config_loader.resolve_path(
            cfg.get("report_json", "results/eval_report.json")
        )
        self.report_markdown = config_loader.resolve_path(
            cfg.get("report_markdown", "results/eval_report.md")
        )
        self.eval_set_path = config_loader.resolve_path(
            cfg.get("eval_set", "queries/multilingual_eval_set.json")
        )

        self.language_agent = LanguageAgent()

        # Multilingual encoder, so query/answer similarity is meaningful even
        # when the answer is in German and the source was English.
        model_name = config_loader.get(
            "embedding.model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        print(f"Loading evaluation encoder '{model_name}'...")
        self.encoder = SentenceTransformer(model_name)

        # The judge is optional so the embedding-based metrics can still be
        # collected quickly without paying for an LLM call per query.
        self.judge = JudgeAgent() if use_judge else None

        print("✅ EvalAgent initialized.")

    # --- embedding-based metrics ------------------------------------------

    def _cosine(self, a: str, b: str) -> float:
        if not a or not b or not a.strip() or not b.strip():
            return 0.0
        vectors = self.encoder.encode([a, b])
        norm_a = np.linalg.norm(vectors[0])
        norm_b = np.linalg.norm(vectors[1])
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vectors[0], vectors[1]) / (norm_a * norm_b))

    def answer_relevancy(self, query: str, answer: str) -> float:
        """How semantically close the answer is to the question asked."""
        return round(self._cosine(query, answer), 4)

    def context_utilisation(self, context: List[Dict[str, Any]], answer: str) -> float:
        """Groundedness proxy: mean best-match similarity of answer sentences
        to context sentences.

        This is the cheap, deterministic counterpart to the LLM judge's
        `groundedness` score. A sentence invented out of nothing has no close
        neighbour in the context and drags the mean down.
        """
        if not context or not answer or not answer.strip():
            return 0.0

        answer_sentences = [s.strip() for s in _SENTENCE_RE.split(answer) if s.strip()]
        if not answer_sentences:
            return 0.0

        context_sentences: List[str] = []
        for doc in context:
            context_sentences.extend(
                s.strip() for s in _SENTENCE_RE.split(doc.get("text", "")) if s.strip()
            )
        if not context_sentences:
            return 0.0

        answer_vectors = self.encoder.encode(answer_sentences)
        context_vectors = self.encoder.encode(context_sentences)

        # Normalise once, then a single matmul gives every pairwise cosine.
        answer_vectors = answer_vectors / (
            np.linalg.norm(answer_vectors, axis=1, keepdims=True) + 1e-9
        )
        context_vectors = context_vectors / (
            np.linalg.norm(context_vectors, axis=1, keepdims=True) + 1e-9
        )
        similarity = answer_vectors @ context_vectors.T

        return round(float(np.mean(np.max(similarity, axis=1))), 4)

    # --- per-query evaluation ---------------------------------------------

    def evaluate_case(self, case: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Scores a single query given the pipeline's output for it."""
        query = case["query"]
        expected_language = case.get("language") or self.language_agent.detect(query)
        relevant_ids = case.get("relevant_doc_ids", []) or []
        reference_answer = case.get("reference_answer")

        answer = outcome.get("answer", "") or ""
        retrieved = outcome.get("retrieved_docs", []) or []
        retrieved_ids = [doc.get("id") for doc in retrieved if doc.get("id")]

        record: Dict[str, Any] = {
            "id": case.get("id"),
            "query": query,
            "language": expected_language,
            "answer": answer,
            "latency_ms": outcome.get("latency_ms"),
            "retrieved_ids": retrieved_ids,
            "retrieved_languages": sorted(
                {doc.get("language", "unknown") for doc in retrieved}
            ),
        }

        # 1. Retrieval metrics -- only meaningful with ground-truth labels.
        if relevant_ids:
            retrieval: Dict[str, Any] = {"labelled": True}
            for k in self.k_values:
                retrieval[f"hit@{k}"] = hit_at_k(retrieved_ids, relevant_ids, k)
                retrieval[f"precision@{k}"] = round(
                    precision_at_k(retrieved_ids, relevant_ids, k), 4
                )
                retrieval[f"recall@{k}"] = round(
                    recall_at_k(retrieved_ids, relevant_ids, k), 4
                )
                retrieval[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4)
            retrieval["mrr"] = round(reciprocal_rank(retrieved_ids, relevant_ids), 4)
            record["retrieval"] = retrieval
        else:
            record["retrieval"] = {"labelled": False}

        # 2. Generation metrics.
        generation: Dict[str, Any] = {
            "answer_relevancy": self.answer_relevancy(query, answer),
            "context_utilisation": self.context_utilisation(retrieved, answer),
            "citations": citation_density(answer),
        }
        if reference_answer:
            generation["answer_similarity_to_reference"] = round(
                self._cosine(reference_answer, answer), 4
            )
        record["generation"] = generation

        # 3. Multilingual check.
        record["language_check"] = self.language_agent.response_matches_language(
            answer, expected_language
        )

        # 4. LLM-as-Judge rubric.
        if self.judge and answer.strip() and retrieved:
            record["judge"] = self.judge.run(query, retrieved, answer, expected_language)
        else:
            record["judge"] = None

        return record

    # --- aggregation -------------------------------------------------------

    def aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Builds overall and per-language summaries."""

        def summarise(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not subset:
                return {}

            summary: Dict[str, Any] = {"queries": len(subset)}

            labelled = [r for r in subset if r["retrieval"].get("labelled")]
            if labelled:
                retrieval_summary: Dict[str, Any] = {"labelled_queries": len(labelled)}
                for k in self.k_values:
                    for metric in ("hit", "precision", "recall", "ndcg"):
                        key = f"{metric}@{k}"
                        retrieval_summary[key] = _mean(
                            [r["retrieval"].get(key) for r in labelled]
                        )
                retrieval_summary["mrr"] = _mean([r["retrieval"].get("mrr") for r in labelled])
                summary["retrieval"] = retrieval_summary

            summary["generation"] = {
                "answer_relevancy": _mean(
                    [r["generation"]["answer_relevancy"] for r in subset]
                ),
                "context_utilisation": _mean(
                    [r["generation"]["context_utilisation"] for r in subset]
                ),
                "cited_sentence_ratio": _mean(
                    [r["generation"]["citations"]["cited_sentence_ratio"] for r in subset]
                ),
            }

            reference_scores = [
                r["generation"].get("answer_similarity_to_reference") for r in subset
            ]
            if any(s is not None for s in reference_scores):
                summary["generation"]["answer_similarity_to_reference"] = _mean(
                    reference_scores
                )

            matches = [r["language_check"]["match"] for r in subset if r.get("language_check")]
            summary["language_match_rate"] = (
                round(sum(1 for m in matches if m) / len(matches), 4) if matches else None
            )

            judged = [r["judge"] for r in subset if r.get("judge") and r["judge"].get("judge_ok")]
            if judged:
                judge_summary: Dict[str, Any] = {
                    "judged_queries": len(judged),
                    "overall_score": _mean([j["overall_score"] for j in judged]),
                    "pass_rate": round(
                        sum(1 for j in judged if j.get("passed")) / len(judged), 4
                    ),
                }
                dimensions = judged[0]["scores"].keys()
                judge_summary["by_dimension"] = {
                    dimension: _mean([j["scores"][dimension]["score"] for j in judged])
                    for dimension in dimensions
                }
                summary["judge"] = judge_summary

            latencies = [r["latency_ms"] for r in subset if r.get("latency_ms")]
            summary["latency_ms"] = {
                "mean": _mean(latencies),
                "max": round(max(latencies), 2) if latencies else None,
            }

            return summary

        languages = sorted({r["language"] for r in records})
        return {
            "overall": summarise(records),
            "by_language": {
                lang: summarise([r for r in records if r["language"] == lang])
                for lang in languages
            },
        }

    # --- driving the pipeline ---------------------------------------------

    def load_eval_set(self, path: Path = None) -> List[Dict[str, Any]]:
        path = path or self.eval_set_path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = data["cases"] if isinstance(data, dict) else data
        print(f"Loaded {len(cases)} evaluation cases from {path}")
        return cases

    def run(
        self,
        run_fn: Callable[[str, str], Dict[str, Any]],
        cases: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs every case through `run_fn` and scores the outputs.

        Args:
            run_fn: Callable taking (query, language) and returning a dict with
                at least `answer` and `retrieved_docs`.
            cases: Eval cases; loaded from the configured eval set when omitted.
        """
        cases = cases if cases is not None else self.load_eval_set()

        records: List[Dict[str, Any]] = []
        for i, case in enumerate(cases, start=1):
            query = case["query"]
            language = case.get("language") or self.language_agent.detect(query)
            print(f"\n=== EVAL CASE {i}/{len(cases)} [{language}] {query[:60]} ===")

            start = time.time()
            try:
                outcome = run_fn(query, language)
            except Exception as e:
                print(f"❌ Pipeline failed on this case: {e}")
                outcome = {"answer": "", "retrieved_docs": [], "error": str(e)}
            outcome.setdefault("latency_ms", round((time.time() - start) * 1000, 2))

            records.append(self.evaluate_case(case, outcome))

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "generation_model": config_loader.get("llm.model"),
                "judge_model": config_loader.get("llm.judge_model"),
                # Flagged in the report because self-judged scores are inflated
                # and should not be compared against independently judged runs.
                "self_judged": (
                    config_loader.get("llm.judge_model") == config_loader.get("llm.model")
                ),
                "embedding_model": config_loader.get("embedding.model"),
                "index": config_loader.get("retrieval.index_name"),
                "alpha": config_loader.get("retrieval.alpha"),
                "k_values": self.k_values,
            },
            "summary": self.aggregate(records),
            "cases": records,
        }
        return report

    # --- reporting ---------------------------------------------------------

    def save_report(self, report: Dict[str, Any]) -> None:
        self.report_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ JSON report written to {self.report_json}")

        markdown = self.render_markdown(report)
        with open(self.report_markdown, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"✅ Markdown report written to {self.report_markdown}")

    def render_markdown(self, report: Dict[str, Any]) -> str:
        """Renders a human-readable summary table."""
        lines: List[str] = ["# RAG Evaluation Report", ""]
        lines.append(f"_Generated: {report['generated_at']}_")
        lines.append("")

        cfg = report["config"]
        if cfg.get("self_judged"):
            lines.append(
                "> ⚠️ **Self-judged run.** The judge and the generator are the same "
                "model, so rubric scores are inflated by self-preference bias and "
                "are not comparable with independently judged runs."
            )
            lines.append("")

        lines.append("## Configuration")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("| --- | --- |")
        for key, value in cfg.items():
            lines.append(f"| {key} | `{value}` |")
        lines.append("")

        def fmt(value: Any) -> str:
            return "n/a" if value is None else (
                f"{value:.3f}" if isinstance(value, float) else str(value)
            )

        summary = report["summary"]
        sections = [("Overall", summary["overall"])] + [
            (f"Language: {lang}", data) for lang, data in summary["by_language"].items()
        ]

        for title, data in sections:
            if not data:
                continue
            lines.append(f"## {title}")
            lines.append("")
            lines.append(f"Queries: **{data.get('queries', 0)}**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")

            retrieval = data.get("retrieval", {})
            for key, value in retrieval.items():
                if key != "labelled_queries":
                    lines.append(f"| retrieval / {key} | {fmt(value)} |")

            for key, value in data.get("generation", {}).items():
                lines.append(f"| generation / {key} | {fmt(value)} |")

            lines.append(f"| language match rate | {fmt(data.get('language_match_rate'))} |")

            judge = data.get("judge", {})
            if judge:
                lines.append(f"| judge / overall | {fmt(judge.get('overall_score'))} |")
                lines.append(f"| judge / pass rate | {fmt(judge.get('pass_rate'))} |")
                for dimension, value in judge.get("by_dimension", {}).items():
                    lines.append(f"| judge / {dimension} | {fmt(value)} |")

            latency = data.get("latency_ms", {})
            lines.append(f"| latency ms (mean) | {fmt(latency.get('mean'))} |")
            lines.append("")

        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the multilingual RAG pipeline.")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-Judge pass (much faster; embedding metrics only).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N cases."
    )
    parser.add_argument(
        "--eval-set", type=str, default=None, help="Path to an alternative eval set."
    )
    args = parser.parse_args()

    print("--- Starting Eval Agent ---")

    # Imported here rather than at module scope so that `--help` and the pure
    # metric functions do not require Pinecone credentials or a running Ollama.
    from orchestrator import build_pipeline

    agent = EvalAgent(use_judge=not args.no_judge)

    cases = agent.load_eval_set(Path(args.eval_set) if args.eval_set else None)
    if args.limit:
        cases = cases[: args.limit]

    pipeline = build_pipeline()

    def run_fn(query: str, language: str) -> Dict[str, Any]:
        final_state = pipeline.invoke_query(query, language=language)
        return {
            "answer": final_state.get("final_response", ""),
            "retrieved_docs": final_state.get("retrieved_docs", []),
            "latency_ms": final_state.get("run_latency_ms"),
        }

    report = agent.run(run_fn, cases)
    agent.save_report(report)

    overall = report["summary"]["overall"]
    print("\n--- SUMMARY ---")
    print(f"Queries evaluated: {overall.get('queries')}")
    print(f"Language match rate: {overall.get('language_match_rate')}")
    if overall.get("judge"):
        print(f"Judge overall score: {overall['judge'].get('overall_score')}")

    print("--- Eval Agent Finished ---")
