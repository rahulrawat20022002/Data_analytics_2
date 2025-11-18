import os
import json
from pathlib import Path
from typing import List, Dict, Any
import re
import ollama
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

class VerifierAgent:
    LLM_MODEL_NAME = "mistral:7b"
    SIMILARITY_MODEL = "all-MiniLM-L6-v2"
    def __init__(self):
        try:
            ollama.show(self.LLM_MODEL_NAME)
        except Exception:
            print(f"❌ Error: Ollama model '{self.LLM_MODEL_NAME}' not found.")
            print(f"Please run 'ollama pull {self.LLM_MODEL_NAME}' in your terminal.")
            raise
        self.similarity_model = SentenceTransformer(self.SIMILARITY_MODEL)
        print("✅ VerifierAgent initialized.")
    def _load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {file_path} not found.")
            return []
    def _load_text_data(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"❌ Error: {file_path} not found.")
            return ""
    def check_semantic_alignment(self, query: str, summary: str, threshold: float = 0.5) -> Dict[str, Any]:
        print("Running semantic alignment check...")
        embeddings = self.similarity_model.encode([query, summary])
        cos_sim = semantic_search(embeddings[0], embeddings[1], top_k=1)[0][0]
        is_aligned = cos_sim['score'] >= threshold
        return {
            "check": "semantic_alignment",
            "aligned": is_aligned,
            "score": cos_sim['score'],
            "threshold": threshold,
            "pass": "✅" if is_aligned else "❌"
        }
    def check_factuality_nli(self, context_docs: List[Dict[str, Any]], summary: str) -> Dict[str, Any]:
        print("Running factuality (LLM-as-Judge) check...")
        context_str = " ".join([doc['text'] for doc in context_docs])
        sentences = [s.strip() for s in re.split(r'[.!?]', summary) if s.strip()]
        if not sentences:
            return {"check": "factuality_nli", "contradictions": 0, "total_sentences": 0, "pass": "✅"}
        contradictions = 0
        nli_results = []
        for sentence in sentences:
            if not sentence:
                continue
            prompt = f"""
            [INST]
            You are a fact-checker. Your task is to determine if the "Statement" is
            *directly supported* by the "Context".
            You must respond with *only* one of three labels:
            "entailment", "neutral", or "contradiction".
            Context: "{context_str}"
            Statement: "{sentence}"
            Label:
            [/INST]
            """
            try:
                response = ollama.chat(
                    model=self.LLM_MODEL_NAME,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                best_label = response['message']['content'].strip().lower()
                if "contradiction" in best_label:
                    best_label = "contradiction"
                elif "entailment" in best_label:
                    best_label = "entailment"
                else:
                    best_label = "neutral"
                nli_results.append({"sentence": sentence, "label": best_label})
                if best_label == "contradiction":
                    contradictions += 1
            except Exception as e:
                print(f"  - NLI error for sentence: {e}")
        is_factual = contradictions == 0
        return {
            "check": "factuality_nli",
            "contradictions": contradictions,
            "total_sentences": len(sentences),
            "results": nli_results,
            "pass": "✅" if is_factual else "❌"
        }
    def check_citations(self, summary: str) -> Dict[str, Any]:
        print("Running citation check...")
        citations_found = re.findall(r"\[Source \d+", summary)
        has_citations = len(citations_found) > 0
        return {
            "check": "citation_check",
            "citations_found": len(citations_found),
            "pass": "✅" if has_citations else "❌"
        }
    def run(self, query: str, context: List[Dict[str, Any]], summary: str) -> List[Dict[str, Any]]:
        metrics = []
        metrics.append(self.check_semantic_alignment(query, summary))
        metrics.append(self.check_factuality_nli(context, summary))
        metrics.append(self.check_citations(summary))
        return metrics