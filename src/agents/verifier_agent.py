import os
import json
from pathlib import Path
from typing import List, Dict, Any
import re

# We now use the InferenceClient for NLI
from huggingface_hub import InferenceClient

# For Semantic Alignment Check
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

class VerifierAgent:
    """
    This agent is responsible for Task 6:
    1. Checking factuality (using an LLM as a judge).
    2. Checking semantic alignment (cosine similarity).
    3. Checking for citations.
    """
    
    # We will use Mistral for NLI, not the old BART model
    LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    SIMILARITY_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, api_key: str):
        # This client will be used for our NLI check
        self.llm_client = InferenceClient(token=api_key)
        self.similarity_model = SentenceTransformer(self.SIMILARITY_MODEL)
        print("✅ VerifierAgent initialized.")

    def _load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Helper to load JSON data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {file_path} not found.")
            return []

    def _load_text_data(self, file_path: str) -> str:
        """Helper to load raw text data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"❌ Error: {file_path} not found.")
            return ""

    def check_semantic_alignment(self, query: str, summary: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Checks if the summary is semantically aligned with the query.
        """
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
        """
        Checks if the summary is factually consistent with the context
        using our Mistral LLM as a judge.
        """
        print("Running factuality (LLM-as-Judge) check...")
        
        # --- THIS IS THE FIX ---
        # The key is 'text', not 'processed_text'.
        context_str = " ".join([doc['text'] for doc in context_docs])
        # -----------------------
        
        sentences = [s.strip() for s in re.split(r'[.!?]', summary) if s.strip()]
        if not sentences:
            return {"check": "factuality_nli", "contradictions": 0, "total_sentences": 0, "pass": "✅"}

        contradictions = 0
        nli_results = []

        for sentence in sentences:
            if not sentence:
                continue
            
            # Create a prompt to ask the LLM to fact-check
            prompt = f"""
            [INST]
            You are a fact-checker. Your task is to determine if the "Statement" is
            *directly supported* by the "Context".
            
            You must respond with *only* one of three labels:
            "entailment", "neutral", or "contradiction".

            - "entailment": The Context directly supports the Statement.
            - "contradiction": The Context directly contradicts the Statement.
            - "neutral": The Context does not contain enough information to support or contradict the Statement.

            Context: "{context_str}"

            Statement: "{sentence}"

            Label:
            [/INST]
            """
            
            try:
                # Use the chat_completion method we know works
                response = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.LLM_MODEL_NAME,
                    max_tokens=10, # We only need one word
                )
                
                best_label = response.choices[0].message.content.strip().lower()
                
                # Clean up the label
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
        """
        Checks if the summary contains citations (e.g., [Source 1...]).
        """
        print("Running citation check...")
        citations_found = re.findall(r"\[Source \d+", summary)
        
        has_citations = len(citations_found) > 0
        
        return {
            "check": "citation_check",
            "citations_found": len(citations_found),
            "pass": "✅" if has_citations else "❌"
        }

    def run(self, query: str, context: List[Dict[str, Any]], summary: str) -> List[Dict[str, Any]]:
        """
        Runs all verification checks.
        """
        metrics = []
        metrics.append(self.check_semantic_alignment(query, summary))
        metrics.append(self.check_factuality_nli(context, summary))
        metrics.append(self.check_citations(summary))
        
        return metrics

def save_metrics(metrics: List[Dict[str, Any]], output_file: str):
    """Saves the verification metrics to the specified JSON file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    
    # Load API Key
    HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or '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' # Your key
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not set")

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # --- INPUTS ---
    SUMMARY_FILE = PROJECT_ROOT / "results" / "final_policy_brief.txt"
    
    # --- OUTPUT ---
    OUTPUT_METRICS_FILE = PROJECT_ROOT / "results" / "metrics.json"
    
    print("--- Starting Verifier Agent (Test Mode) ---")
    
    try:
        agent = VerifierAgent(api_key=HF_API_KEY)
        
        test_query = "What are the main policies regarding climate change?"
        
        # Use the *actual* context that the summary was based on
        test_context = [
             {
                "id": "chunk_90", "score": 0.5496,
                # THIS IS THE KEY: 'text'
                "text": "CHAPTER 1 gLObaL PROSPECTS aND POLICIES... emissions reduction targets, countries need a holistic set of mitigation instruments, ideally including carbon pricing, public infrastructure investment in clean en...",
                "metadata": {"source": "ch1.pdf", "page": 22}
            },
            {
                "id": "chunk_1234", "score": 0.5210,
                # THIS IS THE KEY: 'text'
                "text": "The European Union has committed to its 'Green Deal', aiming for carbon neutrality by 2050. This involves significant investment in renewable energy and a circular economy. However, some critics point to the high cost and the challenge of implementation across all member states.",
                "metadata": {"source": "eu_policy.pdf", "page": 5}
            }
        ]
        
        test_summary = agent._load_text_data(str(SUMMARY_FILE))
        
        if not test_context or not test_summary:
            raise FileNotFoundError("Could not load input files. Run previous agents first.")

        # 2. Run verification
        all_metrics = agent.run(test_query, test_context, test_summary)
        
        # 3. Save metrics
        save_metrics(all_metrics, str(OUTPUT_METRICS_FILE))
        
        print(f"\n✅ Successfully ran verification checks.")
        print(f"Saved metrics to {OUTPUT_METRICS_FILE}")
        print("\n--- METRICS ---")
        print(json.dumps(all_metrics, indent=2))
            
    except Exception as e:
        print(f"\n--- Verifier Agent Failed ---")
        print(f"Error: {e}")
        
    print("--- Verifier Agent Finished ---")