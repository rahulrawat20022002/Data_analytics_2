import os
import json
from pathlib import Path
from typing import List, Dict, Any

# We use the same official client
from huggingface_hub import InferenceClient

class EvaluatorAgent:
    """
    This agent is responsible for "LLM-as-Judge" evaluation.
    It loads a comparison file and uses an LLM to judge which
    retrieval method produced better results.
    """
    
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    
    def __init__(self, api_key: str):
        self.client = InferenceClient(token=api_key)
        print(f"✅ EvaluatorAgent initialized. Using model: {self.MODEL_NAME}")

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Formats a list of retrieved docs for the prompt."""
        formatted_str = ""
        for i, doc in enumerate(results):
            formatted_str += f"  Result {i+1} (Score: {doc.get('score', doc.get('rerank_score', 0)):.4f}):\n"
            formatted_str += f"  Text: {doc['text'][:200]}...\n" # Show snippet
            formatted_str += f"  Source: {doc['metadata']['source']}\n\n"
        return formatted_str.strip()

    def run(self, comparison_data: Dict[str, Any]) -> str:
        """
        Runs the LLM-as-Judge evaluation.
        
        Args:
            comparison_data: The dictionary loaded from 'retrieval_comparison.json'.
        
        Returns:
            A string containing the LLM's evaluation.
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
        Conclude your evaluation by stating "Winner: [Baseline (Hybrid) or Advanced (Reranked)]"
        and provide a brief justification.
        
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
        
        Your expert evaluation:
        [/INST]
        """
        
        try:
            print("Calling Hugging Face API for evaluation...")
            
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL_NAME,
                max_tokens=500, # Give it room for a good explanation
            )
            
            evaluation = response.choices[0].message.content
            print("✅ Evaluation complete.")
            return evaluation
            
        except Exception as e:
            print(f"❌ Error during LLM evaluation: {e}")
            return f"Error: Could not generate evaluation. {e}"

if __name__ == "__main__":
    
    # Load API Key
    HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY") # Your key
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not set")

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Input file (from RetrieverExperimentAgent)
    INPUT_FILE = PROJECT_ROOT / "results" / "retrieval_comparison.json"
    
    # We'll just print the evaluation. 
    # The 'metrics.json' is more for the VerifierAgent.
    
    print("--- Starting Evaluator Agent ---")
    
    try:
        agent = EvaluatorAgent(api_key=HF_API_KEY)
        
        # 1. Load the comparison data
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                comparison_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {INPUT_FILE} not found. Run 'retriever_experiment_agent.py' first.")
            exit()

        # 2. Run evaluation
        evaluation_text = agent.run(comparison_data)
        
        print("\n--- LLM-AS-JUDGE EVALUATION ---")
        print(evaluation_text)
            
    except Exception as e:
        print(f"\n--- Evaluator Agent Failed ---")
        print(f"Error: {e}")
        
    print("--- Evaluator Agent Finished ---")