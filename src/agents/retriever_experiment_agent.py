import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import our baseline retriever to get the initial candidates
from retriever_agent import RetrieverAgent

# Import the CrossEncoder model
from sentence_transformers import CrossEncoder

# --- NEW IMPORTS ---
import matplotlib.pyplot as plt
import pandas as pd
# -------------------
load_dotenv()


class RetrieverExperimentAgent:
    """
    This agent is responsible for Task 8: Advanced Retrieval.
    It implements a Cross-Encoder Reranking strategy.
    """
    
    def __init__(self, baseline_retriever: RetrieverAgent):
        """
        Initializes the agent with the baseline retriever.
        """
        self.baseline_retriever = baseline_retriever
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ RetrieverExperimentAgent (Reranker) initialized.")

    def search(self, query: str, top_k: int = 5, candidate_k: int = 25) -> List[Dict[str, Any]]:
        """
        Performs an advanced search.
        """
        print(f"Running advanced search. Fetching {candidate_k} candidates...")
        
        # 1. Retrieve (Fast)
        self.baseline_retriever.alpha = 0.5
        candidate_docs = self.baseline_retriever.search(query, top_k=candidate_k)
        
        if not candidate_docs:
            print("No candidates found by baseline retriever.")
            return []
            
        print(f"Got {len(candidate_docs)} candidates. Reranking with CrossEncoder...")

        # 2. Rerank (Smart)
        doc_texts = [doc['text'] for doc in candidate_docs]
        query_pairs = [[query, doc_text] for doc_text in doc_texts]
        
        scores = self.reranker.predict(query_pairs)
        
        reranked_results = []
        for i, doc in enumerate(candidate_docs):
            doc['rerank_score'] = float(scores[i]) # <-- Fixed float32 error
            doc['original_hybrid_score'] = float(doc['score']) # <-- Fixed float32 error
            reranked_results.append(doc)
            
        # Sort the candidates by the new rerank_score
        sorted_results = sorted(reranked_results, key=lambda x: x['rerank_score'], reverse=True)
        
        print("✅ Reranking complete.")
        
        # Return the new top_k results
        return sorted_results[:top_k]

def save_comparison(baseline: List[Dict], advanced: List[Dict], query: str, output_file: str):
    """
    Saves the results from both pipelines to the comparison JSON.
    """
    comparison_data = {
        "query": query,
        "baseline_results (hybrid)": baseline,
        "advanced_results (reranked)": advanced
    }
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

# --- NEW PLOTTING FUNCTION ---
def plot_comparison(comparison_data: Dict[str, Any], output_file: str):
    """
    Plots a bar chart comparing the scores of baseline vs advanced.
    """
    print(f"Plotting retrieval comparison to {output_file}...")
    
    try:
        # Extract scores
        baseline_scores = [doc['score'] for doc in comparison_data['baseline_results (hybrid)']]
        advanced_scores = [doc['rerank_score'] for doc in comparison_data['advanced_results (reranked)']]
        
        # Create a DataFrame
        data = {
            'Baseline Score': pd.Series(baseline_scores),
            'Advanced Score': pd.Series(advanced_scores)
        }
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 7))
        ax = df.plot(kind='bar', 
                     title=f"Retrieval Score Comparison\nQuery: {comparison_data['query'][:50]}...",
                     rot=0)
        
        ax.set_xlabel("Document Rank (Top 5)")
        ax.set_ylabel("Relevance Score")
        ax.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=100)
        print("✅ Retrieval plot saved.")
        plt.close()
        
    except Exception as e:
        print(f"❌ Error plotting retrieval comparison: {e}")

# -----------------------------
        
if __name__ == "__main__":
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    OUTPUT_FILE_JSON = PROJECT_ROOT / "results" / "retrieval_comparison.json"
    # --- NEW PLOT FILE PATH ---
    OUTPUT_FILE_PNG = PROJECT_ROOT / "results" / "plots" / "retrieval_plot.png"
    
    print("--- Starting Retriever Experiment Agent ---")
    
    try:
        # --- FIX: Load keys to pass to the baseline agent ---
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # PASTE YOUR KEY
        PINECONE_REGION = os.getenv("PINECONE_REGION") or 'us-east-1'
        
        if "YOUR_PINECONE" in PINECONE_API_KEY or "..." in PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set in retriever_experiment_agent.py. Set it in the script or as env variable.")

        # 1. Initialize the baseline agent first
        baseline_agent = RetrieverAgent(api_key=PINECONE_API_KEY, region=PINECONE_REGION, alpha=0.5)
        # ------------------------
        
        # 2. Initialize the experiment agent
        experiment_agent = RetrieverExperimentAgent(baseline_retriever=baseline_agent)
        
        # 3. Run a test query
        test_query = "What are the main policies regarding climate change and renewable energy?"

        print("\n--- Running Baseline (Hybrid) Search ---")
        baseline_results = baseline_agent.search(test_query, top_k=5)
        
        print("\n--- Running Advanced (Reranking) Search ---")
        advanced_results = experiment_agent.search(test_query, top_k=5, candidate_k=25)

        # 4. Save JSON comparison
        print(f"\nSaving comparison to {OUTPUT_FILE_JSON}...")
        save_comparison(baseline_results, advanced_results, test_query, str(OUTPUT_FILE_JSON))

        # --- 5. CREATE AND SAVE PLOT (NEW) ---
        plot_comparison({
            "query": test_query,
            "baseline_results (hybrid)": baseline_results,
            "advanced_results (reranked)": advanced_results
        }, str(OUTPUT_FILE_PNG))
        # -------------------------------------

        print(f"\n✅ Experiment complete. Compare results in {OUTPUT_FILE_JSON} and {OUTPUT_FILE_PNG}.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    
    print("--- Retriever Experiment Agent Finished ---")