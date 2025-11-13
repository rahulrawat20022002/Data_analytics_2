if __name__ == "__main__":
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    OUTPUT_FILE = PROJECT_ROOT / "results" / "retrieval_comparison.json"
    
    print("--- Starting Retriever Experiment Agent ---")
    
    try:
        # --- THIS IS THE FIX ---
        # Load keys to pass to the baseline agent
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or 'pcsk_3qfgyB_LkDmDwdc6gWHQtfaZgDX4jm2Q6BREGDSeG18jNrbz2GJFE9kjoU5Rmth2io1CmV'
        PINECONE_REGION = os.getenv("PINECONE_REGION") or 'us-east-1'
        
        if "YOUR_PINECONE" in PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set. Set it in the script or as env variable.")

        # 1. Initialize the baseline agent first
        baseline_agent = RetrieverAgent(api_key=PINECONE_API_KEY, region=PINECONE_REGION, alpha=0.5)
        # ------------------------
        
        # 2. Initialize the experiment agent, giving it the baseline
        experiment_agent = RetrieverExperimentAgent(baseline_retriever=baseline_agent)
        
        # 3. Run a test query
        test_query = "What are the main policies regarding climate change and renewable energy?"

        # --- Run Baseline ---
        print("\n--- Running Baseline (Hybrid) Search ---")
        baseline_results = baseline_agent.search(test_query, top_k=5)
        
        # --- Run Advanced ---
        print("\n--- Running Advanced (Reranking) Search ---")
        advanced_results = experiment_agent.search(test_query, top_k=5, candidate_k=25)

        # --- 4. Compare and Save ---
        print(f"\nSaving comparison to {OUTPUT_FILE}...")
        save_comparison(baseline_results, advanced_results, test_query, str(OUTPUT_FILE))

        print("\n--- Top 3 Baseline Results ---")
        for i, res in enumerate(baseline_results[:3]):
            print(f"Rank {i+1} (Score: {res['score']:.4f}) Text: {res['text'][:100]}...")

        print("\n--- Top 3 Advanced Results (Reranked) ---")
        for i, res in enumerate(advanced_results[:3]):
            print(f"Rank {i+1} (Score: {res['rerank_score']:.4f}) Text: {res['text'][:100]}...")

        print(f"\n✅ Experiment complete. Compare results in {OUTPUT_FILE}.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    
    print("--- Retriever Experiment Agent Finished ---")