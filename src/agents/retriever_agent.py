import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Pinecone client for dense vector search
from pinecone import Pinecone, ServerlessSpec 

# BM25 for sparse keyword search
from rank_bm25 import BM25Okapi

class RetrieverAgent:
    """
    This agent is responsible for Task 3: Hybrid Retrieval.
    (Updated to accept API key in __init__)
    """
    
    def __init__(self, api_key: str, region: str, alpha: float = 0.5):
        """
        Initializes the agent, Pinecone connection, and loads/builds indices.
        
        Args:
            api_key: The Pinecone API key.
            region: The Pinecone region (e.g., "us-east-1").
            alpha: The weighting factor for hybrid search.
        """
        print("Initializing RetrieverAgent...")
        self.alpha = alpha
        
        # --- PATHS ---
        # --- THIS IS THE FIX ---
        # It now goes up 3 levels to the project root
        self.project_root = Path(__file__).parent.parent.parent
        self.data_file = self.project_root / "results" / "classical_output.json"
        self.vector_file = self.project_root / "results" / "sbert_embeddings.npy"
        
        # --- PINEONE SETUP ---
        self.PINECONE_API_KEY = api_key
        self.PINECONE_REGION = region
        # ---------------------
        
        self.index_name = "policy-rag-index"
        
        if self.PINECONE_API_KEY is None or "YOUR_API_KEY" in self.PINECONE_API_KEY:
             raise ValueError("A valid Pinecone API Key was not passed to RetrieverAgent.")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        
        # Load text data
        print("Loading processed text data...")
        self.documents = self._load_documents()
        self.corpus = [doc['processed_text'] for doc in self.documents]

        # 1. Build BM25 (Sparse) Index
        print("Building BM25 sparse index...")
        self.bm25 = self._build_bm25_index(self.corpus)
        
        # 2. Get/Build Pinecone (Dense) Index
        print("Connecting to Pinecone dense index...")
        self.pinecone_index = self._get_pinecone_index()
        
        print(f"✅ RetrieverAgent initialized. Ready for hybrid search (alpha={self.alpha}).")

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Loads the processed chunks from 'classical_output.json'."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {self.data_file} not found. Run preprocessor_agent.py first.")
            return []

    def _build_bm25_index(self, corpus: List[str]) -> BM25Okapi:
        """Creates a BM25 index from the document corpus."""
        if not corpus:
            print("⚠️ Warning: Corpus is empty. BM25 index will be empty.")
            # Return a dummy/empty BM25 object
            return BM25Okapi([])
            
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        return BM25Okapi(tokenized_corpus)

    def _get_pinecone_index(self):
        """Connects to Pinecone, checks if the index exists, and upserts data if needed."""
        
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Index '{self.index_name}' not found. Creating a new one...")
            print("This can take a minute...")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=384, # all-MiniLM-L6-v2 dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=self.PINECONE_REGION
                )
            )
        
        index = self.pc.Index(self.index_name)
        
        # Check if index is already populated
        if not self.documents:
            print("Documents not loaded, skipping Pinecone population check.")
            return index
            
        stats = index.describe_index_stats()
        if stats['total_vector_count'] >= len(self.documents):
            print(f"Pinecone index '{self.index_name}' is already populated.")
            return index

        # If not populated, load vectors and upsert
        print("Loading SBERT embeddings for upsert...")
        try:
            vectors = np.load(self.vector_file)
        except FileNotFoundError:
            print(f"❌ Error: {self.vector_file} not found. Run embedding_agent.py first.")
            return None

        print(f"Upserting {len(self.documents)} vectors to Pinecone... (This will take time)")
        
        vectors_to_upsert = []
        for i, doc in enumerate(self.documents):
            vector_id = doc['id']
            vector = vectors[i].tolist()
            metadata = {
                "text": doc['processed_text'],
                "source": doc['metadata'].get('source', 'unknown'),
                "page": int(doc['metadata'].get('page', 0))
            }
            vectors_to_upsert.append({"id": vector_id, "values": vector, "metadata": metadata})

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            index.upsert(vectors=batch)
            print(f"  ...upserted batch {i//batch_size + 1}")
            
        print("✅ Pinecone upsert complete.")
        return index

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search.
        """
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2') 
        
        query_vector = sbert_model.encode(query).tolist()
        
        dense_results = self.pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # Check if BM25 index is empty
        if not self.corpus:
            print("BM25 index is empty, returning dense-only results.")
            # Format dense results to match final output
            final_results = []
            for match in dense_results['matches']:
                # Find the original doc from 'self.documents'
                doc_id = match['id']
                doc_index = int(doc_id.split('_')[1])
                final_results.append({
                    "id": doc_id,
                    "score": float(match['score']),
                    "text": self.documents[doc_index]['processed_text'],
                    "metadata": self.documents[doc_index]['metadata']
                })
            return final_results

        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        max_bm25 = np.max(bm25_scores)
        if max_bm25 == 0: max_bm25 = 1e-6
        norm_bm25 = bm25_scores / max_bm25
        
        dense_scores = {}
        for match in dense_results['matches']:
            dense_scores[match['id']] = match['score']
        
        hybrid_scores = {}
        for i, doc in enumerate(self.documents):
            doc_id = doc['id']
            sparse_score = norm_bm25[i]
            dense_score = dense_scores.get(doc_id, 0.0) 
            hybrid_score = (self.alpha * dense_score) + ((1 - self.alpha) * sparse_score)
            
            if hybrid_score > 0:
                hybrid_scores[doc_id] = hybrid_score

        sorted_docs = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc_index = int(doc_id.split('_')[1])
            final_results.append({
                "id": doc_id,
                "score": float(score), # Convert to standard float
                "text": self.documents[doc_index]['processed_text'],
                "metadata": self.documents[doc_index]['metadata']
            })
            
        return final_results

if __name__ == "__main__":
    
    print("--- Starting Retriever Agent (Setup Mode) ---")
    
    # This test mode will fail unless keys are set as env variables
    try:
        TEST_API_KEY = os.getenv("PINECONE_API_KEY")
        TEST_REGION = os.getenv("PINECONE_REGION", "us-east-1")
        
        if not TEST_API_KEY:
            raise ValueError("Set PINECONE_API_KEY env var to test this agent.")
            
        retriever = RetrieverAgent(api_key=TEST_API_KEY, region=TEST_REGION, alpha=0.5)
        
        print("\n--- Running Test Query ---")
        test_query = "What are the main policies regarding climate change and renewable energy?"
        
        results = retriever.search(test_query, top_k=3)
        
        print(f"\nResults for query: '{test_query}'")
        for i, res in enumerate(results):
            print(f"\nRank {i+1} (Score: {res['score']:.4f})")
            print(f"Source: {res['metadata']['source']}, Page: {res['metadata']['page']}")
            print(f"Text: {res['text'][:250]}...")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    
    print("--- Retriever Agent Finished ---")