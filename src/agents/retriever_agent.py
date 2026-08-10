import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
# Pinecone client for dense vector search
from pinecone import Pinecone, ServerlessSpec

# BM25 for sparse keyword search
from rank_bm25 import BM25Okapi

# Multilingual encoder, loaded once at init rather than per query
from sentence_transformers import SentenceTransformer

import config_loader
from language_agent import LanguageAgent


load_dotenv()
class RetrieverAgent:
    """
    This agent is responsible for Task 3: Multilingual Hybrid Retrieval.

    Dense retrieval uses a multilingual sentence encoder, so queries in one
    language match semantically equivalent passages in another. Sparse (BM25)
    retrieval stays lexical and therefore mostly same-language -- the hybrid
    alpha controls how much cross-lingual reach the final ranking gets.
    """

    def __init__(
        self,
        api_key: str,
        region: str = None,
        alpha: float = None,
        model_name: str = None,
        index_name: str = None,
    ):
        """
        Initializes the agent, Pinecone connection, and loads/builds indices.

        Args:
            api_key: The Pinecone API key.
            region: The Pinecone region (e.g., "us-east-1").
            alpha: The weighting factor for hybrid search.
            model_name: Override the multilingual embedding model.
            index_name: Override the Pinecone index name.
        """
        print("Initializing RetrieverAgent (multilingual)...")

        # --- CONFIG ---
        self.alpha = alpha if alpha is not None else config_loader.get("retrieval.alpha", 0.5)
        self.PINECONE_REGION = region or config_loader.get("retrieval.region", "us-east-1")
        self.model_name = model_name or config_loader.get(
            "embedding.model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.index_name = index_name or config_loader.get(
            "retrieval.index_name", "policy-rag-index-multilingual"
        )
        self.dimension = config_loader.get("embedding.dimension", 384)

        # --- PATHS ---
        # It goes up 3 levels to the project root
        self.project_root = Path(__file__).parent.parent.parent
        self.data_file = self.project_root / "results" / "classical_output.json"
        self.vector_file = self.project_root / "results" / "multilingual_embeddings.npy"

        # --- PINECONE SETUP ---
        self.PINECONE_API_KEY = api_key
        # ---------------------

        if self.PINECONE_API_KEY is None or "YOUR_API_KEY" in self.PINECONE_API_KEY:
             raise ValueError("A valid Pinecone API Key was not passed to RetrieverAgent.")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)

        self.language_agent = LanguageAgent()

        # Load the encoder ONCE. This used to be constructed inside search(),
        # which re-initialized the model on every single query.
        print(f"Loading multilingual encoder '{self.model_name}'...")
        self.encoder = SentenceTransformer(self.model_name)

        # Load text data
        print("Loading processed text data...")
        self.documents = self._load_documents()
        self.corpus = [doc['processed_text'] for doc in self.documents]
        # id -> position lookup, replacing the brittle `int(id.split('_')[1])`
        # parsing that assumed a specific id format and offset.
        self.doc_index_by_id = {doc['id']: i for i, doc in enumerate(self.documents)}

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
        """Creates a BM25 index from the document corpus.

        Uses the LanguageAgent's Unicode-aware tokenizer instead of a plain
        `.split(" ")`, which left punctuation attached to words and did not
        case-fold -- both of which hurt German recall noticeably.
        """
        if not corpus:
            print("⚠️ Warning: Corpus is empty. BM25 index will be empty.")
            # Return a dummy/empty BM25 object
            return BM25Okapi([[]])

        tokenized_corpus = [self.language_agent.tokenize(doc) for doc in corpus]
        return BM25Okapi(tokenized_corpus)

    def _get_pinecone_index(self):
        """Connects to Pinecone, checks if the index exists, and upserts data if needed."""
        
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Index '{self.index_name}' not found. Creating a new one...")
            print("This can take a minute...")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension, # 384 for paraphrase-multilingual-MiniLM-L12-v2
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=config_loader.get("retrieval.cloud", "aws"),
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
        
        # Guard against upserting English vectors into the multilingual index
        # (or vice versa) -- a silent mismatch that degrades recall without
        # ever raising an error.
        if vectors.shape[0] != len(self.documents):
            print(
                f"❌ Vector/document count mismatch: {vectors.shape[0]} vectors vs "
                f"{len(self.documents)} documents. Re-run embedding_agent.py."
            )
            return None
        if vectors.shape[1] != self.dimension:
            print(
                f"❌ Embedding dimension mismatch: file has {vectors.shape[1]}, "
                f"index expects {self.dimension}. Re-run embedding_agent.py with "
                f"the configured model '{self.model_name}'."
            )
            return None

        vectors_to_upsert = []
        for i, doc in enumerate(self.documents):
            vector_id = doc['id']
            vector = vectors[i].tolist()
            metadata = {
                "text": doc['processed_text'],
                "source": doc['metadata'].get('source', 'unknown'),
                "page": int(doc['metadata'].get('page', 0)),
                # Stored so retrieval can optionally be filtered by language.
                "language": doc.get('language', 'unknown')
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

    def _format_result(self, doc_id: str, score: float) -> Optional[Dict[str, Any]]:
        """Builds a result dict from a document id, or None if the id is unknown."""
        doc_index = self.doc_index_by_id.get(doc_id)
        if doc_index is None:
            # The index contains a vector we have no local text for -- usually a
            # stale index left over from a previous corpus.
            return None
        doc = self.documents[doc_index]
        return {
            "id": doc_id,
            "score": float(score),
            "text": doc['processed_text'],
            "language": doc.get('language', 'unknown'),
            "metadata": doc['metadata'],
        }

    def search(
        self,
        query: str,
        top_k: int = None,
        language: str = None,
        restrict_to_language: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Performs a multilingual hybrid search.

        Args:
            query: The (any-language) search query.
            top_k: Number of results to return.
            language: Query language. Detected automatically when omitted.
            restrict_to_language: When True, only return same-language chunks.
                Off by default -- the whole point of a shared multilingual
                vector space is that a German query can surface an English
                source document.
        """
        if top_k is None:
            top_k = config_loader.get("retrieval.top_k", 5)

        if language is None:
            language = self.language_agent.detect(query)

        # Encode with the pre-loaded multilingual model.
        query_vector = self.encoder.encode(query).tolist()

        query_kwargs: Dict[str, Any] = {
            "vector": query_vector,
            # Over-fetch on the dense side so the hybrid re-ranking has more
            # than top_k candidates to blend with the sparse scores.
            "top_k": max(top_k * 4, top_k),
            "include_metadata": True,
        }
        if restrict_to_language:
            query_kwargs["filter"] = {"language": {"$eq": language}}

        dense_results = self.pinecone_index.query(**query_kwargs)

        # Check if BM25 index is empty
        if not self.corpus:
            print("BM25 index is empty, returning dense-only results.")
            # Format dense results to match final output
            final_results = []
            for match in dense_results['matches']:
                formatted = self._format_result(match['id'], match['score'])
                if formatted:
                    final_results.append(formatted)
            return final_results[:top_k]

        tokenized_query = self.language_agent.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        max_bm25 = np.max(bm25_scores) if len(bm25_scores) else 0.0
        if max_bm25 == 0: max_bm25 = 1e-6
        norm_bm25 = bm25_scores / max_bm25

        dense_scores = {}
        for match in dense_results['matches']:
            dense_scores[match['id']] = match['score']

        hybrid_scores = {}
        for i, doc in enumerate(self.documents):
            doc_id = doc['id']
            if restrict_to_language and doc.get('language') != language:
                continue
            sparse_score = norm_bm25[i] if i < len(norm_bm25) else 0.0
            dense_score = dense_scores.get(doc_id, 0.0)
            hybrid_score = (self.alpha * dense_score) + ((1 - self.alpha) * sparse_score)

            if hybrid_score > 0:
                hybrid_scores[doc_id] = hybrid_score

        sorted_docs = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)

        final_results = []
        for doc_id, score in sorted_docs[:top_k]:
            formatted = self._format_result(doc_id, score)
            if formatted:
                final_results.append(formatted)

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

        # The same question in English and German should retrieve overlapping
        # documents -- that overlap is the signal that cross-lingual retrieval
        # is actually working rather than just not crashing.
        test_queries = [
            "What are the main policies regarding climate change and renewable energy?",
            "Welche Maßnahmen gibt es zum Klimawandel und zu erneuerbaren Energien?",
        ]

        retrieved_ids = []
        for test_query in test_queries:
            print(f"\n--- Running Test Query ---")
            results = retriever.search(test_query, top_k=3)
            retrieved_ids.append({res['id'] for res in results})

            print(f"\nResults for query: '{test_query}'")
            for i, res in enumerate(results):
                print(f"\nRank {i+1} (Score: {res['score']:.4f})")
                print(f"Source: {res['metadata']['source']}, Page: {res['metadata']['page']}")
                print(f"Language: {res.get('language', 'unknown')}")
                print(f"Text: {res['text'][:250]}...")

        if len(retrieved_ids) == 2:
            overlap = retrieved_ids[0] & retrieved_ids[1]
            print(f"\n--- Cross-lingual overlap: {len(overlap)}/3 documents shared ---")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    
    print("--- Retriever Agent Finished ---")