import json
from pathlib import Path
import warnings
from typing import List, Dict, Any

# Embedding models
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# Visualization & Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

import config_loader

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class EmbeddingAgent:
    """
    This agent is responsible for the rest of Task 2:
    1. Generating various embeddings (TF-IDF, Word2Vec, SBERT).
    2. Visualizing and comparing them using PCA/t-SNE.
    """
    def __init__(self, sbert_model_name: str = None):
        # Multilingual sentence encoder, read from config.yaml. The default
        # (paraphrase-multilingual-MiniLM-L12-v2) maps ~50 languages into one
        # shared vector space, so a German query can retrieve English chunks
        # and vice versa. It is also 384-dim, matching the existing index spec.
        if sbert_model_name is None:
            sbert_model_name = config_loader.get(
                "embedding.model", "paraphrase-multilingual-MiniLM-L12-v2"
            )
        self.model_name = sbert_model_name
        self.sbert_model = SentenceTransformer(sbert_model_name)

        # Initialize other models.
        # NOTE: stop_words='english' is deliberately dropped -- applying an
        # English stop list to a multilingual corpus strips nothing from German
        # text while quietly biasing the English side of the comparison.
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

        print(f"EmbeddingAgent initialized with multilingual SBERT model: {sbert_model_name}")

    def load_data(self, input_file: str) -> List[str]:
        """
        Loads the 'classical_output.json' and extracts the processed text.
        """
        print(f"Loading processed data from '{input_file}'...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file}")
            return []
        
        # Use the 'processed_text' which is clean and PII-redacted
        documents = [chunk.get("processed_text", "") for chunk in data]
        # Also get token lists for Word2Vec
        self.tokenized_docs = [chunk.get("tokens", []) for chunk in data]
        
        print(f"Loaded {len(documents)} documents for embedding.")
        return documents

    def create_embeddings(self, documents: List[str]) -> Dict[str, np.ndarray]:
        """
        Creates TF-IDF, Word2Vec, and SBERT embeddings.
        """
        embeddings = {}
        
        # 1. SBERT Embeddings (Dense)
        print("Generating SBERT embeddings... (This may take a moment)")
        embeddings['sbert'] = self.sbert_model.encode(documents, show_progress_bar=True)
        
        # 2. TF-IDF Embeddings (Sparse)
        print("Generating TF-IDF embeddings...")
        # We use .toarray() to make it dense for visualization,
        # but in practice, you'd keep it sparse.
        embeddings['tfidf'] = self.tfidf_vectorizer.fit_transform(documents).toarray()

        # 3. Word2Vec Embeddings (Average)
        print("Generating Word2Vec embeddings...")
        # Train a Word2Vec model on the tokenized documents
        w2v_model = Word2Vec(sentences=self.tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)
        
        # Average Word2Vec vectors for each document
        doc_vectors = []
        for doc_tokens in self.tokenized_docs:
            vectors = [w2v_model.wv[word] for word in doc_tokens if word in w2v_model.wv]
            if vectors:
                doc_vectors.append(np.mean(vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(w2v_model.vector_size)) # Fallback for empty docs
        embeddings['word2vec'] = np.array(doc_vectors)
        
        print("All embeddings generated.")
        return embeddings

    def visualize_embeddings(self, embeddings: Dict[str, np.ndarray], output_file: str):
        """
        Uses PCA and t-SNE to reduce dimensions and plots the results.
        """
        print("Visualizing embeddings using PCA & t-SNE...")
        
        # We have too many points (9417) for t-SNE, it will be very slow.
        # Let's sample N points for a clearer plot.
        n_samples = min(2000, len(self.tokenized_docs))
        indices = np.random.choice(len(self.tokenized_docs), n_samples, replace=False)
        
        # Set up the plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Embedding Comparison (PCA vs. t-SNE) on 2000 Samples', fontsize=16)
        
        # Apply dimensionality reduction
        for i, (name, emb) in enumerate(embeddings.items()):
            print(f"  Processing {name}...")
            # Sample the embeddings
            sampled_emb = emb[indices]
            
            # 1. PCA
            pca = PCA(n_components=2, random_state=42)
            emb_pca = pca.fit_transform(sampled_emb)
            
            # 2. t-SNE
            # Run PCA first to 50 components, then t-SNE (standard practice)
            if sampled_emb.shape[1] > 50:
                pca_50 = PCA(n_components=50, random_state=42)
                emb_pca_50 = pca_50.fit_transform(sampled_emb)
            else:
                emb_pca_50 = sampled_emb # Use as-is if < 50 features (like Word2Vec)
                
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
            emb_tsne = tsne.fit_transform(emb_pca_50)
            
            # Plot PCA
            ax = axes[0, i]
            ax.scatter(emb_pca[:, 0], emb_pca[:, 1], s=5, alpha=0.5)
            ax.set_title(f'{name.upper()} - PCA')
            
            # Plot t-SNE
            ax = axes[1, i]
            ax.scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=5, alpha=0.5)
            ax.set_title(f'{name.upper()} - t-SNE')

        # Save the plot
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_file, dpi=150)
        print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    
    # Define project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Input file
    INPUT_FILE = PROJECT_ROOT / "results" / "classical_output.json"
    
    # Output file (as per your results folder structure)
    OUTPUT_PLOT_FILE = PROJECT_ROOT / "results" / "plots" / "embedding_map_multilingual.png"

    # Output file for SBERT embeddings (for the *next* agent to use).
    # Written to a new filename so the previous English-only
    # 'sbert_embeddings.npy' stays intact for comparison.
    OUTPUT_EMBEDDINGS_FILE = PROJECT_ROOT / "results" / "multilingual_embeddings.npy"

    print("--- Starting Embedding Agent ---")
    
    agent = EmbeddingAgent()
    
    # 1. Load data
    documents = agent.load_data(str(INPUT_FILE))
    
    if documents:
        # 2. Create embeddings
        all_embeddings = agent.create_embeddings(documents)
        
        # 3. Save the SBERT embeddings for our RAG pipeline
        # We save only the SBERT ones as they are the ones we'll use for retrieval
        np.save(OUTPUT_EMBEDDINGS_FILE, all_embeddings['sbert'])
        print(f"SBERT embeddings saved to {OUTPUT_EMBEDDINGS_FILE} for retrieval.")
        
        # 4. Create and save the comparison visualization
        agent.visualize_embeddings(all_embeddings, str(OUTPUT_PLOT_FILE))
    
    else:
        print("\nNo data was processed.")
        
    print("--- Embedding Agent Finished ---")