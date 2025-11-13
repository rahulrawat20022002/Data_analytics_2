import os
import json
from pathlib import Path
from typing import List, Dict, Any
import warnings

# For vectorizing text
from sklearn.feature_extraction.text import TfidfVectorizer
# For topic modeling (as per Task 2)
from sklearn.decomposition import LatentDirichletAllocation, NMF

# For visualizing the topics (as per your agent table)
import pyLDAvis
import pyLDAvis.lda_model

class TopicModelAgent:
    """
    This agent is responsible for Task 2:
    1. Loading the processed lemmas from 'classical_output.json'.
    2. Vectorizing the text using TF-IDF.
    3. Running LDA (or NMF) to identify topics.
    4. Saving a visualization of the topics.
    """
    def __init__(self, n_topics: int = 10, n_top_words: int = 15):
        """
        Initializes the agent.
        
        Args:
            n_topics: The number of topics to identify (Task 2 says 10+).
            n_top_words: The number of top words to show for each topic.
        """
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        
        # Initialize the vectorizer
        # We use the lemmas, so no need for built-in stop_words or lemmatization
        self.vectorizer = TfidfVectorizer(
            max_df=0.95, 
            min_df=2,    # Ignore words that appear in less than 2 documents
            stop_words='english' # Still good to filter common stop words
        )
        
        # Initialize the LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_method='online',
            random_state=42,
            n_jobs=-1  # Use all available CPUs
        )
        print(f"✅ TopicModelAgent initialized for {self.n_topics} topics.")

    def load_data(self, input_file: str) -> List[str]:
        """
        Loads the 'classical_output.json' and extracts the lemmas.
        """
        print(f"Loading processed data from '{input_file}'...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: Input file not found at {input_file}")
            print("Please run the 'preprocessor_agent.py' first.")
            return []
        
        # We need to join the list of lemmas for each chunk back into a string
        # so the vectorizer can process it.
        documents = [" ".join(chunk.get("lemmas", [])) for chunk in data]
        print(f"Loaded {len(documents)} documents for topic modeling.")
        return documents

    def run(self, documents: List[str]):
        """
        Runs the full topic modeling pipeline.
        """
        if not documents:
            print("No documents to process. Aborting.")
            return
            
        print("Vectorizing text with TF-IDF...")
        tfidf_data = self.vectorizer.fit_transform(documents)
        
        print("Fitting LDA model...")
        self.lda_model.fit(tfidf_data)
        
        print("\n--- Top Topics Found by LDA ---")
        self.print_top_words(self.lda_model, self.vectorizer.get_feature_names_out())
        
        # The agent returns the fitted models and data 
        # for the orchestrator or for visualization.
        return self.lda_model, tfidf_data, self.vectorizer

    def print_top_words(self, model, feature_names):
        """Helper function to print the top words for each topic."""
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-self.n_top_words - 1:-1]]
            print(f"Topic #{topic_idx + 1}: {' | '.join(top_words)}")
            
    def save_visualization(self, lda_model, tfidf_data, vectorizer, output_file):
        """
        Saves the pyLDAvis visualization to an HTML file.
        Note: pyLDAvis is designed for raw counts, not TF-IDF, so we must
        re-vectorize with CountVectorizer for the visualization.
        """
        print("\nPreparing visualization...")
        
        # We need the raw documents from the load_data step
        documents = self.load_data(INPUT_CHUNKS_FILE) # Reload for this
        if not documents:
            return

        # pyLDAvis works best with CountVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        count_vectorizer = CountVectorizer(
            max_df=0.95, 
            min_df=2,
            stop_words='english'
        )
        count_data = count_vectorizer.fit_transform(documents)
        
        # Fit a new LDA model on the count data (required for pyLDAvis)
        lda_vis_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        print("Fitting new LDA model for visualization...")
        lda_vis_model.fit(count_data)

        # Suppress warnings from pyLDAvis
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
        print(f"Saving pyLDAvis visualization to {output_file}...")
        panel = pyLDAvis.lda_model.prepare(
            lda_vis_model, 
            count_data, 
            count_vectorizer, 
            mds='tsne' # Use t-SNE for dimensionality reduction
        )
        
        # Ensure the output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        pyLDAvis.save_html(panel, str(output_file))


if __name__ == "__main__":
    
    # Define project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Input file (created by the previous agent)
    INPUT_CHUNKS_FILE = PROJECT_ROOT / "results" / "classical_output.json"
    
    # Output file (visualization)
    # Your README shows 'embedding_map.png', but pyLDAvis saves as HTML.
    # Let's save it in the 'plots' folder.
    OUTPUT_VIS_FILE = PROJECT_ROOT / "results" / "plots" / "topic_model_visualization.html"
    
    print("--- Starting Topic Model Agent ---")
    
    agent = TopicModelAgent(n_topics=15) # Identify 15 topics
    
    # Load the data
    documents = agent.load_data(str(INPUT_CHUNKS_FILE))
    
    if documents:
        # Run the agent
        lda_model, tfidf_data, vectorizer = agent.run(documents)
        
        # Save the visualization
        agent.save_visualization(lda_model, tfidf_data, vectorizer, str(OUTPUT_VIS_FILE))
        
        print(f"\n✅ Successfully ran topic modeling.")
        print(f"Visualization saved to {OUTPUT_VIS_FILE}")
    else:
        print("\nNo data was processed.")
        
    print("--- Topic Model Agent Finished ---")