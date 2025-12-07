import os
import json
import spacy # For tokenization, lemmatization, NER
import re    # For PII redaction (regex)
from pathlib import Path
from typing import List, Dict, Any

# Load the spaCy model we just downloaded
# We disable components we don't need to speed up processing
nlp = spacy.load("en_core_web_sm", disable=["parser"]) 

class PreprocessorAgent:
    """
    This agent is responsible for the second part of Task 1:
    1. Loading the raw chunks from the IngestionAgent.
    2. Performing tokenization, lemmatization, and NER.
    3. Redacting PII.
    4. Saving the structured output.
    """
    def __init__(self):
        # Define PII patterns using regular expressions
        # This is the "regex" technique from your README
        self.pii_patterns = {
            "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
            "PHONE": re.compile(r"(\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}"),
            # Add more patterns as needed (e.g., social security numbers)
        }
        print("✅ PreprocessorAgent initialized.")

    def redact_pii(self, text: str) -> str:
        """
        Redacts PII from a text string using regex.
        """
        for pii_type, pattern in self.pii_patterns.items():
            text = pattern.sub(f"[{pii_type}_REDACTED]", text)
        return text

    def process_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single chunk of text.
        """
        text = chunk_data["text"]
        
        # 1. Redact PII (as per Task 1 & 6)
        clean_text = self.redact_pii(text)
        
        # Process the clean text with spaCy
        doc = nlp(clean_text)
        
        # 2. Tokenization & Lemmatization (as per Task 1)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # 3. Named Entity Recognition (NER)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        # 4. POS Tagging (Bonus, as per Task 1)
        pos_tags = [{"text": token.text, "pos": token.pos_} for token in doc]

        # Return a structured dictionary for this chunk
        return {
            "id": chunk_data["id"],
            "original_text": chunk_data["text"],
            "processed_text": clean_text,
            "tokens": tokens,
            "lemmas": lemmas,
            "entities": entities,
            "pos_tags": pos_tags,
            "metadata": chunk_data["metadata"]
        }

    def run(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Loads the input file, processes all chunks, and returns the data.
        """
        print(f"Loading chunks from '{input_file}'...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_chunks = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: Input file not found at {input_file}")
            print("Please run the 'pdf_ingestion_agent.py' first.")
            return []
        
        print(f"Loaded {len(raw_chunks)} chunks. Starting processing...")
        
        processed_data = []
        for i, chunk in enumerate(raw_chunks):
            if (i + 1) % 500 == 0:
                print(f"  ...processed {i+1}/{len(raw_chunks)} chunks")
            processed_data.append(self.process_chunk(chunk))
            
        print("✅ Processing complete.")
        return processed_data

def save_processed_data(data: List[Dict[str, Any]], output_file: str):
    """
    Saves the final processed data to the specified JSON file.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    
    # Define project paths
    # Assumes this script is in /src/agents/
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Input file (created by the previous agent)
    INPUT_CHUNKS_FILE = PROJECT_ROOT / "results" / "ingested_chunks.json"
    
    # Output file (as specified in your Task 1 README)
    OUTPUT_FILE = PROJECT_ROOT / "results" / "classical_output.json"
    
    print("--- Starting Preprocessor Agent ---")
    
    agent = PreprocessorAgent()
    
    # Run the agent
    processed_chunks = agent.run(str(INPUT_CHUNKS_FILE))
    
    if processed_chunks:
        # Save the output
        save_processed_data(processed_chunks, str(OUTPUT_FILE))
        print(f"\n✅ Successfully processed data.")
        print(f"Saved {len(processed_chunks)} processed chunks to {OUTPUT_FILE}")
    else:
        print("\nNo data was processed. Check input file.")
        
    print("--- Preprocessor Agent Finished ---")