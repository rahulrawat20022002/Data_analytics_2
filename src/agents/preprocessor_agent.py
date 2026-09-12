import os
import json
import spacy # For tokenization, lemmatization, NER
import re    # For PII redaction (regex)
from pathlib import Path
from typing import List, Dict, Any

from _1language_agent import LanguageAgent

class PreprocessorAgent:
    """
    This agent is responsible for the second part of Task 1:
    1. Loading the raw chunks from the IngestionAgent.
    2. Detecting the language of each chunk.
    3. Performing tokenization, lemmatization, and NER with a language-matched
       spaCy pipeline.
    4. Redacting PII.
    5. Saving the structured output.
    """
    def __init__(self):
        # Define PII patterns using regular expressions
        # This is the "regex" technique from your README
        # These are script-agnostic, so they work unchanged across languages.
        self.pii_patterns = {
            "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
            "PHONE": re.compile(r"(\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}"),
            # Add more patterns as needed (e.g., social security numbers)
        }
        self.language_agent = LanguageAgent()
        # spaCy pipelines are loaded on first use and cached, so a corpus that
        # is 95% English never pays to load the German model until it needs it.
        self._pipelines: Dict[str, Any] = {}
        print("PreprocessorAgent initialized (multilingual).")

    def _get_pipeline(self, lang: str):
        """Returns a cached spaCy pipeline for `lang`, falling back sensibly.

        Order of preference:
          1. The configured model for the language (e.g. de_core_news_sm).
          2. The blank multilingual "xx" pipeline -- tokenizes correctly but
             provides no lemmas or entities.
        """
        if lang in self._pipelines:
            return self._pipelines[lang]

        model_name = self.language_agent.spacy_model_for(lang)
        pipeline = None

        if model_name:
            try:
                # The parser is not needed for tokens/lemmas/NER and is slow.
                pipeline = spacy.load(model_name, disable=["parser"])
                print(f"  - Loaded spaCy pipeline '{model_name}' for '{lang}'.")
            except OSError:
                print(
                    f"spaCy model '{model_name}' not installed for '{lang}'. "
                    f"Run: python -m spacy download {model_name}"
                )

        if pipeline is None:
            print(f"  - Falling back to blank multilingual pipeline for '{lang}'.")
            pipeline = spacy.blank("xx")

        self._pipelines[lang] = pipeline
        return pipeline

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

        # 2. Detect language so the right spaCy pipeline is used. Doing this on
        #    the redacted text keeps placeholder tokens out of the decision.
        lang, lang_confidence = self.language_agent.detect_with_confidence(clean_text)
        nlp = self._get_pipeline(lang)

        # Process the clean text with spaCy
        doc = nlp(clean_text)

        # 3. Tokenization & Lemmatization (as per Task 1)
        #    The blank "xx" fallback has no lemmatizer or tagger, so guard on
        #    what the loaded pipeline actually provides.
        has_lemmas = doc.has_annotation("LEMMA")
        has_pos = doc.has_annotation("TAG")

        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        if has_lemmas:
            lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        else:
            # Lowercased surface forms are the best available stand-in.
            lemmas = [token.lower() for token in tokens]

        # 4. Named Entity Recognition (NER)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # 5. POS Tagging (Bonus, as per Task 1)
        pos_tags = (
            [{"text": token.text, "pos": token.pos_} for token in doc] if has_pos else []
        )

        # Return a structured dictionary for this chunk
        return {
            "id": chunk_data["id"],
            "original_text": chunk_data["text"],
            "processed_text": clean_text,
            "language": lang,
            "language_confidence": round(lang_confidence, 4),
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
            print(f"Error: Input file not found at {input_file}")
            print("Please run the 'pdf_ingestion_agent.py' first.")
            return []
        
        print(f"Loaded {len(raw_chunks)} chunks. Starting processing...")
        
        processed_data = []
        for i, chunk in enumerate(raw_chunks):
            if (i + 1) % 500 == 0:
                print(f"  ...processed {i+1}/{len(raw_chunks)} chunks")
            processed_data.append(self.process_chunk(chunk))

        print("Processing complete.")
        self._report_language_distribution(processed_data)
        return processed_data

    def _report_language_distribution(self, processed_data: List[Dict[str, Any]]) -> None:
        """Prints how the corpus splits across languages -- a quick sanity check
        that multilingual ingestion actually saw more than one language."""
        counts: Dict[str, int] = {}
        for chunk in processed_data:
            lang = chunk.get("language", "unknown")
            counts[lang] = counts.get(lang, 0) + 1

        total = len(processed_data) or 1
        print("\nLanguage distribution across corpus:")
        for lang, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {lang}: {count} chunks ({100 * count / total:.1f}%)")

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
        print(f"\nSuccessfully processed data.")
        print(f"Saved {len(processed_chunks)} processed chunks to {OUTPUT_FILE}")
    else:
        print("\nNo data was processed. Check input file.")
        
    print("--- Preprocessor Agent Finished ---")