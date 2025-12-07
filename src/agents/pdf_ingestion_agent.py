import os
import json
from pathlib import Path
from typing import List, Dict, Any

# As per your spec, we'll use a loader that handles PyMuPDF/pdfplumber.
# PyPDFDirectoryLoader is a great choice for this.
from langchain_community.document_loaders import PyPDFDirectoryLoader

# This import was updated
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain's base document schema
from langchain_core.documents import Document

class PDFIngestionAgent:
    """
    This agent is responsible for Task 1:
    1. Loading all PDFs from a specified directory.
    2. Chunking the loaded documents semantically.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the agent with a text splitter configuration.
        
        Args:
            chunk_size: The target size for each text chunk (in characters).
            chunk_overlap: The number of characters to overlap between chunks
                           to maintain context.
        """
        # This splitter is recommended for generic text and tries to
        # split on semantic boundaries (paragraphs, sentences) first.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], # Define priority of separators
            length_function=len,
        )
        print("✅ PDFIngestionAgent initialized.")

    def run(self, input_dir: str) -> List[Document]:
        """
        Loads all PDFs from the input directory and splits them into chunks.

        Args:
            input_dir: Path to the directory containing PDFs (e.g., "data/pdfs").

        Returns:
            A list of 'Document' objects, each representing a text chunk.
        """
        if not os.path.exists(input_dir) or not os.listdir(input_dir):
            print(f"⚠️ Warning: Input directory '{input_dir}' is empty or doesn't exist.")
            # --- THIS PRINT STATEMENT IS NOW DYNAMIC ---
            print(f"Please add your 10+ policy PDFs to this folder.")
            return []

        print(f"Loading PDFs from '{input_dir}'...")
        
        # PyPDFDirectoryLoader finds and loads all .pdf files in the given path.
        # It uses pypdf by default, which aligns with your PyPDFLoader/pdfplumber spec.
        loader = PyPDFDirectoryLoader(input_dir, recursive=True)
        
        try:
            documents = loader.load()
            if not documents:
                print(f"No PDFs found in {input_dir}.")
                return []
        except Exception as e:
            print(f"❌ Error loading PDFs: {e}")
            return []
            
        print(f"Loaded {len(documents)} pages from all PDF files.")
        
        # Now, chunk the documents
        # This is the "Chunk semantically (RecursiveCharacterTextSplitter)" step
        chunks = self.splitter.split_documents(documents)
        
        print(f"Split documents into {len(chunks)} chunks.")
        
        # The Orchestrator will pass these chunks to the next agent
        return chunks

def save_chunks_to_json(chunks: List[Document], output_file: str):
    """
    Helper function to save the chunked documents to a JSON file for inspection
    or for the next agent to pick up.
    """
    # Convert Document objects to a serializable list of dicts
    serializable_chunks = []
    for i, chunk in enumerate(chunks):
        serializable_chunks.append({
            "id": f"chunk_{i}",
            "text": chunk.page_content,
            "metadata": chunk.metadata  # Contains source file and page number
        })
            
    # Ensure the output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)

# This allows you to run this agent as a standalone script for testing
if __name__ == "__main__":
    
    # Define project paths
    # Assumes this script is in /src/agents/
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # --- THIS IS THE MAIN FIX ---
    # Path updated to point to your 'Data' folder
    INPUT_PDF_DIR ='/home/lenovo/Desktop/Examination_Copy/Data'
    # ----------------------------

    OUTPUT_DIR = PROJECT_ROOT / "results"
    
    # This is an intermediate file. The 'classical_output.json' from Task 1
    # will likely be created *after* the PreprocessorAgent runs.
    OUTPUT_CHUNKS_FILE = OUTPUT_DIR / "ingested_chunks.json"
    
    # Create directories if they don't exist
    # Note: We don't create INPUT_PDF_DIR, we assume it exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Starting PDF Ingestion Agent ---")
    
    agent = PDFIngestionAgent(chunk_size=1000, chunk_overlap=200)
    
    # Run the agent
    document_chunks = agent.run(str(INPUT_PDF_DIR))
    
    if document_chunks:
        # Save the output for review
        save_chunks_to_json(document_chunks, OUTPUT_CHUNKS_FILE)
        print(f"\n✅ Successfully processed PDFs.")
        print(f"Saved {len(document_chunks)} chunks to {OUTPUT_CHUNKS_FILE}")
    else:
        # --- THIS PRINT STATEMENT IS NOW DYNAMIC ---
        print(f"\nNo documents were processed. Please check your '{INPUT_PDF_DIR}' folder.")
        
    print("--- PDF Ingestion Agent Finished ---")