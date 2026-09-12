import os
import json
from pathlib import Path
from typing import List, Dict, Any
import ollama

import config_loader
from _1language_agent import LanguageAgent

class SummarizerAgent:
    def __init__(self):
        self.MODEL_NAME = config_loader.get("llm.model", "mistral:7b")
        try:
            ollama.show(self.MODEL_NAME)
        except Exception:
            print(f"Error: Ollama model '{self.MODEL_NAME}' not found.")
            print(f"Please run 'ollama pull {self.MODEL_NAME}' in your terminal.")
            raise
        self.language_agent = LanguageAgent()
        print(f"SummarizerAgent initialized. Using Ollama model: {self.MODEL_NAME}")
    #code->
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            source_file = os.path.basename(doc['metadata'].get('source', 'unknown'))
            page = doc['metadata'].get('page', 'N/A')
            doc_lang = doc.get('language', 'unknown')
            citation_tag = f"[Source {i+1}: file={source_file}, page={page}]"
            context_str += f"{citation_tag} (language: {doc_lang})\n"
            context_str += f"Text: \"{doc['text']}\"\n\n"
        return context_str.strip()
    def run(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        language: str = None,
    ) -> str:
        """Summarizes retrieved context, answering in the query's language.

        Args:
            query: The user query.
            retrieved_docs: Hybrid-retrieval results, possibly in mixed languages.
            language: Target output language code. Detected from the query when
                omitted.
        """
        if language is None:
            language = self.language_agent.detect(query)

        language_name = self.language_agent.name_of(language)
        print(f"Summarizing context for query: '{query}' (output language: {language_name})")
        formatted_context = self._format_context(retrieved_docs)
        prompt = f"""
        [INST]
        You are a professional policy analyst. Your task is to write a structured
        summary that answers the user's query.
        Follow these instructions:
        1. Base your answer *only* on the provided "Sources" context.
        2. Do not use any prior knowledge.
        3. For every claim or piece of information you write, you *must* cite the
           source using the `[Source X: file=..., page=...]` tag.
        4. If the context does not contain the answer, state that.
        5. Structure your answer logically (e.g., use bullet points).
        6. {self.language_agent.instruction_for(language)}
        7. The sources may be written in a different language than your answer.
           Translate the meaning faithfully into {language_name}; never invent
           detail that is not present in the source text.
        ---
        QUERY:
        "{query}"
        ---
        SOURCES:
        {formatted_context}
        ---
        STRUCTURED SUMMARY (in {language_name}):
        [/INST]
        """
        try:
            print(f"Calling Ollama model: {self.MODEL_NAME}...")
            response = ollama.chat(
                model=self.MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}]
            )
            summary = response['message']['content']
            print("Summary generated successfully.")
            return summary
        except Exception as e:
            print(f"Error during LLM summarization: {e}")
            return "Error: Could not generate summary."