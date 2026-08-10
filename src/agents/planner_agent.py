import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import ollama
from dotenv import load_dotenv

import config_loader
from language_agent import LanguageAgent

load_dotenv()
class PlannerAgent:
    def __init__(self):
        self.MODEL_NAME = config_loader.get("llm.model", "mistral:7b")
        try:
            ollama.show(self.MODEL_NAME)
        except Exception:
            print(f"❌ Error: Ollama model '{self.MODEL_NAME}' not found.")
            print(f"Please run 'ollama pull {self.MODEL_NAME}' in your terminal.")
            raise
        self.language_agent = LanguageAgent()
        print(f"✅ PlannerAgent initialized. Using Ollama model: {self.MODEL_NAME}")
    def detect_language(self, query: str) -> str:
        """Delegates to LanguageAgent so detection rules live in one place."""
        lang = self.language_agent.detect(query)
        print(f"Language detected: {lang}")
        return lang
    def decompose_query(self, query: str, language: str = "en") -> Dict[str, Any]:
        print(f"Decomposing complex query: '{query}'")
        language_name = self.language_agent.name_of(language)
        prompt = f"""
        [INST]
        You are a research assistant. Your task is to decompose a complex policy question
        into a list of 3-5 specific, answerable sub-queries.
        Write the sub-queries in {language_name}, the same language as the question.
        Respond ONLY with a single, valid JSON object. Do not include any introductory
        text, explanations, or markdown code fences.
        The JSON object must follow this format:
        {{
          "original_query": "The user's original question",
          "sub_queries": [
            "Specific sub-query 1",
            "Specific sub-query 2"
          ]
        }}
        Complex Question: "{query}"
        JSON response:
        [/INST]
        """
        try:
            print(f"Calling Ollama model: {self.MODEL_NAME}...")
            response = ollama.chat(
                model=self.MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            generated_text = response['message']['content']
            plan = json.loads(generated_text)
            if "sub_queries" not in plan or not plan["sub_queries"]:
                 raise ValueError("LLM returned an invalid plan.")
            print("✅ Query decomposed successfully.")
            return plan
        except Exception as e:
            print(f"❌ Error during LLM decomposition: {e}")
            return {"original_query": query, "sub_queries": [query]}
    def run(self, complex_query: str, language: str = None) -> Dict[str, Any]:
        lang = language or self.detect_language(complex_query)
        plan = self.decompose_query(complex_query, language=lang)
        plan['language'] = lang
        return plan