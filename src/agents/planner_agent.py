import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from langdetect import detect, LangDetectException
import ollama
from dotenv import load_dotenv

load_dotenv()
class PlannerAgent:
    MODEL_NAME = "mistral:7b"
    def __init__(self):
        try:
            ollama.show(self.MODEL_NAME)
        except Exception:
            print(f"❌ Error: Ollama model '{self.MODEL_NAME}' not found.")
            print(f"Please run 'ollama pull {self.MODEL_NAME}' in your terminal.")
            raise
        print(f"✅ PlannerAgent initialized. Using Ollama model: {self.MODEL_NAME}")
    def detect_language(self, query: str) -> str:
        try:
            lang = detect(query)
            print(f"Language detected: {lang}")
            return lang
        except LangDetectException:
            print("Could not detect language. Defaulting to 'en'.")
            return 'en'
    def decompose_query(self, query: str) -> Dict[str, Any]:
        print(f"Decomposing complex query: '{query}'")
        prompt = f"""
        [INST]
        You are a research assistant. Your task is to decompose a complex policy question 
        into a list of 3-5 specific, answerable sub-queries.
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
    def run(self, complex_query: str) -> Dict[str, Any]:
        lang = self.detect_language(complex_query)
        plan = self.decompose_query(complex_query)
        plan['language'] = lang 
        return plan