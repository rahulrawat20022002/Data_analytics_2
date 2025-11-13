# 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



import os
import json
import re       
from pathlib import Path
from typing import List, Dict, Any
from langdetect import detect, LangDetectException

# --- NEW IMPORT ---
# We now use the official client, not 'requests'
from huggingface_hub import InferenceClient

class PlannerAgent:
    """
    This agent is responsible for Task 4:
    1. Detecting query language.
    2. Decomposing complex questions into sub-queries using an LLM.
    (Updated to use huggingface_hub.InferenceClient)
    """
    
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Using v0.2
    
    def __init__(self):
        # Try to get the API key from the environment
        api_key_from_env = os.getenv("HUGGINGFACE_API_KEY")
        
        # ⬇️ *** OR, PASTE YOUR KEY HERE (less secure) *** ⬇️
        pasted_api_key = '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' # Your key
        # ----------------------------------------------------

        self.api_key = pasted_api_key if pasted_api_key else api_key_from_env
            
        if not self.api_key:
            print("❌ Error: HUGGINGFACE_API_KEY not found.")
            raise ValueError("HUGGINGFACE_API_KEY not set")
            
        # --- INITIALIZE THE NEW CLIENT ---
        self.client = InferenceClient(token=self.api_key)
        print(f"✅ PlannerAgent initialized. Using model: {self.MODEL_NAME}")

    def detect_language(self, query: str) -> str:
        """ Detects the language of the query. """
        try:
            lang = detect(query)
            print(f"Language detected: {lang}")
            return lang
        except LangDetectException:
            print("Could not detect language. Defaulting to 'en'.")
            return 'en'

    def decompose_query(self, query: str) -> Dict[str, Any]:
        """ Uses the Hugging Face API to decompose a complex query. """
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
            print("Calling Hugging Face API... (This may take a moment for the model to load)")
            
            # --- THIS IS THE NEW, SIMPLER WAY TO CALL THE API ---
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL_NAME,
                max_tokens=300,
            )
            
            # Get the text from the response object
            generated_text = response.choices[0].message.content
            
            # --- Parsing logic is the same ---
            json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
            
            if not json_match:
                raise ValueError(f"No JSON object found in LLM response: {generated_text}")
                
            json_string = json_match.group(0)
            plan = json.loads(json_string)
            
            if "sub_queries" not in plan or not plan["sub_queries"] or plan["sub_queries"][0] == query:
                 raise ValueError(f"LLM returned a weak plan: {json_string}")
            
            print("✅ Query decomposed successfully.")
            return plan
            
        except Exception as e:
            print(f"❌ Error during LLM decomposition: {e}")
            return {
                "original_query": query,
                "sub_queries": [query] # Fallback
            }

    def run(self, complex_query: str) -> Dict[str, Any]:
        """ Runs the full planning pipeline. """
        lang = self.detect_language(complex_query)
        plan = self.decompose_query(complex_query)
        plan['language'] = lang 
        
        return plan

def save_plan(plan: Dict[str, Any], output_file: str):
    """Saves the generated plan to the specified JSON file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    OUTPUT_FILE = PROJECT_ROOT / "results" / "plan.json"
    
    print("--- Starting Planner Agent ---")
    
    try:
        agent = PlannerAgent()
        test_query = "How do EU and US policies on artificial intelligence differ, particularly regarding data privacy, ethics, and economic investment in the sector?"
        
        generated_plan = agent.run(test_query)
        
        if generated_plan and len(generated_plan.get("sub_queries", [])) > 1:
            save_plan(generated_plan, str(OUTPUT_FILE))
            print(f"\n✅ Successfully generated plan.")
            print(f"Saved plan to {OUTPUT_FILE}")
            print("\nGenerated Plan:")
            print(json.dumps(generated_plan, indent=2))
        else:
            print("\nFailed to generate a valid plan (fallback was used).")
            print("Generated Plan (Fallback):")
            print(json.dumps(generated_plan, indent=2))
            
    except Exception as e:
        print(f"\n--- Planner Agent Failed ---")
        print(f"Error: {e}")
        
    print("--- Planner Agent Finished ---")