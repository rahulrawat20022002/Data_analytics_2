import os
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class MemoryAgent:
    """
    This agent is responsible for Task 7:
    1. Logging pipeline parameters (latency, confidence, etc.) to a persistent file.
    
    The "auto-tuning" aspect will be handled by the Orchestrator,
    which can read this memory to adjust parameters.
    """
    
    def __init__(self, memory_file: str):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the file exists and is a valid JSON list
        if not self.memory_file.exists():
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
        print(f"MemoryAgent initialized. Logging to {self.memory_file}")

    def _load_memory(self) -> List[Dict[str, Any]]:
        """Loads the current memory log from the JSON file."""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                if not isinstance(memory, list):
                    return [] # Start fresh if file is corrupt
                return memory
        except json.JSONDecodeError:
            print("Warning: Memory file corrupted. Starting fresh.")
            return []

    def log_run(self, run_data: Dict[str, Any]):
        """
        Logs a new entry to the memory file.
        
        Args:
            run_data: A dictionary containing metrics for a single run,
                      e.g., {"query": ..., "latency_ms": ..., "confidence": ...}
        """
        print(f"Logging new run to memory...")
        memory = self._load_memory()
        
        # Add a timestamp to the log entry
        run_data["timestamp"] = datetime.now().isoformat()
        
        memory.append(run_data)
        
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, indent=2)
            print("Log entry saved.")
        except Exception as e:
            print(f"Error saving memory: {e}")

    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Returns all logs for analysis."""
        return self._load_memory()
        
    def get_last_n_runs(self, n: int) -> List[Dict[str, Any]]:
        """Returns the last N run logs."""
        return self._load_memory()[-n:]

if __name__ == "__main__":
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # As per your results screenshot
    MEMORY_FILE = PROJECT_ROOT / "results" / "memory.json"
    
    print("--- Starting Memory Agent (Test Mode) ---")
    
    agent = MemoryAgent(memory_file=MEMORY_FILE)
    
    # --- Test: Log two simulated runs ---
    
    # Run 1
    test_run_1 = {
        "query": "What is the policy on data privacy?",
        "retriever_alpha": 0.5,
        "retrieval_k": 5,
        "latency_ms": 1250.5,
        "confidence_score": 0.92, # From VerifierAgent (semantic alignment)
        "factuality_pass": True   # From VerifierAgent (NLI)
    }
    agent.log_run(test_run_1)
    
    # Run 2
    test_run_2 = {
        "query": "What are the main policies regarding climate change?",
        "retriever_alpha": 0.7,
        "retrieval_k": 3,
        "latency_ms": 980.0,
        "confidence_score": 0.88,
        "factuality_pass": True
    }
    agent.log_run(test_run_2)
    
    # --- Test: Read logs back ---
    print("\n--- Current Memory Log ---")
    all_logs = agent.get_all_logs()
    print(json.dumps(all_logs, indent=2))
    
    print(f"\nTotal runs logged: {len(all_logs)}")
    print("--- Memory Agent Finished ---")