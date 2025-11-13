import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# For plotting
import matplotlib.pyplot as plt
import networkx as nx

class VisualizerAgent:
    """
    This agent is responsible for Task 7:
    1. Plotting confidence trajectories from the memory file.
    2. Plotting the agent graph (workflow).
    """
    
    def __init__(self, memory_file: str, plot_dir: str):
        self.memory_file = Path(memory_file)
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.memory_file.exists():
            print(f"⚠️ Warning: Memory file not found at {self.memory_file}")
            self.memory_data = pd.DataFrame()
        else:
            # Load the memory logs into a pandas DataFrame for easy plotting
            try:
                self.memory_data = pd.read_json(self.memory_file)
                print(f"✅ VisualizerAgent initialized. Loaded {len(self.memory_data)} logs.")
            except ValueError:
                print(f"⚠️ Warning: Memory file {self.memory_file} is empty or corrupt.")
                self.memory_data = pd.DataFrame()

    def plot_confidence_trajectory(self):
        """
        Plots the 'confidence_score' over time.
        Saves to 'confidence_curve.png'.
        """
        if "confidence_score" not in self.memory_data.columns:
            print("❌ Cannot plot confidence: 'confidence_score' not found in memory logs.")
            return

        print("Plotting confidence trajectory...")
        
        plt.figure(figsize=(10, 6))
        
        # Plot confidence score
        self.memory_data['confidence_score'].plot(
            kind='line', 
            marker='o', 
            title='Pipeline Confidence Score Over Time'
        )
        
        # Add an average line
        avg_confidence = self.memory_data['confidence_score'].mean()
        plt.axhline(
            y=avg_confidence, 
            color='r', 
            linestyle='--', 
            label=f'Average ({avg_confidence:.2f})'
        )
        
        plt.xlabel("Run Index")
        plt.ylabel("Confidence Score (Semantic Alignment)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_file = self.plot_dir / "confidence_curve.png"
        plt.savefig(output_file)
        print(f"✅ Confidence plot saved to {output_file}")
        plt.close()

    def plot_agent_graph(self):
        """
        Plots a static graph of the agent pipeline.
        Saves to 'agent_graph.png'.
        """
        print("Plotting agent graph...")
        
        G = nx.DiGraph() # A Directed Graph

        # Add nodes for each agent
        agents = [
            "User Query", "Guardrails (In)", "Planner", "Retriever", 
            "Summarizer", "DebateAgents", "Verifier", "Guardrails (Out)", 
            "Memory", "Final Response"
        ]
        G.add_nodes_from(agents)
        
        # Define the edges (the flow)
        edges = [
            ("User Query", "Guardrails (In)"),
            ("Guardrails (In)", "Planner"),
            ("Planner", "Retriever"),
            ("Retriever", "Summarizer"),
            ("Summarizer", "DebateAgents"),
            ("DebateAgents", "Verifier"),
            ("Verifier", "Guardrails (Out)"),
            ("Guardrails (Out)", "Final Response"),
            # Memory logging happens throughout (simplified here)
            ("Verifier", "Memory") 
        ]
        G.add_edges_from(edges)
        
        # Draw the graph
        plt.figure(figsize=(12, 10))
        
        # Use a layout that spreads nodes out
        pos = nx.spring_layout(G, k=0.9, iterations=50) 
        
        nx.draw(
            G, 
            pos, 
            with_labels=True, 
            node_size=3000, 
            node_color="#a0cbe2", 
            font_size=10, 
            font_weight="bold", 
            arrows=True, 
            arrowstyle="->",
            arrowsize=20
        )
        
        plt.title("RAG Agent Pipeline Workflow", size=15)
        
        output_file = self.plot_dir / "agent_graph.png"
        plt.savefig(output_file, bbox_inches="tight")
        print(f"✅ Agent graph saved to {output_file}")
        plt.close()

    def run(self):
        """
        Runs all visualization tasks.
        """
        if self.memory_data.empty:
            print("Memory file is empty. Skipping confidence plot.")
        else:
            self.plot_confidence_trajectory()
            
        # The agent graph is static and can always be drawn
        self.plot_agent_graph()

if __name__ == "__main__":
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Input file (from MemoryAgent)
    MEMORY_FILE = PROJECT_ROOT / "results" / "memory.json"
    
    # Output directory (as per your results screenshot)
    PLOT_DIR = PROJECT_ROOT / "results" / "plots"
    
    print("--- Starting Visualizer Agent ---")
    
    agent = VisualizerAgent(memory_file=MEMORY_FILE, plot_dir=PLOT_DIR)
    agent.run()
    
    print("--- Visualizer Agent Finished ---")