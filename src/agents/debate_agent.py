import os
import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Annotated
import operator

# For building the agent graph
from langgraph.graph import StateGraph, END

# We use the same official client
from huggingface_hub import InferenceClient

# --- 1. Define the State ---
# The state is the "memory" of our graph. It's what gets passed
# from one agent (node) to the next.

class DebateState(TypedDict):
    """
    Represents the state of our debate.
    
    Attributes:
        query: The user's original question.
        context: The list of retrieved documents.
        messages: The history of the debate, appended to by each agent.
        max_turns: The maximum number of turns for the debate.
    """
    query: str
    context: List[Dict[str, Any]]
    messages: Annotated[List[Dict[str, str]], operator.add] # 'operator.add' appends messages
    max_turns: int

# --- 2. Define the Agent Nodes ---

class DebateAgents:
    """
    Contains the logic for the two debating agents and the consensus agent.
    """
    
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    
    def __init__(self, api_key: str):
        self.client = InferenceClient(token=api_key)
        print("✅ DebateAgents initialized.")
        
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Formats the context for the prompts."""
        context_str = ""
        for i, doc in enumerate(context):
            source_file = os.path.basename(doc['metadata'].get('source', 'unknown'))
            page = doc['metadata'].get('page', 'N/A')
            citation_tag = f"[Source {i+1}: file={source_file}, page={page}]"
            context_str += f"{citation_tag}\nText: \"{doc['text']}\"\n\n"
        return context_str.strip()

    def _call_llm(self, prompt: str) -> str:
        """Helper function to call the LLM."""
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL_NAME,
                max_tokens=400,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            return f"Error: Could not get response."

    def agent_a_node(self, state: DebateState) -> Dict[str, Any]:
        """
        Agent A: The "Proponent"
        Takes an optimistic or "pro-policy" stance.
        """
        print("--- Agent A Turn ---")
        context_str = self._format_context(state['context'])
        debate_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state['messages']])
        
        prompt = f"""
        [INST]
        You are Policy Analyst A (Proponent).
        Your task is to argue for the *positive aspects, effectiveness, and benefits* of the policies related to the query.
        
        - Use *only* the provided context and cite your sources.
        - Your goal is to build a strong, optimistic case.
        - Respond directly to Analyst B if they have spoken.
        
        QUERY: {state['query']}
        CONTEXT:
        {context_str}
        
        DEBATE HISTORY:
        {debate_history}
        
        Your analysis (Proponent view):
        [/INST]
        """
        response = self._call_llm(prompt)
        return {"messages": [{"role": "Analyst A (Proponent)", "content": response}]}

    def agent_b_node(self, state: DebateState) -> Dict[str, Any]:
        """
        Agent B: The "Critic"
        Takes a critical or "con-policy" stance, pointing out risks/challenges.
        """
        print("--- Agent B Turn ---")
        context_str = self._format_context(state['context'])
        debate_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state['messages']])
        
        prompt = f"""
        [INST]
        You are Policy Analyst B (Critic).
        Your task is to argue about the *risks, challenges, and limitations* of the policies related to the query.
        
        - Use *only* the provided context and cite your sources.
        - Your goal is to build a strong, critical case.
        - Respond directly to Analyst A.
        
        QUERY: {state['query']}
        CONTEXT:
        {context_str}
        
        DEBATE HISTORY:
        {debate_history}
        
        Your analysis (Critic view):
        [/INST]
        """
        response = self._call_llm(prompt)
        return {"messages": [{"role": "Analyst B (Critic)", "content": response}]}

    def consensus_node(self, state: DebateState) -> Dict[str, Any]:
        """
        Consensus Agent: The "Editor"
        Reads the full debate and writes a final, balanced brief.
        """
        print("--- Consensus Agent Turn ---")
        context_str = self._format_context(state['context'])
        debate_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state['messages']])
        
        prompt = f"""
        [INST]
        You are the Chief Policy Editor.
        Your job is to write a final, balanced, and structured policy brief
        that answers the query.
        
        - Use the *full debate history* and the *original context* to form your answer.
        - Synthesize the arguments from both Analyst A and B.
        - Cite sources from the original context.
        
        QUERY: {state['query']}
        CONTEXT:
        {context_str}
        
        DEBATE HISTORY:
        {debate_history}
        
        Your Final, Balanced Policy Brief (with citations):
        [/INST]
        """
        response = self._call_llm(prompt)
        # This is the final output
        return {"messages": [{"role": "Editor (Consensus)", "content": response}]}

# --- 3. Define the Graph Edges ---

def should_continue(state: DebateState) -> str:
    """
    This function is our "router." It decides where to go next.
    If the debate has had enough turns, it goes to "consensus".
    Otherwise, it continues the debate.
    """
    turns = len(state['messages'])
    if turns >= state['max_turns']:
        print("Debate finished. Moving to consensus.")
        return "end_debate"
    else:
        print(f"Debate continuing. Turn {turns + 1}")
        return "continue"

# --- 4. Main script to build and run the graph ---

if __name__ == "__main__":
    
    # Load API Key
    HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or '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' # Your key
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not set")

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    OUTPUT_FILE = PROJECT_ROOT / "results" / "final_policy_brief.txt"
    
    print("--- Starting Debate Agent ---")
    
    # 1. Initialize the agents
    agents = DebateAgents(api_key=HF_API_KEY)
    
    # 2. Build the graph
    workflow = StateGraph(DebateState)
    
    # Add the nodes
    workflow.add_node("agent_a", agents.agent_a_node)
    workflow.add_node("agent_b", agents.agent_b_node)
    workflow.add_node("consensus", agents.consensus_node)
    
    # 3. Define the flow (edges)
    workflow.set_entry_point("agent_a")
    
    # From A, decide where to go
    workflow.add_conditional_edges(
        "agent_a",
        should_continue,
        {
            "continue": "agent_b",
            "end_debate": "consensus"
        }
    )
    
    # From B, decide where to go
    workflow.add_conditional_edges(
        "agent_b",
        should_continue,
        {
            "continue": "agent_a",
            "end_debate": "consensus"
        }
    )
    
    # The consensus node is the end
    workflow.add_edge("consensus", END)
    
    # 4. Compile the graph
    app = workflow.compile()
    
    # 5. Run the graph with test data
    
    # We use the same test data as the summarizer
    test_query = "What are the main policies regarding climate change?"
    test_docs = [
        {
            "id": "chunk_90", "score": 0.5496,
            "text": "CHAPTER 1 gLObaL PROSPECTS aND POLICIES... emissions reduction targets, countries need a holistic set of mitigation instruments, ideally including carbon pricing, public infrastructure investment in clean en...",
            "metadata": {"source": "ch1.pdf", "page": 22}
        },
        {
            "id": "chunk_1234", "score": 0.5210,
            "text": "The European Union has committed to its 'Green Deal', aiming for carbon neutrality by 2050. This involves significant investment in renewable energy and a circular economy. However, some critics point to the high cost and the challenge of implementation across all member states.",
            "metadata": {"source": "eu_policy.pdf", "page": 5}
        }
    ]
    
    # Set the initial state for the graph
    initial_state = {
        "query": test_query,
        "context": test_docs,
        "messages": [],
        "max_turns": 4 # Let them argue for 4 turns (A -> B -> A -> B)
    }
    
    print("Invoking graph...")
    # The 'stream' method lets us see the output from each step
    final_state = None
    for s in app.stream(initial_state, {"recursion_limit": 10}):
        print(s)
        print("---")
        final_state = s

    # 6. Save the final consensus
    if final_state:
        # Get the last message, which is from the "Editor (Consensus)"
        final_brief = final_state['consensus']['messages'][-1]['content']
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(final_brief)
        
        print(f"\n✅ Successfully generated and saved final policy brief.")
        print(f"Saved to {OUTPUT_FILE}")
        print("\n--- FINAL BRIEF ---")
        print(final_brief)
    
    print("--- Debate Agent Finished ---")