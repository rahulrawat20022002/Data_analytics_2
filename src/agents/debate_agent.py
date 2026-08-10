import os
import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
import ollama

import config_loader
from language_agent import LanguageAgent

class DebateState(TypedDict):
    query: str
    context: List[Dict[str, Any]]
    messages: Annotated[List[Dict[str, str]], operator.add]
    max_turns: int
    language: str

class DebateAgents:
    def __init__(self):
        self.MODEL_NAME = config_loader.get("llm.model", "mistral:7b")
        try:
            ollama.show(self.MODEL_NAME)
        except Exception:
            print(f"❌ Error: Ollama model '{self.MODEL_NAME}' not found.")
            print(f"Please run 'ollama pull {self.MODEL_NAME}' in your terminal.")
            raise
        self.language_agent = LanguageAgent()
        print(f"✅ DebateAgents initialized. Using Ollama model: {self.MODEL_NAME}")

    def _language_directive(self, state: DebateState) -> str:
        """Language instruction for debate turns.

        The consensus turn produces the user-facing brief, so every turn is held
        to the same language -- otherwise the editor has to translate mid-debate
        and detail gets lost.
        """
        language = state.get('language', 'en')
        return self.language_agent.instruction_for(language)
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        context_str = ""
        for i, doc in enumerate(context):
            source_file = os.path.basename(doc['metadata'].get('source', 'unknown'))
            page = doc['metadata'].get('page', 'N/A')
            citation_tag = f"[Source {i+1}: file={source_file}, page={page}]"
            context_str += f"{citation_tag}\nText: \"{doc['text']}\"\n\n"
        return context_str.strip()
    def _call_llm(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            return "Error: Could not get response."
    def agent_a_node(self, state: DebateState) -> Dict[str, Any]:
        print("--- Agent A Turn ---")
        context_str = self._format_context(state['context'])
        debate_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state['messages']])
        prompt = f"""
        [INST]
        You are Policy Analyst A (Proponent)... (rest of prompt)
        {self._language_directive(state)}
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
        print("--- Agent B Turn ---")
        context_str = self._format_context(state['context'])
        debate_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state['messages']])
        prompt = f"""
        [INST]
        You are Policy Analyst B (Critic)... (rest of prompt)
        {self._language_directive(state)}
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
        print("--- Consensus Agent Turn ---")
        context_str = self._format_context(state['context'])
        debate_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state['messages']])
        prompt = f"""
        [INST]
        You are the Chief Policy Editor... (rest of prompt)
        {self._language_directive(state)}
        QUERY: {state['query']}
        CONTEXT:
        {context_str}
        DEBATE HISTORY:
        {debate_history}
        Your Final, Balanced Policy Brief (with citations):
        [/INST]
        """
        response = self._call_llm(prompt)
        return {"messages": [{"role": "Editor (Consensus)", "content": response}]}

def should_continue(state: DebateState) -> str:
    turns = len(state['messages'])
    if turns >= state['max_turns']:
        print("Debate finished. Moving to consensus.")
        return "end_debate"
    else:
        print(f"Debate continuing. Turn {turns + 1}")
        return "continue"