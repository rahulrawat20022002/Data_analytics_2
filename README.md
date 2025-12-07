# 🤖 PolicyNeural: Multi-Agent RAG Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-Active-green)

> **An intelligent, multi-agent system designed to ingest, analyze, debate, and verify complex policy documents using Hybrid RAG (Retrieval-Augmented Generation).**

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Results & Visualization](#-results--visualization)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

**PolicyNeural** goes beyond simple "Question & Answer" bots. It leverages a pipeline of specialized AI agents to handle the entire lifecycle of document analysis—from ingestion and topic modeling to semantic retrieval and automated debate.

The system uses a **Hybrid RAG** approach (combining sparse BM25 retrieval with dense vector search via Pinecone) to ensure high-precision answers. It goes a step further by employing debate agents that argue opposing viewpoints to reduce bias, and a verifier agent that fact-checks answers against source texts using Natural Language Inference (NLI).

---

## 🚀 Key Features

* **🤖 Multi-Agent Orchestration:** A modular architecture where specialized agents (Ingestion, Retrieval, Debate, Verification) collaborate via a **LangGraph** state machine.
* **🔎 Hybrid Retrieval Engine:** Combines **BM25** (keyword matching) and **Pinecone** (dense semantic embeddings) with a configurable alpha threshold ($S_{final} = \alpha \cdot S_{dense} + (1-\alpha) \cdot S_{sparse}$).
* **⚖️ Automated Debate:** Two distinct agents (DebateAgent A & B) argue opposing positions on policy questions to reach a balanced consensus.
* **🛡️ Verification & Guardrails:** * **VerifierAgent:** Checks factuality (NLI), semantic alignment ($\cos \ge 0.8$), and temporal consistency.
    * **GuardrailsAgent:** Filters PII (Personally Identifiable Information) and prevents prompt injection attacks.
* **🧠 Adaptive Memory:** A `MemoryAgent` logs parameters ($\alpha$, $k$) and auto-tunes them for improved factual precision.

---

## 🛠 System Architecture & Agents

The project is structured around the following specialized agents:

| Agent | Function | Techniques |
| :--- | :--- | :--- |
| **PDFIngestionAgent** | Parse PDFs, extract text/tables, chunk semantically | PyMuPDF / pdfplumber + spaCy |
| **PreprocessorAgent** | Tokenize, lemmatize, NER, redact PII | spaCy, regex |
| **TopicModelAgent** | Topic discovery (LDA / NMF) | scikit-learn / pyLDAvis |
| **EmbeddingAgent** | Build TF-IDF + BERT / SBERT representations | sentence-transformers |
| **RetrieverAgent** | **Baseline Hybrid RAG** (BM25 + dense Pinecone) | LangChain + rank-bm25 + Pinecone |
| **PlannerAgent** | Query decomposition & multilingual routing | langdetect, LLM planning |
| **SummarizerAgent** | Structured summary with citations | Flan-T5 / Mistral-7B |
| **DebateAgents A & B** | Argue opposing positions → consensus | LangGraph loops |
| **VerifierAgent** | NLI & semantic alignment checks | facebook/bart-large-mnli |
| **EvaluatorAgent** | Judge factuality / clarity / bias | LLM-as-judge |
| **GuardrailsAgent** | PII redaction & prompt injection defence | regex / policy filters |
| **VisualizerAgent** | Plot topics / confidence / agent graph | matplotlib / networkx |
| **Orchestrator** | Connect agents → pipeline | LangGraph state machine |

---

## 📂 Folder Structure

The project follows a modular structure where each agent is self-contained:

```bash
PolicyNeural/
├── data/
│   └── pdfs/                  # Place your 10+ policy PDF files here
├── results/                   # Output plots, logs, and JSON summaries
│   ├── metrics.json           # Factual accuracy and latency logs
│   ├── retrieval_ablation.json
│   └── plots/                 # Visualizations
├── src/
│   └── agents/
│       ├── pdf_ingestion_agent.py
│       ├── preprocessor_agent.py
│       ├── topic_model_agent.py
│       ├── embedding_agent.py
│       ├── retriever_agent.py
│       ├── planner_agent.py
│       ├── summarizer_agent.py
│       ├── debate_agent.py
│       ├── verifier_agent.py
│       ├── guardrails_agent.py
│       ├── memory_agent.py
│       ├── visualizer_agent.py
│       └── orchestrator.py    # MAIN ENTRY POINT
├── requirements.txt           # Project dependencies
└── README.md
```
## ⚡ Getting Started
1. Clone the Repository -: ```bash git clone [https://github.com/rahulrawat20022002/Data_analytics_2.git](https://github.com/rahulrawat20022002/Data_analytics_2.git)
  
2.   Create a Virtual Environment -:```bash python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies -: ```bash pip install -r requirements.txt

4. Configure Environment
Create a .env file in the root directory and add your API keys -: ```bash PINECONE_API_KEY=your_key_here
PINECONE_ENV=us-east-1-aws
OPENAI_API_KEY=your_key_here (if using OpenAI models)
HUGGINGFACEHUB_API_TOKEN=your_token_here


🏃 Usage
Run the Full Pipeline
To ingest PDFs, train the topic model, and start the agent orchestrator: ```bash python src/agents/orchestrator.py

Run Specific Tasks:
You can test individual components by running their specific agents:

Ingest PDFs: python src/agents/pdf_ingestion_agent.py

Run Retrieval Test: python src/agents/retriever_agent.py



📊 Results & Visualization
The system automatically generates insights in the results/ folder:

Topic Modeling: embedding_map.png (t-SNE/PCA projections).

Retrieval Diagnostics: retrieval_ablation.json (Comparing Sparse vs. Dense performance).

Confidence Trajectories: Plots tracking agent confidence across debate rounds.





🔬 Advanced Research (Task 8)
This project includes a research-grade challenge module in src/agents/retriever_experiment_agent.py. It implements alternative architectures beyond standard RAG, such as:

GraphRAG (Entity graphs with NetworkX)

Cross-Encoder Reranking

ColBERT (Late Interaction)



👤 Author
Rahul Rawat

LinkedIn: [LinkedIn](https://www.linkedin.com/in/rahulrawat2r/)

GitHub: [Github](https://github.com/rahulrawat20022002)
   
