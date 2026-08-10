# 🤖 PolicyNeural: Multi-Agent RAG Pipeline

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
- [Multilingual RAG](#-multilingual-rag)
- [Evaluation & LLM-as-Judge](#-evaluation--llm-as-judge)
- [Results & Visualization](#-results--visualization)

---

## 🔍 Overview

**PolicyNeural** goes beyond simple "Question & Answer" bots. It leverages a pipeline of specialized AI agents to handle the entire lifecycle of document analysis—from ingestion and topic modeling to semantic retrieval and automated debate.

The system uses a **Hybrid RAG** approach (combining sparse BM25 retrieval with dense vector search via Pinecone) to ensure high-precision answers. It goes a step further by employing debate agents that argue opposing viewpoints to reduce bias, and a verifier agent that fact-checks answers against source texts using Natural Language Inference (NLI).

Retrieval and generation are **multilingual** (English and German as first-class languages), and every run is scored by an **LLM-as-Judge** rubric plus a full **evaluation harness** that reports retrieval, generation and language metrics side by side.

---

## 🚀 Key Features

* **🤖 Multi-Agent Orchestration:** A modular architecture where specialized agents (Ingestion, Retrieval, Debate, Verification) collaborate via a **LangGraph** state machine.
* **🔎 Hybrid Retrieval Engine:** Combines **BM25** (keyword matching) and **Pinecone** (dense semantic embeddings) with a configurable alpha threshold ($S_{final} = \alpha \cdot S_{dense} + (1-\alpha) \cdot S_{sparse}$).
* **⚖️ Automated Debate:** Two distinct agents (DebateAgent A & B) argue opposing positions on policy questions to reach a balanced consensus.
* **🛡️ Verification & Guardrails:** * **VerifierAgent:** Checks factuality (NLI), semantic alignment ($\cos \ge 0.8$), and temporal consistency.
    * **GuardrailsAgent:** Filters PII (Personally Identifiable Information) and prevents prompt injection attacks.
* **🧠 Adaptive Memory:** A `MemoryAgent` logs parameters ($\alpha$, $k$) and auto-tunes them for improved factual precision.
* **🌍 Multilingual RAG:** Queries and documents in **English and German** share a single vector space (`paraphrase-multilingual-MiniLM-L12-v2`), so a German question retrieves relevant English sources and is answered back in German.
* **⚖️ LLM-as-Judge:** A `JudgeAgent` scores every final answer on a five-part rubric — groundedness, relevance, completeness, citation quality and language quality.
* **📏 Evaluation Harness:** An `EvalAgent` runs a labelled query set end-to-end and reports retrieval metrics (hit@k, precision@k, recall@k, MRR, nDCG@k) alongside generation and language metrics, broken down per language.

---

## 🛠 System Architecture & Agents

The project is structured around the following specialized agents:

| Agent | Function | Techniques |
| :--- | :--- | :--- |
| **PDFIngestionAgent** | Parse PDFs, extract text/tables, chunk semantically | PyMuPDF / pdfplumber + spaCy |
| **PreprocessorAgent** | Detect language, tokenize, lemmatize, NER, redact PII | spaCy (en/de + `xx` fallback), regex |
| **LanguageAgent** | Language detection, output-language directives, Unicode tokenization | langdetect |
| **TopicModelAgent** | Topic discovery (LDA / NMF) | scikit-learn / pyLDAvis |
| **EmbeddingAgent** | Build TF-IDF + BERT / SBERT representations | sentence-transformers |
| **RetrieverAgent** | **Multilingual Hybrid RAG** (BM25 + dense Pinecone) | rank-bm25 + Pinecone + multilingual SBERT |
| **PlannerAgent** | Query decomposition in the query's language | langdetect, LLM planning |
| **SummarizerAgent** | Structured summary with citations, in the query's language | Mistral-7B via Ollama |
| **DebateAgents A & B** | Argue opposing positions → consensus | LangGraph loops |
| **VerifierAgent** | NLI, semantic alignment & output-language checks | Ollama + multilingual SBERT |
| **JudgeAgent** | **LLM-as-Judge** rubric scoring of the final answer | Ollama (JSON mode, temp 0) |
| **EvaluatorAgent** | LLM-as-Judge A/B comparison of retrieval strategies | Ollama (JSON mode, temp 0) |
| **EvalAgent** | **Evaluation harness**: retrieval + generation + language metrics | numpy, SBERT, JudgeAgent |
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
├── queries/
│   ├── policy_queries.json
│   └── multilingual_eval_set.json  # Labelled EN/DE evaluation cases
├── src/
│   └── agents/
│       ├── config_loader.py
│       ├── language_agent.py
│       ├── pdf_ingestion_agent.py
│       ├── preprocessor_agent.py
│       ├── topic_model_agent.py
│       ├── embedding_agent.py
│       ├── retriever_agent.py
│       ├── planner_agent.py
│       ├── summarizer_agent.py
│       ├── debate_agent.py
│       ├── verifier_agent.py
│       ├── judge_agent.py     # LLM-as-Judge (answer rubric)
│       ├── evaluator_agent.py # LLM-as-Judge (retrieval A/B)
│       ├── eval_agent.py      # Evaluation harness
│       ├── guardrails_agent.py
│       ├── memory_agent.py
│       ├── visualizer_agent.py
│       └── orchestrator.py    # MAIN ENTRY POINT
├── config.yaml                # Models, languages, index, eval settings
├── requirements.txt           # Project dependencies
└── README.md
```
## ⚡ Getting Started
1. Clone the Repository -:
   ```bash
   git clone [https://github.com/rahulrawat20022002/Data_analytics_2.git](https://github.com/rahulrawat20022002/Data_analytics_2.git)
  
2.   Create a Virtual Environment -:
      ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies -:
    ```bash
   pip install -r requirements.txt

4. Configure Environment
Create a .env file in the root directory. Pinecone is the only external service —
generation, judging and evaluation all run locally on Ollama:
   ```bash
   PINECONE_API_KEY=your_key_here
   PINECONE_REGION=us-east-1
   ```

5. Pull the local model and the German spaCy pipeline -:
   ```bash
   ollama pull mistral:7b
   python -m spacy download de_core_news_sm
   ```

Models, languages, index name and evaluation thresholds are all set in
`config.yaml` — no need to edit agent code to change them.


## 🏃 Usage
Run the Full Pipeline
To ingest PDFs, train the topic model, and start the agent orchestrator:
   ```bash
   python src/agents/orchestrator.py
```


## Run Specific Tasks:
You can test individual components by running their specific agents:

Ingest PDFs: python src/agents/pdf_ingestion_agent.py

Run Retrieval Test: python src/agents/retriever_agent.py

Test language detection: python src/agents/language_agent.py

Test the judge: python src/agents/judge_agent.py

---

## 🌍 Multilingual RAG

Queries in English or German are detected automatically, routed through a single
shared vector space, and answered in the language they were asked in.

| Stage | Behaviour |
| :--- | :--- |
| Detection | `LanguageAgent` (seeded `langdetect`); falls back to the default below 0.60 confidence |
| Preprocessing | `en_core_web_sm` / `de_core_news_sm`, with a blank `xx` pipeline as fallback |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~50 languages) |
| Sparse retrieval | Unicode-aware, case-folded tokenization (replaces `text.split(" ")`) |
| Generation | Planner, summarizer and debate agents are all pinned to the query's language |
| Verification | Output language is checked independently of the LLM's own opinion |

Retrieval is deliberately **not** filtered by language — the point of a shared
multilingual space is that a German query can surface a relevant English source.
Pass `restrict_to_language=True` to `RetrieverAgent.search()` to opt out.

### ⚠️ Re-indexing is required

The multilingual encoder produces vectors that are **not compatible** with the
existing `all-MiniLM-L6-v2` index, even though both are 384-dim. Re-embed and
upsert into the fresh index before running anything:

```bash
python src/agents/preprocessor_agent.py   # adds a `language` field per chunk
python src/agents/embedding_agent.py      # writes results/multilingual_embeddings.npy
python src/agents/retriever_agent.py      # creates + populates policy-rag-index-multilingual
```

The previous English index (`policy-rag-index`) and `sbert_embeddings.npy` are
left untouched, so you can A/B the two.

---

## 📏 Evaluation & LLM-as-Judge

Two distinct judges, plus a metrics harness:

* **`JudgeAgent`** — scores a *final answer* 1–5 on groundedness, relevance,
  completeness, citation quality and language quality. Runs at `temperature 0`
  in JSON mode so scores are reproducible. It also runs `compare()` for pairwise
  A/B, scoring each answer independently to avoid position bias.
* **`EvaluatorAgent`** — scores a *retrieval strategy* A/B (hybrid vs reranked).
* **`EvalAgent`** — the harness that ties it together.

Run the full evaluation:

```bash
python src/agents/eval_agent.py                 # full run, with the LLM judge
python src/agents/eval_agent.py --no-judge      # embedding metrics only (fast)
python src/agents/eval_agent.py --limit 4       # smoke test on the first 4 cases
```

Metrics produced, aggregated overall **and per language**:

| Family | Metrics |
| :--- | :--- |
| Retrieval | hit@k, precision@k, recall@k, MRR, nDCG@k |
| Generation | answer relevancy, context utilisation (groundedness proxy), citation density |
| Judge | per-dimension rubric scores, overall score, pass rate |
| Multilingual | language match rate (did a German question get a German answer?) |

Output lands in `results/eval_report.json` and `results/eval_report.md`.

Retrieval metrics need ground truth: add chunk ids to `relevant_doc_ids` in
`queries/multilingual_eval_set.json`. Cases left unlabelled still produce
generation, language and judge metrics, so the harness is useful before any
labelling work is done.



## 📊 Results & Visualization
The system automatically generates insights in the results/ folder:

Topic Modeling: embedding_map.png (t-SNE/PCA projections).
<img width="3000" height="1800" alt="image" src="https://github.com/user-attachments/assets/a30021b4-42a2-4c20-861b-07dad862d595" />


Retrieval Diagnostics: retrieval_ablation.json (Comparing Sparse vs. Dense performance).
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/06b7eb30-4b6b-4473-906d-bbf924f74968" />






## 🔬 Advanced Research (Task 8)
This project includes a research-grade challenge module in src/agents/retriever_experiment_agent.py. It implements alternative architectures beyond standard RAG, such as:

GraphRAG (Entity graphs with NetworkX)

Cross-Encoder Reranking

ColBERT (Late Interaction)



👤 Author
Rahul Rawat

LinkedIn: [LinkedIn](https://www.linkedin.com/in/rahulrawat2r/)

GitHub: [Github](https://github.com/rahulrawat20022002)
   
