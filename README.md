## Task

You will build and evaluate a **multi-agent RAG pipeline** that:
- Loads and processes **real-world policy PDFs** (OECD, IMF, UN, etc.)
- Retrieves relevant context via **Hybrid + Advanced RAG retrieval**
- Summarizes or answers complex cross-document questions
- Evaluates factual precision and source alignment


The system must combine **classical NLP**, **transformer embeddings**, and **agentic reasoning** within a reproducible GitHub repository.



## Architecture
You will implement the following agents:

| Agent | Function | Techniques |
|--------|-----------|-------------|
| **PDFIngestionAgent** | Parse PDFs, extract text & tables, chunk semantically | PyMuPDF / pdfplumber + spaCy |
| **PreprocessorAgent** | Tokenize / lemmatize / NER / redact PII | spaCy, regex |
| **TopicModelAgent** | Topic discovery (LDA / NMF) | scikit-learn / pyLDAvis |
| **EmbeddingAgent** | Build TF-IDF + BERT / SBERT representations | `sentence-transformers` |
| **RetrieverAgent** | **Baseline Hybrid RAG** (BM25 + dense Pinecone) | LangChain + rank-bm25 + Pinecone (Free Tier) |
| **RetrieverExperimentAgent** | **Advanced RAG** (GraphRAG, Cross-Encoder, Self-RAG …) | networkx / CrossEncoder / ColBERT |
| **PlannerAgent** | Query decomposition & multilingual routing | langdetect, LLM planning |
| **SummarizerAgent** | Structured summary with citations `[src: file p.X]` | Flan-T5 / Mistral-7B |
| **DebateAgents A & B** | Argue opposing positions → consensus | LangGraph loops |
| **VerifierAgent** | NLI & semantic alignment checks | `facebook/bart-large-mnli` |
| **EvaluatorAgent** | Judge factuality / clarity / bias | LLM-as-judge |
| **GuardrailsAgent** | PII redaction & prompt injection defence | regex / policy filters |
| **MemoryAgent** | Persist α β k weights & adapt | JSON store |
| **VisualizerAgent** | Plot topics / confidence / agent graph | matplotlib / networkx |
| **Orchestrator** | Connect agents → pipeline | LangGraph state machine |

---

### **Task 1 — PDF Ingestion & Pre-Processing**
- Parse all PDFs under `data/pdfs/` using `pdfplumber` or `PyPDFLoader`.
- Chunk semantically (`RecursiveCharacterTextSplitter`).
- Perform tokenization, POS tagging, lemmatization, NER.
- Save outputs to `results/classical_output.json`.

### **Task 2 — Topic Modeling & Representation Diagnostics**
- Run **LDA / NMF** to identify 10+ topics.
- Train **Word2Vec or GloVe**, compare to BERT/SBERT embeddings.
- Visualize via t-SNE / PCA → `results/embedding_map.png`.


### **Task 3 — Hybrid Retrieval (BM25 + Pinecone)**
- Implement in `src/agents/retriever_agent.py`.
- Combine sparse (BM25) and dense (Pinecone) similarity:
  \[
  S_\text{final} = \alpha S_\text{dense} + (1-\alpha) S_\text{sparse}
  \]
- Store retrieval diagnostics in `results/retrieval_ablation.json`.

### **Task 4 — Planning & Multilingual Query Routing**
- Detect query language (EN/DE).
- Decompose complex policy questions into sub-queries via `PlannerAgent`.
- Save to `results/plan.json`.

### **Task 5 — Synthesis & Debate**
- **SummarizerAgent:** produce structured summaries with citations `[src: file.pdf, p.X]`.
- **DebateAgents A/B:** argue contrasting positions; consensus → `results/final_policy_brief.txt`.

### **Task 6 — Verification & Guardrails**
- **VerifierAgent:** check factuality (NLI), semantic alignment (cos ≥ 0.8), and temporal consistency.
- **GuardrailsAgent:** redact PII and filter injected prompts.
- Metrics → `results/metrics.json`.

### **Task 7 — Adaptivity & Visualization**
- **MemoryAgent:** log parameters `(α, k, latency, confidence)`.
- Auto-tune for improved factual precision.
- **VisualizerAgent:** plot confidence trajectories & agent graph → `results/plots/`.

### **Task 8 — Advanced Retrieval Architectures (Beyond Hybrid + Pinecone)**
> *Research-grade challenge*

Implement **one** alternative retrieval architecture in  
`src/agents/retriever_experiment_agent.py`.

#### Options
| Paradigm | Description | Hints |
|-----------|-------------|-------|
| **Cross-Encoder Reranking** | Re-score top-k results with a transformer (`cross-encoder/ms-marco-MiniLM-L-6-v2`). | `sentence-transformers` |
| **GraphRAG** | Build entity graphs with `spaCy` + `networkx`; retrieve subgraphs. | entity co-occurrence edges |
| **Self-RAG** | Generate → re-retrieve → refine in a feedback loop. | two-stage generation |
|  **ColBERT / Late Interaction** | Fine-grained token-level matching for semantic precision. | `colbert-ai` |
| **Long-Context RAG** | Use long-context LLMs (e.g., Mistral 7B Instruct, LongT5). | full-context inference |
| **Multi-Retriever Ensemble** | Combine multiple retrievers with learned weights. | adaptive fusion |

#### Deliverables
| File | Description |
|------|--------------|
| `src/agents/retriever_experiment_agent.py` | your advanced retriever |
| `results/retrieval_comparison.json` | metrics vs baseline |
| `results/retrieval_plot.png` | visualization |

