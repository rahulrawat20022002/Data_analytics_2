📖 Project OverviewAnalyzing real-world policy documents requires more than just simple keyword search. PolicyNeural leverages a pipeline of specialized AI agents to handle the entire lifecycle of document analysis—from ingestion and topic modeling to semantic retrieval and automated debate.The system uses a Hybrid RAG approach (combining sparse BM25 retrieval with dense vector search via Pinecone) to ensure high-precision answers. It goes beyond simple QA by including agents that plan complex queries, debate opposing viewpoints to reduce bias, and verify facts against the source text.




🚀 Key FeaturesMulti-Agent Orchestration: A modular architecture where specialized agents (Ingestion, Retrieval, Debate, Verification) collaborate to solve complex user queries.Hybrid Retrieval Engine: Combines BM25 (keyword matching) and Pinecone (dense semantic embeddings) with a configurable alpha threshold ($S_{final} = \alpha \cdot S_{dense} + (1-\alpha) \cdot S_{sparse}$) to maximize retrieval accuracy.Automated Debate & Consensus: Two distinct agents argue opposing viewpoints on a policy question to generate a balanced, unbiased consensus.Robust Guardrails: Integrated PII (Personally Identifiable Information) redaction and prompt injection defense to ensure security and privacy.Self-Correction: A VerifierAgent checks the factual alignment of generated answers using Natural Language Inference (NLI) techniques.




🛠️ System Architecture & Tech StackThe project is structured around the following specialized agents:AgentRole & ResponsibilityKey TechnologiesPDFIngestionAgentParses PDF directories and performs semantic chunking.PyMuPDF, RecursiveCharacterTextSplitterPreprocessorAgentTokenization, lemmatization, NER, and PII redaction.spaCy, regexTopicModelAgentDiscovers latent topics within the policy corpus.scikit-learn (LDA/NMF), pyLDAvisEmbeddingAgentGenerates vector representations of text chunks.sentence-transformers (SBERT), TF-IDFRetrieverAgentPerforms Hybrid RAG search (Sparse + Dense).LangChain, Pinecone, rank-bm25PlannerAgentDecomposes complex user queries into sub-tasks.LLM Planning, langdetectSummarizerAgentGenerates structured summaries with specific citations.Flan-T5 / Mistral-7BDebateAgents (A&B)Argue contrasting positions to reduce hallucination/bias.LangGraph loopsVerifierAgentChecks semantic alignment ($\cos \ge 0.8$) and factuality.facebook/bart-large-mnliOrchestratorManages the state and flow between all agents.LangGraph State Machine




📂 Repository StructureBash├── data/
│   └── pdfs/                
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
│       └── orchestrator.py   
├── results/                   
└── README.md



📊 Performance & VisualizationThe system includes a VisualizerAgent that generates insights into the model's performance:Topic Modeling: t-SNE/PCA projections of the document embedding space.Retrieval Diagnostics: Ablation studies comparing Sparse vs. Dense retrieval performance.Confidence Trajectories: Tracking agent confidence scores across the debate rounds.(Optional: Add a screenshot of your results/retrieval_plot.png here)



⚙️ Installation & UsageClone the repository:Bashgit clone https://github.com/yourusername/PolicyNeural.git
cd PolicyNeural
Install dependencies:Bashpip install -r requirements.txt
Set up Environment Variables:Create a .env file with your API keys:Code snippetPINECONE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # If using OpenAI models
Run the Pipeline:To process PDFs and start the agent orchestrator:Bashpython src/agents/orchestrator.py




🔬 Advanced Research (Task 8)Beyond standard RAG, this project includes a RetrieverExperimentAgent designed to test state-of-the-art retrieval architectures, including:GraphRAG: Building entity graphs for structured retrieval.Cross-Encoder Reranking: Re-scoring top-k results for higher precision.ColBERT: Late interaction models for fine-grained token matching.


👤 Author Rahul Rawat - AI Engineer & Researcher[Link to your Portfolio/LinkedIn]
