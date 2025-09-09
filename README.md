# Enterprise-Rag
Production-Ready Knowledge Assistant with Multi-Modal Understanding

Here's a complete README for your Enterprise-RAG solution, in straightforward English and with a compelling explanation anyone can follow:

***

# Enterprise-Rag  
**Production-Ready Knowledge Assistant with Multi-Modal Understanding**

***

## What is this?

Enterprise-Rag is a modern AI system purpose-built to help organizations find answers from their own documents, data, and knowledge bases. If you’ve got a collection of PDFs, text files, images, and are tired of manual searching or unreliable AI guesses, this system gives you fast, accurate, and referenced answers — all without sending your sensitive data to external APIs.

Simply put: it’s like having a “ChatGPT for your organization’s documents,” but with enterprise safety, transparency, and full local control.

***

## How does it work?

1. **Document Ingestion:**  
   First, we load your files (PDFs, text, even images) and split them into small, manageable chunks for deep indexing.

2. **Embedding & Indexing:**  
   Each chunk is transformed into “vectors” using modern AI models (like Sentence Transformers). These vectors let the system understand meaning, not just keywords.
   
   To make sure keywords and acronyms aren’t missed, we also use a classic BM25 algorithm. Together, these cover both semantic and precise keyword searches.

3. **Storage:**  
   Vector search is powered by FAISS, an open-source library famous for its speed and reliability. BM25 is kept in memory for blazing-fast keyword matches.

4. **Hybrid Retrieval:**  
   When you ask a question, the system runs two queries — one semantic (FAISS) and one keyword-based (BM25). Their results are combined using RRF (Reciprocal Rank Fusion). This means you get the best possible context, whether your question is fuzzy or exact.

5. **AI Answer Generation:**  
   Context chunks are passed to Ollama, running a local Llama 3.1 model. Answers are grounded only in retrieved content, and the system ensures citations for every claim — so you always know where the information comes from.

6. **Evaluation & Metrics:**  
   Built-in evaluation tools using RAGAS let you measure how trustworthy, precise, and relevant the answers are, with key metrics:
   - Faithfulness
   - Answer Similarity
   - Context Precision

***

## Main Features

- **Multi-modal ingestion:** Handles text, PDFs, and images (with optional AI captioning)
- **Privacy-first:** All processing happens locally, nothing leaves your organization
- **Hybrid search:** Semantic (vector-based) + precise keyword search
- **Citations & guardrails:** Answers always point to source documents; AI never invents info
- **Abstention logic:** If the system doesn’t find related information, it admits “I don’t know from the KB”
- **Flexible evaluation:** You get real metrics to track and improve quality over time
- **User-friendly API:** FastAPI powers the REST endpoints, easy to plug into any dashboard or workflow

***

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally
- Pull a model in Ollama (like `ollama pull llama3.1`)
- Virtual environment set up (`python -m venv .venv` and activate)

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -U langchain-huggingface sentence-transformers
```

### 3. Configure .env  
Sample:
```
FAISS_INDEX_PATH=./faiss_index
COLLECTION_NAME=enterprise_kg
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
MAX_CHUNK_TOKENS=800
CHUNK_OVERLAP=120
TOP_K=8
HYBRID_ALPHA=0.5
```

### 4. Index Your Documents

Put your documents (PDFs, text files, images) in a `data/` folder.  
Then run:
```bash
python -m app.scripts.index_documents --data_dir ./data
```

### 5. Start the API

```bash
python -m uvicorn app.api.main:app --reload
```

Visit [localhost:8000/docs](http://localhost:8000/docs) to try it out!

### 6. Evaluate

For metrics and QA, launch:
```bash
streamlit run app/evaluation/dashboard.py
```
Upload sample queries and see detailed results for faithfulness, similarity, and context precision.
<img width="1456" height="607" alt="Screenshot (13)" src="https://github.com/user-attachments/assets/c342f17f-feba-4602-a547-b72ba023d0ef" />

***

## Tech Breakdown

- **Document Handling:** PDF/text/image loaders, chunking, optional FastVLM for image captioning
- **Vector Search:** FAISS with local persistence
- **Keyword Search:** BM25 in-memory index, rebuilt from stored chunks
- **Hybrid Ranking:** RRF — combines vector and keyword hits for the best context
- **LLM Generation:** Local Ollama runs Llama 3.1 for efficient, private answer generation
- **API Layer:** FastAPI for REST integration, Streamlit for dashboard/evaluation

***

## Why does this matter?

Most chatbots and document Q&A systems you find online struggle to combine deep language understanding with accurate document referencing. They either hallucinate, get tripped up by acronyms, or can’t cite where information came from.

**Enterprise-Rag solves that:**
- Never invents answers — only answers from your documents
- Can understand both general and highly specific questions
- Shows citations for every response, so users trust the output  
- Works with images, not just text  
- Scalable. Local-first, high performance, secure

It's a robust, transparent, and scalable answer engine for your organization’s collective knowledge.

***

## Troubleshooting

- **Empty Answers?**
  - Did you run the indexer?  
  - Is `FAISS_INDEX_PATH` correct in `.env`?
  - Is your data meaningful (check document types)?

- **Low Score in Hybrid Retrieval?**
  - That's normal for RRF! It ranks by relevance, not raw score magnitudes.

- **Ollama 404 or Connection Issue?**
  - Make sure you started Ollama (`ollama serve`) and model is available (`ollama list`).

- **Image Handling Not Working?**
  - By default, only text is processed. Enable FastVLM/captioning if you want image content indexed.

***

## Future Directions

- Integrate table/text extraction models for images and diagrams
- Advanced semantic chunking (group paragraphs by meaning)
- More robust handling for huge datasets
- Plug in authentication and authorization for team workflows
- Docker/Kubernetes ready deployment scripts

***

## License

MIT 

***

This project is open for feedback, issues, and improvements. It’s built for real-world use, not just demos or research slides! Whether you’re an engineer, data scientist, or tech lead, Enterprise-Rag will make your document knowledge always accessible and reliable.

