# 📚 Book RAG System

A state-of-the-art, localized **Retrieval-Augmented Generation (RAG)** application designed for semantic questioning over a personal library of books (PDFs and plain text files). 

The system features a decoupled architecture with a **FastAPI backend** managing document ingestion, semantic vector embedding generation, and LLM orchestration, paired with a **modern, premium React SPA dashboard** styled in Vanilla CSS.

---

## 🚀 Key Features

* **Real-time Ingestion**: Upload, parse, and index `.pdf` and `.txt` files directly from the UI.
* **Decoupled Multi-Book Filtering**: Perform semantic search scopes globally or filter search queries to a specific book.
* **Hallucination Safeguards**: Restrictive prompt construction directives ensure the LLM answers *only* when the information exists in the retrieved book passages.
* **React Interface**: Beautifully designed UI utilizing dark mode, glassmorphism, micro-animations, and responsive layouts.
* **Idempotent Data Pipeline**: Deterministic document chunk ID generation prevents duplicate database records on re-ingestion.
* **Flexible LLM Support**: Dynamically route queries to local open-source models (**Ollama** or **LM Studio** / custom OpenAI-compatible endpoints) or cloud services (**OpenAI API**).
* **Local Persistence**: Data is written directly to disk using a local **ChromaDB SQLite** backend.

---

## 🛠️ Technology Stack

* **Frontend**: React, Vite, Vanilla CSS
* **Backend**: FastAPI, Uvicorn, Pydantic Settings
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
* **Vector Store**: ChromaDB (with disk-backed SQLite storage)
* **Testing**: Pytest

---

## 📂 Directory Layout

```text
Book-RAG/
├── config.yaml              # App configuration (chunk size, model types, temperature, ports)
├── requirements.txt         # Python dependencies
├── architecture.md          # Technical pipeline descriptions
├── data/
│   ├── raw/                 # Ingested PDF & TXT documents
│   └── vector_store/        # Persistent ChromaDB SQLite binary files
├── src/
│   ├── main.py              # FastAPI server entry point
│   ├── config.py            # Strongly typed settings loader (Pydantic)
│   ├── ingestion/           # Document parsing and recursive character chunking modules
│   └── services/            # Database connectivity, LLM integrations, and RAG orchestrator
├── app/                     # React dashboard folder (Vite, Javascript, Vanilla CSS)
└── tests/                   # Pytest automated test suites
```

---

## 🏗️ System Architecture

The application is split into two asynchronous lifecycle stages: the **Ingestion Pipeline** (Data Parsing & Vector Storage) and the **Inference Pipeline** (Retrieval & LLM Generation).

### System Topology Diagram

```
                   INGESTION PIPELINE (Offline / Upload)
                   
 [ Raw E-Books ]        +--------------------+
 (PDF, TXT) ----------> | 1. Document Parser | (Extracts & cleans raw text)
                        +--------------------+
                                  |
                                  v
                        +--------------------+
                        | 2. Text Chunker    | (Recursive character split with overlap)
                        +--------------------+
                                  |
                                  v
                        +--------------------+
                        | 3. Embedding Model | (sentence-transformers -> 384 floats)
                        +--------------------+
                                  |
                                  v
                        +--------------------+
                        |  Persistent DB     | (ChromaDB SQLite & HNSW index files)
                        +--------------------+

                   INFERENCE PIPELINE (Online / Search)
                   
  [ User Query ] 
        |
        v
  +--------------------+
  | 1. Embedding Model | (Vectorizes the question using same model)
  +--------------------+
        |
        v
  +--------------------+      Similarity Search      +---------------------+
  |  Query Vector      | --------------------------> | ChromaDB Vector Store|
  +--------------------+      (Cosine Distance)      +---------------------+
                                                                |
                                                                v
  +--------------------+     Injects Context         +---------------------+
  | 2. Prompt Builder  | <-------------------------- |  Top-K Chunks (k=3) |
  |                    |                             |  + Metadata Filter  |
  +--------------------+                             +---------------------+
        |
        v  (Sends compiled prompt)
  +--------------------+
  | 3. LLM Generator   | 
  | (Ollama / OpenAI)  |
  +--------------------+
        |
        v
  [ Synthesized Answer ]
```

### Component Design & Responsibilities

1. **Document Parsers (parser.py)**: Decoupled handlers that ingest raw file paths, detect file extensions, extract characters cleanly, and return a standardized metadata-rich schema.
2. **Chunking Engine (chunker.py)**: Recursively partitions large texts based on paragraph boundaries, newlines, and spaces. Enforces overlap boundaries to avoid losing context.
3. **Vector Service (vector_db.py)**: Directs the interface to ChromaDB. Builds deterministic chunk IDs to prevent duplicate record insertion and executes semantic similarity lookups.
4. **LLM Bindings (llm.py)**: Decouples API invocations for Ollama and OpenAI-compatible local APIs (LM Studio) using uniform settings.
5. **RAG Orchestrator (rag_engine.py)**: Glues retrieval and generation components together. Collects local document contexts, formats security prompts, and executes the synthesis request.

---

## ⚙️ Getting Started

### Prerequisites
* Python 3.10 or higher
* Node.js (v16+)
* Either **LM Studio** (running local server) or **Ollama** installed locally.

### 1. Backend Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   cd Book-RAG
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify your settings in `config.yaml`:
   ```yaml
   persist_directory: "data/vector_store"
   raw_data_directory: "data/raw"
   
   # Embedding configuration
   embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2"
   
   # LLM settings: 'ollama' or 'openai' (LM Studio uses 'openai' with a custom base URL)
   llm_provider: "openai"
   llm_model: "qwen2.5-7b-instruct"  # Update with your local model name
   llm_temperature: 0.0
   openai_base_url: "http://localhost:1234/v1" # Leave blank to use official OpenAI API keys
   ```

5. Run the FastAPI server:
   ```bash
   python src/main.py
   ```
   The backend API will be available at `http://127.0.0.1:8000`.

### 2. Frontend Setup
1. Navigate to the `app` directory:
   ```bash
   cd app
   ```
2. Install Node packages:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your web browser.

### 3. Running Tests
Ensure your virtual environment is active and run the tests:
```bash
pytest
```

---

## 📈 Scope for Improvement & Next Phases

To scale this application into a production-grade system, the following roadmap is recommended:

### 1. Layout-Aware PDF Parsing (Layout AI)
* **Current state**: Standard PDF parsing extracts sequential character lines, which can break on multi-column pages, tables, headers, and footnotes.
* **Improvement**: Integrate advanced layout parsers (e.g., `pdfplumber`, `PyMuPDF` with font-size structure rules, or **LlamaParse**) to convert PDFs into structured Markdown before chunking. 

### 2. Advanced Search & Reranking (Two-Stage Retrieval)
* **Current state**: Retrieves the top-k chunks directly using Cosine similarity.
* **Improvement**:
  * **Hybrid Search**: Combine vector search (for conceptual meaning) with **BM25 keyword search** (for exact words, codes, and names).
  * **Reranking Node**: Query a larger subset (e.g. top 25 chunks) and run them through a cross-encoder model (e.g. `BAAI/bge-reranker-large` or Cohere Rerank) to reorder and retrieve the absolute top 3 most factual context pages.

### 3. Parent-Child Chunking Strategy
* **Current state**: Chunks are uniform sizes (e.g., 500 characters).
* **Improvement**: Store small child chunks (e.g., 150 characters) in the vector database for precision matching, but map them to larger parent paragraphs (e.g., 1000 characters). When a child chunk is matched, feed the larger parent context to the LLM to provide richer context.

### 4. Conversational History & Session Memory
* **Current state**: Queries are stateless; each question is handled in isolation.
* **Improvement**: Implement chat session management storing history in a local SQLite database or Redis, passing a sliding window of historical summary conversations alongside the RAG context to support follow-up questions.

### 5. Asynchronous Ingestion Workers
* **Current state**: Uploading and parsing large books blocks the FastAPI request-response thread.
* **Improvement**: Hand off ingestion tasks to background workers (using Celery, RQ, or FastAPI `BackgroundTasks`) and provide real-time ingestion status indicators (e.g., "Parsing...", "Embedding...", "Indexed") on the React dashboard.
