import argparse
import os
import sys
from typing import Optional

# Add project root to sys.path to allow direct execution of this script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import core modules
from src.config import load_settings
from src.ingestion.parser import parse_txt, parse_pdf
from src.ingestion.chunker import chunk_text
from src.services.vector_db import add_documents, get_vector_store
from src.services.rag_engine import execute_rag_query

app = FastAPI(title="Book RAG API Server", version="1.0.0")

# Enable CORS for React frontend (default local Vite runs on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact React origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    book_title: Optional[str] = None

class IngestResponse(BaseModel):
    status: str
    processed_books: list[str]
    total_chunks: int

# Ingestion helper
def run_batch_ingestion() -> tuple[list[str], int]:
    """Scans data/raw/ and processes plain text and PDF books into the database.
    
    Returns:
        A tuple containing (list of processed book names, total chunks ingested).
    """
    settings = load_settings()
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        return [], 0
        
    processed_books = []
    total_chunks = 0
    
    files = os.listdir(raw_dir)
    for file in files:
        file_path = os.path.join(raw_dir, file)
        if os.path.isdir(file_path) or file == ".gitkeep":
            continue
            
        _, ext = os.path.splitext(file.lower())
        try:
            if ext == ".txt":
                doc = parse_txt(file_path)
            elif ext == ".pdf":
                doc = parse_pdf(file_path)
            else:
                print(f"Skipping unsupported file format: {file}")
                continue
                
            # Chunk parsed text
            chunks = chunk_text(doc["text"], settings.chunk_size, settings.chunk_overlap)
            
            # Combine chunk metadata with document metadata
            final_chunks = []
            for chunk in chunks:
                combined_metadata = {**doc["metadata"], **chunk["metadata"]}
                final_chunks.append({
                    "text": chunk["text"],
                    "metadata": combined_metadata
                })
                
            # Persist to database
            add_documents(final_chunks)
            processed_books.append(doc["metadata"]["book_title"])
            total_chunks += len(final_chunks)
            print(f"Successfully ingested book: {doc['metadata']['book_title']} ({len(final_chunks)} chunks)")
            
        except Exception as e:
            print(f"Error processing book '{file}': {e}")
            
    return processed_books, total_chunks

# API Endpoints
@app.post("/api/query")
def api_query(payload: QueryRequest):
    """Answers a user's question, scoped optionally to a specific book."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        answer, sources = execute_rag_query(payload.query, payload.book_title)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest", response_model=IngestResponse)
def api_ingest():
    """Triggers batch processing of all raw books in data/raw/."""
    try:
        processed_books, total_chunks = run_batch_ingestion()
        return IngestResponse(
            status="success",
            processed_books=processed_books,
            total_chunks=total_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/books")
def api_get_books():
    """Retrieves list of unique book titles currently present in the database index."""
    try:
        client = get_vector_store()
        collection = client.get_or_create_collection(name="book_chunks")
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        # Collect distinct titles
        titles = set()
        for meta in metadatas:
            if meta and "book_title" in meta:
                titles.add(meta["book_title"])
        return {"books": sorted(list(titles))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Book RAG System Entrypoint")
    parser.add_argument("--ingest", action="store_true", help="Run offline batch book ingestion and exit")
    args = parser.parse_args()
    
    if args.ingest:
        print("Starting batch book ingestion...")
        books, chunks = run_batch_ingestion()
        print(f"Batch ingestion complete. Processed {len(books)} books, generated {chunks} chunks.")
        sys.exit(0)
    else:
        # Run FastAPI Server
        print("Starting RAG API Server...")
        uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
