import pytest
from src.services.vector_db import get_embedding

def test_get_embedding_success():
    """Verify that get_embedding generates a vector with correct dimensions."""
    phrase = "This is a test sentence for validating our local embedding model connection."
    embedding = get_embedding(phrase)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Dimension of sentence-transformers/all-MiniLM-L6-v2
    assert all(isinstance(x, float) for x in embedding)

def test_get_embedding_empty():
    """Verify that get_embedding handles empty strings gracefully."""
    embedding = get_embedding("")
    assert embedding == []

def test_initialize_vector_store():
    """Verify that initialize_vector_store correctly initializes persistent storage on disk."""
    import tempfile
    import shutil
    import os
    from src.services.vector_db import initialize_vector_store
    
    # Create temporary directory for DB
    temp_dir = tempfile.mkdtemp()
    
    try:
        client = initialize_vector_store(temp_dir)
        # Create a test collection
        collection = client.create_collection("test_initialization")
        
        # Verify collection works
        assert collection.name == "test_initialization"
        
        # Verify that directory is populated
        assert os.path.exists(temp_dir)
        files = os.listdir(temp_dir)
        assert len(files) > 0
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                # SQLite locks the file on Windows until process exit, ignore
                pass

def test_add_documents():
    """Verify that add_documents correctly embeds and ingests document chunks with idempotency."""
    import tempfile
    import shutil
    import os
    from src.services.vector_db import initialize_vector_store, add_documents
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize temp DB
        client = initialize_vector_store(temp_dir)
        collection = client.get_or_create_collection("book_chunks")
        
        # Initial count
        assert collection.count() == 0
        
        # Ingest 3 distinct mock chunk items
        mock_chunks = [
            {
                "text": "This is chunk number one representing general knowledge.",
                "metadata": {"book_title": "test_book", "chunk_index": 0, "source": "dummy.txt"}
            },
            {
                "text": "This is chunk number two detailing technical architectural specifications.",
                "metadata": {"book_title": "test_book", "chunk_index": 1, "source": "dummy.txt"}
            },
            {
                "text": "This is chunk number three listing testing verification procedures.",
                "metadata": {"book_title": "test_book", "chunk_index": 2, "source": "dummy.txt"}
            }
        ]
        
        add_documents(mock_chunks)
        
        # Verify collection count increased by exactly 3
        assert collection.count() == 3
        
        # Verify idempotency: re-adding the same chunks does not duplicate them
        add_documents(mock_chunks)
        assert collection.count() == 3
        
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass
