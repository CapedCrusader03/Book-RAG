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
