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
