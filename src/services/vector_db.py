import os

_model = None

def _get_model():
    """Lazy loads and caches the SentenceTransformer model instance."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from src.config import load_settings
        try:
            settings = load_settings()
            model_name = settings.embedding_model_name
        except Exception as e:
            # Fallback to standard model in case config is missing or invalid
            print(f"Warning: Failed to load config, falling back to default embedding model: {e}")
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        _model = SentenceTransformer(model_name)
    return _model

def get_embedding(text: str) -> list[float]:
    """Generates a dense vector embedding for the input text.
    
    Args:
        text: The input text to vectorize.
        
    Returns:
        A list of floats representing the embedding vector.
    """
    if not text:
        # Prevent calling encoder on empty string
        return []
        
    model = _get_model()
    # encode returns a numpy array, convert it to standard python list of floats
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
