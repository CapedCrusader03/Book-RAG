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

_client = None

def initialize_vector_store(persist_directory: str):
    """Initializes the local ChromaDB persistent client pointing to the specified directory.
    
    Args:
        persist_directory: Path to the persistent database storage on disk.
        
    Returns:
        The initialized chromadb.PersistentClient instance.
    """
    import chromadb
    global _client
    os.makedirs(persist_directory, exist_ok=True)
    _client = chromadb.PersistentClient(path=persist_directory)
    return _client

def get_vector_store() -> "chromadb.PersistentClient":
    """Returns the initialized vector store client, loading settings from config if not initialized."""
    global _client
    if _client is None:
        from src.config import load_settings
        settings = load_settings()
        initialize_vector_store(settings.persist_directory)
    return _client

def add_documents(chunks: list[dict]) -> None:
    """Ingests a list of document chunks, generates embeddings, and persists them to ChromaDB.
    
    This operation is designed to be idempotent: chunk IDs are generated deterministically
    based on the book title and chunk index, allowing safe re-runs without duplication.
    
    Args:
        chunks: A list of dictionaries, each containing:
                - "text": The content string of the chunk.
                - "metadata": A dictionary containing chunk metadata attributes
                              (e.g., source, book_title, chunk_index, start_char, end_char).
    """
    if not chunks:
        return
        
    client = get_vector_store()
    collection = client.get_or_create_collection(name="book_chunks")
    
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        
        # Build deterministic ID for idempotency: book_title + chunk_index
        book_title = metadata.get("book_title", "unknown")
        chunk_idx = metadata.get("chunk_index", i)
        chunk_id = f"{book_title}_chunk_{chunk_idx}"
        
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata)
        embeddings.append(get_embedding(text))
        
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
