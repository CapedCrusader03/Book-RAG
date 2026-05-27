def assemble_prompt(query: str, context_chunks: list[str]) -> str:
    """Concatenates the context chunks and query into a strict system-instructed prompt.
    
    Args:
        query: The user's input question.
        context_chunks: A list of relevant text fragments to use as context.
        
    Returns:
        The formatted prompt string.
    """
    context_text = "\n\n".join([f"--- Context Segment ---\n{chunk}" for chunk in context_chunks])
    
    prompt = f"""You are a helpful assistant specialized in answering questions about books. 
Using the provided context passages below, answer the user's question.

CRITICAL INSTRUCTIONS:
1. Base your answer ONLY on the provided context passages. Do not assume or extrapolate.
2. If the context does not contain the answer, you must state exactly: "I do not know." or "The context does not contain this information."
3. Do not mention the context or passages in your final answer unless helpful.

[START CONTEXT]
{context_text}
[END CONTEXT]

Question: {query}
Answer:"""
    return prompt

def execute_rag_query(query: str, book_title: str = None) -> tuple[str, list[dict]]:
    """Orchestrates the complete RAG pipeline.
    
    Retrieves matching text segments from the vector database, formats the system
    prompt, calls the LLM, and returns the synthesized answer along with the source chunks.
    
    Args:
        query: The user's input question.
        book_title: Optional filter to restrict retrieval to a specific book.
        
    Returns:
        A tuple of (answer_text, retrieved_chunks_list).
    """
    from src.services.vector_db import similarity_search
    from src.services.llm import generate_answer
    
    metadata_filter = None
    if book_title:
        metadata_filter = {"book_title": book_title}
        
    # Query database for top 3 matching chunks
    retrieved_chunks = similarity_search(query, k=3, metadata_filter=metadata_filter)
    
    if not retrieved_chunks:
        return "I do not know.", []
        
    # Extract text from chunks and assemble prompt
    context_texts = [chunk["text"] for chunk in retrieved_chunks]
    prompt = assemble_prompt(query, context_texts)
    
    # Generate completion from configured LLM provider
    try:
        answer = generate_answer(prompt)
        return answer, retrieved_chunks
    except Exception as e:
        print(f"Error executing RAG query generation: {e}")
        raise e
