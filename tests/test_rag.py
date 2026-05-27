import pytest
from unittest.mock import patch, MagicMock
from src.services.rag_engine import assemble_prompt, execute_rag_query
from src.services.llm import generate_answer

def test_assemble_prompt():
    """Verify prompt formatting contains boundaries, contexts, and query."""
    query = "What is the capital of Mars?"
    contexts = ["Mars has no capital.", "Olympus Mons is on Mars."]
    
    prompt = assemble_prompt(query, contexts)
    
    assert "[START CONTEXT]" in prompt
    assert "[END CONTEXT]" in prompt
    assert "Mars has no capital." in prompt
    assert "Olympus Mons is on Mars." in prompt
    assert f"Question: {query}" in prompt
    assert "Answer:" in prompt

@patch("src.services.llm.load_settings")
@patch("httpx.post")
def test_generate_answer_ollama(mock_post, mock_load_settings):
    """Verify Ollama routing works and processes json response correctly."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "ollama"
    mock_settings.llm_model = "llama3"
    mock_settings.llm_temperature = 0.0
    mock_load_settings.return_value = mock_settings
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Green"}
    mock_post.return_value = mock_response
    
    ans = generate_answer("Respond with exactly Green")
    assert ans == "Green"
    mock_post.assert_called_once()

@patch("src.services.llm.load_settings")
@patch("src.services.llm.OpenAI")
def test_generate_answer_openai(mock_openai_class, mock_load_settings):
    """Verify OpenAI routing works and processes chat completions correctly."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "openai"
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_temperature = 0.0
    mock_load_settings.return_value = mock_settings
    
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Green"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client
    
    ans = generate_answer("Respond with exactly Green")
    assert ans == "Green"
    mock_client.chat.completions.create.assert_called_once()

@patch("src.services.vector_db.similarity_search")
@patch("src.services.llm.generate_answer")
def test_execute_rag_query_success(mock_generate_answer, mock_similarity_search):
    """Verify that execute_rag_query coordinates retrieval and generation loops."""
    mock_similarity_search.return_value = [
        {"text": "Apples are red.", "metadata": {"book_title": "fruit_facts"}}
    ]
    mock_generate_answer.return_value = "Apples are typically red in color."
    
    answer, sources = execute_rag_query("What color are apples?")
    
    assert answer == "Apples are typically red in color."
    assert len(sources) == 1
    assert sources[0]["metadata"]["book_title"] == "fruit_facts"
    mock_similarity_search.assert_called_once_with("What color are apples?", k=3, metadata_filter=None)
    mock_generate_answer.assert_called_once()
