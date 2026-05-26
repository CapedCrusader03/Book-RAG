import os
import tempfile
import pytest
import yaml
from pydantic import ValidationError
from src.config import load_settings, Settings

def test_load_settings_success():
    """Verify that settings are loaded correctly with valid default configuration."""
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 50
    assert settings.persist_directory == "data/vector_store"
    assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3"
    assert settings.llm_temperature == 0.0

def test_load_settings_file_not_found():
    """Verify that a FileNotFoundError is raised when config path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_settings("non_existent_file.yaml")

def test_load_settings_invalid_provider():
    """Verify validation error is raised for incorrect provider type."""
    bad_data = {
        "chunk_size": 256,
        "chunk_overlap": 20,
        "persist_directory": "tmp/store",
        "embedding_model_name": "test-embed",
        "llm_provider": "unsupported_provider",  # Should trigger error
        "llm_model": "gpt-4",
        "llm_temperature": 0.5
    }
    
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(bad_data, f)
        temp_path = f.name
        
    try:
        with pytest.raises(ValidationError):
            load_settings(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_load_settings_invalid_ranges():
    """Verify range checking for chunk size and temperature."""
    bad_data = {
        "chunk_size": -10,  # Should be gt 0
        "chunk_overlap": 10,
        "persist_directory": "tmp/store",
        "embedding_model_name": "test-embed",
        "llm_provider": "ollama",
        "llm_model": "llama3",
        "llm_temperature": 1.5  # Should be <= 1.0
    }
    
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(bad_data, f)
        temp_path = f.name
        
    try:
        with pytest.raises(ValidationError):
            load_settings(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
