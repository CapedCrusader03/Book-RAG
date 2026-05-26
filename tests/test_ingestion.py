import os
import tempfile
import pytest
from src.ingestion.parser import parse_txt

def test_parse_txt_success():
    """Verify that parse_txt extracts text content and correct metadata."""
    sample_content = "Hello, this is a sample book content.\nLine two."
    
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(sample_content)
        temp_path = f.name
        
    try:
        result = parse_txt(temp_path)
        assert isinstance(result, dict)
        assert result["text"] == sample_content
        assert result["metadata"]["source"] == temp_path
        filename = os.path.basename(temp_path)
        expected_title, _ = os.path.splitext(filename)
        assert result["metadata"]["book_title"] == expected_title
        assert result["metadata"]["file_type"] == "txt"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_parse_txt_non_existent():
    """Verify that FileNotFoundError is raised for non-existent files."""
    with pytest.raises(FileNotFoundError):
        parse_txt("non_existent_file_path.txt")

def test_parse_txt_invalid_encoding():
    """Verify that invalid UTF-8 characters are handled gracefully using replacement."""
    # Write bytes that are invalid in UTF-8 (e.g., latin-1 character 0xff)
    invalid_bytes = b"Hello \xff world!"
    
    with tempfile.NamedTemporaryFile("wb+", suffix=".txt", delete=False) as f:
        f.write(invalid_bytes)
        temp_path = f.name
        
    try:
        result = parse_txt(temp_path)
        assert "Hello" in result["text"]
        assert "world!" in result["text"]
        # The invalid byte 0xff should have been replaced with the Unicode replacement character \ufffd
        assert "\ufffd" in result["text"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
