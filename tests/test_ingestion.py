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

def test_parse_pdf_non_existent():
    """Verify that FileNotFoundError is raised for non-existent PDF files."""
    with pytest.raises(FileNotFoundError):
        from src.ingestion.parser import parse_pdf
        parse_pdf("non_existent_file_path.pdf")

def test_parse_pdf_success():
    """Verify that parse_pdf extracts text content and metadata from mock pages."""
    from unittest.mock import MagicMock, patch
    from src.ingestion.parser import parse_pdf
    
    with patch("src.ingestion.parser.PdfReader") as mock_pdf_reader, \
         patch("os.path.exists") as mock_exists:
         
        mock_exists.return_value = True
        
        # Mock the page extraction
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "This is page 1 content."
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "This is page 2 content."
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page_1, mock_page_2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        result = parse_pdf("dummy_book.pdf")
        
        assert isinstance(result, dict)
        assert result["text"] == "This is page 1 content.\nThis is page 2 content."
        assert result["metadata"]["source"] == "dummy_book.pdf"
        assert result["metadata"]["book_title"] == "dummy_book"
        assert result["metadata"]["file_type"] == "pdf"
        
        mock_pdf_reader.assert_called_once_with("dummy_book.pdf")

def test_parse_pdf_empty_page_warning(capsys):
    """Verify that parse_pdf warns loudly when pages yield no text."""
    from unittest.mock import MagicMock, patch
    from src.ingestion.parser import parse_pdf
    
    with patch("src.ingestion.parser.PdfReader") as mock_pdf_reader, \
         patch("os.path.exists") as mock_exists:
         
        mock_exists.return_value = True
        
        # Mock the page extraction - page 1 has text, page 2 has no text
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "Page one content."
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = None  # Scanned or empty page
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page_1, mock_page_2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        result = parse_pdf("empty_pages.pdf")
        
        assert result["text"] == "Page one content."
        assert result["metadata"]["book_title"] == "empty_pages"
        
        # Verify loud logging/warning print
        captured = capsys.readouterr()
        assert "Warning: Page 2 of PDF 'empty_pages.pdf' yielded no extractable text." in captured.out

