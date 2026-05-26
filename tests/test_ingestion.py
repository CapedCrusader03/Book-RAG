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

def test_chunk_text_validation_errors():
    """Verify ValueError is raised for invalid chunk parameter inputs."""
    from src.ingestion.chunker import chunk_text
    
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        chunk_text("dummy", chunk_size=0, chunk_overlap=10)
        
    with pytest.raises(ValueError, match="chunk_overlap must be 0 or greater"):
        chunk_text("dummy", chunk_size=10, chunk_overlap=-1)
        
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        chunk_text("dummy", chunk_size=100, chunk_overlap=100)
        
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        chunk_text("dummy", chunk_size=100, chunk_overlap=120)

def test_chunk_text_success_delimiters():
    """Verify basic text segmentation using paragraph, newline, and space delimiters."""
    from src.ingestion.chunker import chunk_text
    
    text = "Paragraph 1 line A.\nParagraph 1 line B.\n\nParagraph 2 content."
    # With chunk_size=30, chunk_overlap=10:
    # "Paragraph 1 line A.\nParagraph 1 line B." is 40 chars -> exceeds 30.
    # Split by \n: "Paragraph 1 line A.\n" (20 chars) and "Paragraph 1 line B." (19 chars).
    # "Paragraph 1 line A.\n" and "Paragraph 1 line B." cannot be merged into one chunk because length would be 39 > 30.
    # So chunk 1 = "Paragraph 1 line A.\n" (len 20)
    # Since chunk_overlap=10, we try to overlap. But individual segments are 20 and 19. No single segment is <= 10.
    # So overlap is 0. Next chunk starts at "Paragraph 1 line B.\n\n".
    # Let's test chunk_text with simple parameters to verify standard splits.
    chunks = chunk_text(text, chunk_size=45, chunk_overlap=15)
    assert len(chunks) > 0
    # Reconstruct the original text (minus paragraph boundaries maybe, but let's assert no character loss)
    # Check offsets
    for chunk in chunks:
        chunk_txt = chunk["text"]
        start = chunk["metadata"]["start_char"]
        end = chunk["metadata"]["end_char"]
        assert text[start:end] == chunk_txt
        assert len(chunk_txt) <= 45

def test_chunk_text_sliding_window_characters():
    """Verify that adjacent chunks share exactly the expected overlap when splitting by characters."""
    from src.ingestion.chunker import chunk_text
    
    # 2,000 character test block of 'A' (no delimiters match, forces character splitting)
    text = "A" * 2000
    chunk_size = 500
    chunk_overlap = 100
    
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Verify:
    # 1. We have exactly 5 chunks:
    #    Chunk 0: 0 to 500
    #    Chunk 1: 400 to 900
    #    Chunk 2: 800 to 1300
    #    Chunk 3: 1200 to 1700
    #    Chunk 4: 1600 to 2000
    assert len(chunks) == 5
    
    for i, chunk in enumerate(chunks):
        assert len(chunk["text"]) <= chunk_size
        assert chunk["metadata"]["chunk_index"] == i
        
        expected_start = i * (chunk_size - chunk_overlap)
        expected_end = expected_start + chunk_size if i < 4 else 2000
        
        assert chunk["metadata"]["start_char"] == expected_start
        assert chunk["metadata"]["end_char"] == expected_end
        assert chunk["text"] == text[expected_start:expected_end]
        
    # Verify exact overlap at boundaries
    for i in range(len(chunks) - 1):
        curr_chunk_end = chunks[i]["metadata"]["end_char"]
        next_chunk_start = chunks[i+1]["metadata"]["start_char"]
        
        # Intersection between Chunk i and Chunk i+1
        # Chunk i ends at curr_chunk_end (500)
        # Chunk i+1 starts at next_chunk_start (400)
        # Overlap = curr_chunk_end - next_chunk_start
        overlap_len = curr_chunk_end - next_chunk_start
        assert overlap_len == chunk_overlap


