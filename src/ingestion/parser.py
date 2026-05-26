import os
from pypdf import PdfReader

def parse_txt(file_path: str) -> dict:
    """Parses a plain text file, returning standard dictionary schema.
    
    Args:
        file_path: Absolute or relative path to the text file.
        
    Returns:
        A dict containing:
        - "text": The extracted and stripped text string.
        - "metadata": A dict with "source", "book_title", and "file_type".
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Read file using UTF-8 with character replacement fallback for robustness
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
        
    # Normalize whitespaces: clean trailing whitespace of document
    text = text.strip()
    
    filename = os.path.basename(file_path)
    book_title, _ = os.path.splitext(filename)
    
    return {
        "text": text,
        "metadata": {
            "source": file_path,
            "book_title": book_title,
            "file_type": "txt"
        }
    }

def parse_pdf(file_path: str) -> dict:
    """Parses a Portable Document Format (PDF) file, returning standard dictionary schema.
    
    Args:
        file_path: Absolute or relative path to the PDF file.
        
    Returns:
        A dict containing:
        - "text": The extracted and concatenated text string.
        - "metadata": A dict with "source", "book_title", and "file_type".
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    reader = PdfReader(file_path)
    text_parts = []
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
        else:
            # Loud logging/printing warning for pages with no text (e.g., scanned image pages)
            print(f"Warning: Page {i + 1} of PDF '{file_path}' yielded no extractable text.")
            
    full_text = "\n".join(text_parts).strip()
    
    filename = os.path.basename(file_path)
    book_title, _ = os.path.splitext(filename)
    
    return {
        "text": full_text,
        "metadata": {
            "source": file_path,
            "book_title": book_title,
            "file_type": "pdf"
        }
    }
