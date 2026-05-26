import os

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
