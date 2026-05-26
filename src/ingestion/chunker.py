def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Splits continuous text sequences into smaller overlapping chunks.
    
    Uses a recursive strategy splitting by paragraphs (\\n\\n), newlines (\\n),
    spaces ( ), and finally characters if necessary, while enforcing chunk_size
    and chunk_overlap limits.
    
    Args:
        text: The raw input text string.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Overlap character length between consecutive chunks.
        
    Returns:
        A list of dicts, where each dict has:
        - "text": The chunked text slice.
        - "metadata": A dict containing:
          - "chunk_index": Sequential index of the chunk (0-indexed).
          - "start_char": The start character index in the original text.
          - "end_char": The end character index in the original text.
          
    Raises:
        ValueError: If configuration values are invalid (chunk_size <= 0,
                     chunk_overlap < 0, or chunk_overlap >= chunk_size).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")
        
    if not text:
        return []
        
    separators = ["\n\n", "\n", " ", ""]
    
    def split_recursive(txt: str, offset: int, seps: list[str]) -> list[tuple[str, int]]:
        if len(txt) <= chunk_size:
            return [(txt, offset)]
            
        if not seps:
            # Character-level fallback split size
            split_size = chunk_overlap if chunk_overlap > 0 else chunk_size
            return [(txt[i:i+split_size], offset + i) for i in range(0, len(txt), split_size)]
            
        sep = seps[0]
        if sep == "":
            split_size = chunk_overlap if chunk_overlap > 0 else chunk_size
            return [(txt[i:i+split_size], offset + i) for i in range(0, len(txt), split_size)]
            
        parts = txt.split(sep)
        result = []
        current_offset = offset
        
        for i, part in enumerate(parts):
            # Re-inject separator except for the last part to preserve layout/spacing
            part_str = part + sep if i < len(parts) - 1 else part
            if not part_str:
                continue
            if len(part_str) <= chunk_size:
                result.append((part_str, current_offset))
            else:
                result.extend(split_recursive(part_str, current_offset, seps[1:]))
            current_offset += len(part_str)
            
        return result

    # 1. Split the text into base semantic segments
    segments = split_recursive(text, 0, separators)
    
    # 2. Recombine segments into overlapping chunks
    chunks = []
    current_segs = []
    current_len = 0
    chunk_index = 0
    
    for seg, off in segments:
        # If adding this segment would exceed chunk_size, save current chunk and shift
        if current_len + len(seg) > chunk_size and current_segs:
            chunk_text_str = "".join([s[0] for s in current_segs])
            chunks.append({
                "text": chunk_text_str,
                "metadata": {
                    "chunk_index": chunk_index,
                    "start_char": current_segs[0][1],
                    "end_char": current_segs[-1][1] + len(current_segs[-1][0])
                }
            })
            chunk_index += 1
            
            # Pop segments until:
            # 1. Total length is <= chunk_overlap
            # 2. Total length + new segment length <= chunk_size
            while current_segs and (current_len > chunk_overlap or current_len + len(seg) > chunk_size):
                popped_seg, _ = current_segs.pop(0)
                current_len -= len(popped_seg)
                
        current_segs.append((seg, off))
        current_len += len(seg)
        
    # Append the final remaining chunk
    if current_segs:
        chunk_text_str = "".join([s[0] for s in current_segs])
        chunks.append({
            "text": chunk_text_str,
            "metadata": {
                "chunk_index": chunk_index,
                "start_char": current_segs[0][1],
                "end_char": current_segs[-1][1] + len(current_segs[-1][0])
            }
        })
        
    return chunks
