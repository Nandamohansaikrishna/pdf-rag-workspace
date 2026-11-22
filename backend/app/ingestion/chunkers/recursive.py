from typing import List, Dict, Any

def chunk_page(page_data: Dict[str, Any], chunk_size: int = 1024, chunk_overlap: int = 256) -> List[Dict[str, Any]]:
    """Splits text and maps word bounding boxes to each chunk."""
    text = page_data["text"]
    words = page_data["words"]
    page_num = page_data["page_number"]
    
    # 1. Simple text chunking
    raw_chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            raw_chunks.append({"text": text[start:], "start": start, "end": len(text)})
            break
        
        # Try to break at a newline or period
        break_point = max(text.rfind('\n', start, end), text.rfind('.', start, end))
        if break_point != -1 and break_point > start + chunk_size // 2:
            end = break_point + 1
        else:
            break_point = text.rfind(' ', start, end) # Fallback to space
            if break_point != -1 and break_point > start:
                end = break_point + 1
                
        raw_chunks.append({"text": text[start:end], "start": start, "end": end})
        start = end - chunk_overlap
        if start <= raw_chunks[-1]["start"]: start = raw_chunks[-1]["end"]

    # 2. Map word boxes to chunks
    final_chunks = []
    for rc in raw_chunks:
        covered_boxes = []
        c_start, c_end = rc["start"], rc["end"]
        current_char = 0
        
        for w in words:
            w_text = w['text']
            idx = text.find(w_text, current_char)
            if idx != -1:
                current_char = idx + len(w_text)
                if idx < c_end and (idx + len(w_text)) > c_start:
                    covered_boxes.append([w['x0'], w['top'], w['x1'], w['bottom']])
        
        final_chunks.append({
            "page": page_num,
            "text": rc["text"].strip(),
            "char_start": c_start,
            "char_end": c_end,
            "boxes": covered_boxes
        })
    return final_chunks