import pymupdf
from typing import List, Dict, Any


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Reads PDF and extracts text and word-level coordinates per page using PyMuPDF."""
    pages_data: List[Dict[str, Any]] = []
    with pymupdf.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            raw_words = page.get_text("words")
            words = [
                {
                    "text": w[4],
                    "x0": w[0],
                    "top": w[1],
                    "x1": w[2],
                    "bottom": w[3],
                }
                for w in raw_words
            ]
            text = page.get_text("text") or ""
            pages_data.append({
                "page_number": page_num,
                "text": text,
                "words": words,
            })
    return pages_data