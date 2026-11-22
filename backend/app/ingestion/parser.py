import pdfplumber
from typing import List, Dict, Any

def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Reads PDF and extracts text and word-level coordinates per page."""
    pages_data = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            text = page.extract_text() or ""
            pages_data.append({
                "page_number": page_num,
                "text": text,
                "words": words  # Contains 'text', 'x0', 'top', 'x1', 'bottom'
            })
    return pages_data