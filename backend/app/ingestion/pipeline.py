from backend.app.ingestion.parser import parse_pdf
from backend.app.ingestion.chunkers.recursive import chunk_page
from backend.app.retrieval.vector_store import upsert_chunks


def ingest_document(file_path: str, collection_name: str = "documents") -> int:
    pages_data = parse_pdf(file_path)
    all_chunks = []
    for page in pages_data:
        all_chunks.extend(chunk_page(page))

    if not all_chunks:
        raise ValueError("No selectable text found. Run OCR on the PDF and upload it again.")

    upsert_chunks(collection_name, all_chunks)
    print(f"Ingested {len(all_chunks)} chunks from {file_path} into '{collection_name}'")
    return len(all_chunks)


if __name__ == "__main__":
    ingest_document("sample_docs/t5.pdf")