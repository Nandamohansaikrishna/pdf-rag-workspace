import sys
from backend.app.ingestion.pipeline import ingest_document
from backend.app.generation.answer import answer_question

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    
    if command == "ingest" and len(sys.argv) == 3:
        ingest_document(sys.argv[2])
    elif command == "ask" and len(sys.argv) == 3:
        result = answer_question(sys.argv[2])
        print("\n=== ANSWER ===")
        print(result["answer"])
        print("\n=== SOURCES ===")
        for s in result["sources"]:
            print(f"- Page {s['page']}")
    else:
        print("Usage:\n  python scripts/ask.py ingest <pdf_path>\n  python scripts/ask.py ask \"<question>\"")
        