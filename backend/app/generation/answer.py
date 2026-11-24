import os
from backend.app.retrieval.vector_store import search
from backend.app.generation.prompts import SYSTEM_PROMPT
from backend.app.generation.providers.ollama import OllamaProvider
from backend.app.generation.providers.groq_provider import generate_groq_answer
from backend.app.core.config import settings


def answer_question(
    query: str,
    collection_name: str = "documents",
    provider_override: str | None = None,
):
    # 1. Two-stage retrieval (Qdrant + Hybrid BM25 / Cross-Encoder Reranker)
    results = search(collection_name, query, top_k=5)
    if not results:
        return {
            "answer": "No matching passages found in this PDF. Try rephrasing your question.",
            "sources": [],
            "model": None,
            "top_score": 0.0,
        }

    context_parts, sources = [], []
    for idx, res in enumerate(results):
        payload = res["payload"]
        context_parts.append(
            f"--- Passage {idx + 1} (Page {payload['page']}) ---\n{payload['text']}"
        )
        sources.append(
            {
                "page": payload["page"],
                "text": payload["text"],
                "boxes": payload.get("boxes", []),
                "score": float(res["score"]),
                "cross_score": float(res["cross_score"]),
            }
        )

    # 2. Format system prompt with retrieved context
    formatted_system_prompt = SYSTEM_PROMPT.format(context="\n\n".join(context_parts))

    # 3. Smart Routing: UI override -> env var -> .env default
    provider_choice = (
        provider_override or os.getenv("LLM_PROVIDER") or settings.LLM_PROVIDER
    ).lower().strip()
    groq_api_key = (os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY or "").strip()
    groq_model = os.getenv("GROQ_MODEL") or settings.GROQ_MODEL or "openai/gpt-oss-20b"
    ollama_model = os.getenv("OLLAMA_MODEL") or settings.OLLAMA_MODEL or "qwen2.5:latest"
    disable_ollama = os.getenv("DISABLE_OLLAMA", "").lower() in ("1", "true", "yes")

    answer = None
    model_used = None

    if provider_choice == "groq":
        if groq_api_key:
            print(f"Routing request to Groq Cloud ({groq_model})...")
            answer = generate_groq_answer(
                prompt=query,
                system_message=formatted_system_prompt,
                model=groq_model,
            )
            if answer:
                model_used = groq_model
        else:
            print("No GROQ_API_KEY found in config or environment.")

    # Automatic fallback to local Ollama unless disabled in cloud mode
    if not answer and not disable_ollama:
        print(f"Routing request to Local Ollama ({ollama_model})...")
        try:
            provider = OllamaProvider()
            answer = provider.generate(prompt=query, system=formatted_system_prompt)
            if answer:
                suffix = " (fallback)" if provider_choice == "groq" else ""
                model_used = f"{ollama_model}{suffix}"
        except Exception as exc:
            print(f"Local Ollama Error: {exc}")

    if not answer:
        answer = (
            "Could not reach Groq or local Ollama. "
            "Check your API key or ensure Ollama is running. Retrieved passages are shown below."
        )
        model_used = None

    return {
        "answer": answer,
        "sources": sources,
        "model": model_used,
        "top_score": float(results[0]["score"]),
    }