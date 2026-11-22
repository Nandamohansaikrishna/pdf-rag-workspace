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
    # 1. Two-stage retrieval (Qdrant + Cross-Encoder Reranker)
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

    # 3. Smart Routing: UI override takes priority, otherwise fall back to .env setting
    provider_choice = (provider_override or settings.LLM_PROVIDER).lower().strip()
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")

    answer = None
    model_used = None

    if provider_choice == "groq":
        if settings.GROQ_API_KEY:
            print(f"Routing request to Groq Cloud ({groq_model})...")
            answer = generate_groq_answer(
                prompt=query, system_message=formatted_system_prompt
            )
            if answer:
                model_used = groq_model
        else:
            print("No GROQ_API_KEY found in config. Defaulting to local Ollama...")

    # Automatic fallback to local Ollama if Groq wasn't chosen, failed, or lacked an API key
    if not answer:
        print(f"Routing request to Local Ollama ({ollama_model})...")
        try:
            provider = OllamaProvider()
            answer = provider.generate(prompt=query, system=formatted_system_prompt)
            if answer:
                suffix = " (fallback)" if provider_choice == "groq" else ""
                model_used = f"{ollama_model}{suffix}"
        except Exception as exc:
            print(f"Local Ollama Error: {exc}")
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