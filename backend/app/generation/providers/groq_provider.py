import os
from groq import Groq
from backend.app.core.config import settings

_groq_client: Groq | None = None
_groq_api_key: str | None = None


def _get_groq_client(api_key: str) -> Groq:
    global _groq_client, _groq_api_key
    if _groq_client is None or _groq_api_key != api_key:
        _groq_client = Groq(api_key=api_key, timeout=30.0)
        _groq_api_key = api_key
    return _groq_client


def generate_groq_answer(
    prompt: str,
    system_message: str = "You are a helpful AI assistant.",
    model: str | None = None,
) -> str | None:
    """Generates an answer using the Groq API for cloud deployment."""
    api_key = (os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY or "").strip()
    if not api_key:
        print("Groq API Error: GROQ_API_KEY is not set.")
        return None

    selected_model = (
        model
        or os.getenv("GROQ_MODEL")
        or settings.GROQ_MODEL
        or "openai/gpt-oss-20b"
    )
    client = _get_groq_client(api_key)

    kwargs = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "model": selected_model,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    if "gpt-oss" in selected_model:
        kwargs["reasoning_effort"] = (
            os.getenv("GROQ_REASONING_EFFORT")
            or settings.GROQ_REASONING_EFFORT
            or "low"
        )

    try:
        try:
            chat_completion = client.chat.completions.create(**kwargs)
        except TypeError:
            # Fallback if an older local groq SDK doesn't support reasoning_effort
            kwargs.pop("reasoning_effort", None)
            chat_completion = client.chat.completions.create(**kwargs)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API Error: {e}")
        return None