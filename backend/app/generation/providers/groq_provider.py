import os
from groq import Groq
from backend.app.core.config import settings

_groq_client = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=api_key, timeout=30.0)
    return _groq_client


def call_groq(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
    model = os.getenv("GROQ_MODEL") or settings.GROQ_MODEL or "openai/gpt-oss-20b"
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = (
            os.getenv("GROQ_REASONING_EFFORT") or settings.GROQ_REASONING_EFFORT or "low"
        )

    response = _get_groq_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""