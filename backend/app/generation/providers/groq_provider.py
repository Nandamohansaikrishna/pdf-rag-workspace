import os
from groq import Groq
from backend.app.core.config import settings


def generate_groq_answer(
    prompt: str,
    system_message: str = "You are a helpful AI assistant.",
    model: str | None = None,
) -> str | None:
    """Generates an answer using the Groq API for cloud deployment."""
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Groq API Error: GROQ_API_KEY is not set.")
        return None

    selected_model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    client = Groq(api_key=api_key, timeout=30.0)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            model=selected_model,
            temperature=0.0,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API Error: {e}")
        return None