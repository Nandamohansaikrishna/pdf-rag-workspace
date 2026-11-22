import httpx
from backend.app.core.config import settings
from backend.app.generation.providers.base import LLMProvider

class OllamaProvider(LLMProvider):
    def generate(self, prompt: str, system: str = "") -> str:
        url = f"{settings.OLLAMA_HOST}/api/generate"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False
        }
        response = httpx.post(url, json=payload, timeout=None)
        response.raise_for_status()
        return response.json()["response"]
    