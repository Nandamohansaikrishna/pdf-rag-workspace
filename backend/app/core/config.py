from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_PROVIDER: str = "ollama"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:latest"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_REASONING_EFFORT: str = "low"
    API_SECRET_KEY: str = ""
    REFUSAL_THRESHOLD: float = 0.0

    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = ""

    QDRANT_PATH: str = "./qdrant_data"
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Cloud optimization settings
    HF_TOKEN: str = ""
    EMBED_MODE: str = "auto"  # "auto" (HF API if HF_TOKEN set, else local), "hf_api", or "local"
    DISABLE_RERANKER: bool = False


settings = Settings()