from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OPENAI = "openai"


class EmbeddingProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    CHROMA_DEFAULT = "chroma_default"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Project metadata
    PROJECT_NAME: str = "RAG Technical Documentation Assistant"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # LLM Settings
    LLM_PROVIDER: LLMProvider = LLMProvider.GEMINI
    LLM_MODEL_NAME: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None

    # Embeddings
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.CHROMA_DEFAULT
    EMBEDDING_MODEL_NAME: Optional[str] = None

    # Vector Database
    CHROMA_PERSIST_DIRECTORY: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "technical_documentation"

    # Storage Paths
    FEEDBACK_DB_PATH: str = "data/feedback.db"
    CORPUS_DIRECTORY: str = "data/corpus"

    # RAG Tuning Parameters
    RETRIEVAL_TOP_K: int = 4
    MAX_RETRIES: int = 2
    CONFIDENCE_THRESHOLD: float = 0.6
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 120

    # Feature Toggles
    ENABLE_HALLUCINATION_CHECK: bool = True
    ENABLE_WEB_SEARCH_FALLBACK: bool = True

    # Server & Logging
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    def get_resolved_llm_model(self) -> str:
        """Resolve model name dynamically if left blank."""
        if self.LLM_MODEL_NAME:
            return self.LLM_MODEL_NAME
        match self.LLM_PROVIDER:
            case LLMProvider.GEMINI:
                return "gemini-1.5-flash"
            case LLMProvider.GROQ:
                return "llama-3.3-70b-versatile"
            case LLMProvider.OPENAI:
                return "gpt-4o-mini"
            case _:
                return "gemini-1.5-flash"

    def get_resolved_embedding_model(self) -> str:
        """Resolve embedding model dynamically based on provider."""
        if self.EMBEDDING_MODEL_NAME:
            return self.EMBEDDING_MODEL_NAME
        match self.EMBEDDING_PROVIDER:
            case EmbeddingProvider.GOOGLE:
                return "models/text-embedding-004"
            case EmbeddingProvider.OPENAI:
                return "text-embedding-3-small"
            case EmbeddingProvider.HUGGINGFACE:
                return "all-MiniLM-L6-v2"
            case _:
                return "default"


settings = Settings()
