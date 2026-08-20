from typing import Any
from langchain_core.embeddings import Embeddings
from app.config import EmbeddingProvider, settings
from app.core.logging import logger


class ChromaDefaultEmbeddings(Embeddings):
    """
    Local embedding wrapper utilizing ChromaDB's built-in ONNX all-MiniLM-L6-v2 embedding function.
    Provides zero-cost, zero-network, high-speed local embeddings.
    """
    def __init__(self):
        import chromadb.utils.embedding_functions as ef
        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._ef(texts)
        return [list(map(float, emb)) for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._ef([text])
        return [float(x) for x in embeddings[0]]


def get_embedding_model() -> Embeddings:
    """
    Factory function returning the configured embedding model.
    Falls back gracefully to local ChromaDefaultEmbeddings if cloud keys are not configured.
    """
    provider = settings.EMBEDDING_PROVIDER
    model_name = settings.get_resolved_embedding_model()

    if provider == EmbeddingProvider.GOOGLE and settings.GOOGLE_API_KEY:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            logger.info(f"Initialized Google Gemini Embeddings ({model_name})")
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=settings.GOOGLE_API_KEY
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Google embeddings: {e}. Falling back to local default embeddings.")

    elif provider == EmbeddingProvider.OPENAI and settings.OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info(f"Initialized OpenAI Embeddings ({model_name})")
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=settings.OPENAI_API_KEY
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}. Falling back to local default embeddings.")

    logger.info("Using zero-cost, high-speed local DefaultEmbeddingFunction (all-MiniLM-L6-v2)")
    return ChromaDefaultEmbeddings()
