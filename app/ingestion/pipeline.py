from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.ingestion.chunking import TechnicalDocumentChunker
from app.ingestion.loader import DocumentLoader
from app.retrieval.vectorstore import VectorStoreManager, vector_store_manager


class IngestionPipeline:
    """Orchestrates loading, technical chunking, embedding generation, and vector store upsertion."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        chunker: Optional[TechnicalDocumentChunker] = None
    ):
        self.vector_store = vector_store or vector_store_manager
        self.chunker = chunker or TechnicalDocumentChunker()

    def ingest_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Loads and indexes a single file."""
        text, metadata = DocumentLoader.load_file(file_path)
        is_md = Path(file_path).suffix.lower() in [".md", ".markdown"]
        chunks = self.chunker.chunk_document(text, metadata, is_markdown=is_md)
        num_added = self.vector_store.add_documents(chunks)
        return {
            "source": metadata.get("source"),
            "doc_title": metadata.get("doc_title"),
            "chunks_created": num_added,
            "status": "indexed"
        }

    def ingest_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Loads and indexes in-memory uploaded file bytes."""
        text, metadata = DocumentLoader.load_bytes(file_bytes, filename)
        is_md = filename.lower().endswith((".md", ".markdown"))
        chunks = self.chunker.chunk_document(text, metadata, is_markdown=is_md)
        num_added = self.vector_store.add_documents(chunks)
        return {
            "source": metadata.get("source"),
            "doc_title": metadata.get("doc_title"),
            "chunks_created": num_added,
            "status": "indexed"
        }

    async def ingest_url(self, url: str) -> Dict[str, Any]:
        """Fetches, cleans, chunks, and indexes content from a URL."""
        text, metadata = await DocumentLoader.load_url(url)
        chunks = self.chunker.chunk_document(text, metadata, is_markdown=False)
        num_added = self.vector_store.add_documents(chunks)
        return {
            "source": metadata.get("source"),
            "doc_title": metadata.get("doc_title"),
            "chunks_created": num_added,
            "status": "indexed"
        }

    def ingest_text(self, text: str, source_name: str, doc_title: Optional[str] = None) -> Dict[str, Any]:
        """Indexes raw text snippet directly."""
        metadata = {
            "source": source_name,
            "source_type": "raw_text",
            "doc_title": doc_title or source_name
        }
        chunks = self.chunker.chunk_document(text, metadata, is_markdown=True)
        num_added = self.vector_store.add_documents(chunks)
        return {
            "source": source_name,
            "doc_title": doc_title or source_name,
            "chunks_created": num_added,
            "status": "indexed"
        }

    def ingest_directory(self, dir_path: str | Path) -> List[Dict[str, Any]]:
        """Scans a directory and indexes all markdown and text files."""
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            raise IngestionError(f"Directory not found: {dir_path}")

        results = []
        for file in sorted(path.glob("*")):
            if file.is_file() and file.suffix.lower() in [".md", ".markdown", ".txt", ".pdf"]:
                try:
                    res = self.ingest_file(file)
                    results.append(res)
                except Exception as e:
                    logger.error(f"Failed to ingest file {file.name}: {e}")
                    results.append({
                        "source": file.name,
                        "error": str(e),
                        "status": "failed"
                    })
        return results


# Global singleton
ingestion_pipeline = IngestionPipeline()
