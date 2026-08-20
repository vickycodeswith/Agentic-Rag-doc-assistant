import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.core.logging import logger
from app.retrieval.embeddings import get_embedding_model


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage, indexing, and similarity retrieval."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[Embeddings] = None
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_model = embedding_model or get_embedding_model()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaDB vector store at '{self.persist_directory}' [Collection: {self.collection_name}]")

    def add_documents(self, documents: List[Document]) -> int:
        """Embeds and inserts a batch of LangChain Document objects into ChromaDB."""
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [
            doc.metadata.get("chunk_id") or f"chunk_{abs(hash(doc.page_content))}_{idx}"
            for idx, doc in enumerate(documents)
        ]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.embed_documents(texts)

        # Batch upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"Successfully indexed {len(documents)} chunks in collection '{self.collection_name}'.")
        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Searches the collection for nearest neighbors and converts cosine distance to a similarity score (0.0 to 1.0).
        """
        top_k = k or settings.RETRIEVAL_TOP_K
        query_embedding = self.embedding_model.embed_query(query)

        count = self.collection.count()
        if count == 0:
            logger.warning("Vector store collection is empty. Returning 0 results.")
            return []

        search_k = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_k,
            where=filter_dict if filter_dict else None,
            include=["documents", "metadatas", "distances"]
        )

        matched_docs: List[Tuple[Document, float]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return matched_docs

        for doc_text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity. (ranges [0, 2])
            # Higher similarity score is better (1.0 = exact match).
            similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
            doc = Document(page_content=doc_text, metadata=meta or {})
            matched_docs.append((doc, similarity_score))

        return matched_docs

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Aggregates and returns summary statistics of all indexed documents in the collection.
        Used by the GET /documents endpoint.
        """
        count = self.collection.count()
        if count == 0:
            return []

        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])

        # Group by source
        grouped: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            src = meta.get("source", "Unknown Source")
            if src not in grouped:
                grouped[src] = {
                    "source": src,
                    "title": meta.get("doc_title") or meta.get("title") or src,
                    "source_type": meta.get("source_type", "file"),
                    "total_chunks": 0,
                    "sections": set(),
                    "last_ingested": meta.get("ingested_at")
                }
            grouped[src]["total_chunks"] += 1
            if "section_title" in meta and meta["section_title"]:
                grouped[src]["sections"].add(meta["section_title"])

        # Convert set to list
        summary = []
        for item in grouped.values():
            item["sections"] = list(item["sections"])[:10]  # sample top 10 sections
            summary.append(item)

        return summary

    def delete_by_source(self, source: str) -> int:
        """Deletes all chunks associated with a specific document source."""
        existing = self.collection.get(where={"source": source})
        ids_to_del = existing.get("ids", [])
        if ids_to_del:
            self.collection.delete(ids=ids_to_del)
            logger.info(f"Deleted {len(ids_to_del)} chunks for source: {source}")
            return len(ids_to_del)
        return 0

    def get_total_chunk_count(self) -> int:
        """Returns total number of chunks currently indexed in ChromaDB."""
        return self.collection.count()


# Singleton instance
vector_store_manager = VectorStoreManager()
