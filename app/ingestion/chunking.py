import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import settings
from app.core.logging import logger


class TechnicalDocumentChunker:
    """
    Intelligent two-stage chunking strategy designed specifically for technical documentation:
    1. Header-Aware Splitting: Preserves hierarchical outline (H1, H2, H3) and attaches section paths.
    2. Recursive Splitting: Splits large subsections while respecting code blocks (```), paragraphs, and signatures.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Headers to preserve as structural boundaries
        self.headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )

        # Code-aware recursive splitter with custom separators
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n```\n",     # Preserve complete code blocks
                "\n```",
                "\n\n",        # Paragraph breaks
                "\n",          # Line breaks
                " ",           # Word boundaries
                "",
            ],
            keep_separator=True,
        )

    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        is_markdown: bool = True
    ) -> List[Document]:
        """
        Transforms raw text into a list of enriched LangChain Documents with unique IDs and section metadata.
        """
        if not text.strip():
            return []

        intermediate_docs: List[Document] = []
        if is_markdown:
            try:
                intermediate_docs = self.markdown_splitter.split_text(text)
            except Exception as e:
                logger.warning(f"Markdown header splitting failed: {e}. Falling back to recursive splitting.")
                intermediate_docs = [Document(page_content=text, metadata={})]
        else:
            intermediate_docs = [Document(page_content=text, metadata={})]

        final_chunks: List[Document] = []
        chunk_index = 0
        timestamp = datetime.now(timezone.utc).isoformat()
        source_name = metadata.get("source", "unknown")

        for intermediate_doc in intermediate_docs:
            sub_chunks = self.recursive_splitter.split_documents([intermediate_doc])
            
            for chunk in sub_chunks:
                # Merge original metadata with header metadata from intermediate doc
                merged_meta = dict(metadata)
                merged_meta.update(chunk.metadata)

                # Extract human-readable section title
                header_parts = [
                    merged_meta.get(h_name)
                    for _, h_name in self.headers_to_split_on
                    if merged_meta.get(h_name)
                ]
                section_title = " > ".join(header_parts) if header_parts else "General"
                
                # Deterministic chunk ID
                content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:10]
                chunk_id = f"{source_name}_chunk_{chunk_index}_{content_hash}"

                merged_meta.update({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "section_title": section_title,
                    "char_length": len(chunk.page_content),
                    "ingested_at": timestamp,
                    "source": source_name,
                })

                final_chunks.append(
                    Document(page_content=chunk.page_content.strip(), metadata=merged_meta)
                )
                chunk_index += 1

        logger.info(f"Chunked '{source_name}' ({len(text)} chars) into {len(final_chunks)} chunks.")
        return final_chunks
