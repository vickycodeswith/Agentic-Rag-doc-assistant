import io
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import httpx
from pypdf import PdfReader

from app.core.exceptions import IngestionError
from app.core.logging import logger


class DocumentLoader:
    """Loads and extracts clean text content and metadata from files, URLs, or raw text."""

    @staticmethod
    def load_file(file_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """Loads text and metadata from a local file (.md, .txt, .pdf)."""
        path = Path(file_path)
        if not path.exists():
            raise IngestionError(f"File not found: {file_path}")

        source_name = path.name
        ext = path.suffix.lower()

        try:
            if ext in [".md", ".markdown"]:
                text = path.read_text(encoding="utf-8")
                return text, {
                    "source": source_name,
                    "source_type": "markdown_file",
                    "file_path": str(path.resolve()),
                    "doc_title": path.stem.replace("_", " ").title()
                }
            elif ext in [".txt", ".rst"]:
                text = path.read_text(encoding="utf-8")
                return text, {
                    "source": source_name,
                    "source_type": "text_file",
                    "file_path": str(path.resolve()),
                    "doc_title": path.stem.replace("_", " ").title()
                }
            elif ext == ".pdf":
                reader = PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(pages)
                return text, {
                    "source": source_name,
                    "source_type": "pdf_file",
                    "file_path": str(path.resolve()),
                    "doc_title": path.stem.replace("_", " ").title(),
                    "total_pages": len(reader.pages)
                }
            else:
                # Default fallback
                text = path.read_text(encoding="utf-8", errors="ignore")
                return text, {
                    "source": source_name,
                    "source_type": "generic_file",
                    "file_path": str(path.resolve()),
                    "doc_title": path.stem.title()
                }
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise IngestionError(f"Failed to read file {source_name}: {str(e)}")

    @staticmethod
    def load_bytes(file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Loads text content from uploaded file bytes (for POST /ingest multipart uploads)."""
        path = Path(filename)
        ext = path.suffix.lower()

        try:
            if ext == ".pdf":
                reader = PdfReader(io.BytesIO(file_bytes))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(pages)
                return text, {
                    "source": filename,
                    "source_type": "pdf_upload",
                    "doc_title": path.stem.replace("_", " ").title(),
                    "total_pages": len(reader.pages)
                }
            else:
                text = file_bytes.decode("utf-8", errors="ignore")
                return text, {
                    "source": filename,
                    "source_type": "file_upload",
                    "doc_title": path.stem.replace("_", " ").title()
                }
        except Exception as e:
            logger.error(f"Error processing uploaded bytes for {filename}: {e}")
            raise IngestionError(f"Failed to process uploaded file {filename}: {str(e)}")

    @staticmethod
    async def load_url(url: str) -> Tuple[str, Dict[str, Any]]:
        """Scrapes web page, strips scripts/navigation, and extracts markdown/clean text."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise IngestionError(f"Invalid URL provided: {url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 RAGBot/1.0"
        }

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove non-content tags
            for element in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg"]):
                element.decompose()

            # Find main article or content container if available
            main_content = soup.find("main") or soup.find("article") or soup.find("body") or soup

            title = soup.title.string.strip() if soup.title and soup.title.string else url

            # Clean text lines
            lines = [line.strip() for line in main_content.get_text(separator="\n").splitlines()]
            clean_text = "\n".join(line for line in lines if line)

            return clean_text, {
                "source": url,
                "source_type": "url",
                "doc_title": title,
                "domain": parsed.netloc
            }
        except Exception as e:
            logger.error(f"Failed to fetch content from URL {url}: {e}")
            raise IngestionError(f"Failed to fetch URL {url}: {str(e)}")
