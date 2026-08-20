import argparse
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.ingestion.pipeline import ingestion_pipeline
from app.retrieval.vectorstore import vector_store_manager

console = Console()


def run_ingestion(corpus_dir: str = settings.CORPUS_DIRECTORY):
    console.print(f"[bold cyan]🔍 Starting Document Ingestion Pipeline[/bold cyan]")
    console.print(f"Target Directory: [yellow]{corpus_dir}[/yellow]")
    console.print(f"Embedding Provider: [green]{settings.EMBEDDING_PROVIDER.value}[/green]")
    console.print(f"Vector Database: [green]ChromaDB ({settings.CHROMA_PERSIST_DIRECTORY})[/green]\n")

    start_time = time.perf_counter()
    results = ingestion_pipeline.ingest_directory(corpus_dir)
    elapsed = round(time.perf_counter() - start_time, 2)

    table = Table(title="Document Ingestion Summary", show_header=True, header_style="bold magenta")
    table.add_column("Document Source", style="cyan", no_wrap=True)
    table.add_column("Document Title", style="white")
    table.add_column("Chunks Created", justify="right", style="green")
    table.add_column("Status", style="bold green")

    total_chunks = 0
    for r in results:
        chunks = r.get("chunks_created", 0)
        total_chunks += chunks
        table.add_row(
            r.get("source", "N/A"),
            r.get("doc_title", "N/A"),
            str(chunks),
            r.get("status", "failed")
        )

    console.print(table)
    console.print(f"\n[bold green]✅ Ingestion Completed in {elapsed}s![/bold green] Total chunks indexed: [bold cyan]{total_chunks}[/bold cyan]")
    console.print(f"Total collection count in ChromaDB: [bold yellow]{vector_store_manager.get_total_chunk_count()}[/bold yellow]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest technical documentation corpus into ChromaDB.")
    parser.add_argument("--dir", type=str, default=settings.CORPUS_DIRECTORY, help="Path to corpus directory")
    args = parser.parse_args()
    run_ingestion(args.dir)
