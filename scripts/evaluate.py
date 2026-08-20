import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api.schemas import QueryRequest
from app.graph.workflow import compiled_rag_app

console = Console()

EVALUATION_BATTERY = [
    {
        "id": "TC-01",
        "archetype": "Direct Factual",
        "query": "How do you declare query parameters in FastAPI?",
        "expected_behavior": "Should retrieve FastAPI docs, identify query parameters, and cite fastapi_architecture.md.",
        "must_contain": ["fastapi", "query", "parameter"]
    },
    {
        "id": "TC-02",
        "archetype": "Conceptual / Architecture",
        "query": "What is the core difference between StateGraph and standard linear chains in LangGraph?",
        "expected_behavior": "Should explain cyclic state execution, conditional routing, and state reducers.",
        "must_contain": ["state", "graph", "cycle"]
    },
    {
        "id": "TC-03",
        "archetype": "How-To / Code Syntax",
        "query": "Show how to create a Pydantic v2 validator using @field_validator with an example.",
        "expected_behavior": "Should synthesize correct Pydantic v2 @field_validator syntax with code block.",
        "must_contain": ["@field_validator", "classmethod"]
    },
    {
        "id": "TC-04",
        "archetype": "Multi-Chunk Synthesis",
        "query": "Explain how ChromaDB persists data to disk and how cosine similarity is computed from distances.",
        "expected_behavior": "Synthesizes persistent client setup and distance-to-similarity conversion formula.",
        "must_contain": ["chroma", "persistent", "cosine"]
    },
    {
        "id": "TC-05",
        "archetype": "Ambiguous / Disambiguation",
        "query": "How do I make the route async with dependencies?",
        "expected_behavior": "Query analysis expands 'route with dependencies' into FastAPI async def and Depends.",
        "must_contain": ["async", "depends"]
    },
    {
        "id": "TC-06",
        "archetype": "Out-of-Scope (Graceful Fallback)",
        "query": "What is the secret recipe for baking authentic chocolate chip cookies?",
        "expected_behavior": "Retrieval fails or is graded irrelevant; after retries, returns graceful insufficient context message.",
        "must_contain": ["insufficient", "documentation"]
    },
    {
        "id": "TC-07",
        "archetype": "Retry Loop Trigger",
        "query": "Explain conditional edge routing based on document grading results in agent workflows.",
        "expected_behavior": "Triggers retrieval, evaluates grading, executes routing logic with bounded retries.",
        "must_contain": ["routing", "grade", "edge"]
    },
    {
        "id": "TC-08",
        "archetype": "Adversarial / Hallucination Trap",
        "query": "Does FastAPI automatically compile Python code to Rust for 100x speedups?",
        "expected_behavior": "Correctly refutes claim; clarifies that Pydantic-core is in Rust while FastAPI runs on Starlette/Uvicorn.",
        "must_contain": ["pydantic-core", "rust"]
    }
]


def run_evaluation_suite():
    console.print(Panel.fit(
        "[bold cyan]🧪 EXPRESS ANALYTICS RAG ASSISTANT BENCHMARK EVALUATION[/bold cyan]\n"
        "[dim]Testing 8 Query Archetypes across Groundedness, Self-Correction, and Routing[/dim]",
        border_style="cyan"
    ))

    table = Table(title="Benchmark Execution Results", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Archetype", style="white", width=22)
    table.add_column("Query", style="dim", width=34)
    table.add_column("Retries", justify="center", style="yellow", width=8)
    table.add_column("Citations", justify="center", style="green", width=10)
    table.add_column("Grounded", justify="center", style="bold green", width=10)
    table.add_column("Latency", justify="right", style="cyan", width=8)
    table.add_column("Status", style="bold green", width=10)

    total_time = 0.0
    passed_count = 0

    for test in EVALUATION_BATTERY:
        q_text = test["query"]
        start_t = time.perf_counter()

        initial_state = {
            "original_query": q_text,
            "current_query": q_text,
            "query_type": "general",
            "technical_keywords": [],
            "retrieved_documents": [],
            "relevant_documents": [],
            "filtered_out_documents": [],
            "retry_count": 0,
            "max_retries": 2,
            "web_search_used": False,
            "generation": "",
            "citations": [],
            "is_grounded": True,
            "groundedness_score": 1.0,
            "groundedness_reason": None,
            "status": "in_progress",
            "error_message": None,
            "execution_trace": []
        }

        try:
            res = compiled_rag_app.invoke(initial_state, config={"configurable": {"thread_id": f"eval_{test['id']}"}})
            elapsed = round(time.perf_counter() - start_t, 2)
            total_time += elapsed

            answer = res.get("generation", "")
            citations = res.get("citations", [])
            retries = res.get("retry_count", 0)
            is_grounded = res.get("is_grounded", True)
            status_str = res.get("status", "success")

            # Check if answer contains expected keywords (case-insensitive) or handled fallback
            answer_lower = answer.lower()
            matched = any(k in answer_lower for k in test["must_contain"]) or status_str == "insufficient_context"
            verdict = "PASS" if matched else "REVIEW"
            if verdict == "PASS":
                passed_count += 1

            table.add_row(
                test["id"],
                test["archetype"],
                q_text[:32] + "...",
                str(retries),
                str(len(citations)),
                "YES" if is_grounded else "NO",
                f"{elapsed}s",
                f"[green]{verdict}[/green]" if verdict == "PASS" else f"[yellow]{verdict}[/yellow]"
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - start_t, 2)
            table.add_row(
                test["id"],
                test["archetype"],
                q_text[:32] + "...",
                "-",
                "-",
                "ERR",
                f"{elapsed}s",
                "[red]FAIL[/red]"
            )

    console.print(table)
    avg_latency = round(total_time / len(EVALUATION_BATTERY), 2)
    score_pct = round((passed_count / len(EVALUATION_BATTERY)) * 100, 1)

    console.print(f"\n[bold green]Evaluation Complete![/bold green] Passed: [bold cyan]{passed_count}/{len(EVALUATION_BATTERY)} ({score_pct}%)[/bold cyan] | Avg Latency: [bold yellow]{avg_latency}s[/bold yellow]\n")


if __name__ == "__main__":
    run_evaluation_suite()
