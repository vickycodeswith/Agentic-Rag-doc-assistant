from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.core.logging import logger
from app.graph.edges import decide_after_hallucination_check, decide_to_generate
from app.graph.nodes import (
    document_grading_node,
    generation_node,
    hallucination_check_node,
    query_analysis_node,
    retrieval_node,
    rewrite_query_node,
    web_search_fallback_node,
)
from app.graph.state import GraphState


def create_rag_graph() -> StateGraph:
    """
    Constructs the self-corrective RAG StateGraph workflow.
    """
    workflow = StateGraph(GraphState)

    # 1. Register all nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("grade_documents", document_grading_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search_fallback", web_search_fallback_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # 2. Add structural deterministic edges
    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # 3. Add Document Grading Conditional Routing
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search_fallback": "web_search_fallback",
        }
    )

    # 4. Add Cycle Edges
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search_fallback", "generate")
    workflow.add_edge("generate", "hallucination_check")

    # 5. Add Hallucination Check Conditional Edge
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination_check,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
        }
    )

    return workflow


# In-memory checkpointer for conversation memory
checkpointer = MemorySaver()

# Compiled application workflow
compiled_rag_app = create_rag_graph().compile(checkpointer=checkpointer)


def get_mermaid_graph_ascii() -> str:
    """Returns ASCII representation of the compiled graph."""
    try:
        return compiled_rag_app.get_graph().draw_ascii()
    except Exception:
        return "Graph ASCII visualization unavailable."


def get_mermaid_graph_markdown() -> str:
    """Returns Mermaid markdown code for the compiled graph."""
    try:
        return compiled_rag_app.get_graph().draw_mermaid()
    except Exception:
        return """
graph TD
    START --> query_analysis
    query_analysis --> retrieve
    retrieve --> grade_documents
    grade_documents -->|Relevant Docs Found| generate
    grade_documents -->|No Docs & Retry Left| rewrite_query
    grade_documents -->|No Docs & Retries Exhausted| web_search_fallback
    rewrite_query --> retrieve
    web_search_fallback --> generate
    generate --> hallucination_check
    hallucination_check --> END
"""
