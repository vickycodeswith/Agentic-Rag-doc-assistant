import pytest
from app.graph.edges import decide_to_generate, decide_after_hallucination_check
from app.graph.state import GraphState, DocumentChunk
from app.graph.workflow import compiled_rag_app


def test_decide_to_generate_with_relevant_docs(sample_relevant_chunk):
    state: GraphState = {
        "original_query": "Test query",
        "current_query": "Test query",
        "query_type": "general",
        "technical_keywords": [],
        "retrieved_documents": [sample_relevant_chunk],
        "relevant_documents": [sample_relevant_chunk],
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

    decision = decide_to_generate(state)
    assert decision == "generate"


def test_decide_to_generate_with_no_docs_triggers_rewrite():
    state: GraphState = {
        "original_query": "Test query",
        "current_query": "Test query",
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

    decision = decide_to_generate(state)
    assert decision == "rewrite_query"


def test_decide_to_generate_exhausted_retries():
    state: GraphState = {
        "original_query": "Test query",
        "current_query": "Test query",
        "query_type": "general",
        "technical_keywords": [],
        "retrieved_documents": [],
        "relevant_documents": [],
        "filtered_out_documents": [],
        "retry_count": 2,
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

    decision = decide_to_generate(state)
    # Either web_search_fallback or generate
    assert decision in ["web_search_fallback", "generate"]


def test_full_graph_execution():
    initial_state = {
        "original_query": "How do you declare query parameters in FastAPI?",
        "current_query": "How do you declare query parameters in FastAPI?",
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

    res = compiled_rag_app.invoke(initial_state, config={"configurable": {"thread_id": "test_session_1"}})
    assert "generation" in res
    assert "citations" in res
    assert "execution_trace" in res
    assert len(res["execution_trace"]) > 0
