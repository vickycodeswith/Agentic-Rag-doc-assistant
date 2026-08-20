import pytest
from app.graph.nodes import document_grading_node
from app.graph.prompts import GradeChunkOutput, DOCUMENT_GRADING_PROMPT
from app.graph.state import DocumentChunk, GraphState


def test_structured_grading_pydantic_schema():
    valid_grade = GradeChunkOutput(
        is_relevant=True,
        confidence=0.95,
        reasoning="The chunk directly describes how to declare path parameters in FastAPI."
    )
    assert valid_grade.is_relevant is True
    assert valid_grade.confidence == 0.95

    # Test score bounds
    with pytest.raises(Exception):
        GradeChunkOutput(is_relevant=True, confidence=1.5, reasoning="Invalid confidence")


def test_document_grading_prompt_formatting():
    formatted = DOCUMENT_GRADING_PROMPT.format_messages(
        question="How do dependencies work in FastAPI?",
        document_content="FastAPI uses Depends() for DI.",
        section_title="Dependencies",
        source="fastapi_architecture.md"
    )
    assert len(formatted) == 2
    assert "FastAPI uses Depends()" in formatted[1].content


def test_document_grading_node_filtering(sample_relevant_chunk, sample_irrelevant_chunk):
    state: GraphState = {
        "original_query": "How do dependencies work in FastAPI?",
        "current_query": "How do dependencies work in FastAPI?",
        "query_type": "how-to",
        "technical_keywords": ["FastAPI", "Depends"],
        "retrieved_documents": [sample_relevant_chunk, sample_irrelevant_chunk],
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

    result = document_grading_node(state)
    assert "relevant_documents" in result
    assert "filtered_out_documents" in result
    assert len(result["execution_trace"]) == 1
    assert result["execution_trace"][0].node_name == "document_grading"
