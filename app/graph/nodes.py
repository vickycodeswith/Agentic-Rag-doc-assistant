from datetime import datetime, timezone
from typing import Any, Dict, List
from langchain_core.documents import Document

from app.config import settings
from app.core.logging import log_graph_event, logger
from app.graph.prompts import (
    DOCUMENT_GRADING_PROMPT,
    GENERATION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    QUERY_REWRITE_PROMPT,
    GradeChunkOutput,
    HallucinationCheckOutput,
    QueryAnalysisOutput,
    RewriteQueryOutput,
)
from app.graph.state import Citation, DocumentChunk, ExecutionTraceStep, GraphState
from app.models.llm_factory import get_llm
from app.retrieval.vectorstore import vector_store_manager
from app.services.web_search_service import web_search_service


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def query_analysis_node(state: GraphState) -> Dict[str, Any]:
    """Node 1: Analyzes, classifies, and rewrites the raw user question for dense retrieval."""
    original_query = state["original_query"]
    logger.info(f"--- [Node 1: Query Analysis] Analyzing: '{original_query}' ---")

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(QueryAnalysisOutput)

    try:
        chain = QUERY_ANALYSIS_PROMPT | structured_llm
        result: QueryAnalysisOutput = chain.invoke({"question": original_query})
        rewritten = result.rewritten_query or original_query
        q_type = result.query_type or "general"
        keywords = result.technical_keywords or []
    except Exception as e:
        logger.warning(f"Query analysis failed: {e}. Using raw query fallback.")
        rewritten = original_query
        q_type = "general"
        keywords = []

    trace_step = ExecutionTraceStep(
        node_name="query_analysis",
        step_number=len(state.get("execution_trace", [])) + 1,
        details={
            "original_query": original_query,
            "rewritten_query": rewritten,
            "query_type": q_type,
            "keywords": keywords,
        },
        timestamp=_now()
    )
    log_graph_event("query_analysis", trace_step.details)

    return {
        "current_query": rewritten,
        "query_type": q_type,
        "technical_keywords": keywords,
        "execution_trace": state.get("execution_trace", []) + [trace_step]
    }


def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """Node 2: Performs vector similarity search against ChromaDB."""
    search_query = state.get("current_query") or state["original_query"]
    logger.info(f"--- [Node 2: Retrieval] Searching ChromaDB for: '{search_query}' (Top-k: {settings.RETRIEVAL_TOP_K}) ---")

    raw_results = vector_store_manager.similarity_search_with_score(
        query=search_query,
        k=settings.RETRIEVAL_TOP_K
    )

    retrieved_chunks: List[DocumentChunk] = []
    for doc, score in raw_results:
        retrieved_chunks.append(
            DocumentChunk(
                page_content=doc.page_content,
                metadata=doc.metadata,
                similarity_score=round(score, 4)
            )
        )

    trace_step = ExecutionTraceStep(
        node_name="retrieval",
        step_number=len(state.get("execution_trace", [])) + 1,
        details={
            "search_query": search_query,
            "chunks_retrieved": len(retrieved_chunks),
            "sources": [c.metadata.get("source") for c in retrieved_chunks]
        },
        timestamp=_now()
    )
    log_graph_event("retrieval", trace_step.details)

    return {
        "retrieved_documents": retrieved_chunks,
        "execution_trace": state.get("execution_trace", []) + [trace_step]
    }


def document_grading_node(state: GraphState) -> Dict[str, Any]:
    """Node 3: Evaluates retrieved chunks for factual relevance and filters out noise."""
    question = state["original_query"]
    retrieved = state.get("retrieved_documents", [])
    logger.info(f"--- [Node 3: Document Grading] Grading {len(retrieved)} retrieved chunks ---")

    llm = get_llm(temperature=0.0)
    structured_grader = llm.with_structured_output(GradeChunkOutput)
    grading_chain = DOCUMENT_GRADING_PROMPT | structured_grader

    relevant_docs: List[DocumentChunk] = []
    filtered_docs: List[DocumentChunk] = []

    for chunk in retrieved:
        section = chunk.metadata.get("section_title", "General")
        source = chunk.metadata.get("source", "Document")
        
        try:
            grade: GradeChunkOutput = grading_chain.invoke({
                "question": question,
                "document_content": chunk.page_content,
                "section_title": section,
                "source": source
            })
            chunk.is_relevant = grade.is_relevant
            chunk.relevance_score = grade.confidence
            chunk.grading_reason = grade.reasoning

            if grade.is_relevant:
                relevant_docs.append(chunk)
            else:
                filtered_docs.append(chunk)
        except Exception as e:
            logger.warning(f"Error grading chunk from {source}: {e}. Defaulting to relevant.")
            chunk.is_relevant = True
            chunk.relevance_score = 0.5
            chunk.grading_reason = "Grading fallback due to parsing error"
            relevant_docs.append(chunk)

    trace_step = ExecutionTraceStep(
        node_name="document_grading",
        step_number=len(state.get("execution_trace", [])) + 1,
        details={
            "total_graded": len(retrieved),
            "relevant_count": len(relevant_docs),
            "filtered_count": len(filtered_docs),
            "relevant_sources": [d.metadata.get("source") for d in relevant_docs]
        },
        timestamp=_now()
    )
    log_graph_event("document_grading", trace_step.details)

    return {
        "relevant_documents": relevant_docs,
        "filtered_out_documents": filtered_docs,
        "execution_trace": state.get("execution_trace", []) + [trace_step]
    }


def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    """Node 5: Self-correction rewrite node when retrieval finds no relevant documents."""
    current_retry = state.get("retry_count", 0) + 1
    original = state["original_query"]
    current = state.get("current_query", original)
    logger.info(f"--- [Node 5: Rewrite Query] Attempt {current_retry} of {state.get('max_retries', settings.MAX_RETRIES)} ---")

    llm = get_llm(temperature=0.2)
    structured_rewriter = llm.with_structured_output(RewriteQueryOutput)
    chain = QUERY_REWRITE_PROMPT | structured_rewriter

    try:
        res: RewriteQueryOutput = chain.invoke({
            "original_query": original,
            "current_query": current,
            "retry_count": current_retry,
            "max_retries": state.get("max_retries", settings.MAX_RETRIES)
        })
        improved_query = res.improved_query or original
        strategy = res.search_strategy_explanation
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}. Modifying terms simply.")
        improved_query = f"{original} technical guide documentation"
        strategy = "Fallback simple expansion"

    trace_step = ExecutionTraceStep(
        node_name="rewrite_query",
        step_number=len(state.get("execution_trace", [])) + 1,
        details={
            "retry_count": current_retry,
            "previous_query": current,
            "new_query": improved_query,
            "strategy": strategy
        },
        timestamp=_now()
    )
    log_graph_event("rewrite_query", trace_step.details)

    return {
        "current_query": improved_query,
        "retry_count": current_retry,
        "execution_trace": state.get("execution_trace", []) + [trace_step]
    }


def web_search_fallback_node(state: GraphState) -> Dict[str, Any]:
    """Fallback Node: Executes external search when vector database lacks relevant context."""
    query = state["original_query"]
    logger.info(f"--- [Fallback Node: Web Search] Querying web search fallback for: '{query}' ---")

    search_results = web_search_service.search(query=query)
    web_chunks: List[DocumentChunk] = []

    for res in search_results:
        web_chunks.append(
            DocumentChunk(
                page_content=res["content"],
                metadata={
                    "source": res["url"],
                    "doc_title": res["title"],
                    "section_title": "Web Search Result",
                    "source_type": "web_search"
                },
                similarity_score=0.8,
                is_relevant=True,
                relevance_score=0.8,
                grading_reason="External web fallback result"
            )
        )

    trace_step = ExecutionTraceStep(
        node_name="web_search_fallback",
        step_number=len(state.get("execution_trace", [])) + 1,
        details={
            "results_count": len(web_chunks),
            "sources": [c.metadata.get("source") for c in web_chunks]
        },
        timestamp=_now()
    )
    log_graph_event("web_search_fallback", trace_step.details)

    return {
        "relevant_documents": web_chunks,
        "web_search_used": True,
        "execution_trace": state.get("execution_trace", []) + [trace_step]
    }


def generation_node(state: GraphState) -> Dict[str, Any]:
    """Node 4: Synthesizes final grounded answer and formats structured citations."""
    question = state["original_query"]
    relevant_docs = state.get("relevant_documents", [])
    logger.info(f"--- [Node 4: Generation] Synthesizing answer with {len(relevant_docs)} context chunks ---")

    # If absolutely no context is available
    if not relevant_docs:
        fallback_msg = (
            "I could not find sufficient technical documentation in the indexed corpus to answer your question accurately. "
            "Please check if the required documentation has been ingested, or try rephrasing your question with specific framework terms."
        )
        trace_step = ExecutionTraceStep(
            node_name="generation",
            step_number=len(state.get("execution_trace", [])) + 1,
            details={"status": "insufficient_context", "answer_length": len(fallback_msg)},
            timestamp=_now()
        )
        return {
            "generation": fallback_msg,
            "citations": [],
            "status": "insufficient_context",
            "is_grounded": True,
            "groundedness_score": 1.0,
            "execution_trace": state.get("execution_trace", []) + [trace_step]
        }

    # Format context blocks with header tags
    formatted_context_blocks = []
    citations: List[Citation] = []
    seen_citations = set()

    for idx, doc in enumerate(relevant_docs):
        src = doc.metadata.get("source", "Unknown Source")
        sec = doc.metadata.get("section_title", "General")
        title = doc.metadata.get("doc_title", src)
        cid = doc.metadata.get("chunk_id")

        formatted_context_blocks.append(
            f"--- Context Chunk {idx + 1} [Source: {src} | Section: {sec}] ---\n{doc.page_content}\n"
        )

        citation_key = f"{src}_{sec}"
        if citation_key not in seen_citations:
            citations.append(Citation(
                source=src,
                doc_title=title,
                section_title=sec,
                chunk_id=cid
            ))
            seen_citations.add(citation_key)

    context_str = "\n".join(formatted_context_blocks)
    llm = get_llm(temperature=0.1)
    chain = GENERATION_PROMPT | llm

    try:
        response = chain.invoke({"question": question, "context": context_str})
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        answer_text = f"An error occurred while generating the answer: {str(e)}"

    trace_step = ExecutionTraceStep(
        node_name="generation",
        step_number=len(state.get("execution_trace", [])) + 1,
        details={
            "citations_count": len(citations),
            "sources": [c.source for c in citations],
            "answer_length": len(answer_text)
        },
        timestamp=_now()
    )
    log_graph_event("generation", trace_step.details)

    return {
        "generation": answer_text,
        "citations": citations,
        "status": "success" if not state.get("web_search_used") else "fallback_used",
        "execution_trace": state.get("execution_trace", []) + [trace_step]
    }


def hallucination_check_node(state: GraphState) -> Dict[str, Any]:
    """Bonus Node: Self-RAG Groundedness check verifying factual faithfulness to context."""
    if not settings.ENABLE_HALLUCINATION_CHECK or not state.get("relevant_documents"):
        return {
            "is_grounded": True,
            "groundedness_score": 1.0,
            "groundedness_reason": "Check disabled or empty context."
        }

    logger.info("--- [Bonus Node: Hallucination Check] Verifying answer faithfulness ---")
    generation = state.get("generation", "")
    context_str = "\n\n".join([d.page_content for d in state.get("relevant_documents", [])])

    llm = get_llm(temperature=0.0)
    structured_checker = llm.with_structured_output(HallucinationCheckOutput)
    chain = HALLUCINATION_CHECK_PROMPT | structured_checker

    try:
        res: HallucinationCheckOutput = chain.invoke({
            "context": context_str,
            "generation": generation
        })
        is_grounded = res.is_grounded
        score = res.groundedness_score
        reason = res.reasoning
    except Exception as e:
        logger.warning(f"Hallucination check failed: {e}. Defaulting to grounded.")
        is_grounded = True
        score = 0.9
        reason = "Checker fallback"

    trace_step = ExecutionTraceStep(
        node_name="hallucination_check",
        step_number=len(state.get("execution_trace", [])) + 1,
        details={
            "is_grounded": is_grounded,
            "groundedness_score": score,
            "reason": reason
        },
        timestamp=_now()
    )
    log_graph_event("hallucination_check", trace_step.details)

    return {
        "is_grounded": is_grounded,
        "groundedness_score": score,
        "groundedness_reason": reason,
        "execution_trace": state.get("execution_trace", []) + [trace_step]
    }
