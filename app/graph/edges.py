from app.config import settings
from app.core.logging import logger
from app.graph.state import GraphState


def decide_to_generate(state: GraphState) -> str:
    """
    Core Conditional Edge:
    Evaluates document grading outcome to determine whether to generate, rewrite query, or fallback.
    """
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.MAX_RETRIES)

    # 1. If we have relevant documents, proceed directly to generation
    if len(relevant_docs) > 0:
        logger.info(f"Routing -> 'generate' ({len(relevant_docs)} relevant documents found)")
        return "generate"

    # 2. If no relevant documents and retry budget remains, attempt query rewriting
    if retry_count < max_retries:
        logger.info(f"Routing -> 'rewrite_query' (0 relevant docs. Retry {retry_count + 1}/{max_retries})")
        return "rewrite_query"

    # 3. If retries exhausted, check if external web search fallback is enabled
    if settings.ENABLE_WEB_SEARCH_FALLBACK:
        logger.info("Routing -> 'web_search_fallback' (Retries exhausted, engaging web search)")
        return "web_search_fallback"

    # 4. Graceful termination with insufficient context message
    logger.info("Routing -> 'generate' (Retries exhausted, triggering graceful fallback response)")
    return "generate"


def decide_after_hallucination_check(state: GraphState) -> str:
    """
    Self-RAG Groundedness Edge:
    Ensures the final answer contains zero ungrounded hallucinations.
    """
    is_grounded = state.get("is_grounded", True)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.MAX_RETRIES)

    if is_grounded:
        return "end"

    # If ungrounded and retries left, trigger query rewrite to seek better grounding context
    if retry_count < max_retries:
        logger.warning("Answer ungrounded. Routing back to -> 'rewrite_query'")
        return "rewrite_query"

    return "end"
