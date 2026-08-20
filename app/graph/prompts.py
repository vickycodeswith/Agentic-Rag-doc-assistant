from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


# ------------------------------------------------------------------------------
# Pydantic Output Schemas for Structured LLM Calls
# ------------------------------------------------------------------------------

class QueryAnalysisOutput(BaseModel):
    """Structured output for Node 1: Query Analysis & Classification."""
    rewritten_query: str = Field(
        description="Expanded, unambiguous search query optimized for semantic dense retrieval with technical keywords."
    )
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api_reference", "general"] = Field(
        description="Classification category of the user question."
    )
    technical_keywords: List[str] = Field(
        default_factory=list,
        description="Specific framework symbols, function names, classes, or keywords extracted from the question."
    )


class GradeChunkOutput(BaseModel):
    """Structured output for Node 3: Document Grading."""
    is_relevant: bool = Field(
        description="True if the document chunk contains information directly relevant to answering the user question, False otherwise."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in the relevance judgment between 0.0 and 1.0."
    )
    reasoning: str = Field(
        description="Brief technical explanation of why this chunk is or is not relevant to the query."
    )


class RewriteQueryOutput(BaseModel):
    """Structured output for Node 5: Query Rewriting."""
    improved_query: str = Field(
        description="A completely reformulated search query designed to retrieve missing technical context that failed in previous attempts."
    )
    search_strategy_explanation: str = Field(
        description="Reasoning behind why this new formulation will overcome previous retrieval gaps."
    )


class HallucinationCheckOutput(BaseModel):
    """Structured output for Self-RAG Groundedness Check."""
    is_grounded: bool = Field(
        description="True if every claim and fact in the answer is directly supported by the retrieved document context, False if it hallucinates."
    )
    groundedness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Factual consistency score from 0.0 to 1.0."
    )
    reasoning: str = Field(
        description="Analysis of whether any claims lack evidence in the provided context."
    )


# ------------------------------------------------------------------------------
# Chat Prompt Templates
# ------------------------------------------------------------------------------

QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical query analyzer for an AI/ML documentation assistant.
Your task is to analyze the user's raw question, classify its intent, extract key technical terms, and rewrite/expand it to maximize vector similarity retrieval.

Guidelines:
1. Clarify ambiguity and expand abbreviations (e.g., 'DI' -> 'Dependency Injection', 'k8s' -> 'Kubernetes', 'Pydantic v2 validators').
2. Maintain technical specificity without altering the core intent.
3. Classify into: 'conceptual', 'how-to', 'troubleshooting', 'api_reference', or 'general'."""),
    ("human", "User Question: {question}")
])


DOCUMENT_GRADING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a rigorous technical document grader evaluating retrieved chunks for a RAG system.
Evaluate whether the provided document chunk contains substantive, useful information to answer the user question.

Grading Criteria:
- Score as RELEVANT (is_relevant=True) if the chunk contains concepts, code examples, API definitions, or explanations directly addressing the question.
- Score as IRRELEVANT (is_relevant=False) if the chunk is merely tangential, generic boilerplate, or discusses unrelated topics.
- Be discerning: Do not accept chunks that merely share words without addressing the user's actual question."""),
    ("human", """User Question: {question}

Document Chunk:
[Section: {section_title} | Source: {source}]
{document_content}""")
])


QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimizer for a technical documentation retrieval engine.
The previous search query failed to retrieve relevant documents from the vector database.
Analyze the user's original question and the previously attempted query, then formulate an alternative, more effective search query.

Techniques:
- Replace vague phrasing with precise framework terminology, function names, and standard technical phrasing.
- Strip extraneous conversational filler words.
- Expand acronyms or use synonyms."""),
    ("human", """Original Question: {original_query}
Previous Attempted Query: {current_query}
Retry Attempt: {retry_count} of {max_retries}

Formulate an improved search query:""")
])


GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior technical documentation assistant.
Generate a clear, technically precise, and actionable answer grounded EXCLUSIVELY in the provided context chunks.

STRICT INSTRUCTIONS:
1. Grounding: Answer ONLY using facts and code explicitly present in the retrieved context. Do NOT fabricate features or rely on unverified external assumptions.
2. Citations: Every key claim or code snippet MUST include inline citations citing the source and section: `[Doc: <source_file> | Section: <section_title>]`.
3. Code Quality: When providing code snippets, ensure they match the exact syntax and patterns from the context.
4. Transparency: If the context partially answers the question, answer what is known and state clearly what is missing.
5. Formatting: Use clean Markdown with headers, bullet points, and syntax-highlighted code blocks."""),
    ("human", """User Question: {question}

Retrieved Technical Context:
{context}

Generate the grounded technical answer:""")
])


HALLUCINATION_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict factual hallucination auditor for an enterprise RAG assistant.
Your task is to verify whether the generated answer is completely faithful to and supported by the provided context chunks.

Criteria:
- `is_grounded=True`: All facts, arguments, code examples, and assertions are directly backed by the context.
- `is_grounded=False`: The answer contains statements, claims, or code signatures that are absent from or contradict the context."""),
    ("human", """Retrieved Context:
{context}

Generated Answer:
{generation}

Audit the answer for factual grounding:""")
])
