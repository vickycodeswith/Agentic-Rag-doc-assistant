from typing import Any, Optional, Type
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from app.config import LLMProvider, settings
from app.core.exceptions import ModelProviderError
from app.core.logging import logger


class MockChatModel(BaseChatModel):
    """
    Deterministic mock chat model used for offline testing and fallback when API keys are not present.
    """
    model_name: str = "mock-model"

    def _extract_text(self, messages: Any) -> str:
        """Helper to extract text from list of BaseMessages, ChatPromptValue, or strings."""
        if hasattr(messages, "to_messages"):
            msg_list = messages.to_messages()
            return " ".join(m.content for m in msg_list if hasattr(m, "content"))
        elif isinstance(messages, list):
            return " ".join(
                m.content if hasattr(m, "content") else str(m)
                for m in messages
            )
        elif hasattr(messages, "text"):
            return messages.text
        return str(messages)

    def _generate(self, messages: Any, stop: list[str] | None = None, **kwargs) -> Any:
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        full_text = self._extract_text(messages).lower()
        
        # Extract last message content if available
        if hasattr(messages, "to_messages"):
            msg_list = messages.to_messages()
            last_msg = msg_list[-1].content.lower() if msg_list else full_text
        elif isinstance(messages, list) and messages:
            last_msg = messages[-1].content.lower() if hasattr(messages[-1], "content") else str(messages[-1]).lower()
        else:
            last_msg = full_text

        # Extract clean target question from prompt text
        target_q = last_msg
        for line in last_msg.splitlines():
            if "user question:" in line:
                target_q = line.replace("user question:", "").strip()
                break
            elif "original question:" in line:
                target_q = line.replace("original question:", "").strip()
                break

        # 1. Document grading check
        if "grading criteria" in full_text or "score as relevant" in full_text:
            if "cookie" in target_q or "cake" in target_q or "baking" in target_q:
                content = '{"is_relevant": false, "confidence": 0.99, "reasoning": "Document discusses software architecture, not baking recipes."}'
            else:
                content = '{"is_relevant": true, "confidence": 0.95, "reasoning": "Chunk contains technical documentation answering the question."}'

        # 2. Query analysis check
        elif "query analyzer" in full_text or "expert technical query analyzer" in full_text:
            if "cookie" in target_q or "baking" in target_q:
                content = '{"rewritten_query": "baking authentic chocolate chip cookies recipe", "query_type": "general", "technical_keywords": ["cookies", "recipe"]}'
            elif "rust" in target_q:
                content = '{"rewritten_query": "FastAPI execution model Pydantic-core Rust Starlette", "query_type": "troubleshooting", "technical_keywords": ["FastAPI", "Rust", "pydantic-core"]}'
            elif "pydantic" in target_q or "@field_validator" in target_q:
                content = '{"rewritten_query": "Pydantic v2 @field_validator method syntax example", "query_type": "how-to", "technical_keywords": ["Pydantic", "@field_validator"]}'
            elif "chroma" in target_q:
                content = '{"rewritten_query": "ChromaDB persistence PersistentClient cosine similarity distance", "query_type": "conceptual", "technical_keywords": ["ChromaDB", "PersistentClient", "cosine"]}'
            elif "stategraph" in target_q or "difference between" in target_q:
                content = '{"rewritten_query": "LangGraph StateGraph cycles state reducer TypedDict", "query_type": "conceptual", "technical_keywords": ["LangGraph", "StateGraph", "cycle"]}'
            elif "conditional edge" in target_q or "routing based on document grading" in target_q:
                content = '{"rewritten_query": "LangGraph conditional edges routing document grading StateGraph", "query_type": "conceptual", "technical_keywords": ["routing", "grade", "edge"]}'
            elif "async" in target_q and "depend" in target_q:
                content = '{"rewritten_query": "FastAPI async def route dependencies Depends", "query_type": "how-to", "technical_keywords": ["FastAPI", "async", "Depends"]}'
            else:
                content = '{"rewritten_query": "FastAPI query parameter declaration and validation Query", "query_type": "how-to", "technical_keywords": ["FastAPI", "Query", "parameter"]}'

        # 3. Hallucination check
        elif "hallucination auditor" in full_text or "factual grounding" in full_text:
            content = '{"is_grounded": true, "groundedness_score": 0.98, "reasoning": "Every factual statement and code snippet is grounded in the retrieved documentation context."}'

        # 4. Query rewriting
        elif "query optimizer" in full_text or "search query" in full_text:
            content = '{"improved_query": "LangGraph conditional edges routing document grading StateGraph", "search_strategy_explanation": "Added precise graph edge terminology and node names."}'

        # 5. Default answer generation
        else:
            if "cookie" in target_q or "baking" in target_q:
                content = "I could not find sufficient technical documentation in the indexed corpus to answer your question accurately. Please check if the required documentation has been ingested."
            elif "rust" in target_q:
                content = "FastAPI does not automatically compile Python code to Rust. However, its validation layer (Pydantic v2) is powered by `pydantic-core`, which is written in Rust for performance. [Doc: pydantic_v2_core.md | Section: Overview and Core Engine]"
            elif "pydantic" in target_q or "@field_validator" in target_q:
                content = """In Pydantic v2, validators are declared using the `@field_validator` decorator on a `@classmethod`.

```python
from pydantic import BaseModel, field_validator

class DocumentQuery(BaseModel):
    query_text: str

    @field_validator("query_text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query text cannot be empty")
        return v
```
[Doc: pydantic_v2_core.md | Section: Validators in Pydantic v2]"""
            elif "stategraph" in target_q or "difference between" in target_q or "langgraph" in target_q:
                content = "In LangGraph, `StateGraph` represents an explicit stateful computation graph supporting cycles and state reducers, unlike standard linear chains. State is defined with `TypedDict`. [Doc: langgraph_state_routing.md | Section: Core Concepts: StateGraph and TypedDict]"
            elif "chroma" in target_q:
                content = "ChromaDB persists data using `chromadb.PersistentClient(path='./data/chroma_db')`. Cosine similarity is computed from distances as `Similarity = 1 - distance`. [Doc: chromadb_vector_engine.md | Section: Client Initialization and Modes]"
            elif "routing" in target_q or "edge" in target_q:
                content = "In LangGraph, conditional edges evaluate the state after document grading to dynamically route to either `generate` or `rewrite_query`. [Doc: langgraph_state_routing.md | Section: Nodes and Conditional Routing]"
            elif "async" in target_q or "dependencies" in target_q:
                content = "In FastAPI, routes can be declared using `async def` and inject dependencies using `Depends(dependency_function)`. [Doc: fastapi_architecture.md | Section: Dependency Injection System]"
            else:
                content = "In FastAPI, query parameters are declared as function parameters that are not part of the path. You can validate them using `Query()`. [Doc: fastapi_architecture.md | Section: Query Parameters and Validation]"

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def with_structured_output(self, schema: Type[BaseModel], **kwargs):
        """Mock implementation of with_structured_output for test safety."""
        from langchain_core.runnables import RunnableLambda

        def mock_structured_call(messages):
            import json
            raw_text = self._generate(messages).generations[0].message.content
            try:
                data = json.loads(raw_text)
                return schema(**data)
            except Exception:
                return schema.model_construct()

        return RunnableLambda(mock_structured_call)


def get_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None
) -> BaseChatModel:
    """
    Instantiates and returns the configured Chat LLM.
    Supports Google Gemini, Groq, and OpenAI with automatic fallback if credentials are absent.
    """
    selected_provider = provider or settings.LLM_PROVIDER
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    max_t = max_tokens or settings.LLM_MAX_TOKENS
    model = model_name or settings.get_resolved_llm_model()

    if selected_provider == LLMProvider.GEMINI:
        if not settings.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY is not set. Falling back to MockChatModel for offline safety.")
            return MockChatModel()
        
        target_model = model
        if target_model in ["gemini-1.5-flash", "models/gemini-1.5-flash"]:
            target_model = "gemini-2.5-flash"

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info(f"Initialized Google Gemini Chat LLM ({target_model})")
            return ChatGoogleGenerativeAI(
                model=target_model,
                temperature=temp,
                max_output_tokens=max_t,
                google_api_key=settings.GOOGLE_API_KEY
            )
        except Exception as e:
            logger.warning(f"Failed with {target_model}: {e}. Retrying with 'gemini-flash-latest'...")
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model="gemini-flash-latest",
                    temperature=temp,
                    max_output_tokens=max_t,
                    google_api_key=settings.GOOGLE_API_KEY
                )
            except Exception as e2:
                logger.error(f"Failed to initialize Gemini model: {e2}")
                raise ModelProviderError(f"Gemini initialization error: {e2}")

    elif selected_provider == LLMProvider.GROQ:
        if not settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is not set. Falling back to MockChatModel.")
            return MockChatModel()
        try:
            from langchain_groq import ChatGroq
            logger.info(f"Initialized Groq Chat LLM ({model})")
            return ChatGroq(
                model_name=model,
                temperature=temp,
                max_tokens=max_t,
                groq_api_key=settings.GROQ_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq model: {e}")
            raise ModelProviderError(f"Groq initialization error: {e}")

    elif selected_provider == LLMProvider.OPENAI:
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set. Falling back to MockChatModel.")
            return MockChatModel()
        try:
            from langchain_openai import ChatOpenAI
            logger.info(f"Initialized OpenAI Chat LLM ({model})")
            return ChatOpenAI(
                model_name=model,
                temperature=temp,
                max_tokens=max_t,
                openai_api_key=settings.OPENAI_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI model: {e}")
            raise ModelProviderError(f"OpenAI initialization error: {e}")

    logger.warning("No valid LLM provider configured. Using MockChatModel.")
    return MockChatModel()
