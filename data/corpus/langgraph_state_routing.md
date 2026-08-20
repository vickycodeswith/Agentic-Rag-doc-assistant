# LangGraph StateGraph Architecture & Routing Mechanics

## 1. Introduction to LangGraph
LangGraph is a library built on top of LangChain for building stateful, multi-actor applications with LLMs. Unlike traditional linear DAG chains (such as `RunnableSequence`), LangGraph supports cycles, branching, conditional execution, and dynamic state modification.

## 2. Core Concepts: StateGraph and TypedDict
Every LangGraph application centers around a state schema. The state is represented as a Python `TypedDict` or Pydantic model where each key holds specific application state.

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    query: str
    documents: List[str]
    retry_count: int
    answer: str
```

### 2.1 State Reducers
Keys in `AgentState` can define custom reducer functions via `Annotated[List[str], operator.add]` to append messages or overwrite values deterministically.

## 3. Nodes and Conditional Routing
### 3.1 Node Definitions
A node is simply a Python function that takes the current `state` as input and returns a dictionary of updated state fields:
```python
def retrieve_node(state: AgentState) -> dict:
    docs = vector_store.search(state["query"])
    return {"documents": docs}
```

### 3.2 Conditional Edges
Conditional edges evaluate the current state and return the name of the next node to execute. This is essential for Self-Corrective RAG and agent loops:
```python
def route_after_grading(state: AgentState) -> str:
    if len(state["documents"]) > 0:
        return "generate"
    elif state["retry_count"] < 3:
        return "rewrite_query"
    else:
        return "fallback"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("fallback", fallback_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", route_after_grading, {
    "generate": "generate",
    "rewrite_query": "rewrite_query",
    "fallback": "fallback"
})
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)
```

## 4. Preventing Infinite Loops
In cyclic graphs, bounded termination is mandatory. The `retry_count` is incremented on every retry attempt in `rewrite_query`. When `retry_count >= max_retries`, the conditional edge routes to `fallback` instead of looping back to `retrieve`.

## 5. Checkpointing and Persistence
LangGraph includes persistence checkpointers such as `MemorySaver` or `SqliteSaver`. Checkpointers record the graph state at each super-step, enabling conversational memory (`thread_id`), state inspection, time-travel, and human-in-the-loop approvals.
