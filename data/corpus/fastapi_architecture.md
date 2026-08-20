# FastAPI Architecture & Technical Reference

## 1. Overview and Core Philosophy
FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints. It is built on top of Starlette for the web routing layer and Pydantic for the data validation layer. Underneath, it uses ASGI (Asynchronous Server Gateway Interface) standards via Uvicorn.

## 2. Dependency Injection System
FastAPI includes an extensible Dependency Injection (DI) system that allows developers to declare dependencies in path operation functions using `Depends`.

### 2.1 Basic Usage
```python
from typing import Annotated
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
    return commons
```

### 2.2 Sub-dependencies and Hierarchical Trees
Dependencies can have their own sub-dependencies. FastAPI builds a Directed Acyclic Graph (DAG) of dependencies and executes them in optimal topological order. Results are cached per request by default (`use_cache=True`).

### 2.3 Yield Dependencies for Resource Lifecycle
For managing resources such as database sessions or connection pools:
```python
async def get_db():
    db = DBSession()
    try:
        yield db
    finally:
        db.close()
```

## 3. Path Operations and Concurrency
FastAPI handles both synchronous (`def`) and asynchronous (`async def`) endpoints seamlessly:
- When using `async def`, FastAPI runs the endpoint directly on the main event loop thread. If the function contains blocking I/O (e.g. `time.sleep()`), it will block all incoming requests.
- When using standard `def`, FastAPI executes the function in an external threadpool worker managed by `anyio`, preventing the event loop from being blocked.

## 4. Query Parameters and Validation
Query parameters are declared as function arguments that are not part of the path parameters.
```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/search/")
async def search_items(
    q: str = Query(..., min_length=3, max_length=50, description="The search term"),
    tags: list[str] = Query(default=[])
):
    return {"query": q, "tags": tags}
```

## 5. Middleware and Exception Handling
### 5.1 Global Exception Handlers
FastAPI enables custom exception handlers using `@app.exception_handler`:
```python
from fastapi import Request
from fastapi.responses import JSONResponse

class ItemNotFoundError(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

@app.exception_handler(ItemNotFoundError)
async def item_not_found_handler(request: Request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "Item not found", "item_id": exc.item_id}
    )
```
