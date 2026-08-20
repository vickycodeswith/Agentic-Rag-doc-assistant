# ChromaDB Vector Engine Architecture & Retrieval Reference

## 1. Introduction to ChromaDB
ChromaDB is an open-source, AI-native embedding database designed for developer productivity and local-first vector search. It handles vector indexing, metadata storage, filtering, and nearest neighbor search.

## 2. Client Initialization and Modes
ChromaDB supports two primary execution modes:
- **Ephemeral (In-Memory)**: `chromadb.EphemeralClient()` for transient testing.
- **Persistent (On-Disk)**: `chromadb.PersistentClient(path="./data/chroma_db")` for saving embeddings across application restarts.

```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./data/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
```

## 3. Collections and Distance Metrics
Collections group embeddings, documents, and metadata. When creating a collection, developers specify the distance function in `metadata={"hnsw:space": "cosine"}` (or `"l2"`, `"ip"`).

```python
collection = client.get_or_create_collection(
    name="technical_documentation",
    metadata={"hnsw:space": "cosine"}
)
```

## 4. Ingestion: Adding Documents and Metadata
Documents are inserted along with unique IDs, embeddings, and structured metadata:
```python
collection.add(
    ids=["doc_1_chunk_0", "doc_1_chunk_1"],
    documents=["FastAPI is built on Starlette...", "Pydantic performs validation..."],
    metadatas=[
        {"source": "fastapi_architecture.md", "section": "Overview", "chunk_index": 0},
        {"source": "pydantic_v2_core.md", "section": "Validators", "chunk_index": 1}
    ],
    embeddings=[[0.02, 0.91, -0.44], [0.15, -0.22, 0.88]]
)
```

## 5. Querying and Metadata Filtering
ChromaDB performs approximate nearest neighbor search using HNSW indexing and supports boolean filter operators (`$where`, `$and`, `$or`):
```python
results = collection.query(
    query_embeddings=[[0.01, 0.85, -0.39]],
    n_results=4,
    where={"source": "fastapi_architecture.md"},
    include=["documents", "metadatas", "distances"]
)
```

### 5.1 Distance to Cosine Similarity Conversion
For cosine space (`hnsw:space: cosine`), ChromaDB returns cosine distances ($D \in [0, 2]$). The cosine similarity score is calculated as:
$$\text{Similarity} = 1 - \frac{D}{2} \quad \text{or} \quad \text{Similarity} = 1 - D$$
where distance 0 represents identical vectors.
