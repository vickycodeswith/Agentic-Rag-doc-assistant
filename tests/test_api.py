import pytest


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data
    assert "indexed_chunks" in data


@pytest.mark.asyncio
async def test_query_endpoint_success(async_client):
    payload = {
        "query": "How do you declare query parameters in FastAPI?",
        "max_retries": 2
    }
    response = await async_client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "retry_count" in data
    assert "latency_seconds" in data
    assert data["query"] == payload["query"]


@pytest.mark.asyncio
async def test_query_endpoint_validation_error(async_client):
    # Empty query should fail validation
    payload = {"query": "   "}
    response = await async_client.post("/api/v1/query", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_documents_endpoint(async_client):
    response = await async_client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert isinstance(data["documents"], list)


@pytest.mark.asyncio
async def test_feedback_submission_and_summary(async_client):
    feedback_payload = {
        "rating": "up",
        "query": "Test query for feedback",
        "answer": "Test answer",
        "comment": "Very accurate explanation!"
    }
    post_res = await async_client.post("/api/v1/feedback", json=feedback_payload)
    assert post_res.status_code == 201
    post_data = post_res.json()
    assert post_data["rating"] == "up"
    assert post_data["status"] == "success"

    # Get feedback summary
    summary_res = await async_client.get("/api/v1/feedback")
    assert summary_res.status_code == 200
    summary_data = summary_res.json()
    assert summary_data["total_feedback"] >= 1
    assert summary_data["positive_count"] >= 1


@pytest.mark.asyncio
async def test_graph_visualization_endpoint(async_client):
    response = await async_client.get("/api/v1/graph/visualize")
    assert response.status_code == 200
    assert len(response.text) > 0
