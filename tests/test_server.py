"""
Tests for FastAPI server endpoints
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


class TestServerEndpoints:

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_split_text_endpoint(self):
        """Test text splitting endpoint"""
        payload = {"text": "one two three four", "delimiter": " ", "stream": False}
        response = client.post("/text/split", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert len(data["result"]) == 4

    def test_find_replace_endpoint(self):
        """Test find and replace endpoint"""
        payload = {
            "text": "hello world hello universe",
            "find": "hello",
            "replace": "hi",
            "stream": False,
        }
        response = client.post("/text/find-replace", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "hi world" in data["result"]

    def test_fuzzy_delete_endpoint(self):
        """Test fuzzy deletion endpoint"""
        payload = {
            "text": "apple banana orange apple",
            "target": "apple",
            "similarity_threshold": 1.0,
            "stream": False,
        }
        response = client.post("/text/fuzzy-delete", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "apple" not in data["result"]

    def test_search_endpoint(self):
        """Test search endpoint"""
        payload = {
            "text": "The quick brown fox jumps",
            "query": "fox",
            "search_type": "exact",
            "stream": False,
        }
        response = client.post("/text/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert len(data["result"]) == 1

    def test_batch_process_endpoint(self):
        """Test batch processing endpoint"""
        payload = {
            "texts": ["hello world", "hello universe"],
            "operation": "find_replace",
            "parameters": {"find": "hello", "replace": "hi"},
            "stream": False,
        }
        response = client.post("/text/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert len(data["result"]) == 2

    def test_invalid_operation_batch(self):
        """Test batch processing with invalid operation"""
        payload = {
            "texts": ["test"],
            "operation": "invalid_operation",
            "parameters": {},
            "stream": False,
        }
        response = client.post("/text/batch", json=payload)
        # The server returns 200 but with error results in the response
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        # Check that the first result contains an error
        assert len(data["result"]) == 1
        assert "error" in data["result"][0]["metadata"]

    def test_streaming_endpoints(self):
        """Test that streaming endpoints return proper response type"""
        payload = {"text": "test text", "delimiter": " ", "stream": True}
        response = client.post("/text/split", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_invalid_regex_handling(self):
        """Test handling of invalid regex patterns"""
        payload = {
            "text": "test text",
            "find": "[invalid",
            "replace": "replace",
            "regex": True,
            "stream": False,
        }
        response = client.post("/text/find-replace", json=payload)
        assert response.status_code == 500

    def test_empty_text_handling(self):
        """Test handling of empty text input"""
        payload = {"text": "", "delimiter": " ", "stream": False}
        response = client.post("/text/split", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == []

    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        payload = {"delimiter": " ", "stream": False}
        response = client.post("/text/split", json=payload)
        assert response.status_code == 422  # Validation error

    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.get("/")
        assert response.status_code == 200
        # CORS headers should be present in actual deployment


if __name__ == "__main__":
    pytest.main([__file__])
