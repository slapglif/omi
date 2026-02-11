"""
test_nim_unit.py - Unit tests for NVIDIA NIM batch API (mock-based, no API key required)

These tests use mocks and do NOT require NIM_API_KEY.
They verify the batch API call format and chunking logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Skip entire module if dependencies unavailable
pytest.importorskip("requests")
pytest.importorskip("numpy")

from omi.embeddings import NIMEmbedder

# Constants
REQUIRED_DIM = 1024


class TestNIMBatchAPIUnit:
    """Unit tests for batch API - use mocks, no API key needed"""

    def test_nim_batch_api_format(self):
        """
        Unit test: Verify batch API request format (mocked, no real API key needed)

        Tests that _embed_nim sends the correct batch format:
        - input: list of strings (not individual strings)
        - encoding_format: "float"
        - model: correct model name
        """
        # Mock the requests.Session and connection test
        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            # Mock successful connection test
            mock_test_response = Mock()
            mock_test_response.status_code = 200
            mock_session.post.return_value = mock_test_response

            # Create embedder with dummy key (mocked, won't actually call API)
            embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=False)

            # Now mock the actual embedding call
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1] * REQUIRED_DIM},
                    {"embedding": [0.2] * REQUIRED_DIM},
                    {"embedding": [0.3] * REQUIRED_DIM}
                ]
            }

            mock_session.post.return_value = mock_response
            mock_session.post.reset_mock()  # Reset call count after init

            texts = ["text1", "text2", "text3"]
            result = embedder._embed_nim(texts)

            # Verify API was called once with batch format
            assert mock_session.post.call_count == 1

            # Extract the call arguments
            call_args = mock_session.post.call_args
            json_payload = call_args.kwargs['json']

            # Verify batch API format
            assert json_payload["model"] == "baai/bge-m3", "Should use correct model"
            assert isinstance(json_payload["input"], list), "Input should be a list"
            assert len(json_payload["input"]) == 3, "Should send all texts in batch"
            assert json_payload["input"] == texts, "Should send exact texts"
            assert json_payload["encoding_format"] == "float", "Should request float encoding"

            # Verify result format
            assert isinstance(result, list), "Should return list for batch input"
            assert len(result) == 3, "Should return 3 embeddings"
            assert all(len(emb) == REQUIRED_DIM for emb in result), "All embeddings should have correct dimension"

    def test_nim_batch_api_single_vs_batch(self):
        """
        Unit test: Verify _embed_nim handles single string differently from batch

        Single string: returns single embedding (List[float])
        Batch (list): returns list of embeddings (List[List[float]])
        """
        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            # Mock successful connection test
            mock_test_response = Mock()
            mock_test_response.status_code = 200
            mock_session.post.return_value = mock_test_response

            embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=False)

            # Mock response for single string
            mock_response_single = Mock()
            mock_response_single.status_code = 200
            mock_response_single.json.return_value = {
                "data": [{"embedding": [0.1] * REQUIRED_DIM}]
            }

            mock_session.post.return_value = mock_response_single
            mock_session.post.reset_mock()

            # Test single string input
            result_single = embedder._embed_nim("single text")

            # Verify single string returns List[float], not List[List[float]]
            assert isinstance(result_single, list), "Should return list"
            assert isinstance(result_single[0], float), "Should return List[float] for single input"
            assert len(result_single) == REQUIRED_DIM, "Should have correct dimensions"

            # Verify API received list (even for single input)
            json_payload = mock_session.post.call_args.kwargs['json']
            assert json_payload["input"] == ["single text"], "Should send single text as list to API"

            # Mock response for batch
            mock_response_batch = Mock()
            mock_response_batch.status_code = 200
            mock_response_batch.json.return_value = {
                "data": [
                    {"embedding": [0.1] * REQUIRED_DIM},
                    {"embedding": [0.2] * REQUIRED_DIM}
                ]
            }

            mock_session.post.return_value = mock_response_batch
            mock_session.post.reset_mock()

            # Test batch input
            result_batch = embedder._embed_nim(["text1", "text2"])

            # Verify batch returns List[List[float]]
            assert isinstance(result_batch, list), "Should return list"
            assert isinstance(result_batch[0], list), "Should return List[List[float]] for batch input"
            assert len(result_batch) == 2, "Should return 2 embeddings"

            # Verify API received batch as-is
            json_payload = mock_session.post.call_args.kwargs['json']
            assert json_payload["input"] == ["text1", "text2"], "Should send batch as list to API"

    def test_nim_batch_api_empty_batch(self):
        """
        Unit test: Verify handling of edge case - empty batch
        """
        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            # Mock successful connection test
            mock_test_response = Mock()
            mock_test_response.status_code = 200
            mock_session.post.return_value = mock_test_response

            embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=False)

            # Mock response for empty batch
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}

            mock_session.post.return_value = mock_response
            mock_session.post.reset_mock()

            result = embedder._embed_nim([])

            # Empty batch should return empty list
            assert result == [], "Empty batch should return empty list"

    def test_nim_embed_batch_chunking(self):
        """
        Unit test: Verify embed_batch chunks correctly with batch_size parameter
        """
        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            # Mock successful connection test
            mock_test_response = Mock()
            mock_test_response.status_code = 200
            mock_session.post.return_value = mock_test_response

            embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=False)

            # Mock _embed_nim to track batch sizes
            batches_received = []

            def mock_embed_nim(texts):
                batches_received.append(len(texts))
                return [[0.1] * REQUIRED_DIM for _ in texts]

            embedder._embed_nim = mock_embed_nim

            # Test with 7 texts and batch_size=3
            texts = [f"text{i}" for i in range(7)]
            result = embedder.embed_batch(texts, batch_size=3)

            # Should make 3 batches: [3, 3, 1]
            assert batches_received == [3, 3, 1], f"Expected batches [3, 3, 1], got {batches_received}"
            assert len(result) == 7, "Should return all 7 embeddings"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
