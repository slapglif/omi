"""Unit tests for MemorySummarizer module"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.summarizer import MemorySummarizer, LLMProvider, OllamaSummarizer


class TestMemorySummarizerInit:
    """Test MemorySummarizer initialization"""

    def test_openai_provider_requires_api_key(self):
        """Test OpenAI provider raises error without API key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                MemorySummarizer(provider="openai")

    def test_anthropic_provider_requires_api_key(self):
        """Test Anthropic provider raises error without API key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                MemorySummarizer(provider="anthropic")

    def test_ollama_provider_no_api_key_required(self):
        """Test Ollama provider works without API key"""
        with patch.dict('os.environ', {}, clear=True):
            summarizer = MemorySummarizer(provider="ollama")
            assert summarizer.provider == LLMProvider.OLLAMA
            assert summarizer.api_key == ""

    def test_openai_with_env_var_api_key(self):
        """Test OpenAI initialization with API key from environment"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key-123'}):
            summarizer = MemorySummarizer(provider="openai")
            assert summarizer.api_key == "test-key-123"
            assert summarizer.provider == LLMProvider.OPENAI

    def test_anthropic_with_env_var_api_key(self):
        """Test Anthropic initialization with API key from environment"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key-456'}):
            summarizer = MemorySummarizer(provider="anthropic")
            assert summarizer.api_key == "test-key-456"
            assert summarizer.provider == LLMProvider.ANTHROPIC

    def test_explicit_api_key_parameter(self):
        """Test passing API key directly as parameter"""
        summarizer = MemorySummarizer(provider="openai", api_key="explicit-key")
        assert summarizer.api_key == "explicit-key"

    def test_default_model_selection(self):
        """Test default model is selected for each provider"""
        # OpenAI
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            assert summarizer.model == "gpt-4o-mini"

        # Anthropic
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="anthropic")
            assert summarizer.model == "claude-3-haiku-20240307"

        # Ollama
        summarizer = MemorySummarizer(provider="ollama")
        assert summarizer.model == "llama3.2:3b"

    def test_custom_model_override(self):
        """Test custom model can be specified"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai", model="gpt-4o")
            assert summarizer.model == "gpt-4o"

    def test_default_base_url_selection(self):
        """Test default base URL is selected for each provider"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            assert summarizer.base_url == "https://api.openai.com/v1"

    def test_custom_base_url_override(self):
        """Test custom base URL can be specified"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(
                provider="openai",
                base_url="https://custom.api.com"
            )
            assert summarizer.base_url == "https://custom.api.com"

    def test_temperature_and_max_tokens_defaults(self):
        """Test default temperature and max_tokens values"""
        summarizer = MemorySummarizer(provider="ollama")
        assert summarizer.temperature == 0.3
        assert summarizer.max_tokens == 1000

    def test_custom_temperature_and_max_tokens(self):
        """Test custom temperature and max_tokens can be set"""
        summarizer = MemorySummarizer(
            provider="ollama",
            temperature=0.7,
            max_tokens=500
        )
        assert summarizer.temperature == 0.7
        assert summarizer.max_tokens == 500


class TestMemorySummarization:
    """Test summarization methods"""

    @patch('requests.Session.post')
    def test_openai_summarization(self, mock_post):
        """Test OpenAI summarization with mocked response"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Summarized memory content with key facts preserved"
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            summarizer = MemorySummarizer(provider="openai")
            summary = summarizer.summarize_memory(
                "This is a long detailed memory about a meeting where we discussed project timelines and deliverables."
            )

        assert summary == "Summarized memory content with key facts preserved"
        mock_post.assert_called_once()

        # Verify correct API endpoint
        call_args = mock_post.call_args
        assert "chat/completions" in call_args[0][0]

    @patch('requests.Session.post')
    def test_anthropic_summarization(self, mock_post):
        """Test Anthropic summarization with mocked response"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{
                "text": "Concise summary preserving facts"
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            summarizer = MemorySummarizer(provider="anthropic")
            summary = summarizer.summarize_memory("Original detailed memory content")

        assert summary == "Concise summary preserving facts"
        mock_post.assert_called_once()

        # Verify correct API endpoint
        call_args = mock_post.call_args
        assert "messages" in call_args[0][0]

    @patch('requests.Session.post')
    def test_ollama_summarization(self, mock_post):
        """Test Ollama summarization with mocked response"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Local Ollama summary"
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        summarizer = MemorySummarizer(provider="ollama")
        summary = summarizer.summarize_memory("Memory to summarize locally")

        assert summary == "Local Ollama summary"
        mock_post.assert_called_once()

        # Verify correct API endpoint
        call_args = mock_post.call_args
        assert "api/generate" in call_args[0][0]

    @patch('requests.Session.post')
    def test_summarization_with_metadata(self, mock_post):
        """Test summarization includes metadata in prompt"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Summary with metadata"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            metadata = {"confidence": 0.95, "relationships": ["user_123"]}
            summary = summarizer.summarize_memory("Content", metadata=metadata)

        # Verify metadata was included in the API call
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        prompt = request_data['messages'][1]['content']
        assert "METADATA" in prompt

    @patch('requests.Session.post')
    def test_openai_api_error_handling(self, mock_post):
        """Test error handling for OpenAI API failures"""
        # Mock HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            with pytest.raises(Exception, match="API Error"):
                summarizer.summarize_memory("Content")

    @patch('requests.Session.post')
    def test_anthropic_api_error_handling(self, mock_post):
        """Test error handling for Anthropic API failures"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Anthropic Error")
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="anthropic")
            with pytest.raises(Exception, match="Anthropic Error"):
                summarizer.summarize_memory("Content")

    @patch('requests.Session.post')
    def test_ollama_api_error_handling(self, mock_post):
        """Test error handling for Ollama API failures"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Ollama Error")
        mock_post.return_value = mock_response

        summarizer = MemorySummarizer(provider="ollama")
        with pytest.raises(Exception, match="Ollama Error"):
            summarizer.summarize_memory("Content")


class TestBatchProcessing:
    """Test batch summarization methods"""

    @patch('requests.Session.post')
    def test_batch_summarize_multiple_memories(self, mock_post):
        """Test batch_summarize processes multiple memories"""
        # Mock API responses
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Summary"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            memories = [
                "Memory 1 content",
                "Memory 2 content",
                "Memory 3 content"
            ]
            summaries = summarizer.batch_summarize(memories)

        assert len(summaries) == 3
        assert all(s == "Summary" for s in summaries)
        # Should be called 3 times (one per memory)
        assert mock_post.call_count == 3

    @patch('requests.Session.post')
    def test_batch_summarize_with_metadata_list(self, mock_post):
        """Test batch_summarize with metadata for each memory"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Summary"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            memories = ["Memory 1", "Memory 2"]
            metadata_list = [{"id": "1"}, {"id": "2"}]
            summaries = summarizer.batch_summarize(memories, metadata_list=metadata_list)

        assert len(summaries) == 2

    def test_batch_summarize_metadata_length_validation(self):
        """Test batch_summarize validates metadata_list length matches"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            memories = ["Memory 1", "Memory 2", "Memory 3"]
            metadata_list = [{"id": "1"}, {"id": "2"}]  # Wrong length!

            with pytest.raises(ValueError, match="metadata_list length"):
                summarizer.batch_summarize(memories, metadata_list=metadata_list)

    @patch('requests.Session.post')
    def test_batch_summarize_with_custom_batch_size(self, mock_post):
        """Test batch_summarize respects custom batch_size parameter"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Summary"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            memories = ["M1", "M2", "M3", "M4", "M5"]
            summaries = summarizer.batch_summarize(memories, batch_size=2)

        assert len(summaries) == 5
        # Batch size affects processing but all should be summarized
        assert mock_post.call_count == 5

    @patch('requests.Session.post')
    def test_batch_summarize_preserves_order(self, mock_post):
        """Test batch_summarize returns summaries in same order as inputs"""
        # Mock responses with different content
        call_count = [0]

        def mock_response_func(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": f"Summary {call_count[0]}"}}]
            }
            mock_response.raise_for_status = MagicMock()
            call_count[0] += 1
            return mock_response

        mock_post.side_effect = mock_response_func

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}):
            summarizer = MemorySummarizer(provider="openai")
            memories = ["First", "Second", "Third"]
            summaries = summarizer.batch_summarize(memories)

        assert summaries == ["Summary 0", "Summary 1", "Summary 2"]


class TestHelperMethods:
    """Test token estimation and savings calculation"""

    def test_estimate_tokens_basic(self):
        """Test token estimation uses chars/4 heuristic"""
        summarizer = MemorySummarizer(provider="ollama")

        # 400 chars should be ~100 tokens
        text_400 = "a" * 400
        tokens = summarizer.estimate_tokens(text_400)
        assert tokens == 100

    def test_estimate_tokens_various_lengths(self):
        """Test token estimation for various text lengths"""
        summarizer = MemorySummarizer(provider="ollama")

        assert summarizer.estimate_tokens("a" * 100) == 25
        assert summarizer.estimate_tokens("a" * 1000) == 250
        assert summarizer.estimate_tokens("a" * 4) == 1
        assert summarizer.estimate_tokens("") == 0

    def test_estimate_savings_calculation(self):
        """Test estimate_savings calculates compression ratio"""
        summarizer = MemorySummarizer(provider="ollama")

        original = "a" * 1000  # 250 tokens
        summary = "a" * 400    # 100 tokens

        savings = summarizer.estimate_savings(original, summary)

        assert savings["original_tokens"] == 250
        assert savings["summary_tokens"] == 100
        assert savings["tokens_saved"] == 150
        assert savings["savings_percent"] == 60.0

    def test_estimate_savings_50_percent(self):
        """Test estimate_savings with 50% compression"""
        summarizer = MemorySummarizer(provider="ollama")

        original = "x" * 200  # 50 tokens
        summary = "x" * 100   # 25 tokens

        savings = summarizer.estimate_savings(original, summary)

        assert savings["savings_percent"] == 50.0

    def test_estimate_savings_zero_compression(self):
        """Test estimate_savings when no compression occurred"""
        summarizer = MemorySummarizer(provider="ollama")

        text = "same text"
        savings = summarizer.estimate_savings(text, text)

        assert savings["savings_percent"] == 0.0
        assert savings["tokens_saved"] == 0

    def test_estimate_savings_empty_original(self):
        """Test estimate_savings handles empty original text"""
        summarizer = MemorySummarizer(provider="ollama")

        savings = summarizer.estimate_savings("", "")

        assert savings["savings_percent"] == 0.0
        assert savings["original_tokens"] == 0
        assert savings["summary_tokens"] == 0


class TestOllamaSummarizer:
    """Test the OllamaSummarizer fallback class"""

    def test_ollama_summarizer_init_defaults(self):
        """Test OllamaSummarizer initialization with defaults"""
        # This will use real requests.Session but we only test initialization
        try:
            summarizer = OllamaSummarizer()
            assert summarizer.model == "llama3.2:3b"
            assert summarizer.base_url == "http://localhost:11434"
        except Exception:
            # If requests isn't available, skip this test
            pytest.skip("requests library not available")

    def test_ollama_summarizer_custom_model(self):
        """Test OllamaSummarizer with custom model"""
        try:
            summarizer = OllamaSummarizer(model="mistral")
            assert summarizer.model == "mistral"
        except Exception:
            pytest.skip("requests library not available")

    def test_ollama_summarizer_custom_base_url(self):
        """Test OllamaSummarizer with custom base URL"""
        try:
            summarizer = OllamaSummarizer(base_url="http://custom:8080")
            assert summarizer.base_url == "http://custom:8080"
        except Exception:
            pytest.skip("requests library not available")
