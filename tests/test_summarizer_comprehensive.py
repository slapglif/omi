"""
Comprehensive tests for memory summarization (MemorySummarizer, OllamaSummarizer)
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from omi.summarizer import (
    MemorySummarizer,
    OllamaSummarizer,
    LLMProvider,
    LLMConfig,
)


class TestLLMProvider:
    """Test LLMProvider enum"""

    def test_provider_values(self):
        """Test LLMProvider enum values"""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OLLAMA.value == "ollama"


class TestLLMConfig:
    """Test LLMConfig dataclass"""

    def test_config_creation(self):
        """Test creating LLMConfig"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test_key",
            model="gpt-4o-mini",
            timeout=30,
            max_tokens=500
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test_key"
        assert config.model == "gpt-4o-mini"
        assert config.timeout == 30
        assert config.max_tokens == 500


class TestMemorySummarizer:
    """Test MemorySummarizer functionality"""

    def test_init_with_openai(self):
        """Test initialization with OpenAI provider"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(
                provider="openai",
                api_key="test_key",
                model="gpt-4o-mini"
            )
            assert summarizer.provider == LLMProvider.OPENAI
            assert summarizer.api_key == "test_key"
            assert summarizer.model == "gpt-4o-mini"

    def test_init_with_anthropic(self):
        """Test initialization with Anthropic provider"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(
                provider="anthropic",
                api_key="test_key",
                model="claude-3-haiku-20240307"
            )
            assert summarizer.provider == LLMProvider.ANTHROPIC
            assert summarizer.api_key == "test_key"

    def test_init_with_ollama(self):
        """Test initialization with Ollama provider (no API key required)"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(
                provider="ollama",
                model="llama3.2:3b"
            )
            assert summarizer.provider == LLMProvider.OLLAMA
            assert summarizer.api_key == ""

    def test_init_missing_api_key_openai(self):
        """Test initialization fails without OpenAI API key"""
        with pytest.raises(ValueError, match="OPENAI_API_KEY required"):
            MemorySummarizer(provider="openai", api_key=None)

    def test_init_missing_api_key_anthropic(self):
        """Test initialization fails without Anthropic API key"""
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY required"):
            MemorySummarizer(provider="anthropic", api_key=None)

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment"""
        with patch('omi.summarizer.os.getenv', return_value="env_key"):
            with patch('omi.summarizer.requests'):
                summarizer = MemorySummarizer(provider="openai")
                assert summarizer.api_key == "env_key"

    def test_init_session_openai(self):
        """Test HTTP session initialization for OpenAI"""
        with patch('omi.summarizer.requests') as mock_requests:
            mock_session = MagicMock()
            mock_requests.Session.return_value = mock_session

            summarizer = MemorySummarizer(provider="openai", api_key="test_key")

            # Verify headers were set
            assert mock_session.headers.update.called
            call_args = mock_session.headers.update.call_args[0][0]
            assert "Authorization" in call_args
            assert call_args["Authorization"] == "Bearer test_key"

    def test_init_session_anthropic(self):
        """Test HTTP session initialization for Anthropic"""
        with patch('omi.summarizer.requests') as mock_requests:
            mock_session = MagicMock()
            mock_requests.Session.return_value = mock_session

            summarizer = MemorySummarizer(provider="anthropic", api_key="test_key")

            # Verify headers were set
            call_args = mock_session.headers.update.call_args[0][0]
            assert "x-api-key" in call_args
            assert call_args["x-api-key"] == "test_key"

    def test_build_summarization_prompt(self):
        """Test prompt building for summarization"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            prompt = summarizer._build_summarization_prompt(
                "Test memory content",
                metadata={"confidence": 0.9}
            )
            assert "Test memory content" in prompt
            assert "REQUIREMENTS" in prompt
            assert '"confidence": 0.9' in prompt

    def test_build_summarization_prompt_no_metadata(self):
        """Test prompt building without metadata"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            prompt = summarizer._build_summarization_prompt("Test content")
            assert "Test content" in prompt
            assert "SUMMARIZED MEMORY" in prompt

    def test_summarize_memory_openai(self):
        """Test summarize_memory with OpenAI"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Summarized content"
                }
            }]
        }

        with patch('omi.summarizer.requests') as mock_requests:
            mock_session = MagicMock()
            mock_session.post.return_value = mock_response
            mock_requests.Session.return_value = mock_session

            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            result = summarizer.summarize_memory("Original memory content")

            assert result == "Summarized content"
            mock_session.post.assert_called_once()

    def test_summarize_memory_anthropic(self):
        """Test summarize_memory with Anthropic"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{
                "text": "Anthropic summarized content"
            }]
        }

        with patch('omi.summarizer.requests') as mock_requests:
            mock_session = MagicMock()
            mock_session.post.return_value = mock_response
            mock_requests.Session.return_value = mock_session

            summarizer = MemorySummarizer(provider="anthropic", api_key="test_key")
            result = summarizer.summarize_memory("Original memory")

            assert result == "Anthropic summarized content"

    def test_summarize_memory_ollama(self):
        """Test summarize_memory with Ollama"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Ollama summarized content"
        }

        with patch('omi.summarizer.requests') as mock_requests:
            mock_session = MagicMock()
            mock_session.post.return_value = mock_response
            mock_requests.Session.return_value = mock_session

            summarizer = MemorySummarizer(provider="ollama")
            result = summarizer.summarize_memory("Original memory")

            assert result == "Ollama summarized content"

    def test_batch_summarize(self):
        """Test batch_summarize with multiple memories"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Summarized"
                }
            }]
        }

        with patch('omi.summarizer.requests') as mock_requests:
            mock_session = MagicMock()
            mock_session.post.return_value = mock_response
            mock_requests.Session.return_value = mock_session

            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            memories = ["Memory 1", "Memory 2", "Memory 3"]
            results = summarizer.batch_summarize(memories)

            assert len(results) == 3
            assert all(r == "Summarized" for r in results)

    def test_batch_summarize_with_metadata(self):
        """Test batch_summarize with metadata list"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Summarized"
                }
            }]
        }

        with patch('omi.summarizer.requests') as mock_requests:
            mock_session = MagicMock()
            mock_session.post.return_value = mock_response
            mock_requests.Session.return_value = mock_session

            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            memories = ["Memory 1", "Memory 2"]
            metadata = [{"id": 1}, {"id": 2}]
            results = summarizer.batch_summarize(memories, metadata_list=metadata)

            assert len(results) == 2

    def test_batch_summarize_metadata_length_mismatch(self):
        """Test batch_summarize fails with mismatched metadata length"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            memories = ["Memory 1", "Memory 2"]
            metadata = [{"id": 1}]  # Wrong length

            with pytest.raises(ValueError, match="metadata_list length"):
                summarizer.batch_summarize(memories, metadata_list=metadata)

    def test_estimate_tokens(self):
        """Test token estimation"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            text = "A" * 400  # 400 characters
            tokens = summarizer.estimate_tokens(text)
            assert tokens == 100  # 400 / 4

    def test_estimate_savings(self):
        """Test compression savings calculation"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            original = "A" * 1000  # 250 tokens
            summary = "A" * 400    # 100 tokens
            savings = summarizer.estimate_savings(original, summary)

            assert savings["original_tokens"] == 250
            assert savings["summary_tokens"] == 100
            assert savings["savings_percent"] == 60.0
            assert savings["tokens_saved"] == 150

    def test_estimate_savings_empty_original(self):
        """Test savings calculation with empty original"""
        with patch('omi.summarizer.requests'):
            summarizer = MemorySummarizer(provider="openai", api_key="test_key")
            savings = summarizer.estimate_savings("", "summary")
            assert savings["savings_percent"] == 0.0


class TestOllamaSummarizer:
    """Test OllamaSummarizer functionality"""

    def test_init_with_ollama_client(self):
        """Test initialization with ollama client library"""
        with patch('omi.summarizer.ollama') as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.Client.return_value = mock_client

            summarizer = OllamaSummarizer(
                model="llama3.2:3b",
                base_url="http://localhost:11434"
            )

            assert summarizer.model == "llama3.2:3b"
            assert summarizer._use_client is True
            mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")

    def test_init_without_ollama_client(self):
        """Test initialization fallback without ollama library"""
        with patch('omi.summarizer.ollama', side_effect=ImportError):
            with patch('omi.summarizer.requests') as mock_requests:
                mock_session = MagicMock()
                mock_requests.Session.return_value = mock_session

                summarizer = OllamaSummarizer()

                assert summarizer._use_client is False
                assert summarizer._session is not None

    def test_summarize_with_client(self):
        """Test summarization with ollama client"""
        with patch('omi.summarizer.ollama') as mock_ollama:
            mock_client = MagicMock()
            mock_client.generate.return_value = {"response": "Summarized content"}
            mock_ollama.Client.return_value = mock_client

            summarizer = OllamaSummarizer()
            result = summarizer.summarize("Original memory content")

            assert result == "Summarized content"
            mock_client.generate.assert_called_once()

    def test_summarize_without_client(self):
        """Test summarization without ollama client (HTTP fallback)"""
        with patch('omi.summarizer.ollama', side_effect=ImportError):
            with patch('omi.summarizer.requests') as mock_requests:
                mock_session = MagicMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"response": "HTTP summarized"}
                mock_session.post.return_value = mock_response
                mock_requests.Session.return_value = mock_session

                summarizer = OllamaSummarizer()
                result = summarizer.summarize("Original content")

                assert result == "HTTP summarized"
                mock_session.post.assert_called_once()

    def test_default_model(self):
        """Test default model is set correctly"""
        with patch('omi.summarizer.ollama', side_effect=ImportError):
            with patch('omi.summarizer.requests'):
                summarizer = OllamaSummarizer()
                assert summarizer.model == "llama3.2:3b"

    def test_custom_model(self):
        """Test custom model can be set"""
        with patch('omi.summarizer.ollama', side_effect=ImportError):
            with patch('omi.summarizer.requests'):
                summarizer = OllamaSummarizer(model="mistral")
                assert summarizer.model == "mistral"

    def test_custom_base_url(self):
        """Test custom base URL can be set"""
        with patch('omi.summarizer.ollama', side_effect=ImportError):
            with patch('omi.summarizer.requests'):
                summarizer = OllamaSummarizer(base_url="http://custom:8080")
                assert summarizer.base_url == "http://custom:8080"
