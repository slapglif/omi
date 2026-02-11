"""
Memory Summarization via LLM APIs (OpenAI/Anthropic)
Pattern: Cloud API with configurable providers, consistent with embeddings.py setup
"""

import os
import json
from typing import Optional, Dict, List, Union, Any
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """LLM configuration for memory summarization"""
    provider: LLMProvider
    api_key: str
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    max_tokens: int = 1000
    temperature: float = 0.3  # Lower for consistent, factual summaries


class MemorySummarizer:
    """
    LLM-powered memory summarization

    Providers:
    - OpenAI (gpt-4o-mini, gpt-4o)
    - Anthropic (claude-3-haiku, claude-3-5-sonnet)
    - Ollama (local fallback)

    Summarization preserves:
    - Key facts and information
    - Confidence levels
    - Relationship links
    - Semantic meaning

    Usage:
        summarizer = MemorySummarizer(provider="openai")
        summary = summarizer.summarize_memory(memory_content)
    """

    DEFAULT_MODELS: Dict[LLMProvider, str] = {
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",
        LLMProvider.OLLAMA: "llama3.2:3b"
    }

    DEFAULT_BASE_URLS: Dict[LLMProvider, str] = {
        LLMProvider.OPENAI: "https://api.openai.com/v1",
        LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1",
        LLMProvider.OLLAMA: "http://localhost:11434"
    }

    def __init__(self,
                 provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 1000):
        """
        Args:
            provider: LLM provider (openai, anthropic, ollama)
            api_key: API key (or OPENAI_API_KEY/ANTHROPIC_API_KEY env var)
            base_url: API endpoint (optional, uses defaults)
            model: Model name (optional, uses defaults)
            temperature: Sampling temperature (0.0-1.0, lower = more consistent)
            max_tokens: Maximum tokens in response
        """
        # Parse provider
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        self.provider: LLMProvider = provider

        # Get API key from env if not provided
        if api_key is None:
            if self.provider == LLMProvider.OPENAI:
                api_key = os.getenv("OPENAI_API_KEY", "")
            elif self.provider == LLMProvider.ANTHROPIC:
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
            elif self.provider == LLMProvider.OLLAMA:
                api_key = ""  # Ollama doesn't require API key

        # Validate API key for cloud providers
        if self.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC] and not api_key:
            raise ValueError(
                f"{self.provider.value.upper()}_API_KEY required or pass api_key parameter"
            )

        self.api_key: str = api_key
        self.base_url: str = base_url or self.DEFAULT_BASE_URLS[self.provider]
        self.model: str = model or self.DEFAULT_MODELS[self.provider]
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens

        # Initialize HTTP session
        self._session: Any  # requests.Session
        self._init_session()

    def _init_session(self) -> None:
        """Initialize HTTP session with provider-specific headers"""
        import requests
        self._session = requests.Session()

        if self.provider == LLMProvider.OPENAI:
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        elif self.provider == LLMProvider.ANTHROPIC:
            self._session.headers.update({
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            })
        elif self.provider == LLMProvider.OLLAMA:
            self._session.headers.update({
                "Content-Type": "application/json"
            })

    def summarize_memory(self,
                        memory_content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Summarize a single memory, preserving key facts

        Args:
            memory_content: Original memory content to summarize
            metadata: Optional metadata (confidence, relationships, etc.)

        Returns:
            Summarized memory content
        """
        # Build prompt
        prompt = self._build_summarization_prompt(memory_content, metadata)

        # Call appropriate API
        if self.provider == LLMProvider.OPENAI:
            return self._summarize_openai(prompt)
        elif self.provider == LLMProvider.ANTHROPIC:
            return self._summarize_anthropic(prompt)
        elif self.provider == LLMProvider.OLLAMA:
            return self._summarize_ollama(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def batch_summarize(self,
                       memory_contents: List[str],
                       metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
                       batch_size: int = 8) -> List[str]:
        """
        Summarize multiple memories efficiently

        Args:
            memory_contents: List of memory contents to summarize
            metadata_list: Optional list of metadata dicts (same length as memory_contents)
            batch_size: Number of memories to process per batch

        Returns:
            List of summarized memory contents
        """
        results: List[str] = []

        # Handle metadata list - create None list if not provided
        if metadata_list is None:
            metadata_list = [None] * len(memory_contents)

        # Validate metadata_list length
        if len(metadata_list) != len(memory_contents):
            raise ValueError(
                f"metadata_list length ({len(metadata_list)}) must match "
                f"memory_contents length ({len(memory_contents)})"
            )

        # Process in batches
        for i in range(0, len(memory_contents), batch_size):
            batch_contents = memory_contents[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]

            # Summarize each memory in the batch
            batch_results = [
                self.summarize_memory(content, metadata)
                for content, metadata in zip(batch_contents, batch_metadata)
            ]
            results.extend(batch_results)

        return results

    def _build_summarization_prompt(self,
                                    content: str,
                                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Build prompt for memory summarization

        Prompt engineering goals:
        1. Preserve key facts and information
        2. Maintain confidence levels
        3. Note relationship links
        4. Compress verbosity while keeping semantic meaning
        5. Use structured format for consistency
        """
        prompt = f"""Summarize the following memory while preserving all key information.

REQUIREMENTS:
- Keep all factual details, dates, names, and technical specifics
- Maintain confidence levels and certainty markers
- Note any relationship links or references
- Compress verbose descriptions while preserving semantic meaning
- Use concise, structured format

MEMORY TO SUMMARIZE:
{content}
"""

        if metadata:
            prompt += f"\nMETADATA: {json.dumps(metadata)}\n"

        prompt += """
SUMMARIZED MEMORY (concise but complete):"""

        return prompt

    def _summarize_openai(self, prompt: str) -> str:
        """Summarize using OpenAI API"""
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise memory compression assistant. Summarize memories concisely while preserving all key facts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
            timeout=60
        )
        response.raise_for_status()

        data: Any = response.json()
        result: str = data["choices"][0]["message"]["content"].strip()
        return result

    def _summarize_anthropic(self, prompt: str) -> str:
        """Summarize using Anthropic API"""
        response = self._session.post(
            f"{self.base_url}/messages",
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": "You are a precise memory compression assistant. Summarize memories concisely while preserving all key facts.",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=60
        )
        response.raise_for_status()

        data: Any = response.json()
        result: str = data["content"][0]["text"].strip()
        return result

    def _summarize_ollama(self, prompt: str) -> str:
        """Summarize using Ollama (local)"""
        response = self._session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            timeout=60
        )
        response.raise_for_status()

        data: Any = response.json()
        result: str = data["response"].strip()
        return result

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Rough approximation: 1 token ≈ 4 characters
        This is conservative and works across most models
        """
        return len(text) // 4

    def estimate_savings(self, original: str, summary: str) -> Dict[str, Union[int, float]]:
        """
        Calculate compression savings

        Returns:
            Dict with original_tokens, summary_tokens, savings_percent
        """
        original_tokens = self.estimate_tokens(original)
        summary_tokens = self.estimate_tokens(summary)

        if original_tokens == 0:
            savings_percent = 0.0
        else:
            savings_percent = (1 - summary_tokens / original_tokens) * 100

        return {
            "original_tokens": original_tokens,
            "summary_tokens": summary_tokens,
            "savings_percent": round(savings_percent, 1),
            "tokens_saved": original_tokens - summary_tokens
        }


class OllamaSummarizer:
    """
    Local Ollama fallback for summarization

    Models:
    - llama3.2:3b (fast, good quality)
    - mistral (alternative)
    - qwen2.5:7b (higher quality)
    """

    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = "http://localhost:11434"):
        self.model: str = model
        self.base_url: str = base_url
        self.client: Any
        self._use_client: bool
        self._session: Any  # requests.Session

        try:
            import ollama  # type: ignore[import-not-found]
            self.client = ollama.Client(host=base_url)
            self._use_client = True
        except ImportError:
            import requests
            self._use_client = False
            self._session = requests.Session()

    def summarize(self, content: str) -> str:
        """Summarize using Ollama"""
        prompt = f"""Summarize this memory concisely while preserving key facts:

{content}

SUMMARY:"""

        if self._use_client:
            response: Any = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            result: str = response['response'].strip()
            return result
        else:
            import requests
            response_http: Any = self._session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response_http.raise_for_status()
            data: Any = response_http.json()
            result_str: str = data['response'].strip()
            return result_str
