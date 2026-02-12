"""
Memory Summarization via LLM APIs (OpenAI/Anthropic)
Pattern: Cloud API with configurable providers, consistent with embeddings.py setup
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum


def load_compression_config(base_path: Union[str, Path]) -> Dict:
    """Load compression configuration from config.yaml.

    Args:
        base_path: Base directory containing config.yaml

    Returns:
        Dictionary containing compression configuration.
        Returns empty dict if config file doesn't exist or parsing fails.
    """
    if isinstance(base_path, str):
        base_path = Path(base_path)

    config_path = base_path / "config.yaml"

    if not config_path.exists():
        return {}

    try:
        import yaml
        config = yaml.safe_load(config_path.read_text())
        if config and isinstance(config, dict):
            return config.get('compression', {})
        return {}
    except Exception:
        return {}


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

    DEFAULT_MODELS = {
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",
        LLMProvider.OLLAMA: "llama3.2:3b"
    }

    DEFAULT_BASE_URLS = {
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
        self.provider = provider

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

        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URLS[self.provider]
        self.model = model or self.DEFAULT_MODELS[self.provider]
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize HTTP session
        self._session = None
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
                        metadata: Optional[Dict] = None) -> str:
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
                       metadata_list: Optional[List[Optional[Dict]]] = None,
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
        results = []

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
                                    metadata: Optional[Dict] = None) -> str:
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

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

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

        data = response.json()
        return data["content"][0]["text"].strip()

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

        data = response.json()
        return data["response"].strip()

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

    def compress_session_memories(self,
                                  memories: List[Dict],
                                  config: Optional[Dict] = None) -> Dict[str, Union[int, float, List]]:
        """
        Compress a batch of session memories for tiered storage

        This is the main entry point for the compression pipeline workflow.
        Preserves original memories in Graph Palace while creating compressed
        summaries for daily logs and NOW.md.

        Args:
            memories: List of memory dicts with 'content' and optional metadata
            config: Optional compression config (uses defaults if not provided)

        Returns:
            Dict with:
                - original_tokens: Total tokens before compression
                - compressed_tokens: Total tokens after compression
                - savings_percent: Percentage of tokens saved
                - compressed_memories: List of dicts with compressed content and metadata
                - count: Number of memories processed

        Example:
            >>> memories = [
            ...     {"content": "Long memory text...", "type": "fact"},
            ...     {"content": "Another verbose memory...", "type": "experience"}
            ... ]
            >>> result = summarizer.compress_session_memories(memories)
            >>> print(f"Saved {result['savings_percent']}% tokens")
        """
        if not memories:
            return {
                "original_tokens": 0,
                "compressed_tokens": 0,
                "savings_percent": 0.0,
                "compressed_memories": [],
                "count": 0
            }

        # Extract content and metadata from memory dicts
        memory_contents = []
        metadata_list = []
        for mem in memories:
            content = mem.get("content", "")
            memory_contents.append(content)

            # Build metadata dict from memory fields (excluding content)
            metadata = {k: v for k, v in mem.items() if k != "content"}
            metadata_list.append(metadata if metadata else None)

        # Batch compress using existing method
        compressed_contents = self.batch_summarize(
            memory_contents,
            metadata_list=metadata_list,
            batch_size=config.get("batch_size", 8) if config else 8
        )

        # Calculate total compression stats
        total_original_tokens = sum(self.estimate_tokens(c) for c in memory_contents)
        total_compressed_tokens = sum(self.estimate_tokens(c) for c in compressed_contents)

        if total_original_tokens == 0:
            savings_percent = 0.0
        else:
            savings_percent = (1 - total_compressed_tokens / total_original_tokens) * 100

        # Build compressed memory dicts with original metadata
        compressed_memories = []
        for i, compressed_content in enumerate(compressed_contents):
            compressed_mem = memories[i].copy()
            compressed_mem["content"] = compressed_content
            compressed_mem["_original_tokens"] = self.estimate_tokens(memory_contents[i])
            compressed_mem["_compressed_tokens"] = self.estimate_tokens(compressed_content)
            compressed_memories.append(compressed_mem)

        return {
            "original_tokens": total_original_tokens,
            "compressed_tokens": total_compressed_tokens,
            "savings_percent": round(savings_percent, 1),
            "compressed_memories": compressed_memories,
            "count": len(memories)
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
        self.model = model
        self.base_url = base_url

        try:
            import ollama
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
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            return response['response'].strip()
        else:
            import requests
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()['response'].strip()
