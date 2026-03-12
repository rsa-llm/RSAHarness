"""OpenAI (GPT) API adapter for RSA."""

import os
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from rsa.adapters.base import BaseAdapter, GenerationResult

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseAdapter):
    """OpenAI API adapter (GPT, o1, o3, etc.).

    Args:
        model: Model name (e.g. "gpt-4o", "o3").
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        max_workers: Max concurrent API calls.
        api_key: API key. Defaults to OPENAI_API_KEY env var.
        base_url: Optional base URL for OpenAI-compatible APIs.
        extra_kwargs: Additional kwargs passed to chat.completions.create().
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 16384,
        temperature: float = 1.0,
        max_workers: int = 50,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **extra_kwargs,
    ):
        from openai import OpenAI

        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_workers = max_workers
        self.extra_kwargs = extra_kwargs

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        client_kwargs = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def _call(self, prompt: str) -> GenerationResult:
        """Make a single API call."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        kwargs.update(self.extra_kwargs)

        response = self.client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        metadata = {}
        if response.usage:
            metadata["prompt_tokens"] = response.usage.prompt_tokens
            metadata["completion_tokens"] = response.usage.completion_tokens
            metadata["total_tokens"] = response.usage.total_tokens

        return GenerationResult(text=text, metadata=metadata)

    def generate_batch(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate responses concurrently."""
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts))) as executor:
            future_to_idx = {executor.submit(self._call, p): i for i, p in enumerate(prompts)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"OpenAI API error for prompt {idx}: {e}")
                    results[idx] = GenerationResult(text="", metadata={"error": str(e)})

        return results
