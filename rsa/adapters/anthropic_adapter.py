"""Anthropic (Claude) API adapter for RSA."""

import os
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from rsa.adapters.base import BaseAdapter, GenerationResult

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    """Anthropic Claude API adapter.

    Args:
        model: Model name (e.g. "claude-sonnet-4-20250514").
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        thinking: Optional thinking config dict (e.g. {"type": "enabled", "budget_tokens": 10000}).
        max_workers: Max concurrent API calls for batch generation.
        api_key: API key. Defaults to ANTHROPIC_API_KEY env var.
        extra_kwargs: Additional kwargs passed to messages.create().
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 16384,
        temperature: float = 1.0,
        thinking: Optional[Dict[str, Any]] = None,
        max_workers: int = 50,
        api_key: Optional[str] = None,
        **extra_kwargs,
    ):
        import anthropic

        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.max_workers = max_workers
        self.extra_kwargs = extra_kwargs

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)

    def _call(self, prompt: str) -> GenerationResult:
        """Make a single API call."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model_name, "messages": messages, "max_tokens": self.max_tokens}

        if self.thinking:
            kwargs["thinking"] = self.thinking
        else:
            kwargs["temperature"] = self.temperature

        kwargs.update(self.extra_kwargs)

        response = self.client.messages.create(**kwargs)

        # Extract text (skip thinking blocks)
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        metadata = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return GenerationResult(text=text, metadata=metadata)

    def generate_batch(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate responses for a batch of prompts using concurrent API calls."""
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts))) as executor:
            future_to_idx = {executor.submit(self._call, p): i for i, p in enumerate(prompts)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Anthropic API error for prompt {idx}: {e}")
                    results[idx] = GenerationResult(text="", metadata={"error": str(e)})

        return results
