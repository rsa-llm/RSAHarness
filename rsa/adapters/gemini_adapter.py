"""Google Gemini API adapter for RSA."""

import os
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from rsa.adapters.base import BaseAdapter, GenerationResult

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseAdapter):
    """Google Gemini API adapter.

    Args:
        model: Model name (e.g. "gemini-2.5-pro").
        max_output_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        max_workers: Max concurrent API calls.
        api_key: API key. Defaults to GOOGLE_API_KEY env var.
        thinking: Optional thinking config dict.
        extra_kwargs: Additional kwargs for generation config.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        max_output_tokens: int = 16384,
        temperature: float = 1.0,
        max_workers: int = 50,
        api_key: Optional[str] = None,
        thinking: Optional[Dict[str, Any]] = None,
        **extra_kwargs,
    ):
        from google import genai

        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_workers = max_workers
        self.thinking = thinking
        self.extra_kwargs = extra_kwargs

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=key)

    def _call(self, prompt: str) -> GenerationResult:
        """Make a single API call."""
        from google.genai import types

        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        config_params = {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.thinking:
            config_params["thinking"] = self.thinking
        config_params.update(self.extra_kwargs)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_params),
        )

        text = getattr(response, "text", "") or ""
        metadata = {}
        usage = getattr(response, "usage_metadata", None)
        if usage:
            metadata["prompt_tokens"] = getattr(usage, "prompt_token_count", 0)
            metadata["completion_tokens"] = getattr(usage, "candidates_token_count", 0)
            metadata["total_tokens"] = getattr(usage, "total_token_count", 0)

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
                    logger.error(f"Gemini API error for prompt {idx}: {e}")
                    results[idx] = GenerationResult(text="", metadata={"error": str(e)})

        return results
