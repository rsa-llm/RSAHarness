"""Abstract base adapter for RSA inference."""

import abc
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    """Result from a single generation call.

    Attributes:
        text: The generated text response.
        metadata: Optional dict of provider-specific metadata (tokens, cost, etc.).
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(abc.ABC):
    """Abstract base for all RSA inference adapters.

    Subclasses must implement generate_batch() which takes a list of prompts
    and returns a list of GenerationResult objects.

    The adapter handles the model-specific details (chat templates, API calls,
    batching) so the RSA engine only works with strings.
    """

    @abc.abstractmethod
    def generate_batch(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of prompt strings.

        Returns:
            List of GenerationResult, one per prompt, in the same order.
        """
        ...

    def generate(self, prompt: str) -> GenerationResult:
        """Generate a single response. Convenience wrapper around generate_batch."""
        return self.generate_batch([prompt])[0]

    def shutdown(self):
        """Clean up resources. Override if the adapter holds GPU memory or processes."""
        pass
