"""RSA model adapters for local and API-based inference."""

from rsa.adapters.base import BaseAdapter

__all__ = ["BaseAdapter"]

# Lazy imports to avoid requiring all dependencies
def VLLMAdapter(*args, **kwargs):
    from rsa.adapters.vllm_adapter import VLLMAdapter as _VLLMAdapter
    return _VLLMAdapter(*args, **kwargs)

def AnthropicAdapter(*args, **kwargs):
    from rsa.adapters.anthropic_adapter import AnthropicAdapter as _AnthropicAdapter
    return _AnthropicAdapter(*args, **kwargs)

def OpenAIAdapter(*args, **kwargs):
    from rsa.adapters.openai_adapter import OpenAIAdapter as _OpenAIAdapter
    return _OpenAIAdapter(*args, **kwargs)

def GeminiAdapter(*args, **kwargs):
    from rsa.adapters.gemini_adapter import GeminiAdapter as _GeminiAdapter
    return _GeminiAdapter(*args, **kwargs)
