"""vLLM-based local inference adapter for RSA.

Designed for vLLM >= 0.14 which has native data_parallel_size support
via LLM(**kwargs). No manual DP worker management needed.
"""

from typing import List, Union

from rsa.adapters.base import BaseAdapter, GenerationResult


class VLLMAdapter(BaseAdapter):
    """Local vLLM inference adapter.

    Uses vLLM's native data_parallel_size (>= 0.14) for multi-replica inference.

    Args:
        model: HuggingFace model name or path.
        tp_size: Tensor parallel size (GPUs per model replica).
        dp_size: Data parallel size (number of model replicas).
            Total GPUs used = tp_size * dp_size.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        dtype: Model dtype ("bfloat16", "float16", "auto").
        seed: Random seed for vLLM.
        extra_kwargs: Additional kwargs passed to vllm.LLM().
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        tp_size: int = 4,
        dp_size: int = 1,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        dtype: str = "bfloat16",
        seed: int = 1234,
        **extra_kwargs,
    ):
        self.model_name = model
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dtype = dtype
        self.seed = seed
        self.extra_kwargs = extra_kwargs

        self._tokenizer = None
        self._llm = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy init: load model and tokenizer on first use."""
        if self._initialized:
            return

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        from vllm import LLM
        llm_kwargs = dict(
            model=self.model_name,
            tensor_parallel_size=self.tp_size,
            dtype=self.dtype,
            trust_remote_code=True,
            seed=self.seed,
            **self.extra_kwargs,
        )
        if self.dp_size > 1:
            llm_kwargs["data_parallel_size"] = self.dp_size

        self._llm = LLM(**llm_kwargs)
        self._initialized = True

    def _apply_chat_template(self, prompt: Union[str, List[dict]]) -> str:
        """Apply the model's chat template to a prompt.

        Args:
            prompt: Either a plain string (wrapped in a user message) or
                a list of chat message dicts (e.g. [{"role": "system", ...},
                {"role": "user", ...}]) passed directly to the template.
        """
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_batch(self, prompts: List[Union[str, List[dict]]]) -> List[GenerationResult]:
        """Generate responses for a batch of prompts using vLLM.

        Args:
            prompts: List where each element is either a plain string or a list
                of chat message dicts. Strings are wrapped in a user message;
                chat message lists are passed directly to the chat template.
        """
        self._ensure_initialized()

        templated = [self._apply_chat_template(p) for p in prompts]

        from vllm import SamplingParams
        sampling = SamplingParams(
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        outs = self._llm.generate(templated, sampling)
        texts = [o.text for out in outs for o in out.outputs]

        return [GenerationResult(text=t) for t in texts]

    def shutdown(self):
        """Clean up GPU resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._initialized = False
