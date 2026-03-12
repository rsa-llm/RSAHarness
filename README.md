# RSAHarness

Standalone inference harness for **Recursive Self-Aggregation (RSA)** — a multi-step inference algorithm where a language model generates candidate solutions, then iteratively aggregates them to improve quality.

**Paper**: [Recursive Self-Aggregation for LLMs](https://arxiv.org/abs/2509.26626)

## What is RSA?

RSA is a multi-step inference algorithm that improves LLM output quality through iterative self-aggregation:

1. **Generate** a population of N candidate solutions
2. **Aggregate** by sampling K candidates, presenting them alongside the original problem, and generating an improved solution
3. **Repeat** for T steps, with optional island-based diversity

```
              RSA Loop (T=3, M=1, N=4, K=2)

Step 0:  Question --> Gen 0 --> "answer A"
(initial)         --> Gen 1 --> "answer B"
                  --> Gen 2 --> "answer C"
                  --> Gen 3 --> "answer D"

Step 1:  Agg(K=2) --> Gen 0' --> "refined A"
(aggregate)       --> Gen 1' --> "refined B"
                  --> Gen 2' --> "refined C"
                  --> Gen 3' --> "refined D"

Step 2:  Agg(K=2) --> Gen 0'' --> "final A"
(aggregate)       --> Gen 1'' --> "final B"
                  --> Gen 2'' --> "final C"
                  --> Gen 3'' --> "final D"
```

## Install

```bash
pip install rsa-harness
```

With adapter extras:

```bash
pip install "rsa-harness[vllm]"      # local vLLM inference
pip install "rsa-harness[openai]"    # OpenAI API
pip install "rsa-harness[anthropic]" # Anthropic API
pip install "rsa-harness[gemini]"    # Google Gemini API
pip install "rsa-harness[eval]"      # verifiers-based evaluation
pip install "rsa-harness[all]"       # everything
```

## Quick Start

### With vLLM (local GPU inference)

```python
from rsa import RSAEngine
from rsa.adapters import VLLMAdapter

adapter = VLLMAdapter(model="Qwen/Qwen3-4B-Instruct-2507", tp_size=4)
engine = RSAEngine(adapter=adapter, population=8, k=4, loops=10, islands=2, task="math")

results = engine.run(
    questions=["What is the sum of the first 100 positive integers?"],
    ground_truths=["5050"],
)
print(results["final_candidates"])
```

### With OpenAI API

```python
from rsa import RSAEngine
from rsa.adapters import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4o", temperature=0.7)
engine = RSAEngine(adapter=adapter, population=4, k=2, loops=4, task="math")

results = engine.run(
    questions=["Prove that sqrt(2) is irrational."],
)
```

### With Anthropic API

```python
from rsa import RSAEngine
from rsa.adapters import AnthropicAdapter

adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
engine = RSAEngine(adapter=adapter, population=4, k=2, loops=4, task="math")

results = engine.run(questions=["What is 2^10?"], ground_truths=["1024"])
```

### Evaluation with reasoning_gym (requires `[eval]` extra)

```python
from rsa.adapters import VLLMAdapter
from rsa.verifiers_eval import VerifiersEval

adapter = VLLMAdapter(model="Qwen/Qwen3-4B-Instruct-2507", tp_size=4)

# Single-pass evaluation
ev = VerifiersEval("countdown", adapter, num_eval=100)
result = ev.run()
print(f"Accuracy: {result['accuracy']:.1%}")

# RSA evaluation
result = ev.run_rsa(population=4, k=2, loops=4, islands=1)
```

## Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `population` (N) | Candidates per island | `8` |
| `k` (K) | Candidates sampled per aggregation step | `4` |
| `loops` (T) | Total RSA steps | `10` |
| `islands` (M) | Number of islands (must be power of 2) | `1` |
| `task` | Prompt format: `"math"`, `"rg"`, `"supergpqa"`, `"general"` | `"math"` |
| `seed` | Random seed | `1234` |
| `system_prompt` | Optional system prompt for chat models | `None` |

## Custom Adapters

Implement `BaseAdapter` to use any model backend:

```python
from rsa.adapters.base import BaseAdapter, GenerationResult

class MyAdapter(BaseAdapter):
    def generate_batch(self, prompts):
        # Your inference logic here
        return [GenerationResult(text=response) for response in responses]
```

## Citation

```bibtex
@article{venkatraman2025rsa,
  title={Recursive Self-Aggregation for LLMs},
  author={Venkatraman, Siddarth and Bae, Juhan and Pineau, Joelle},
  journal={arXiv preprint arXiv:2509.26626},
  year={2025}
}
```
