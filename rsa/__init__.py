"""RSA (Recursive Self-Aggregation) inference package.

Supports local vLLM inference and external API adapters (Claude, GPT, Gemini).
Includes island-based RSA with pairwise merging.

Usage:
    from rsa import RSAEngine
    from rsa.adapters import VLLMAdapter

    adapter = VLLMAdapter(model="Qwen/Qwen3-4B-Instruct-2507", tp_size=4)
    engine = RSAEngine(adapter=adapter, population=8, k=4, loops=10, islands=2)
    results = engine.run(questions=questions, ground_truths=ground_truths)
"""

from rsa.engine import RSAEngine
from rsa.islands import validate_island_params, get_num_islands, get_merge_schedule

__all__ = ["RSAEngine", "validate_island_params", "get_num_islands", "get_merge_schedule"]
