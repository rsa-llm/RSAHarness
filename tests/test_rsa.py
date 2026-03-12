"""Unit tests for RSA package - no GPU or API keys required."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rsa.islands import validate_island_params, get_num_islands, get_merge_schedule
from rsa.prompts import build_prompt, aggregate_prompt
from rsa.evaluation import (
    last_boxed_only_string, remove_boxed, is_equiv,
    evaluate_candidates_math, evaluate_step,
)
from rsa.adapters.base import BaseAdapter, GenerationResult
from rsa.engine import RSAEngine


# ---- Island logic tests ----

def test_validate_params_valid():
    """Valid configs should not raise."""
    validate_island_params(1, 4, 2, 5)   # M=1, any T
    validate_island_params(2, 4, 4, 6)   # M=2, T divisible by 2
    validate_island_params(4, 4, 2, 6)   # M=4, T divisible by 3
    validate_island_params(8, 4, 2, 8)   # M=8, T divisible by 4
    print("PASS: test_validate_params_valid")


def test_validate_params_invalid():
    """Invalid configs should raise ValueError."""
    errors = []
    try:
        validate_island_params(3, 4, 2, 6)  # M not power of 2
        errors.append("M=3 should fail")
    except ValueError:
        pass
    try:
        validate_island_params(4, 4, 5, 6)  # K > N
        errors.append("K>N should fail")
    except ValueError:
        pass
    try:
        validate_island_params(4, 4, 2, 5)  # T not divisible by 3
        errors.append("T=5,M=4 should fail")
    except ValueError:
        pass
    if errors:
        raise AssertionError(f"Failures: {errors}")
    print("PASS: test_validate_params_invalid")


def test_get_num_islands():
    """Test island count at each step."""
    # M=4, T=6 -> phases of length 2, merges at 2 and 4
    assert get_num_islands(0, 4, 6) == 4
    assert get_num_islands(1, 4, 6) == 4
    assert get_num_islands(2, 4, 6) == 2
    assert get_num_islands(3, 4, 6) == 2
    assert get_num_islands(4, 4, 6) == 1
    assert get_num_islands(5, 4, 6) == 1
    # M=1
    assert get_num_islands(0, 1, 5) == 1
    print("PASS: test_get_num_islands")


def test_merge_schedule():
    """Test merge event schedule."""
    schedule = get_merge_schedule(4, 6)
    assert len(schedule) == 2
    assert schedule[0] == {"at_step": 2, "from": 4, "to": 2}
    assert schedule[1] == {"at_step": 4, "from": 2, "to": 1}
    assert get_merge_schedule(1, 5) == []
    print("PASS: test_merge_schedule")


# ---- Prompt tests ----

def test_build_prompt_no_candidates():
    """First step: just the question."""
    p = build_prompt("What is 2+2?", None, "math")
    assert p == "What is 2+2?"
    print("PASS: test_build_prompt_no_candidates")


def test_build_prompt_with_candidates():
    """Aggregation step: includes candidates."""
    p = build_prompt("What is 2+2?", ["Answer: 4", "Answer: 5"], "math")
    assert "Candidate solutions" in p
    assert "Solution 1" in p
    assert "Solution 2" in p
    assert "\\boxed{}" in p
    print("PASS: test_build_prompt_with_candidates")


def test_build_prompt_single_candidate():
    """Single candidate triggers refinement prompt."""
    p = build_prompt("What is 2+2?", ["Answer: 4"], "math")
    assert "Candidate solution (may contain mistakes)" in p
    assert "Refine" in p
    print("PASS: test_build_prompt_single_candidate")


# ---- Evaluation tests ----

def test_boxed_extraction():
    """Test \\boxed{} extraction."""
    assert remove_boxed(last_boxed_only_string("The answer is \\boxed{42}")) == "42"
    assert remove_boxed(last_boxed_only_string("\\boxed{x^2}")) == "x^2"
    assert last_boxed_only_string("no boxed here") is None
    # Nested braces
    assert remove_boxed(last_boxed_only_string("\\boxed{\\frac{1}{2}}")) == "\\frac{1}{2}"
    print("PASS: test_boxed_extraction")


def test_is_equiv():
    """Test answer equivalence checking."""
    assert is_equiv("42", "42")
    assert is_equiv("42", "42.0")
    assert is_equiv("1/2", "0.5")
    assert is_equiv("1,000", "1000")
    assert not is_equiv("42", "43")
    assert not is_equiv("hello", "world")
    print("PASS: test_is_equiv")


def test_evaluate_candidates_math():
    """Test math candidate evaluation."""
    candidates = [
        "The answer is \\boxed{42}",
        "I think it's \\boxed{43}",
        "So we get \\boxed{42}",
        "Final answer: \\boxed{42}",
    ]
    result = evaluate_candidates_math(candidates, "42")
    assert result["correct_bools"] == [True, False, True, True]
    assert result["mean_acc"] == 0.75
    assert result["pass_at_n"] == 1.0
    assert result["majority_vote"] == 1.0
    print("PASS: test_evaluate_candidates_math")


# ---- Engine test with mock adapter ----

class MockAdapter(BaseAdapter):
    """Returns \\boxed{42} for everything."""
    def generate_batch(self, prompts):
        return [GenerationResult(text=f"The answer is \\boxed{{42}}") for _ in prompts]


def test_engine_basic():
    """Test RSA engine with a mock adapter."""
    adapter = MockAdapter()
    engine = RSAEngine(
        adapter=adapter,
        population=2,
        k=2,
        loops=2,
        islands=1,
        task="math",
        verbose=False,
    )
    result = engine.run(
        questions=["What is 40+2?", "What is 6*7?"],
        ground_truths=["42", "42"],
        num_seeds=1,
    )
    assert "metrics" in result
    assert "final_candidates" in result
    assert len(result["final_candidates"]) == 2
    # All candidates say 42, GT is 42 -> 100% accuracy
    last_step = result["metrics"][-1]
    assert last_step["summary"]["mean_accuracy"]["mean"] == 1.0
    print("PASS: test_engine_basic")


def test_engine_islands():
    """Test RSA engine with islands."""
    adapter = MockAdapter()
    engine = RSAEngine(
        adapter=adapter,
        population=2,
        k=2,
        loops=2,
        islands=2,
        task="math",
        verbose=False,
    )
    result = engine.run(
        questions=["What is 40+2?"],
        ground_truths=["42"],
    )
    assert result["config"]["M"] == 2
    # 2 islands * 2 population = 4 total candidates
    assert len(result["final_candidates"][0]) == 4
    print("PASS: test_engine_islands")


def test_engine_no_ground_truth():
    """RSA works without ground truths (no evaluation)."""
    adapter = MockAdapter()
    engine = RSAEngine(
        adapter=adapter,
        population=2,
        k=2,
        loops=2,
        islands=1,
        verbose=False,
    )
    result = engine.run(
        questions=["Solve this problem..."],
        ground_truths=None,
    )
    assert "metrics" not in result
    assert len(result["final_candidates"]) == 1
    print("PASS: test_engine_no_ground_truth")


if __name__ == "__main__":
    test_validate_params_valid()
    test_validate_params_invalid()
    test_get_num_islands()
    test_merge_schedule()
    test_build_prompt_no_candidates()
    test_build_prompt_with_candidates()
    test_build_prompt_single_candidate()
    test_boxed_extraction()
    test_is_equiv()
    test_evaluate_candidates_math()
    test_engine_basic()
    test_engine_islands()
    test_engine_no_ground_truth()
    print("\n=== ALL UNIT TESTS PASSED ===")
