"""Evaluation functions for RSA results."""

import re
from typing import List, Dict, Any, Optional, Callable
import numpy as np


# ---- Math evaluation (boxed extraction + symbolic equivalence) ----

def last_boxed_only_string(s: str) -> Optional[str]:
    """Extract the last \\boxed{...} expression from a string."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(s):
        if s[i] == "{":
            num_left_braces_open += 1
        if s[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return s[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove \\boxed{...} wrapper, returning inner content."""
    if s is None:
        return ""
    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]
    # Handle \\boxed ... without braces
    if s[:7] == "\\boxed ":
        return s[7:]
    return s


def _normalize_answer(s: str) -> str:
    """Basic normalization for string comparison."""
    s = s.strip()
    # Remove enclosing $ signs
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Remove common LaTeX wrappers
    for wrapper in ["\\text{", "\\mathrm{", "\\textbf{"]:
        if s.startswith(wrapper) and s.endswith("}"):
            s = s[len(wrapper):-1].strip()
    return s


def is_equiv(a: str, b: str) -> bool:
    """Check if two math answers are equivalent.

    Tries direct string comparison first, then numeric comparison.
    For full symbolic equivalence, use verl's is_equiv or sympy.
    """
    a = _normalize_answer(a)
    b = _normalize_answer(b)
    if a == b:
        return True

    # Try numeric comparison
    try:
        # Remove commas from numbers
        a_clean = a.replace(",", "").replace(" ", "")
        b_clean = b.replace(",", "").replace(" ", "")
        return abs(float(a_clean) - float(b_clean)) < 1e-8
    except (ValueError, TypeError):
        pass

    # Try fraction-aware numeric comparison
    try:
        frac_pattern = r"^(-?\d+)\s*/\s*(\d+)$"

        def _to_float(s):
            m = re.match(frac_pattern, s.replace(" ", ""))
            if m:
                return int(m.group(1)) / int(m.group(2))
            return float(s.replace(",", "").replace(" ", ""))

        return abs(_to_float(a) - _to_float(b)) < 1e-8
    except (ValueError, TypeError, ZeroDivisionError):
        pass

    return False


def evaluate_candidates_math(candidates: List[str], gt: str) -> Dict[str, Any]:
    """Evaluate math candidates against ground truth.

    Args:
        candidates: List of candidate solution strings.
        gt: Ground truth answer string.

    Returns:
        Dict with correct_bools, mean_acc, pass_at_n, majority_vote.
    """
    solutions = [
        (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
        for a in candidates
    ]
    extracted = [remove_boxed(s) for s in solutions]

    correct_bools = [bool(is_equiv(e, gt)) for e in extracted]
    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_n = float(1.0 if any(correct_bools) else 0.0)

    # majority vote via clustering
    clusters: List[Dict[str, Any]] = []
    for e in extracted:
        placed = False
        for c in clusters:
            if bool(is_equiv(e, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": e, "count": 1})

    if not clusters:
        majority_vote = 0.0
    else:
        best = max(clusters, key=lambda c: c["count"])
        majority_vote = float(bool(is_equiv(best["rep"], gt)))

    return {
        "correct_bools": correct_bools,
        "extracted": extracted,
        "mean_acc": mean_acc,
        "pass_at_n": pass_at_n,
        "majority_vote": majority_vote,
    }


def make_rg_eval_fn(env):
    """Create an eval_fn for RSAEngine that scores via reasoning_gym.

    Args:
        env: A verifiers.ReasoningGymEnv with parser and rg_dataset attributes.

    Returns:
        A callable(candidates, gt) -> dict compatible with RSAEngine.
    """
    from collections import Counter

    def eval_fn(candidates: List[str], gt: str) -> Dict[str, Any]:
        entry = env.rg_dataset[int(gt)]
        correct_bools = []
        extracted = []

        for candidate in candidates:
            parsed = env.parser.parse_answer(candidate)
            parsed_str = str(parsed).strip() if parsed is not None else None
            extracted.append(parsed_str)
            score = env.rg_dataset.score_answer(answer=parsed_str, entry=entry)
            correct_bools.append(score >= 1.0)

        mean_acc = float(sum(correct_bools)) / max(1, len(correct_bools))
        pass_at_n = float(1.0 if any(correct_bools) else 0.0)

        # Majority vote by exact parsed answer string
        non_none = [e for e in extracted if e is not None]
        if non_none:
            most_common = Counter(non_none).most_common(1)[0][0]
            maj_score = env.rg_dataset.score_answer(
                answer=most_common, entry=entry
            )
            majority_vote = float(maj_score >= 1.0)
        else:
            majority_vote = 0.0

        return {
            "correct_bools": correct_bools,
            "extracted": extracted,
            "mean_acc": mean_acc,
            "pass_at_n": pass_at_n,
            "majority_vote": majority_vote,
        }

    return eval_fn


def evaluate_step(
    data: List[dict],
    M: int,
    N: int,
    T: int,
    step: int,
    task: str = "math",
    eval_fn: Optional[Callable] = None,
) -> dict:
    """Evaluate all problems and return aggregated metrics for this step.

    Args:
        data: List of problem dicts with 'candidates' and 'gt' keys.
        M: Number of islands.
        N: Population per island.
        T: Total steps.
        step: Current step index.
        task: Task type ("math" supported by default).
        eval_fn: Optional custom evaluation function(candidates, gt) -> dict with
            at least 'correct_bools', 'mean_acc', 'pass_at_n', 'majority_vote'.

    Returns:
        Dict of aggregated metrics.
    """
    from rsa.islands import get_num_islands

    total_candidates = M * N
    num_islands = get_num_islands(step, M, T)
    island_size = total_candidates // num_islands

    all_mean_acc = []
    all_majority = []
    all_pass_at_n = []
    all_std_dev = []
    all_island_mean_std = []

    if eval_fn is None:
        if task == "math":
            eval_fn = evaluate_candidates_math
        else:
            raise NotImplementedError(f"No built-in eval for task '{task}'. Provide eval_fn.")

    for problem in data:
        candidates = problem["candidates"]
        gt = problem["gt"]
        result = eval_fn(candidates, gt)

        correct_bools = result["correct_bools"]
        accuracies = [float(b) for b in correct_bools]

        all_mean_acc.append(result["mean_acc"])
        all_majority.append(result["majority_vote"])
        all_pass_at_n.append(result["pass_at_n"])
        all_std_dev.append(float(np.std(accuracies)))

        # Std dev of per-island means
        island_means = []
        for island_idx in range(num_islands):
            start = island_idx * island_size
            end = start + island_size
            island_accs = accuracies[start:end]
            island_means.append(float(np.mean(island_accs)))
        island_mean_std = float(np.std(island_means)) if len(island_means) > 1 else 0.0
        all_island_mean_std.append(island_mean_std)

    return {
        "step": step,
        "num_islands": num_islands,
        "island_size": island_size,
        "n_problems": len(data),
        "mean_accuracy": float(np.mean(all_mean_acc)),
        "majority_vote": float(np.mean(all_majority)),
        "pass_at_n": float(np.mean(all_pass_at_n)),
        "std_dev_across_candidates": float(np.mean(all_std_dev)),
        "std_dev_of_island_means": float(np.mean(all_island_mean_std)),
    }
