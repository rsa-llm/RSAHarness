"""Prompt building for RSA aggregation."""

from typing import List, Optional


def aggregate_prompt(
    question: str,
    candidate_answers: List[str],
    task: str = "math",
) -> str:
    """Build an aggregation prompt from a question and candidate answers.

    Args:
        question: The original problem/question text.
        candidate_answers: List of prior candidate solutions to aggregate.
        task: Task type - "math", "rg" (reasoning gym), "supergpqa", or "general".

    Returns:
        The aggregation prompt string.
    """
    if task == "rg":
        problem_kind = "problem"
        format_hint = "<answer>...</answer>"
    elif task == "supergpqa":
        problem_kind = "multiple-choice problem"
        format_hint = "\\boxed{}. Only include the correct option letter in \\boxed{}; for example \\boxed{A}"
    elif task == "math":
        problem_kind = "math problem"
        format_hint = "\\boxed{}"
    else:
        problem_kind = "problem"
        format_hint = "a clearly marked final answer"

    parts = []
    if len(candidate_answers) == 1:
        parts.append(
            f"You are given a {problem_kind} and a candidate solution. "
            "The candidate may be incomplete or contain errors. "
            "Refine this trajectory and produce an improved, higher-quality solution. "
            "If it is entirely wrong, attempt a new strategy. "
            f"End with the final result in {format_hint}.\n"
        )
    else:
        parts.append(
            f"You are given a {problem_kind} and several candidate solutions. "
            "Some candidates may be incorrect or contain errors. "
            "Aggregate the useful ideas and produce a single, high-quality solution. "
            "Reason carefully; if candidates disagree, choose the correct path. "
            "If all are incorrect, then attempt a different strategy. "
            f"End with the final result in {format_hint}.\n"
        )

    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")

    if len(candidate_answers) == 1:
        parts.append("Candidate solution (may contain mistakes):\n")
        ans_str = (candidate_answers[0] or "").strip()
        parts.append(f"---- Candidate ----\n{ans_str}\n")
        parts.append(
            f"Now refine the candidate into an improved solution. "
            f"Provide clear reasoning and end with the final answer in {format_hint}."
        )
    else:
        parts.append("Candidate solutions (may contain mistakes):\n")
        for i, ans in enumerate(candidate_answers, 1):
            ans_str = (ans or "").strip()
            parts.append(f"---- Solution {i} ----\n{ans_str}\n")
        parts.append(
            f"Now write a single improved solution. "
            f"Provide clear reasoning and end with the final answer in {format_hint}."
        )

    return "\n".join(parts)


def build_prompt(
    question: str,
    candidate_answers: Optional[List[str]] = None,
    task: str = "math",
) -> str:
    """Build a prompt for generation or aggregation.

    Args:
        question: The original question/problem text.
        candidate_answers: If None, returns the raw question (first step).
            Otherwise builds an aggregation prompt.
        task: Task type for format hints.

    Returns:
        The prompt string.
    """
    if candidate_answers is not None:
        return aggregate_prompt(question, candidate_answers, task)
    return question
