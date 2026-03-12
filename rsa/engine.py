"""RSA Engine: orchestrates island-based recursive self-aggregation."""

import json
import math
import os
import random
import time
from typing import List, Dict, Any, Optional, Callable

import numpy as np

from rsa.adapters.base import BaseAdapter
from rsa.islands import validate_island_params, get_num_islands, get_merge_schedule
from rsa.prompts import build_prompt
from rsa.evaluation import evaluate_step


class RSAEngine:
    """Island-based Recursive Self-Aggregation engine.

    Runs RSA on a batch of questions using any BaseAdapter. Supports islands
    (pairwise merging), multi-seed aggregation, and pluggable evaluation.

    Args:
        adapter: A BaseAdapter instance (VLLMAdapter, AnthropicAdapter, etc.).
        population: N - population per island.
        k: K - candidates sampled per aggregation step.
        loops: T - total RSA steps.
        islands: M - number of islands (must be power of 2).
        task: Task type for prompt formatting ("math", "rg", "supergpqa", "general").
        eval_fn: Optional custom evaluation function(candidates, gt) -> dict.
        seed: Base random seed.
        verbose: Print progress to stdout.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        population: int = 8,
        k: int = 4,
        loops: int = 10,
        islands: int = 1,
        task: str = "math",
        eval_fn: Optional[Callable] = None,
        seed: int = 1234,
        verbose: bool = True,
        system_prompt: Optional[str] = None,
    ):
        self.adapter = adapter
        self.population = population
        self.k = k
        self.loops = loops
        self.islands = islands
        self.task = task
        self.eval_fn = eval_fn
        self.seed = seed
        self.verbose = verbose
        self.system_prompt = system_prompt

        validate_island_params(islands, population, k, loops)
        self.total_candidates = islands * population

    def run(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        num_seeds: int = 1,
        output_dir: Optional[str] = None,
        max_problems: int = 0,
    ) -> Dict[str, Any]:
        """Run RSA on a list of questions.

        Args:
            questions: List of question/problem strings.
            ground_truths: Optional list of ground truth answers for evaluation.
                If None, no evaluation is performed.
            num_seeds: Number of independent seeds to run and aggregate.
            output_dir: Optional directory to save metrics JSON.
            max_problems: Limit number of problems (0 = all).

        Returns:
            Dict with:
                - "metrics": per-step aggregated metrics (if ground_truths provided)
                - "final_candidates": list of final candidate lists per question
                - "config": run configuration
        """
        if max_problems > 0:
            questions = questions[:max_problems]
            if ground_truths is not None:
                ground_truths = ground_truths[:max_problems]

        # Build base rows
        base_rows = []
        for i, q in enumerate(questions):
            row = {"orig_prompt": q, "question_idx": i}
            if ground_truths is not None:
                row["gt"] = ground_truths[i]
            base_rows.append(row)

        merge_events = get_merge_schedule(self.islands, self.loops)
        config_str = f"M{self.islands}_N{self.population}_K{self.k}_T{self.loops}"

        if self.verbose:
            self._print_config(base_rows, merge_events, config_str)

        # Accumulators across seeds
        metric_keys = ["mean_accuracy", "majority_vote", "pass_at_n",
                        "std_dev_across_candidates", "std_dev_of_island_means"]
        acc = {key: [[] for _ in range(self.loops)] for key in metric_keys}

        all_seed_candidates = []
        start_time = time.time()

        for s in range(num_seeds):
            random.seed(self.seed + s)
            np.random.seed(self.seed + s)

            data = [{**row, "candidates": None} for row in base_rows]

            if num_seeds > 1 and self.verbose:
                print(f"\n--- Seed {s+1}/{num_seeds} (seed={self.seed + s}) ---")

            for step in range(self.loops):
                is_merge_step = any(ev["at_step"] == step for ev in merge_events)
                num_islands_now = get_num_islands(step, self.islands, self.loops)
                island_size = self.total_candidates // num_islands_now

                if is_merge_step and self.verbose:
                    print(f">>> MERGE at step {step}: {num_islands_now * 2} -> {num_islands_now} islands "
                          f"(island size: {island_size // 2} -> {island_size})")

                if self.verbose:
                    print(f"Step {step}/{self.loops-1}: {num_islands_now} islands, "
                          f"island_size={island_size}, "
                          f"generating {self.total_candidates * len(data)} responses...")

                step_start = time.time()
                self._run_step(step, data)
                step_elapsed = time.time() - step_start

                if ground_truths is not None:
                    metrics = evaluate_step(
                        data, self.islands, self.population, self.loops, step,
                        self.task, self.eval_fn,
                    )
                    if self.verbose:
                        print(f"  mean_acc={metrics['mean_accuracy']:.4f}  "
                              f"majority={metrics['majority_vote']:.4f}  "
                              f"pass@n={metrics['pass_at_n']:.4f}  "
                              f"std_cand={metrics['std_dev_across_candidates']:.4f}  "
                              f"std_islands={metrics['std_dev_of_island_means']:.4f}  "
                              f"({step_elapsed:.1f}s)")
                    for key in metric_keys:
                        acc[key][step].append(metrics[key])
                elif self.verbose:
                    print(f"  ({step_elapsed:.1f}s)")

            all_seed_candidates.append([p["candidates"] for p in data])

        elapsed = time.time() - start_time

        # Build result
        result = {
            "config": {
                "M": self.islands, "N": self.population, "K": self.k,
                "T": self.loops, "task": self.task, "num_seeds": num_seeds,
                "seed": self.seed, "n_problems": len(base_rows),
            },
            "final_candidates": all_seed_candidates[-1] if all_seed_candidates else [],
            "elapsed_seconds": elapsed,
        }

        if ground_truths is not None:
            result["metrics"] = self._aggregate_metrics(acc, metric_keys, merge_events)

        if output_dir:
            self._save_metrics(output_dir, config_str, result, acc, metric_keys,
                               base_rows, merge_events)

        if self.verbose:
            print(f"\nTotal time: {elapsed:.1f}s")
            if ground_truths is not None:
                self._print_summary(acc, metric_keys)

        return result

    def _run_step(self, step: int, data: List[dict]):
        """Run one RSA step: build prompts, generate, assign back."""
        num_islands = get_num_islands(step, self.islands, self.loops)
        island_size = self.total_candidates // num_islands

        prompts = []
        for problem in data:
            question = problem["orig_prompt"]
            candidates = problem["candidates"]

            for i in range(self.total_candidates):
                if candidates is None:
                    # First step: generate from scratch
                    prompt = build_prompt(question, None, self.task)
                else:
                    # Sample K candidates from this candidate's island
                    island_idx = i // island_size
                    island_start = island_idx * island_size
                    island_end = island_start + island_size
                    island_pool = candidates[island_start:island_end]
                    sampled = random.sample(island_pool, self.k)
                    prompt = build_prompt(question, sampled, self.task)

                # Wrap as chat messages when system_prompt is set
                if self.system_prompt and isinstance(prompt, str):
                    prompt = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ]

                prompts.append(prompt)

        # Batch generate
        results = self.adapter.generate_batch(prompts)

        # Assign responses back
        idx = 0
        for problem in data:
            problem["candidates"] = [r.text for r in results[idx:idx + self.total_candidates]]
            idx += self.total_candidates

    def _aggregate_metrics(self, acc, metric_keys, merge_events):
        """Build per-step summary metrics."""
        metrics_list = []
        for step in range(self.loops):
            num_islands = get_num_islands(step, self.islands, self.loops)
            island_size = self.total_candidates // num_islands
            entry = {
                "step": step,
                "num_islands": num_islands,
                "island_size": island_size,
                "summary": {
                    key: {
                        "mean": float(np.mean(acc[key][step])),
                        "std": float(np.std(acc[key][step], ddof=0)),
                    }
                    for key in metric_keys
                },
                "values": {key: acc[key][step] for key in metric_keys},
            }
            metrics_list.append(entry)
        return metrics_list

    def _save_metrics(self, output_dir, config_str, result, acc, metric_keys,
                      base_rows, merge_events):
        """Save metrics to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, f"island_rsa_{config_str}.json")

        # Save per-step entries
        entries = []
        for step in range(self.loops):
            num_islands = get_num_islands(step, self.islands, self.loops)
            island_size = self.total_candidates // num_islands
            entry = {
                "step": step,
                "num_islands": num_islands,
                "island_size": island_size,
                "n_problems": len(base_rows),
                "n_seeds": result["config"]["num_seeds"],
                "values": {key: acc[key][step] for key in metric_keys},
                "summary": {
                    key: {
                        "mean": float(np.mean(acc[key][step])),
                        "std": float(np.std(acc[key][step], ddof=0)),
                    }
                    for key in metric_keys
                },
            }
            entries.append(entry)

        with open(metrics_path, "w") as f:
            json.dump(entries, f, indent=2)

        if self.verbose:
            print(f"Metrics saved to: {metrics_path}")

    def _print_config(self, base_rows, merge_events, config_str):
        """Print run configuration."""
        print("=" * 60)
        print("Island RSA Configuration")
        print("=" * 60)
        print(f"  M (islands): {self.islands}")
        print(f"  N (pop/isl): {self.population}")
        print(f"  K (sample):  {self.k}")
        print(f"  T (steps):   {self.loops}")
        print(f"  Total candidates per question: {self.total_candidates}")
        print(f"  Problems:    {len(base_rows)}")
        print(f"  Task:        {self.task}")

        if merge_events:
            print(f"\n  Merge schedule ({len(merge_events)} merges):")
            num_merges = int(math.log2(self.islands))
            phase_len = self.loops // (num_merges + 1)
            print(f"  Phase length: {phase_len} steps")
            for ev in merge_events:
                isl_before = self.total_candidates // ev["from"]
                isl_after = self.total_candidates // ev["to"]
                print(f"    Step {ev['at_step']}: {ev['from']} islands (size {isl_before}) "
                      f"-> {ev['to']} islands (size {isl_after})")
        print("=" * 60)
        print()

    def _print_summary(self, acc, metric_keys):
        """Print final summary table."""
        print(f"\n{'Step':>4} {'Islands':>7} {'mean_acc':>18} {'majority':>18} "
              f"{'pass@n':>18} {'std_islands':>18}")
        for step in range(self.loops):
            num_islands = get_num_islands(step, self.islands, self.loops)
            row = ""
            for key in ["mean_accuracy", "majority_vote", "pass_at_n", "std_dev_of_island_means"]:
                m = float(np.mean(acc[key][step]))
                sd = float(np.std(acc[key][step], ddof=0))
                row += f"  {m:.4f}\u00b1{sd:.4f}"
            print(f"{step:>4} {num_islands:>7}{row}")
