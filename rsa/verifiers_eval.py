"""Evaluation pipeline for reasoning_gym environments via the verifiers interface.

Uses verifiers.ReasoningGymEnv for prompt formatting, parsing, and scoring.
Bridges to any RSA BaseAdapter for offline batch generation.

Example:
    from rsa.adapters.vllm_adapter import VLLMAdapter
    from rsa.verifiers_eval import VerifiersEval

    adapter = VLLMAdapter(model="Qwen/Qwen3-4B-Instruct-2507", tp_size=4)
    ev = VerifiersEval("countdown", adapter, num_eval=20)
    result = ev.run()
    print(result["mean_reward"])
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from rsa.adapters.base import BaseAdapter


class VerifiersEval:
    """Evaluate an LLM on reasoning_gym environments via verifiers.

    All prompts, parsing, and scoring come from the verifiers/reasoning_gym
    interface — nothing is hardcoded.

    Args:
        env_name: Name of a reasoning_gym environment (e.g. "countdown",
            "game_of_24", "letter_counting") or a list of dataset specs
            for composite envs.
        adapter: A BaseAdapter instance for generation.
        num_eval: Number of evaluation problems to generate.
        seed: Random seed for the environment.
        verbose: Print progress to stdout.
        wandb_project: If set, log metrics to this wandb project.
        wandb_run_name: Optional wandb run name.
        wandb_config: Extra config dict to log to wandb.
    """

    def __init__(
        self,
        env_name,
        adapter: BaseAdapter,
        num_eval: int = 100,
        seed: int = 42,
        verbose: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        from verifiers import ReasoningGymEnv

        self.env = ReasoningGymEnv(
            gym=env_name,
            num_train_examples=0,
            num_eval_examples=num_eval,
            seed=seed,
        )
        self.adapter = adapter
        self.verbose = verbose
        self.env_name = env_name
        self._wandb_run = None

        if wandb_project:
            import wandb
            config = {
                "env_name": env_name,
                "num_eval": num_eval,
                "seed": seed,
                **(wandb_config or {}),
            }
            self._wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config,
            )

    def run(
        self,
        max_problems: int = 0,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run evaluation.

        Args:
            max_problems: Limit number of problems (0 = all).
            output_dir: Optional directory to save results JSON.

        Returns:
            Dict with mean_reward, n_correct, n_problems, per_problem details,
            and timing information.
        """
        dataset = self.env.get_eval_dataset(
            n=max_problems if max_problems > 0 else -1
        )

        prompts = [row["prompt"] for row in dataset]
        answers = [row["answer"] for row in dataset]

        if self.verbose:
            print(f"Evaluating {len(prompts)} problems from '{self.env_name}'")

        start = time.time()
        results = self.adapter.generate_batch(prompts)
        gen_elapsed = time.time() - start

        if self.verbose:
            print(f"Generation done in {gen_elapsed:.1f}s")

        # Parse and score using the verifiers interface
        scores: List[float] = []
        details: List[Dict[str, Any]] = []
        running_correct = 0
        for i, (result, answer_idx) in enumerate(zip(results, answers)):
            parsed = self.env.parser.parse_answer(result.text)
            parsed_str = str(parsed).strip() if parsed is not None else None
            entry = self.env.rg_dataset[int(answer_idx)]
            score = self.env.rg_dataset.score_answer(
                answer=parsed_str, entry=entry
            )
            scores.append(score)
            if score >= 1.0:
                running_correct += 1

            detail = {
                "problem_idx": i,
                "question": entry["question"],
                "ground_truth": entry["answer"],
                "model_response": result.text[:500],
                "parsed_answer": parsed_str,
                "score": score,
            }
            details.append(detail)

            if self.verbose:
                status = "OK" if score >= 1.0 else f"score={score:.2f}"
                print(f"  [{i+1}/{len(prompts)}] {status}  parsed={parsed_str}")

            # Log per-problem metrics to wandb
            if self._wandb_run:
                import wandb
                step_num = i + 1
                log_data = {
                    "running/mean_reward": sum(scores) / len(scores),
                    "running/accuracy": running_correct / step_num,
                    "running/n_correct": running_correct,
                    "running/n_evaluated": step_num,
                }
                # Log 2-3 sample responses per step as text
                if i < 3 or score >= 1.0 and sum(1 for s in scores if s >= 1.0) <= 3:
                    log_data["samples/question"] = wandb.Html(
                        f"<pre>{entry['question'][:300]}</pre>"
                    )
                    log_data["samples/response"] = wandb.Html(
                        f"<pre>{result.text[:500]}</pre>"
                    )
                    log_data["samples/parsed_answer"] = parsed_str or "(none)"
                    log_data["samples/ground_truth"] = entry["answer"]
                    log_data["samples/score"] = score
                wandb.log(log_data, step=step_num)

        elapsed = time.time() - start
        mean_reward = sum(scores) / max(1, len(scores))
        n_correct = sum(1 for s in scores if s >= 1.0)

        output = {
            "env_name": self.env_name,
            "n_problems": len(scores),
            "n_correct": n_correct,
            "mean_reward": mean_reward,
            "accuracy": n_correct / max(1, len(scores)),
            "elapsed_seconds": elapsed,
            "generation_seconds": gen_elapsed,
            "details": details,
        }

        if self.verbose:
            print(f"\n=== Results: {self.env_name} ===")
            print(f"  Problems: {len(scores)}")
            print(f"  Correct:  {n_correct}/{len(scores)} ({output['accuracy']:.1%})")
            print(f"  Mean reward: {mean_reward:.4f}")
            print(f"  Time: {elapsed:.1f}s")

        # Log final summary + results table to wandb
        if self._wandb_run:
            import wandb
            # Summary table with all problems
            table = wandb.Table(columns=[
                "idx", "question", "ground_truth", "parsed_answer",
                "response", "score",
            ])
            for d in details:
                table.add_data(
                    d["problem_idx"],
                    d["question"][:200],
                    d["ground_truth"],
                    d["parsed_answer"],
                    d["model_response"][:300],
                    d["score"],
                )
            wandb.log({
                "final/accuracy": output["accuracy"],
                "final/mean_reward": mean_reward,
                "final/n_correct": n_correct,
                "final/n_problems": len(scores),
                "final/elapsed_seconds": elapsed,
                "final/generation_seconds": gen_elapsed,
                "results_table": table,
            })
            wandb.finish()
            self._wandb_run = None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"{self.env_name}_eval.json")
            with open(path, "w") as f:
                json.dump(output, f, indent=2)
            if self.verbose:
                print(f"  Saved to: {path}")

        return output

    def run_rsa(
        self,
        population: int = 4,
        k: int = 2,
        loops: int = 4,
        islands: int = 1,
        num_seeds: int = 1,
        seed: int = 1234,
        max_problems: int = 0,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run RSA evaluation on the reasoning_gym environment.

        Uses the full RSA loop (population sampling, aggregation, islands)
        with scoring via the verifiers/reasoning_gym interface.

        Args:
            population: N - candidates per island.
            k: K - candidates sampled per aggregation step.
            loops: T - total RSA steps.
            islands: M - number of islands (must be power of 2).
            num_seeds: Independent seeds to run and aggregate.
            seed: Base random seed for RSA.
            max_problems: Limit number of problems (0 = all).
            output_dir: Optional directory to save metrics JSON.

        Returns:
            Dict with metrics (per-step), final_candidates, config, timing.
        """
        from rsa.engine import RSAEngine
        from rsa.evaluation import make_rg_eval_fn

        dataset = self.env.get_eval_dataset(
            n=max_problems if max_problems > 0 else -1
        )

        # Extract question text and ground truth indices
        questions = []
        gt_indices = []
        for row in dataset:
            prompt_messages = row["prompt"]
            user_msg = next(
                m["content"] for m in prompt_messages if m["role"] == "user"
            )
            questions.append(user_msg)
            gt_indices.append(row["answer"])

        system_prompt = self.env.system_prompt

        if self.verbose:
            print(f"RSA eval: {len(questions)} problems from '{self.env_name}'")
            print(f"  Config: M={islands} N={population} K={k} T={loops}")

        eval_fn = make_rg_eval_fn(self.env)

        engine = RSAEngine(
            adapter=self.adapter,
            population=population,
            k=k,
            loops=loops,
            islands=islands,
            task="rg",
            eval_fn=eval_fn,
            seed=seed,
            verbose=self.verbose,
            system_prompt=system_prompt,
        )

        result = engine.run(
            questions=questions,
            ground_truths=gt_indices,
            num_seeds=num_seeds,
            output_dir=output_dir,
        )

        result["env_name"] = self.env_name

        # Log per-step RSA metrics to wandb
        if self._wandb_run and "metrics" in result:
            import wandb

            for step_metrics in result["metrics"]:
                step = step_metrics["step"]
                summary = step_metrics["summary"]
                wandb.log({
                    "rsa/mean_accuracy": summary["mean_accuracy"]["mean"],
                    "rsa/majority_vote": summary["majority_vote"]["mean"],
                    "rsa/pass_at_n": summary["pass_at_n"]["mean"],
                    "rsa/std_dev_across_candidates": summary["std_dev_across_candidates"]["mean"],
                    "rsa/std_dev_of_island_means": summary["std_dev_of_island_means"]["mean"],
                    "rsa/num_islands": step_metrics["num_islands"],
                }, step=step)

            # Log sample final candidates as a table
            final_cands = result.get("final_candidates", [])
            if final_cands:
                table = wandb.Table(columns=[
                    "problem_idx", "question", "ground_truth",
                    "candidate_idx", "response", "parsed_answer", "score",
                ])
                for prob_idx in range(min(5, len(final_cands))):
                    entry = self.env.rg_dataset[int(gt_indices[prob_idx])]
                    cands = final_cands[prob_idx]
                    for cand_idx in range(min(3, len(cands))):
                        parsed = self.env.parser.parse_answer(cands[cand_idx])
                        parsed_str = str(parsed).strip() if parsed is not None else None
                        score = self.env.rg_dataset.score_answer(
                            answer=parsed_str, entry=entry
                        )
                        table.add_data(
                            prob_idx,
                            entry["question"][:200],
                            entry["answer"],
                            cand_idx,
                            cands[cand_idx][:500],
                            parsed_str,
                            score,
                        )
                wandb.log({"rsa_results_table": table})

            # Log final summary
            last = result["metrics"][-1]
            wandb.log({
                "final/rsa_mean_accuracy": last["summary"]["mean_accuracy"]["mean"],
                "final/rsa_majority_vote": last["summary"]["majority_vote"]["mean"],
                "final/rsa_pass_at_n": last["summary"]["pass_at_n"]["mean"],
            })
            wandb.finish()
            self._wandb_run = None

        return result
