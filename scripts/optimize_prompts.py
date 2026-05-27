"""GEPA × CooperBench prompt optimization entry point.

Usage:
    python scripts/optimize_prompts.py [--max-metric-calls N] [--subset NAME]

Environment variables:
    OPENCODE_GO_API_KEY  — API key for OpenCode Go Zen (GLM-5.1 reflection LM)
    VLLM_API_BASE        — Base URL for the local vLLM server (default: http://localhost:8000/v1)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEED_SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer."""

SEED_INSTANCE_TEMPLATE = """## Your Task

{{task}}

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes — see the **Submission** section below for the exact procedure.

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
- To submit your work, follow the **Submission** section below (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`).

## Submission

`patch.txt` is the artifact we evaluate — write whatever unified diff you want to submit to that file.

Write the patch (one common way — `git diff` of your in-place edits):

```bash
git diff -- path/to/file1 path/to/file2 > patch.txt
```

Verify it contains what you intend:

```bash
cat patch.txt
```

Submit (EXACT command required):

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""


def main():
    parser = argparse.ArgumentParser(description="GEPA × CooperBench prompt optimization")
    parser.add_argument("--max-metric-calls", type=int, default=100, help="Evaluation budget (default: 100)")
    parser.add_argument("--subset", type=str, default="9b_easy", help="Dataset subset (default: 9b_easy)")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--logs-dir", type=str, default=None, help="Logs directory")
    parser.add_argument(
        "--task-model", type=str, default="openai/qwen3.5-9b", help="Task LM model name (default: openai/qwen3.5-9b)"
    )
    parser.add_argument(
        "--task-api-base",
        type=str,
        default=None,
        help="Task LM API base URL (default: VLLM_API_BASE env var or http://localhost:8000/v1)",
    )
    parser.add_argument("--run-dir", type=str, default=None, help="GEPA run directory for artifacts")
    parser.add_argument("--step-limit", type=int, default=None, help="Agent step limit per task (default: use solo.yaml default)")
    parser.add_argument(
        "--reflection-minibatch-size", type=int, default=3, help="Train examples per reflection step (default: 3)"
    )
    args = parser.parse_args()

    import gepa

    from cooperbench.gepa_integration.adapter import CooperBenchAdapter
    from cooperbench.gepa_integration.data_loader import CooperBenchDataLoader
    from cooperbench.gepa_integration.reflective_lm import ReflectiveLM

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or f"logs/gepa_solo_{timestamp}"

    task_api_base = args.task_api_base or os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")

    opencode_key = os.environ.get("OPENCODE_GO_API_KEY", "").strip()
    if not opencode_key:
        print("ERROR: OPENCODE_GO_API_KEY environment variable is required for GLM-5.1 reflection LM")
        sys.exit(1)

    reflection_lm = ReflectiveLM(
        model="openai/glm-5.1",
        api_base="https://opencode.ai/zen/go/v1",
        api_key=opencode_key,
        temperature=0.6,
        max_tokens=8192,
        run_dir=run_dir,
        extra_headers={"User-Agent": "opencode/1.0"},
    )

    adapter = CooperBenchAdapter(
        model_name=args.task_model,
        model_api_base=task_api_base,
        model_api_key="EMPTY",
        setting="solo",
        eval_backend="docker",
        dataset_dir=args.dataset_dir,
        logs_dir=args.logs_dir,
        run_name=f"gepa_solo_{timestamp}",
        step_limit=args.step_limit,
    )

    trainset = CooperBenchDataLoader(subset=args.subset, dataset_dir=args.dataset_dir, split="train")
    valset = CooperBenchDataLoader(subset=args.subset, dataset_dir=args.dataset_dir, split="val")

    print("GEPA × CooperBench Optimization")
    print(f"  Task LM:    {args.task_model} @ {task_api_base}")
    print("  Reflection: GLM-5.1 @ OpenCode Go Zen")
    print(f"  Subset:     {args.subset}")
    print(f"  Train:      {len(trainset)} tasks, Val: {len(valset)} tasks")
    print(f"  Budget:     {args.max_metric_calls} metric calls")
    print(f"  Step limit: {args.step_limit or 'default'}")
    print(f"  Minibatch:  {args.reflection_minibatch_size}")
    print(f"  Run dir:    {run_dir}")
    print()

    result = gepa.optimize(
        adapter=adapter,
        seed_candidate={
            "system_prompt": SEED_SYSTEM_PROMPT,
            "instance_template": SEED_INSTANCE_TEMPLATE,
        },
        trainset=trainset,
        valset=valset,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        use_merge=True,
        acceptance_criterion="improvement_or_equal",
        run_dir=run_dir,
        cache_evaluation=True,
    )

    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best candidate idx: {result.best_idx}")
    print(f"Best score: {result.val_aggregate_scores[result.best_idx]:.4f}")

    best = result.best_candidate
    for component, text in best.items():
        print(f"\n--- {component} ---")
        print(text[:500])
        if len(text) > 500:
            print(f"... ({len(text)} chars total)")

    output_path = Path(run_dir) / "best_candidate.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(output_path, "w") as f:
        json.dump(
            {
                "best_idx": result.best_idx,
                "best_candidate": best,
                "best_score": result.val_aggregate_scores[result.best_idx],
                "all_scores": result.val_aggregate_scores,
                "reflection_lm_cost": reflection_lm.total_cost,
                "reflection_lm_tokens_in": reflection_lm.total_tokens_in,
                "reflection_lm_tokens_out": reflection_lm.total_tokens_out,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
