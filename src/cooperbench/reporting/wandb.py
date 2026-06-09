"""Weights & Biases integration for CooperBench.

Logs run execution and evaluation results to a wandb dashboard.
Enabled when ``WANDB_PROJECT`` is set or ``enable()`` is called
with an explicit project name.  Gracefully degrades (no-op) when
wandb is not installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_WANDB_AVAILABLE: bool | None = None


def _check_wandb() -> bool:
    global _WANDB_AVAILABLE
    if _WANDB_AVAILABLE is None:
        try:
            import wandb  # noqa: F401

            _WANDB_AVAILABLE = True
        except ImportError:
            _WANDB_AVAILABLE = False
    return _WANDB_AVAILABLE


class WandBLogger:
    """Log CooperBench run/eval results to Weights & Biases.

    Usage::

        wb = WandBLogger(project="cooperbench", run_name="my-run")
        if wb.is_active:
            wb.log_task({...})
            wb.log_summary({...})
            wb.finish()
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._run: Any = None
        self._is_active = False
        self._step = 0
        self._tasks_table_rows: list[list[Any]] = []
        self._tasks_table_columns = [
            "repo",
            "task_id",
            "features",
            "setting",
            "run_status",
            "steps",
            "cost",
            "duration_s",
            "patch_lines",
            "both_passed",
            "eval_f1",
            "eval_f2",
            "merge_status",
            "error",
        ]

        project = project or os.environ.get("WANDB_PROJECT", "").strip()
        if not project:
            logger.debug("WandBLogger: no project configured, skipping")
            return
        if not _check_wandb():
            logger.warning("WandBLogger: wandb package not installed, skipping")
            return

        import wandb

        try:
            settings = wandb.Settings()
            self._run = wandb.init(
                project=project,
                entity=entity or os.environ.get("WANDB_ENTITY") or None,
                name=run_name,
                config=config or {},
                resume="allow",
                settings=settings,
            )
            self._is_active = True
            logger.info("WandBLogger: initialized run %s/%s", project, run_name or "?")
        except Exception as exc:
            logger.warning("WandBLogger: wandb.init failed: %s", exc)

    @property
    def is_active(self) -> bool:
        return self._is_active

    def log_task(
        self,
        *,
        repo: str,
        task_id: int,
        features: list[int],
        setting: str,
        run_status: str = "",
        steps: int = 0,
        cost: float = 0.0,
        duration_s: float = 0.0,
        patch_lines: int = 0,
        both_passed: bool | None = None,
        eval_f1_passed: bool | None = None,
        eval_f2_passed: bool | None = None,
        merge_status: str = "",
        error: str = "",
    ) -> None:
        """Log a single task's execution + evaluation results."""
        if not self._is_active or self._run is None:
            return

        step = self._step
        self._step += 1

        # Scalar metrics
        scalars: dict[str, Any] = {
            "task/steps": steps,
            "task/cost": cost,
            "task/duration_s": duration_s,
            "task/patch_lines": patch_lines,
            "step": step,
        }
        if both_passed is not None:
            scalars["task/both_passed"] = 1 if both_passed else 0
        if eval_f1_passed is not None:
            scalars["task/f1_passed"] = 1 if eval_f1_passed else 0
        if eval_f2_passed is not None:
            scalars["task/f2_passed"] = 1 if eval_f2_passed else 0
        if merge_status == "conflicts":
            scalars["task/merge_conflict"] = 1
        elif merge_status == "clean":
            scalars["task/merge_conflict"] = 0

        self._run.log(scalars, step=step)

        # Accumulate for table
        features_str = ",".join(str(f) for f in features)
        self._tasks_table_rows.append(
            [
                repo,
                task_id,
                features_str,
                setting,
                run_status,
                steps,
                cost,
                duration_s,
                patch_lines,
                "pass" if both_passed else "fail" if both_passed is not None else "",
                "pass" if eval_f1_passed else "fail" if eval_f1_passed is not None else "",
                "pass" if eval_f2_passed else "fail" if eval_f2_passed is not None else "",
                merge_status,
                error[:500] if error else "",
            ]
        )

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log aggregate run/eval summary metrics and the tasks table."""
        if not self._is_active or self._run is None:
            return

        import wandb

        # Scalar summary metrics
        summary_metrics: dict[str, Any] = {}
        for key in (
            "total_tasks",
            "completed",
            "skipped",
            "failed",
            "total_cost",
            "total_time_seconds",
            "pass_rate",
        ):
            if key in summary:
                summary_metrics[f"summary/{key}"] = summary[key]
        if "eval" in summary:
            ev = summary["eval"]
            for ekey in ("total_evaluated", "passed", "failed", "errors", "skipped", "pass_rate"):
                if ekey in ev:
                    summary_metrics[f"eval/{ekey}"] = ev[ekey]

        self._run.log(summary_metrics)

        # Tasks table
        if self._tasks_table_rows:
            table = wandb.Table(columns=self._tasks_table_columns, data=self._tasks_table_rows)
            self._run.log({"tasks": table})

        # Roll-up aggregates
        if self._tasks_table_rows:
            n_total = len(self._tasks_table_rows)
            n_passed = sum(1 for r in self._tasks_table_rows if r[9] == "pass")
            n_failed = sum(1 for r in self._tasks_table_rows if r[9] == "fail")
            total_steps = sum(r[5] for r in self._tasks_table_rows)
            total_cost = sum(r[6] for r in self._tasks_table_rows)
            total_dur = sum(r[7] for r in self._tasks_table_rows)

            rollup: dict[str, Any] = {
                "aggregate/tasks_total": n_total,
                "aggregate/tasks_passed": n_passed,
                "aggregate/tasks_failed": n_failed,
                "aggregate/pass_rate": n_passed / max(n_total, 1),
                "aggregate/total_steps": total_steps,
                "aggregate/total_cost_usd": total_cost,
                "aggregate/total_duration_s": total_dur,
            }
            if n_total > 0:
                rollup["aggregate/avg_steps"] = total_steps / n_total
                rollup["aggregate/avg_duration_s"] = total_dur / n_total
            self._run.log(rollup)

    def finish(self) -> None:
        """Close the wandb run."""
        if self._is_active and self._run is not None:
            try:
                self._run.finish()
                logger.info("WandBLogger: run finished")
            except Exception as exc:
                logger.warning("WandBLogger: finish failed: %s", exc)
            self._is_active = False
