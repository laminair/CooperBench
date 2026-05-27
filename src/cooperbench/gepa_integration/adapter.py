"""CooperBenchAdapter implementing GEPA's GEPAAdapter protocol.

Bridges GEPA's evolutionary optimization loop with CooperBench's
solo-mode agent execution and Docker-based evaluation pipeline.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from cooperbench.eval.sandbox import test_solo
from cooperbench.gepa_integration.asi import extract_asi, format_asi_for_reflection
from cooperbench.runner.solo import execute_solo
from cooperbench.runner.tasks import DEFAULT_DATASET_DIR

logger = logging.getLogger(__name__)


class CooperBenchAdapter:
    """GEPAAdapter that runs CooperBench solo-mode tasks.

    Each candidate is a dict mapping component names to text:
      - ``"system_prompt"``: The agent's system template
      - ``"instance_template"``: The agent's instance template (optional)

    The adapter writes candidate prompts to a temporary YAML config file,
    runs the mini_swe_agent_v2 agent, evaluates the resulting patch,
    and returns per-task scores.

    Args:
        model_name: LiteLLM model identifier for the task LM
            (e.g. ``"openai/qwen3.5-9b"``).
        model_api_base: API base URL for the task LM
            (e.g. ``"http://localhost:8000/v1"``).
        model_api_key: API key for the task LM.
        setting: ``"solo"`` (only solo is currently supported).
        eval_backend: Evaluation backend (``"docker"``).
        dataset_dir: Root of the CooperBench dataset tree.
        logs_dir: Root for run logs.
        run_name: Prefix for log directories.
        agent_name: Agent framework name (default ``"mini_swe_agent_v2"``).
    """

    def __init__(
        self,
        model_name: str = "openai/qwen3.5-9b",
        model_api_base: str | None = None,
        model_api_key: str = "EMPTY",
        setting: str = "solo",
        eval_backend: str = "docker",
        dataset_dir: Path | str | None = None,
        logs_dir: Path | str | None = None,
        run_name: str = "gepa_optimization",
        agent_name: str = "mini_swe_agent_v2",
        step_limit: int | None = None,
    ):
        if setting != "solo":
            raise NotImplementedError(f"Only solo setting is supported, got: {setting}")

        self.model_name = model_name
        self.model_api_base = model_api_base
        self.model_api_key = model_api_key
        self.setting = setting
        self.eval_backend = eval_backend
        self._dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        self._logs_dir = logs_dir
        self._run_name = run_name
        self._agent_name = agent_name
        self.step_limit = step_limit
        self._call_count = 0
        self.propose_new_texts = None

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ):
        """Evaluate a candidate on a batch of tasks.

        Returns a GEPAResult-compatible EvaluationBatch.
        """
        from gepa.core.adapter import EvaluationBatch

        config_path = self._write_candidate_config(candidate)

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for task in batch:
            self._call_count += 1
            try:
                run_result = execute_solo(
                    repo_name=task["repo"],
                    task_id=task["task_id"],
                    features=task["features"],
                    run_name=self._run_name,
                    agent_name=self._agent_name,
                    model_name=self.model_name,
                    force=True,
                    quiet=True,
                    backend=self.eval_backend,
                    agent_config=str(config_path),
                    dataset_dir=str(self._dataset_dir),
                    logs_dir=self._logs_dir,
                )
            except Exception as e:
                logger.warning(f"Agent run failed for {task}: {e}")
                outputs.append({"error": str(e)})
                scores.append(0.0)
                if capture_traces:
                    trajectories.append({"error": str(e)})
                continue

            eval_result = self._evaluate_patch(task, run_result)

            feature_scores = []
            for feat_id in task["features"]:
                feat_key = f"feature{feat_id}"
                feat_result = eval_result.get(feat_key, {})
                feature_scores.append(1.0 if feat_result.get("passed") else 0.0)

            task_score = sum(feature_scores) / len(feature_scores) if feature_scores else 0.0

            outputs.append(
                {
                    "run_result": run_result,
                    "eval_result": eval_result,
                }
            )
            scores.append(task_score)

            if capture_traces:
                asi = extract_asi(task, run_result or {}, eval_result)
                trajectories.append(asi)

        if config_path.exists():
            config_path.unlink()

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflective dataset from evaluation traces.

        For each component to update, produces a list of records with:
          - Inputs: the task description
          - Generated Outputs: summary of what the agent did
          - Feedback: ASI including test results and failure details
        """
        result: dict[str, list[dict[str, Any]]] = {}
        if eval_batch.trajectories is None:
            return result

        for component in components_to_update:
            records = []
            for i, trajectory in enumerate(eval_batch.trajectories):
                if isinstance(trajectory, dict) and "error" in trajectory:
                    records.append(
                        {
                            "Inputs": {"component": component, "task_index": i},
                            "Generated Outputs": "Agent run failed",
                            "Feedback": trajectory["error"],
                            "Score": 0.0,
                        }
                    )
                    continue

                task_info = trajectory if isinstance(trajectory, dict) else {}
                asi_text = format_asi_for_reflection(task_info) if task_info else "No trace available"
                score = eval_batch.scores[i] if i < len(eval_batch.scores) else 0.0

                records.append(
                    {
                        "Inputs": {
                            "component": component,
                            "task": task_info.get("task", f"task_{i}"),
                            "current_prompt": candidate.get(component, "")[:500],
                        },
                        "Generated Outputs": {
                            "status": task_info.get("status", "Unknown"),
                            "steps": task_info.get("agent_steps", 0),
                            "patch_files": task_info.get("patch_files_modified", 0),
                            "patch_lines_added": task_info.get("patch_lines_added", 0),
                        },
                        "Feedback": asi_text,
                        "Score": score,
                    }
                )

            result[component] = records

        return result

    def _evaluate_patch(
        self,
        task: dict[str, Any],
        run_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Evaluate an agent's patch against each feature's test suite."""
        if not run_result:
            return {}

        result_data = run_result.get("result", run_result)
        patch = result_data.get("patch", "")
        log_dir = run_result.get("log_dir")

        if not patch.strip():
            return {f"feature{fid}": {"passed": False, "error": "Empty patch"} for fid in task["features"]}

        patch_path = None
        if log_dir:
            p = Path(log_dir) / "solo.patch"
            if p.exists():
                patch_path = str(p)

        eval_out = {}
        for feat_id in task["features"]:
            try:
                feat_result = test_solo(
                    repo_name=task["repo"],
                    task_id=task["task_id"],
                    feature_id=feat_id,
                    agent_patch=patch_path or patch,
                    backend=self.eval_backend,
                    dataset_dir=str(self._dataset_dir),
                )
                eval_out[f"feature{feat_id}"] = {
                    "feature_id": feat_id,
                    "passed": feat_result.get("passed", False),
                    "tests_passed": feat_result.get("tests_passed", 0),
                    "tests_failed": feat_result.get("tests_failed", 0),
                    "tests_total": feat_result.get("tests_total", 0),
                    "output": feat_result.get("output", ""),
                }
            except Exception as e:
                logger.warning(f"Evaluation failed for feature {feat_id}: {e}")
                eval_out[f"feature{feat_id}"] = {
                    "feature_id": feat_id,
                    "passed": False,
                    "error": str(e),
                }

        return eval_out

    def _write_candidate_config(self, candidate: dict[str, str]) -> Path:
        """Write candidate prompts to a temporary YAML config file.

        The YAML structure matches what mini_swe_agent_v2's adapter expects
        when ``--agent-config`` is passed.
        """
        config: dict[str, Any] = {}

        agent_cfg: dict[str, Any] = {}
        if "system_prompt" in candidate:
            agent_cfg["system_template"] = candidate["system_prompt"]
        if "instance_template" in candidate:
            agent_cfg["instance_template"] = candidate["instance_template"]
        if self.step_limit is not None:
            agent_cfg["step_limit"] = self.step_limit

        config["agent"] = agent_cfg

        model_cfg: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {"drop_params": True}
        if self.model_api_base:
            model_kwargs["api_base"] = self.model_api_base
        if self.model_api_key:
            model_kwargs["api_key"] = self.model_api_key
        model_cfg["model_kwargs"] = model_kwargs
        config["model"] = model_cfg

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="gepa_candidate_",
            delete=False,
        )
        yaml.dump(config, tmp, default_flow_style=False)
        tmp.close()
        return Path(tmp.name)
