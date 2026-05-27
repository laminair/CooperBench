"""Actionable Side Information (ASI) extraction from CooperBench runs.

Formats agent traces, bash commands, test results, and merge outcomes
as rich textual feedback for GEPA's reflection LM.
"""

from __future__ import annotations

from typing import Any


def extract_asi(
    task_info: dict[str, Any],
    agent_result: dict[str, Any],
    eval_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract Actionable Side Information from a CooperBench run.

    Args:
        task_info: Task dict with ``repo``, ``task_id``, ``features``.
        agent_result: Dict from ``execute_solo()`` containing ``result`` sub-dict.
        eval_result: Dict from ``_evaluate_single()`` with test outcomes, or None.

    Returns:
        Dict with structured ASI suitable for ``make_reflective_dataset()``.
    """
    result = agent_result.get("result", agent_result)

    messages = result.get("messages", [])
    bash_cmds = _extract_bash_commands(messages)
    patch_stats = _patch_stats(result.get("patch", ""))
    test_info = _extract_test_info(eval_result)

    return {
        "task": f"{task_info['repo']}/task{task_info['task_id']}",
        "features": task_info["features"],
        "status": result.get("status", "Unknown"),
        "bash_commands_executed": bash_cmds,
        "num_bash_commands": len(bash_cmds),
        "patch_lines_added": patch_stats["added"],
        "patch_lines_removed": patch_stats["removed"],
        "patch_files_modified": patch_stats["files"],
        "agent_steps": result.get("steps", 0),
        "agent_cost": result.get("cost", 0.0),
        **test_info,
    }


def format_asi_for_reflection(asi: dict[str, Any]) -> str:
    """Format ASI dict into a human-readable string for the reflection prompt.

    This is what gets placed in the ``<side_info>`` placeholder.
    """
    lines = []

    lines.append(f"## Task: {asi.get('task', 'unknown')}")
    lines.append(f"Features: {asi.get('features', [])}")
    lines.append(f"Agent Status: {asi.get('status', 'Unknown')}")
    lines.append(f"Steps taken: {asi.get('agent_steps', 0)}")
    lines.append("")

    if asi.get("test_results"):
        lines.append("## Test Results")
        for feat_id, outcome in asi["test_results"].items():
            status = "PASSED" if outcome.get("passed") else "FAILED"
            lines.append(f"  Feature {feat_id}: {status}")
            if not outcome.get("passed"):
                failed = outcome.get("tests_failed", 0)
                passed = outcome.get("tests_passed", 0)
                total = outcome.get("tests_total", 0)
                lines.append(f"    Tests: {passed}/{total} passed, {failed} failed")
                if outcome.get("output_snippet"):
                    lines.append(f"    Output snippet: {outcome['output_snippet']}")
        lines.append("")

    if asi.get("bash_commands_executed"):
        lines.append("## Bash Commands (last 20)")
        for cmd in asi["bash_commands_executed"][-20:]:
            lines.append(f"  $ {cmd}")
        lines.append("")

    lines.append("## Patch Stats")
    lines.append(f"  Files modified: {asi.get('patch_files_modified', 0)}")
    lines.append(f"  Lines added: {asi.get('patch_lines_added', 0)}")
    lines.append(f"  Lines removed: {asi.get('patch_lines_removed', 0)}")

    return "\n".join(lines)


def _extract_bash_commands(messages: list[dict[str, Any]]) -> list[str]:
    """Extract bash command strings from agent conversation messages."""
    commands = []
    for msg in messages:
        content = msg.get("content", "")
        if not content or not isinstance(content, str):
            continue
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == "bash":
                    import json

                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        cmd = args.get("command", "")
                        if cmd:
                            commands.append(cmd[:200])
                    except (json.JSONDecodeError, TypeError):
                        pass
    return commands


def _patch_stats(patch: str) -> dict[str, int]:
    """Count added/removed lines and modified files from a unified diff."""
    if not patch:
        return {"added": 0, "removed": 0, "files": 0}

    added = 0
    removed = 0
    files = set()

    for line in patch.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].strip()
            if path != "/dev/null" and not path.startswith("a/"):
                files.add(path.lstrip("b/"))
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    return {"added": added, "removed": removed, "files": len(files)}


def _extract_test_info(eval_result: dict[str, Any] | None) -> dict[str, Any]:
    """Extract per-feature test results from eval output."""
    if not eval_result:
        return {"test_results": None}

    test_results = {}

    for key, feat in eval_result.items():
        if not key.startswith("feature") or not isinstance(feat, dict):
            continue
        feat_id = feat.get("feature_id", key.replace("feature", ""))
        output = feat.get("output", "") or ""
        snippet = output[-500:] if len(output) > 500 else output
        test_results[str(feat_id)] = {
            "passed": feat.get("passed", False),
            "tests_passed": feat.get("tests_passed", 0),
            "tests_failed": feat.get("tests_failed", 0),
            "tests_total": feat.get("tests_total", 0),
            "output_snippet": snippet,
        }

    return {"test_results": test_results if test_results else None}
