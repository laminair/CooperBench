"""RoboPhD container runner — executes inside the task Docker container.

This script is copied into the container at /opt/robophd/_runner.py and
invoked by the adapter.  It:

1. Reads the task instruction from /tmp/cb-instruction.txt
2. Sets up a litellm-based LLM function using ROBOPHD_MODEL
3. Imports the agent module from /opt/robophd/agent.py
4. Calls agent.solve(task, repo_path, llm_fn, **kwargs)
5. After the agent finishes, generates patch.txt via ``git diff``
6. Writes cost/token metrics to /tmp/robophd_metrics.json

If COOP_AGENT_ID is set, the runner makes coop messaging functions
(coop_send, coop_recv, coop_broadcast, coop_peek, coop_agents)
available to the agent via keyword arguments.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO_PATH = "/workspace/repo"
INSTRUCTION_PATH = "/tmp/cb-instruction.txt"
METRICS_PATH = "/tmp/robophd_metrics.json"
AGENT_DIR = "/opt/robophd"


def _resolve_model(model_name: str) -> str:
    """Ensure the model name has a litellm provider prefix.

    If OPENAI_BASE_URL is set (indicating a local/OpenAI-compatible server),
    add the 'openai/' prefix if the model name doesn't already have one.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def _setup_llm(model_name: str):
    """Create a litellm-based LLM function for the agent."""
    try:
        import litellm
    except ImportError:
        print("ERROR: litellm not installed; run: pip install litellm", file=sys.stderr)
        sys.exit(1)

    resolved_model = _resolve_model(model_name)

    _total_cost = [0.0]
    _total_input_tokens = [0]
    _total_output_tokens = [0]
    _call_count = [0]
    _messages = []

    def llm_fn(prompt: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Call the configured LLM model via litellm and return the text.

        Handles reasoning models (like Qwen3) that put content in
        reasoning_content by falling back to it when message.content is empty.
        """
        response = litellm.completion(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            drop_params=True,
        )
        msg = response.choices[0].message
        text = msg.content or ""
        if not text.strip():
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if reasoning and isinstance(reasoning, str):
                text = reasoning
        _call_count[0] += 1
        usage = getattr(response, "usage", None)
        if usage:
            _total_input_tokens[0] += getattr(usage, "prompt_tokens", 0) or 0
            _total_output_tokens[0] += getattr(usage, "completion_tokens", 0) or 0
        try:
            cost = litellm.completion_cost(response)
            _total_cost[0] += cost
        except Exception:
            pass
        _messages.append({"role": "user", "content": prompt})
        _messages.append({"role": "assistant", "content": text})
        return text

    llm_fn.cost_usd = lambda: _total_cost[0]
    llm_fn.call_count = lambda: _call_count[0]
    llm_fn.input_tokens = lambda: _total_input_tokens[0]
    llm_fn.output_tokens = lambda: _total_output_tokens[0]
    llm_fn.messages = lambda: list(_messages)

    return llm_fn


def _setup_coop():
    """Create coop messaging function dict if COOP_AGENT_ID is set."""
    agent_id = os.environ.get("COOP_AGENT_ID", "")
    if not agent_id:
        return {}

    def coop_send(recipient, message):
        subprocess.run(
            ["coop-send", recipient, message],
            capture_output=True,
            timeout=30,
        )

    def coop_recv():
        result = subprocess.run(
            ["coop-recv"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            return json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError:
            return []

    def coop_broadcast(message):
        subprocess.run(
            ["coop-broadcast", message],
            capture_output=True,
            timeout=30,
        )

    def coop_peek():
        result = subprocess.run(
            ["coop-peek"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        try:
            return int(result.stdout.strip()) if result.stdout.strip() else 0
        except ValueError:
            return 0

    def coop_agents():
        agents_str = os.environ.get("COOP_AGENTS", "")
        return [a.strip() for a in agents_str.split(",") if a.strip()]

    return {
        "coop_send": coop_send,
        "coop_recv": coop_recv,
        "coop_broadcast": coop_broadcast,
        "coop_peek": coop_peek,
        "coop_agents": coop_agents,
        "agent_id": agent_id,
    }


def main():
    model_name = os.environ.get("ROBOPHD_MODEL", "gpt-4o")
    task = Path(INSTRUCTION_PATH).read_text()

    print(f"[robophd-runner] Starting agent with model={model_name}", file=sys.stderr)
    print(f"[robophd-runner] Task length: {len(task)} chars", file=sys.stderr)

    llm_fn = _setup_llm(model_name)
    coop_kwargs = _setup_coop()

    # Make agent dir importable
    sys.path.insert(0, AGENT_DIR)

    try:
        # Import the agent module
        import agent as agent_module
    except ImportError as e:
        print(f"[robophd-runner] ERROR: Cannot import agent.py: {e}", file=sys.stderr)
        sys.exit(1)

    solve_fn = getattr(agent_module, "solve", None)
    if solve_fn is None:
        print("[robophd-runner] ERROR: agent.py must define a solve() function", file=sys.stderr)
        sys.exit(1)

    try:
        result = solve_fn(
            task=task,
            repo_path=REPO_PATH,
            llm_fn=llm_fn,
            **coop_kwargs,
        )
        print(f"[robophd-runner] Agent returned: {type(result).__name__}", file=sys.stderr)
    except Exception as e:
        print(f"[robophd-runner] Agent raised exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # Generate the patch
    try:
        diff_result = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            cwd=REPO_PATH,
            timeout=30,
        )
        patch_path = Path(f"{REPO_PATH}/patch.txt")
        if diff_result.stdout:
            # Filter out test file changes unless task explicitly requests
            patch_path.write_text(diff_result.stdout)
            print(f"[robophd-runner] Wrote patch.txt ({len(diff_result.stdout.splitlines())} lines)", file=sys.stderr)
        else:
            # Try staged diff
            diff_cached = subprocess.run(
                ["git", "diff", "--cached"],
                capture_output=True,
                text=True,
                cwd=REPO_PATH,
                timeout=30,
            )
            if diff_cached.stdout:
                patch_path.write_text(diff_cached.stdout)
            else:
                # No changes — write empty patch
                patch_path.write_text("")
                print("[robophd-runner] WARNING: No changes detected, empty patch", file=sys.stderr)
    except Exception as e:
        print(f"[robophd-runner] WARNING: git diff failed: {e}", file=sys.stderr)
        Path(f"{REPO_PATH}/patch.txt").write_text("")

    # Write metrics
    metrics = {
        "cost_usd": llm_fn.cost_usd(),
        "llm_calls": llm_fn.call_count(),
        "input_tokens": llm_fn.input_tokens(),
        "output_tokens": llm_fn.output_tokens(),
        "messages": llm_fn.messages(),
    }
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2))
    print(f"[robophd-runner] Metrics: {llm_fn.call_count()} calls, ${llm_fn.cost_usd():.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
