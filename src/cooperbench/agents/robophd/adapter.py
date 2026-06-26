"""RoboPhD agent adapter for CooperBench.

Runs an evolved RoboPhD agent inside the task Docker container.  The
agent directory (supplied via ``--agent-config``) must contain at least
an ``agent.py`` with a ``solve()`` function.  Additional Python files
are also copied and made importable.

The container runner handles:
  - Setting up litellm with the configured model
  - Writing the task instruction to /tmp/cb-instruction.txt
  - Importing and calling the agent's solve() function
  - Producing /workspace/repo/patch.txt via ``git diff``
  - Optional coop/team messaging when COOP_AGENT_ID is set

Design mirrors the claude_code adapter: build environment, write files,
execute, collect outputs.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from cooperbench.agents import AgentResult
from cooperbench.agents._coop import (
    build_git_setup_command,
    build_instruction,
    parse_sent_messages_log,
    rewrite_comm_url_for_container,
)
from cooperbench.agents._coop.runtime import (
    CONTAINER_COOP_MSG_PATH,
    CONTAINER_COOP_SEND_LOG,
    CONTAINER_INSTRUCTION_PATH,
    CONTAINER_REPO_PATH,
    build_environment,
    normalize_patch,
    read_file_from_container,
    write_file_in_container,
)
from cooperbench.agents.registry import register
from cooperbench.team_harness import (
    COOP_TASK_SCRIPT_PATH as TEAM_TASK_SCRIPT_PATH,
)
from cooperbench.team_harness import (
    INSTALL_SNIPPET_PATH as TEAM_INSTALL_SNIPPET_PATH,
)
from cooperbench.team_harness import (
    TeamHarnessConfig,
    TeamSession,
)

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
RUNNER_SCRIPT_PATH = _PACKAGE_DIR / "container_runner.py"
COOP_MSG_SCRIPT_PATH = _PACKAGE_DIR.parent / "_coop" / "coop_msg.py"
COOP_INSTALL_SNIPPET_PATH = _PACKAGE_DIR.parent / "_coop" / "install_snippet.sh"

CONTAINER_AGENT_DIR = "/opt/robophd"
CONTAINER_METRICS_PATH = "/tmp/robophd_metrics.json"
CONTAINER_TEAM_TASK_PATH = "/tmp/cb-coop-task.py"
CONTAINER_TEAM_INSTALL_PATH = "/tmp/cb-team-install.sh"


def _resolve_model_endpoint() -> dict[str, str]:
    """Pick up an OpenAI-compatible API endpoint from the environment.

    Supports OPENAI_API_KEY, OPENAI_BASE_URL, and LITELLM_MODEL
    for vLLM or other local deployments common in CooperBench setups.
    """
    env: dict[str, str] = {}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    model_override = os.environ.get("LITELLM_MODEL", "").strip()
    if model_override:
        env["LITELLM_MODEL"] = model_override
    return env


def _load_agent_files(agent_config: str) -> dict[str, str]:
    """Load all Python files from the agent directory into a name->content map."""
    agent_dir = Path(agent_config)
    if not agent_dir.is_dir():
        raise ValueError(f"Agent config path must be a directory: {agent_config}")
    files: dict[str, str] = {}
    for py_file in sorted(agent_dir.glob("*.py")):
        files[py_file.name] = py_file.read_text()
    if "agent.py" not in files:
        raise ValueError(f"No agent.py found in {agent_config}; expected at least agent.py with a solve() function")
    return files


def _build_run_command(
    model_name: str,
    *,
    coop_env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> str:
    """Compose the in-container shell command that runs the RoboPhD agent."""
    env_exports = ""
    if coop_env:
        env_exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in coop_env.items())
    env_exports += f"export ROBOPHD_MODEL={shlex.quote(model_name)}; "

    model_endpoint = _resolve_model_endpoint()
    for k, v in model_endpoint.items():
        env_exports += f"export {k}={shlex.quote(v)}; "

    return (
        f"{env_exports}"
        f"cd {CONTAINER_REPO_PATH} && "
        f"PYTHONPATH={shlex.quote(CONTAINER_AGENT_DIR)}:$PYTHONPATH "
        f"python3 {shlex.quote(f'{CONTAINER_AGENT_DIR}/_runner.py')} "
        f"2>&1"
    )


@register("robophd")
class RoboPhDRunner:
    """Adapter for evolved RoboPhD agents.

    The ``--agent-config`` flag must point to a directory containing at
    least ``agent.py`` with a ``solve(task, repo_path, llm_fn)`` function.
    Additional ``.py`` files are copied alongside and become importable.

    Supports solo, coop (Redis messaging), and team modes.
    """

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "gpt-4o",
        agents: list[str] | None = None,
        comm_url: str | None = None,
        git_server_url: str | None = None,
        git_enabled: bool = False,
        messaging_enabled: bool = True,
        config: dict | None = None,
        agent_config: str | None = None,
        log_dir: str | None = None,
        team_role: str | None = None,
        team_id: str | None = None,
        task_list_url: str | None = None,
        team_features: TeamHarnessConfig | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        config = config or {}

        # Resolve the agent directory from one of:
        # 1. --agent-config pointing to a YAML file with agent_dir key
        # 2. --agent-config pointing directly to a directory (with agent.py)
        # 3. config["agent_dir"] set programmatically
        #
        # Note: the solo/coop runner tries to load --agent-config as YAML first.
        # If agent_config is a directory, that YAML load will fail with
        # IsADirectoryError — but we catch it here and use the directory directly.
        agent_dir: str | None = config.get("agent_dir")
        if agent_config:
            config_path = Path(agent_config)
            if config_path.is_dir():
                agent_dir = str(config_path.resolve())
            elif config_path.is_file():
                import yaml

                with open(config_path) as f:
                    agent_config_dict = yaml.safe_load(f) or {}
                config.update(agent_config_dict)
                if not agent_dir:
                    agent_dir = config.get("agent_dir")
            else:
                raise ValueError(f"Agent config path does not exist: {agent_config}")
        if not agent_dir:
            raise ValueError(
                "RoboPhD adapter requires --agent-config <path> pointing to a "
                "directory with agent.py, or a YAML file with an 'agent_dir' key."
            )
        agent_dir = str(Path(agent_dir).resolve())

        is_coop = bool(messaging_enabled and comm_url and agents and len(agents) > 1)
        use_git = bool(git_enabled and git_server_url and agents and len(agents) > 1)
        is_team = bool(team_role and team_id and task_list_url and agents and len(agents) > 1)

        team_session: TeamSession | None = None
        if is_team:
            team_session = TeamSession(
                run_id=team_id or "",
                redis_url=task_list_url or "",
                agents=list(agents or []),
                team_volume=str(config.get("team_volume") or ""),
                config=team_features or TeamHarnessConfig(),
            )

        if team_session is not None:
            instruction = team_session.prompt_for(
                task=task,
                agent_id=agent_id,
                git_enabled=use_git,
            )
        else:
            instruction = build_instruction(
                task,
                agents=agents if is_coop else None,
                agent_id=agent_id if is_coop else None,
                git_enabled=use_git,
            )

        coop_env: dict[str, str] = {}
        extra_run_args: list[str] = []
        if is_coop:
            container_url = rewrite_comm_url_for_container(comm_url) or ""
            coop_env = {
                "COOP_REDIS_URL": container_url,
                "COOP_AGENT_ID": agent_id,
                "COOP_AGENTS": ",".join(agents or []),
                "COOP_LOG_PATH": CONTAINER_COOP_SEND_LOG,
            }
            extra_run_args.append("--add-host=host.docker.internal:host-gateway")
        if team_session is not None:
            coop_env.update(team_session.env_for(agent_id))
            extra_run_args.extend(team_session.scratchpad_mount_args())
            if "--add-host=host.docker.internal:host-gateway" not in extra_run_args:
                extra_run_args.append("--add-host=host.docker.internal:host-gateway")

        network = config.get("git_network") if isinstance(config, dict) else None
        backend = config.get("backend", "docker") if isinstance(config, dict) else "docker"
        env = build_environment(
            image,
            network=network,
            extra_run_args=extra_run_args or None,
            backend=backend,
        )

        status = "Error"
        error_msg: str | None = None
        patch_text = ""
        metrics_text = ""
        sent_log_text = ""

        agent_files = _load_agent_files(agent_dir)

        try:
            # 1. Install Python deps (pip install litellm and redis for coop)
            install_cmd = "pip install -q litellm 2>/dev/null; "
            if is_coop:
                install_cmd += "pip install -q redis 2>/dev/null; "
            install_result = env.execute(
                {"command": install_cmd},
                timeout=300,
            )
            if install_result.get("returncode") not in (0, None):
                logger.warning(
                    "pip install had issues: %s",
                    (install_result.get("output") or "")[:500],
                )

            # 2. Copy agent files into the container
            for filename, content in agent_files.items():
                write_file_in_container(
                    env,
                    f"{CONTAINER_AGENT_DIR}/{filename}",
                    content,
                )

            # 3. Copy the runner script
            runner_source = RUNNER_SCRIPT_PATH.read_text()
            write_file_in_container(
                env,
                f"{CONTAINER_AGENT_DIR}/_runner.py",
                runner_source,
            )

            # 4. Drop the coop helper (if coop)
            if is_coop:
                write_file_in_container(
                    env,
                    CONTAINER_COOP_MSG_PATH,
                    COOP_MSG_SCRIPT_PATH.read_text(),
                )
                write_file_in_container(
                    env,
                    "/tmp/cb-coop-install.sh",
                    COOP_INSTALL_SNIPPET_PATH.read_text(),
                )
                env.execute(
                    {"command": "bash /tmp/cb-coop-install.sh"},
                    timeout=60,
                )

            # 5. Install team CLI (if team mode with task_list or protocol)
            install_team_cli = bool(team_session and (team_session.config.task_list or team_session.config.protocol))
            if install_team_cli:
                write_file_in_container(
                    env,
                    CONTAINER_TEAM_TASK_PATH,
                    TEAM_TASK_SCRIPT_PATH.read_text(),
                )
                write_file_in_container(
                    env,
                    CONTAINER_TEAM_INSTALL_PATH,
                    TEAM_INSTALL_SNIPPET_PATH.read_text(),
                )
                env.execute(
                    {"command": "bash /tmp/cb-team-install.sh"},
                    timeout=60,
                )

            # 6. Optional: configure shared git remote
            if use_git:
                git_cmd = build_git_setup_command(
                    agent_id=agent_id,
                    server_url=git_server_url or "",
                )
                git_result = env.execute({"command": git_cmd}, timeout=120)
                if git_result.get("returncode") not in (0, None):
                    logger.warning(
                        "git setup returned non-zero: %s",
                        (git_result.get("output") or "")[:500],
                    )

            # 7. Write instruction file
            write_file_in_container(env, CONTAINER_INSTRUCTION_PATH, instruction)

            # 8. Run the agent
            run_cmd = _build_run_command(
                model_name,
                coop_env=coop_env or None,
                timeout=config.get("timeout", 3600) if isinstance(config, dict) else 3600,
            )
            env.execute({"command": run_cmd}, timeout=config.get("timeout", 3600) if isinstance(config, dict) else 3600)

            # 9. Collect outputs
            patch_text = normalize_patch(read_file_from_container(env, f"{CONTAINER_REPO_PATH}/patch.txt"))
            metrics_text = read_file_from_container(env, CONTAINER_METRICS_PATH)
            if is_coop:
                sent_log_text = read_file_from_container(env, CONTAINER_COOP_SEND_LOG)

        except Exception as e:
            error_msg = str(e)
            logger.exception("RoboPhD adapter run failed")
        finally:
            try:
                env.cleanup()
            except Exception:
                logger.warning("Env cleanup failed", exc_info=True)

        # Parse metrics
        cost = 0.0
        steps = 0
        input_tokens = 0
        output_tokens = 0
        messages: list[dict[str, Any]] = []
        if metrics_text:
            try:
                metrics = json.loads(metrics_text)
                cost = metrics.get("cost_usd", 0.0)
                steps = metrics.get("llm_calls", 0)
                input_tokens = metrics.get("input_tokens", 0)
                output_tokens = metrics.get("output_tokens", 0)
                messages = metrics.get("messages", [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse robophd metrics JSON")

        sent_messages = parse_sent_messages_log(sent_log_text)

        if error_msg is not None:
            status = "Error"
        elif patch_text:
            status = "Submitted"
        else:
            status = "Error"

        if log_dir:
            try:
                log_root = Path(log_dir)
                log_root.mkdir(parents=True, exist_ok=True)
                if metrics_text:
                    (log_root / f"{agent_id}_metrics.json").write_text(metrics_text)
            except OSError:
                logger.warning("Failed to persist RoboPhD logs", exc_info=True)

        return AgentResult(
            status=status,
            patch=patch_text,
            cost=cost,
            steps=steps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            messages=messages,
            sent_messages=sent_messages,
            error=error_msg,
        )
