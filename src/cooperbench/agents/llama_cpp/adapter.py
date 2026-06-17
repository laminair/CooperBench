"""Llama.cpp adapter for CooperBench.

This adapter wraps mini-swe-agent v2 infrastructure configured to use a
local llama.cpp server (llama-server) via litellm with OpenAI-compatible
tool calling.
"""

import logging
import os
from pathlib import Path

import yaml

from cooperbench.agents import AgentResult
from cooperbench.agents.llama_cpp.config import get_config_path
from cooperbench.agents.mini_swe_agent_v2.agents.default import DefaultAgent
from cooperbench.agents.mini_swe_agent_v2.connectors import GitConnector
from cooperbench.agents.mini_swe_agent_v2.connectors.messaging import MessagingConnector
from cooperbench.agents.mini_swe_agent_v2.models.litellm_model import LitellmModel
from cooperbench.agents.registry import register
from cooperbench.team_harness import (
    COOP_TASK_SCRIPT_PATH,
    INSTALL_SNIPPET_PATH,
    TeamHarnessConfig,
    TeamSession,
)

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_CPP_BASE_URL = "http://localhost:8050/v1"
DEFAULT_LLAMA_CPP_API_KEY = "local-llama-cpp"


def _resolve_llama_cpp_base_url() -> str:
    return os.environ.get("LLAMA_CPP_BASE_URL", DEFAULT_LLAMA_CPP_BASE_URL)


def _resolve_llama_cpp_api_key() -> str:
    return os.environ.get("LLAMA_CPP_API_KEY", DEFAULT_LLAMA_CPP_API_KEY)


def recursive_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base, returning a new dict.

    When both values are dicts, merge recursively.  Otherwise override wins.
    """
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = recursive_merge(result[key], value)
        else:
            result[key] = value
    return result


def _install_team_cli_in_container(env) -> None:
    """Drop the team CLI scripts into the agent container."""
    from cooperbench.agents._coop.runtime import write_file_in_container

    try:
        write_file_in_container(env, "/tmp/cb-coop-task.py", COOP_TASK_SCRIPT_PATH.read_text())
        write_file_in_container(env, "/tmp/cb-team-install.sh", INSTALL_SNIPPET_PATH.read_text())
        env.execute(
            {
                "command": (
                    "pip install --quiet --disable-pip-version-check redis >/dev/null 2>&1 "
                    "|| pip3 install --quiet --disable-pip-version-check redis >/dev/null 2>&1 "
                    "|| true; "
                    "bash /tmp/cb-team-install.sh"
                )
            },
            timeout=120,
        )
    except Exception as e:
        logger.warning("team CLI install in container failed: %s", e)


@register("llama_cpp")
class LlamaCppRunner:
    """Adapter for local llama.cpp server via litellm (OpenAI-compatible tool calling)."""

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "openai/Qwen3.6-27B-Q4_K_M.gguf",
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
        **kwargs,
    ) -> AgentResult:
        llama_cpp_base_url = _resolve_llama_cpp_base_url()
        llama_cpp_api_key = _resolve_llama_cpp_api_key()

        # Team-mode setup
        is_team = bool(team_role and team_id and task_list_url and agents and len(agents) > 1)
        team_session: TeamSession | None = None
        if is_team:
            team_session = TeamSession(
                run_id=team_id or "",
                redis_url=task_list_url or "",
                agents=list(agents or []),
                team_volume=str((config or {}).get("team_volume") or ""),
                config=team_features or TeamHarnessConfig(),
            )
            section = team_session.prompt_section(agent_id=agent_id)
            if section:
                task = task + "\n\n---\n\n" + section

        team_poller = team_session.loop_poller(agent_id=agent_id) if team_session else None

        # Load coop or solo config
        is_coop = bool(agents) and len(agents) > 1
        config_name = "coop" if is_coop else "solo"
        config_path = get_config_path(config_name)
        with open(config_path) as f:
            default_config = yaml.safe_load(f)

        # Merge agent_config overrides
        if agent_config:
            try:
                with open(agent_config) as f:
                    overrides = yaml.safe_load(f) or {}
                default_config = recursive_merge(default_config, overrides.get("config", overrides))
            except FileNotFoundError:
                logger.error(f"agent_config file not found: {agent_config}")
            except Exception as e:
                logger.error(f"Error loading agent_config {agent_config}: {e}")

        # Deep-merge passed config overrides
        if config is not None:
            default_config = recursive_merge(default_config, config)

        # Env-var overrides for compaction and context (e.g. when using
        # larger ctx-size on multi-GPU setups)
        trigger_str = os.environ.get("LLAMA_CPP_COMPACTION_TRIGGER")
        if trigger_str:
            default_config.setdefault("agent", {})["compaction_token_trigger"] = int(trigger_str)

        agent_cfg = default_config.get("agent", {})
        model_cfg = default_config.get("model", {})
        env_cfg = default_config.get("environment", {})
        backend = default_config.get("backend", "docker")

        # Inject llama.cpp server URL and API key into model_kwargs.
        # Config YAML provides defaults, env vars override.
        model_kwargs = dict(model_cfg.get("model_kwargs") or {})
        model_kwargs["api_base"] = llama_cpp_base_url
        model_kwargs["api_key"] = llama_cpp_api_key
        model_cfg = {**model_cfg, "model_kwargs": model_kwargs}

        # Create environment
        env_kwargs = {
            "image": image,
            "cwd": "/workspace/repo",
            "timeout": 3600,
        }
        container_env = dict(env_cfg.get("env") or {})
        if team_session is not None:
            container_env.update(team_session.env_for(agent_id))
        if container_env:
            env_kwargs["env"] = container_env

        if backend == "docker":
            from cooperbench.agents.mini_swe_agent_v2.environments.docker import DockerEnvironment

            # Apply run_args / network from YAML environment config so that
            # per-deployment overrides (e.g. host networking on Enroot/HPC)
            # can be set via --agent-config without editing source.
            if env_cfg.get("run_args") is not None:
                env_kwargs["run_args"] = list(env_cfg["run_args"])
            if env_cfg.get("network") is not None:
                env_kwargs["network"] = env_cfg["network"]
            # git_network from the harness-passed config dict (coop infra) takes
            # precedence over env_cfg.network if both are set.
            if config and config.get("git_network"):
                env_kwargs["network"] = config["git_network"]
            if team_session is not None:
                run_args = list(env_kwargs.get("run_args") or ["--rm"])
                if "--add-host=host.docker.internal:host-gateway" not in run_args:
                    run_args.append("--add-host=host.docker.internal:host-gateway")
                run_args.extend(team_session.scratchpad_mount_args())
                env_kwargs["run_args"] = run_args
            env = DockerEnvironment(**env_kwargs)
        else:
            from cooperbench.agents.mini_swe_agent_v2.environments.modal import ModalEnvironment

            env = ModalEnvironment(**env_kwargs)

        # Setup messaging
        comm = None
        use_messaging = messaging_enabled and comm_url and agents and len(agents) > 1
        if use_messaging:
            comm = MessagingConnector(agent_id=agent_id, agents=agents, url=comm_url)

        model = LitellmModel(model_name=model_name, **model_cfg)

        # Setup git
        if git_enabled and git_server_url and agents:
            git_connector = GitConnector(
                agent_id=agent_id,
                agents=agents,
                server_url=git_server_url,
            )
            git_connector.setup(env)

        # Install team CLI if team mode with task_list or protocol features
        if team_session is not None and (team_session.config.task_list or team_session.config.protocol):
            _install_team_cli_in_container(env)

        # Create agent
        extra_vars = {
            "agent_id": agent_id if (agents and len(agents) > 1) else None,
            "agents": agents if agents else [],
            "git_enabled": git_enabled,
            "messaging_enabled": messaging_enabled,
        }

        agent = DefaultAgent(
            model=model,
            env=env,
            comm=comm,
            agent_id=agent_id,
            **agent_cfg,
        )
        agent.extra_template_vars.update(extra_vars)
        if team_poller is not None:
            agent.team_poller = team_poller

        # Run agent
        error_msg = None
        result = {}
        try:
            result = agent.run(task=task)
            status = result.get("exit_status", "Submitted")
        except Exception as e:
            status = "Error"
            error_msg = str(e)

        # Harvest patch
        patch = ""
        try:
            r = env.execute({"command": "cat patch.txt 2>/dev/null"})
            if r.get("returncode") == 0:
                from cooperbench.agents._coop.runtime import normalize_patch

                patch = normalize_patch(r.get("output") or "")
                if not patch or len(patch.strip()) < 50:
                    logger.warning(
                        "[%s] Empty or invalid patch detected (%d chars). "
                        "Agent may have submitted without making changes.",
                        agent_id,
                        len(patch),
                    )
        except Exception:
            pass

        # Save full trajectory if compaction occurred
        if log_dir and agent._compaction_count > 0:
            traj_path = Path(log_dir) / f"{agent_id}_full_traj.json"
            agent.save(traj_path)
            logger.info(
                f"[{agent_id}] Full trajectory with segments saved to {traj_path} "
                f"({agent._compaction_count} compaction(s))"
            )

        # Cleanup
        env.cleanup()

        # Sanitize null content in messages
        sanitized_messages = []
        for msg in agent.messages:
            if msg.get("content") is None:
                msg = {**msg, "content": ""}
            sanitized_messages.append(msg)

        return AgentResult(
            status=status,
            patch=patch,
            cost=agent.cost,
            steps=agent.n_calls,
            messages=sanitized_messages,
            sent_messages=agent.sent_messages,
            error=error_msg,
        )
