"""Tests for DefaultAgent — checkpoint, early-submit, and message-truncation features."""

from unittest.mock import MagicMock

from cooperbench.agents.mini_swe_agent_v2.agents.default import AgentConfig, DefaultAgent


def _make_agent(**overrides) -> DefaultAgent:
    defaults = {
        "system_template": "You are a test agent.",
        "instance_template": "Task: {{ task }}",
        "step_limit": 100,
    }
    defaults.update(overrides)
    model = MagicMock()
    model.format_message = lambda **kw: kw
    model.query = MagicMock(return_value={"role": "assistant", "content": "ok", "extra": {"actions": [], "response": {"usage": {"prompt_tokens": 100}}}})
    model.get_template_vars = MagicMock(return_value={})
    env = MagicMock()
    env.execute = MagicMock(return_value={"output": "", "returncode": 0})
    env.get_template_vars = MagicMock(return_value={})
    return DefaultAgent(model=model, env=env, **defaults)


class TestCheckpointSteps:
    def test_config_default_empty(self):
        cfg = AgentConfig(system_template="x", instance_template="y")
        assert cfg.checkpoint_steps == []

    def test_config_accepts_checkpoint_list(self):
        cfg = AgentConfig(system_template="x", instance_template="y", checkpoint_steps=[25, 50, 75])
        assert cfg.checkpoint_steps == [25, 50, 75]

    def test_checkpoint_injected_at_configured_step(self):
        agent = _make_agent(checkpoint_steps=[2], step_limit=100)
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        agent.query()
        agent.query()
        user_msgs = [m for m in agent.messages if m.get("role") == "user"]
        checkpoint_msgs = [m for m in user_msgs if "[SYSTEM PROGRESS]" in (m.get("content") or "")]
        assert len(checkpoint_msgs) == 1
        assert "98 steps remaining" in checkpoint_msgs[0]["content"]

    def test_no_checkpoint_when_not_configured(self):
        agent = _make_agent(checkpoint_steps=[], step_limit=100)
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        for _ in range(5):
            agent.query()
        user_msgs = [m for m in agent.messages if m.get("role") == "user"]
        checkpoint_msgs = [m for m in user_msgs if "[SYSTEM PROGRESS]" in (m.get("content") or "")]
        assert len(checkpoint_msgs) == 0


class TestEarlySubmitStep:
    def test_config_default_zero(self):
        cfg = AgentConfig(system_template="x", instance_template="y")
        assert cfg.early_submit_step == 0

    def test_early_submit_injects_prompt_when_changes_exist(self):
        agent = _make_agent(early_submit_step=1, step_limit=100)
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        agent.env.execute = MagicMock(side_effect=[
            {"output": "", "returncode": 0},
            {"output": " M src/file.py\n", "returncode": 0},
        ])
        agent.model.query = MagicMock(return_value={
            "role": "assistant",
            "content": "ok",
            "extra": {"actions": [{"tool_name": "bash", "command": "ls"}], "response": {"usage": {"prompt_tokens": 100}}},
        })
        agent.step()
        user_msgs = [m for m in agent.messages if m.get("role") == "user"]
        submit_prompts = [m for m in user_msgs if "uncommitted changes" in (m.get("content") or "")]
        assert len(submit_prompts) == 1

    def test_early_submit_skipped_when_no_changes(self):
        agent = _make_agent(early_submit_step=1, step_limit=100)
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        agent.env.execute = MagicMock(side_effect=[
            {"output": "", "returncode": 0},
            {"output": "", "returncode": 0},
        ])
        agent.model.query = MagicMock(return_value={
            "role": "assistant",
            "content": "ok",
            "extra": {"actions": [{"tool_name": "bash", "command": "ls"}], "response": {"usage": {"prompt_tokens": 100}}},
        })
        agent.step()
        user_msgs = [m for m in agent.messages if m.get("role") == "user"]
        submit_prompts = [m for m in user_msgs if "uncommitted changes" in (m.get("content") or "")]
        assert len(submit_prompts) == 0

    def test_early_submit_disabled_when_zero(self):
        agent = _make_agent(early_submit_step=0, step_limit=100)
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        agent.env.execute = MagicMock(return_value={"output": " M src/file.py\n", "returncode": 0})
        agent.model.query = MagicMock(return_value={
            "role": "assistant",
            "content": "ok",
            "extra": {"actions": [{"tool_name": "bash", "command": "ls"}], "response": {"usage": {"prompt_tokens": 100}}},
        })
        agent.step()
        user_msgs = [m for m in agent.messages if m.get("role") == "user"]
        submit_prompts = [m for m in user_msgs if "uncommitted changes" in (m.get("content") or "")]
        assert len(submit_prompts) == 0


class TestMessageTruncation:
    def test_config_default_zero(self):
        cfg = AgentConfig(system_template="x", instance_template="y")
        assert cfg.message_truncate_chars == 0

    def test_long_message_is_truncated(self):
        comm = MagicMock()
        long_content = "A" * 5000
        comm.receive = MagicMock(return_value=[{"from": "agent2", "content": long_content, "timestamp": "2026-01-01T00:00:00"}])
        agent = _make_agent(message_truncate_chars=1000, step_limit=100)
        agent.comm = comm
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        agent.model.query = MagicMock(return_value={
            "role": "assistant",
            "content": "ok",
            "extra": {"actions": [], "response": {"usage": {"prompt_tokens": 100}}},
        })
        agent.step()
        user_msgs = [m for m in agent.messages if m.get("role") == "user" and "[Message from" in (m.get("content") or "")]
        assert len(user_msgs) == 1
        content = user_msgs[0]["content"]
        assert "[truncated]" in content
        assert len(content) < 5000

    def test_short_message_not_truncated(self):
        comm = MagicMock()
        comm.receive = MagicMock(return_value=[{"from": "agent2", "content": "Hello!", "timestamp": "2026-01-01T00:00:00"}])
        agent = _make_agent(message_truncate_chars=1000, step_limit=100)
        agent.comm = comm
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        agent.model.query = MagicMock(return_value={
            "role": "assistant",
            "content": "ok",
            "extra": {"actions": [], "response": {"usage": {"prompt_tokens": 100}}},
        })
        agent.step()
        user_msgs = [m for m in agent.messages if m.get("role") == "user" and "[Message from" in (m.get("content") or "")]
        assert len(user_msgs) == 1
        assert "[truncated]" not in user_msgs[0]["content"]
        assert "Hello!" in user_msgs[0]["content"]

    def test_truncation_disabled_when_zero(self):
        comm = MagicMock()
        long_content = "A" * 5000
        comm.receive = MagicMock(return_value=[{"from": "agent2", "content": long_content, "timestamp": "2026-01-01T00:00:00"}])
        agent = _make_agent(message_truncate_chars=0, step_limit=100)
        agent.comm = comm
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        agent.model.query = MagicMock(return_value={
            "role": "assistant",
            "content": "ok",
            "extra": {"actions": [], "response": {"usage": {"prompt_tokens": 100}}},
        })
        agent.step()
        user_msgs = [m for m in agent.messages if m.get("role") == "user" and "[Message from" in (m.get("content") or "")]
        assert len(user_msgs) == 1
        assert "[truncated]" not in user_msgs[0]["content"]
        assert long_content in user_msgs[0]["content"]
