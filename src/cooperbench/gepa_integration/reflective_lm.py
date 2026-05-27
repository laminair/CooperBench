"""Reflective LM wrapper that captures reasoning traces from thinking models.

Wraps litellm to call models like GLM-5.1 that return ``reasoning_content``
alongside ``content``. Conforms to GEPA's ``LanguageModel`` protocol
(``(str | list[dict]) -> str``) so it can be used as ``reflection_lm``
in ``gepa.optimize()``.

All reasoning traces are saved to ``<run_dir>/reasoning/`` as JSON artifacts.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReflectiveLM:
    """Language model wrapper that captures reasoning traces.

    Args:
        model: LiteLLM model identifier (e.g. ``"openai/glm-5.1"``).
        api_base: OpenAI-compatible API base URL.
        api_key: API key for the endpoint.
        temperature: Sampling temperature (0.6 recommended for thinking models).
        max_tokens: Maximum tokens for the response.
        run_dir: Directory to save reasoning traces under ``reasoning/``.
        extra_headers: Additional HTTP headers (e.g. ``{"User-Agent": "opencode/1.0"}``).
        num_retries: Number of retries on transient failures.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float = 0.6,
        max_tokens: int = 8192,
        run_dir: str | Path | None = None,
        extra_headers: dict[str, str] | None = None,
        num_retries: int = 3,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}
        self.num_retries = num_retries

        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._lock = threading.Lock()
        self._call_count: int = 0

        self._reasoning_dir: Path | None = None
        if run_dir is not None:
            self._reasoning_dir = Path(run_dir) / "reasoning"
            self._reasoning_dir.mkdir(parents=True, exist_ok=True)

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        import litellm

        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        completion = litellm.completion(
            model=self.model,
            messages=messages,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            num_retries=self.num_retries,
            drop_params=True,
            extra_headers=self.extra_headers,
        )

        choice = completion.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)

        usage = getattr(completion, "usage", None)
        tokens_in = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        tokens_out = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        try:
            cost = litellm.completion_cost(completion_response=completion) or 0.0
        except Exception:
            cost = 0.0

        with self._lock:
            self._call_count += 1
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

        if reasoning and self._reasoning_dir is not None:
            self._save_trace(self._call_count, messages, content, reasoning, tokens_in, tokens_out, cost)

        return content

    def _save_trace(
        self,
        call_idx: int,
        messages: list[dict[str, Any]],
        content: str,
        reasoning: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
    ) -> None:
        if self._reasoning_dir is None:
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"call_{call_idx:04d}_{ts}.json"
        path = self._reasoning_dir / filename

        trace = {
            "call_idx": call_idx,
            "timestamp": ts,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "reasoning_content": reasoning,
            "content": content,
            "usage": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "estimated_cost": cost,
            },
        }

        try:
            with open(path, "w") as f:
                json.dump(trace, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save reasoning trace: {e}")

    def __repr__(self) -> str:
        return f"ReflectiveLM(model={self.model!r}, temperature={self.temperature}, max_tokens={self.max_tokens})"
