from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rlm_runner import MockAdapter, OpenAICompatibleAdapter, RLMConfig, RLMRunner


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _build_real_adapter(model: str) -> OpenAICompatibleAdapter:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()

    provider = os.getenv("WORKFLOW_PROVIDER", "openai")
    if provider != "openai":
        raise RuntimeError("WORKFLOW_PROVIDER must be 'openai' for this example")
    api_base = os.getenv("WORKFLOW_API_BASE") or os.getenv("CUSTOM_API_BASE")
    api_key = os.getenv("WORKFLOW_API_KEY") or os.getenv("CUSTOM_API_KEY")

    if not api_base or not api_key:
        raise RuntimeError(
            "Set WORKFLOW_API_BASE and WORKFLOW_API_KEY (or CUSTOM_API_BASE/CUSTOM_API_KEY)."
        )

    return OpenAICompatibleAdapter(api_base=api_base, api_key=api_key, model=model)


def run_rlm(
    topic: str,
    *,
    email_text: str,
    context: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ = topic
    _ = payload

    workflow_input = context.get("input", {}) if isinstance(context, dict) else {}
    if not isinstance(workflow_input, dict):
        workflow_input = {}

    query = workflow_input.get("rlm_query")
    if not isinstance(query, str) or not query.strip():
        query = email_text

    source_context = workflow_input.get("rlm_context")
    if source_context is None:
        source_context = workflow_input.get("context", "")

    model = workflow_input.get("rlm_model")
    if not isinstance(model, str) or not model.strip():
        model = os.getenv("WORKFLOW_MODEL") or "gemini-3-flash"

    mock_mode = _as_bool(workflow_input.get("rlm_mock"), default=False)

    config = RLMConfig(
        max_turns=_as_int(workflow_input.get("rlm_max_turns"), default=8),
        max_subcalls=_as_int(workflow_input.get("rlm_max_subcalls"), default=24),
        max_subcall_prompt_chars=_as_int(
            workflow_input.get("rlm_max_subcall_prompt_chars"), default=16000
        ),
    )

    adapter = MockAdapter() if mock_mode else _build_real_adapter(model)
    result = RLMRunner(
        adapter=adapter,
        query=query,
        context=source_context,
        config=config,
    ).run()

    trace_path = workflow_input.get("rlm_trace_path")
    if isinstance(trace_path, str) and trace_path.strip():
        trace_file = Path(trace_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return {
        "decision": "completed" if result.answer else "failed",
        "answer": result.answer,
        "termination_reason": result.termination_reason,
        "turns": result.turns,
        "subcalls": result.subcalls,
        "elapsed_ms": result.elapsed_ms,
        "model": model,
        "mock_mode": mock_mode,
        "trace": result.trace,
    }
