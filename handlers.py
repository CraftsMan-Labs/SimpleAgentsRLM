from __future__ import annotations

import contextlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from simple_agents_py import Client

from rlm_runner import (
    MockAdapter,
    RLMConfig,
    RLMRunner,
    SimpleAgentsAdapter,
)


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


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)


def _load_provider_config() -> tuple[str, str, str]:
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

    return provider, api_base, api_key


def _build_real_adapter(model: str) -> SimpleAgentsAdapter:
    provider, api_base, api_key = _load_provider_config()
    client = Client(provider, api_base=api_base, api_key=api_key)
    return SimpleAgentsAdapter(client=client, model=model)


def run_rlm(
    topic: str,
    *,
    email_text: str,
    context: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    if mock_mode:
        adapter = MockAdapter()
        result = RLMRunner(
            adapter=adapter,
            query=query,
            context=source_context,
            config=config,
        ).run()
    else:
        adapter = _build_real_adapter(model)
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


def execute_repl(
    topic: str,
    *,
    email_text: str,
    context: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = payload if isinstance(payload, dict) else {}
    code = request.get("code")
    if not isinstance(code, str) or not code.strip():
        return {
            "stdout": "",
            "error": "Missing or empty code field",
            "state": request.get("state", {}),
        }

    state_obj = request.get("state", {})
    state = state_obj if isinstance(state_obj, dict) else {}

    allowed_builtins: dict[str, Any] = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "zip": zip,
    }

    allowed_modules = {
        "json": json,
        "re": re,
    }

    def restricted_import(
        name: str,
        _globals: Any = None,
        _locals: Any = None,
        _fromlist: Any = (),
        _level: int = 0,
    ) -> Any:
        if name in allowed_modules:
            return allowed_modules[name]
        raise RuntimeError(f"Import not allowed in execute_repl: {name}")

    allowed_builtins["__import__"] = restricted_import

    scope: dict[str, Any] = dict(state)
    scope["__builtins__"] = allowed_builtins

    stdout_buffer = io.StringIO()
    error_text: str | None = None
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, scope, scope)
    except Exception as error:  # noqa: BLE001
        error_text = str(error)

    next_state: dict[str, Any] = {}
    for key, value in scope.items():
        if key == "__builtins__":
            continue
        if callable(value):
            continue
        next_state[key] = value

    stdout_text = stdout_buffer.getvalue()
    if len(stdout_text) > 4000:
        stdout_text = stdout_text[:4000]

    return {
        "stdout": stdout_text,
        "error": error_text,
        "state": next_state,
    }


def llm_query(
    topic: str,
    *,
    email_text: str,
    context: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = payload if isinstance(payload, dict) else {}
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"answer": ""}

    provider, api_base, api_key = _load_provider_config()
    model = request.get("model")
    if not isinstance(model, str) or not model.strip():
        model = os.getenv("WORKFLOW_MODEL") or "gemini-3-flash"

    client = Client(provider, api_base=api_base, api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise sub-LLM call used by an RLM workflow. "
                "Return only the requested answer with no extra preamble."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    response = client.complete(model, messages, max_tokens=1024, temperature=0.0)
    return {"answer": _extract_text(response)}
