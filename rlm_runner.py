from __future__ import annotations

import contextlib
import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from simple_agents_py import Client


class LLMAdapter(Protocol):
    def call_root(self, *, system_prompt: str, user_prompt: str) -> str: ...

    def call_sub(self, *, prompt: str) -> str: ...


@dataclass
class RLMConfig:
    max_turns: int = 8
    max_subcalls: int = 24
    max_subcall_prompt_chars: int = 16000
    max_stdout_chars: int = 4000


@dataclass
class RLMResult:
    answer: str
    termination_reason: str
    turns: int
    subcalls: int
    elapsed_ms: int
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "termination_reason": self.termination_reason,
            "turns": self.turns,
            "subcalls": self.subcalls,
            "elapsed_ms": self.elapsed_ms,
            "trace": self.trace,
        }


@dataclass
class ParsedModelTurn:
    code_blocks: list[str]
    final_var: str | None
    final_text: str | None


class SimpleAgentsAdapter:
    def __init__(
        self,
        *,
        model: str,
        provider: str = "openai",
        api_base: str | None = None,
        api_key: str | None = None,
        client: Client | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._client = client or Client(
            provider,
            api_base=api_base,
            api_key=api_key,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def call_root(self, *, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self._client.complete(
            self._model,
            messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return _extract_text(response)

    def call_sub(self, *, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise sub-LLM inside an RLM. "
                    "Return only the requested answer without extra commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = self._client.complete(
            self._model,
            messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return _extract_text(response)


class MockAdapter:
    """Deterministic adapter for local smoke tests."""

    def call_root(self, *, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return (
            "```repl\n"
            "context_text = str(context)\n"
            "answer_text = (\n"
            '    f"[mock-rlm] query={query}; context_chars={len(context_text)}; "\n'
            '    f"context_preview={context_text[:120]}"\n'
            ")\n"
            "```\n"
            "FINAL_VAR(answer_text)"
        )

    def call_sub(self, *, prompt: str) -> str:
        trimmed = prompt[:120]
        return f"[mock-sub] {trimmed}"


class RLMRunner:
    SYSTEM_PROMPT = (
        "You are the root model in a Recursive Language Model loop. "
        "You must reason via REPL code and only finalize with FINAL(...) or FINAL_VAR(variable_name). "
        "The full context is available in REPL variable `context`; use direct code over `context` whenever possible. "
        "Call llm_query(prompt) only when semantic interpretation is necessary. "
        "Do not emit FINAL(...) inside REPL code blocks. "
        "Prefer exact programmatic computation over estimates."
    )

    _RE_REPL = re.compile(r"```repl\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    _RE_FINAL_VAR = re.compile(r"FINAL_VAR\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
    _RE_FINAL = re.compile(r"FINAL\(\s*(.*?)\s*\)", re.DOTALL)

    def __init__(
        self,
        *,
        adapter: LLMAdapter,
        query: str,
        context: Any,
        config: RLMConfig | None = None,
    ) -> None:
        self._adapter = adapter
        self._query = query
        self._context = context
        self._config = config or RLMConfig()
        self._subcalls = 0
        self._trace: list[dict[str, Any]] = []

    def run(self) -> RLMResult:
        started = time.perf_counter()
        state: dict[str, Any] = {
            "query": self._query,
            "context": self._context,
        }
        last_stdout = ""

        for turn in range(1, self._config.max_turns + 1):
            user_prompt = self._build_turn_prompt(turn=turn, last_stdout=last_stdout)
            model_text = self._adapter.call_root(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            parsed = self._parse_model_turn(model_text)

            turn_record: dict[str, Any] = {
                "turn": turn,
                "model_output_preview": model_text[:600],
                "code_blocks": len(parsed.code_blocks),
                "final_var": parsed.final_var,
                "has_final_text": parsed.final_text is not None,
                "executions": [],
            }

            stdout_chunks: list[str] = []
            for code_block in parsed.code_blocks:
                stdout_text, error_text = self._execute_code_block(code_block, state)
                execution_record = {
                    "code_preview": code_block[:500],
                    "stdout_preview": stdout_text[:400],
                    "error": error_text,
                }
                turn_record["executions"].append(execution_record)
                if stdout_text:
                    stdout_chunks.append(stdout_text)

            last_stdout = "\n".join(stdout_chunks)
            if len(last_stdout) > self._config.max_stdout_chars:
                last_stdout = last_stdout[: self._config.max_stdout_chars]

            self._trace.append(turn_record)

            if parsed.final_var is not None:
                if parsed.final_var not in state:
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    return RLMResult(
                        answer="",
                        termination_reason=(f"final_var_missing:{parsed.final_var}"),
                        turns=turn,
                        subcalls=self._subcalls,
                        elapsed_ms=elapsed_ms,
                        trace=self._trace,
                    )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                return RLMResult(
                    answer=str(state[parsed.final_var]),
                    termination_reason="final_var",
                    turns=turn,
                    subcalls=self._subcalls,
                    elapsed_ms=elapsed_ms,
                    trace=self._trace,
                )

            if parsed.final_text is not None:
                finalized_text = parsed.final_text
                stdout_ints = self._extract_stdout_ints(stdout_chunks)
                if len(stdout_ints) == 1:
                    finalized_text = str(stdout_ints[0])
                    self._trace[-1]["final_text_corrected_from_stdout"] = True
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                return RLMResult(
                    answer=finalized_text,
                    termination_reason="final_text",
                    turns=turn,
                    subcalls=self._subcalls,
                    elapsed_ms=elapsed_ms,
                    trace=self._trace,
                )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RLMResult(
            answer="",
            termination_reason="max_turns_exceeded",
            turns=self._config.max_turns,
            subcalls=self._subcalls,
            elapsed_ms=elapsed_ms,
            trace=self._trace,
        )

    def _build_turn_prompt(self, *, turn: int, last_stdout: str) -> str:
        context_text = str(self._context)
        preview = context_text[:500]
        return (
            f"Turn: {turn}\n"
            f"Query: {self._query}\n"
            f"Context chars: {len(context_text)}\n"
            f"Context preview: {preview}\n"
            f"Subcalls used: {self._subcalls}/{self._config.max_subcalls}\n"
            "Last REPL stdout (truncated):\n"
            f"{last_stdout}\n\n"
            "Respond with one or more ```repl code blocks``` and then FINAL(...) or FINAL_VAR(...)."
        )

    def _parse_model_turn(self, model_text: str) -> ParsedModelTurn:
        code_blocks = self._RE_REPL.findall(model_text)
        non_code_text = self._RE_REPL.sub("", model_text)
        final_var_match = self._RE_FINAL_VAR.search(non_code_text)
        final_match = self._RE_FINAL.search(non_code_text)
        final_var = final_var_match.group(1) if final_var_match is not None else None
        final_text = None
        if final_var is None and final_match is not None:
            final_text = final_match.group(1).strip()
        return ParsedModelTurn(
            code_blocks=code_blocks, final_var=final_var, final_text=final_text
        )

    def _safe_builtins(self) -> dict[str, Any]:
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
            "math": math,
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
            raise RuntimeError(f"Import not allowed in REPL: {name}")

        allowed_builtins["__import__"] = restricted_import
        return allowed_builtins

    def _execute_code_block(
        self, code_block: str, state: dict[str, Any]
    ) -> tuple[str, str | None]:
        scope: dict[str, Any] = dict(state)
        scope["llm_query"] = self._llm_query
        scope["__builtins__"] = self._safe_builtins()

        stdout = io.StringIO()
        error_text: str | None = None

        try:
            with contextlib.redirect_stdout(stdout):
                exec(code_block, scope, scope)
        except Exception as error:  # noqa: BLE001
            error_text = str(error)

        reserved_names = {"llm_query", "__builtins__"}
        for key, value in scope.items():
            if key in reserved_names:
                continue
            state[key] = value

        return stdout.getvalue(), error_text

    @staticmethod
    def _extract_stdout_ints(stdout_chunks: list[str]) -> list[int]:
        values: list[int] = []
        for chunk in stdout_chunks:
            for match in re.findall(r"-?\d+", chunk):
                values.append(int(match))
        return sorted(set(values))

    def _llm_query(self, prompt: str) -> str:
        if self._subcalls >= self._config.max_subcalls:
            raise RuntimeError("max_subcalls_exceeded")
        normalized = str(prompt)
        if len(normalized) > self._config.max_subcall_prompt_chars:
            normalized = normalized[: self._config.max_subcall_prompt_chars]

        response = self._adapter.call_sub(prompt=normalized)
        self._subcalls += 1
        return response


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)
