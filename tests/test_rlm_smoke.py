from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm_runner import MockAdapter, RLMConfig, RLMRunner


class ScriptedAdapter:
    def __init__(self) -> None:
        self._root_calls = 0

    def call_root(self, *, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        self._root_calls += 1
        if self._root_calls == 1:
            return (
                "```repl\n"
                "sub_answer = llm_query('extract key phrase from context')\n"
                "final_result = 'sub=' + sub_answer\n"
                "```\n"
                "FINAL_VAR(final_result)"
            )
        return "FINAL(unreachable)"

    def call_sub(self, *, prompt: str) -> str:
        _ = prompt
        return "mocked-sub"


def test_mock_adapter_completes_with_final_var() -> None:
    runner = RLMRunner(
        adapter=MockAdapter(),
        query="What is this?",
        context="A long context body.",
        config=RLMConfig(max_turns=3),
    )
    result = runner.run()
    assert result.termination_reason == "final_var"
    assert "[mock-rlm]" in result.answer
    assert result.turns == 1


def test_subcall_path_is_executed() -> None:
    runner = RLMRunner(
        adapter=ScriptedAdapter(),
        query="Find thing",
        context={"doc": "abc"},
        config=RLMConfig(max_turns=3, max_subcalls=2),
    )
    result = runner.run()
    assert result.termination_reason == "final_var"
    assert result.subcalls == 1
    assert result.answer == "sub=mocked-sub"
    assert isinstance(result.trace, list)
    assert len(result.trace) == 1
    first_turn: dict[str, Any] = result.trace[0]
    assert first_turn["code_blocks"] == 1
