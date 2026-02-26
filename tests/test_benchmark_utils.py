from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_rlm import _extract_first_int, build_tasks


def test_extract_first_int() -> None:
    assert _extract_first_int("answer: 42") == 42
    assert _extract_first_int("-17 is the value") == -17
    assert _extract_first_int("no integer here") is None


def test_build_tasks_expected_values_are_stable() -> None:
    tasks = build_tasks(seed=7, records=40)
    assert len(tasks) == 3
    assert tasks[0].task_id == "t1_beta_sum"
    assert tasks[0].expected_int > 0
    assert "category=beta" in tasks[0].query
    assert "idx=" in tasks[0].context
