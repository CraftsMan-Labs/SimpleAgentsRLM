from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from simple_agents_py import Client

from rlm_runner import OpenAICompatibleAdapter, RLMConfig, RLMRunner


@dataclass
class BenchmarkTask:
    task_id: str
    description: str
    query: str
    context: str
    expected_int: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini benchmark: direct baseline vs RLM (direct + workflow)"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--records", type=int, default=400)
    parser.add_argument("--baseline-context-chars", type=int, default=2000)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-subcalls", type=int, default=30)
    parser.add_argument("--output", default=None, help="Optional JSON output file")
    return parser.parse_args()


def load_openai_compatible_config() -> tuple[str, str, str, str]:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()

    provider = os.getenv("WORKFLOW_PROVIDER", "openai")
    if provider != "openai":
        raise RuntimeError("WORKFLOW_PROVIDER must be 'openai' for this benchmark")

    api_base = os.getenv("WORKFLOW_API_BASE") or os.getenv("CUSTOM_API_BASE")
    api_key = os.getenv("WORKFLOW_API_KEY") or os.getenv("CUSTOM_API_KEY")
    model = os.getenv("WORKFLOW_MODEL") or "gemini-3-flash"

    if api_base is None or api_key is None:
        raise RuntimeError(
            "Set WORKFLOW_API_BASE and WORKFLOW_API_KEY (or CUSTOM_API_BASE/CUSTOM_API_KEY)."
        )

    return provider, api_base, api_key, model


def _extract_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def _trace_int_candidates(trace: Any) -> list[int]:
    if not isinstance(trace, list):
        return []
    values: list[int] = []
    for turn in trace:
        if not isinstance(turn, dict):
            continue
        executions = turn.get("executions")
        if not isinstance(executions, list):
            continue
        for execution in executions:
            if not isinstance(execution, dict):
                continue
            stdout_preview = execution.get("stdout_preview")
            if not isinstance(stdout_preview, str):
                continue
            value = _extract_first_int(stdout_preview)
            if value is not None:
                values.append(value)
    return values


def _pick_rlm_numeric_answer(raw_text: str, trace: Any) -> int | None:
    raw_value = _extract_first_int(raw_text)
    candidates = _trace_int_candidates(trace)
    if len(candidates) == 0:
        return raw_value
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return raw_value


def build_tasks(seed: int, records: int) -> list[BenchmarkTask]:
    rng = random.Random(seed)
    categories = ["alpha", "beta", "gamma", "delta"]
    rows: list[dict[str, Any]] = []
    for i in range(records):
        category = categories[i % len(categories)]
        value = rng.randint(1, 17)
        urgency = "high" if (i % 7 == 0 or value >= 14) else "low"
        rows.append(
            {
                "idx": i,
                "category": category,
                "value": value,
                "urgency": urgency,
            }
        )

    context_lines = [
        f"idx={row['idx']}|category={row['category']}|value={row['value']}|urgency={row['urgency']}"
        for row in rows
    ]
    context = "\n".join(context_lines)

    beta_sum = sum(int(row["value"]) for row in rows if row["category"] == "beta")
    high_count = sum(
        1 for row in rows if row["urgency"] == "high" and int(row["value"]) % 2 == 0
    )
    gamma_weighted = sum(
        int(row["value"]) * (1 + (int(row["idx"]) % 5))
        for row in rows
        if row["category"] == "gamma"
    )

    return [
        BenchmarkTask(
            task_id="t1_beta_sum",
            description="Linear aggregation over all rows",
            query=(
                "Context is newline-delimited rows in format idx=...|category=...|value=...|urgency=.... "
                "Compute the exact sum of value where category=beta. "
                "Use direct code over `context` (do not estimate). Return integer only."
            ),
            context=context,
            expected_int=beta_sum,
        ),
        BenchmarkTask(
            task_id="t2_high_even_count",
            description="Filter + count across full context",
            query=(
                "From the same row format, count entries where urgency=high AND value is even. "
                "Use direct code over `context` (do not estimate). Return integer only."
            ),
            context=context,
            expected_int=high_count,
        ),
        BenchmarkTask(
            task_id="t3_gamma_weighted",
            description="Weighted aggregation requiring full scan",
            query=(
                "From the same row format, compute exact sum of value*(1+(idx mod 5)) for rows where category=gamma. "
                "Use direct code over `context` (do not estimate). Return integer only."
            ),
            context=context,
            expected_int=gamma_weighted,
        ),
    ]


def run_baseline_direct(
    *,
    adapter: OpenAICompatibleAdapter,
    task: BenchmarkTask,
    baseline_context_chars: int,
) -> dict[str, Any]:
    truncated_context = task.context[:baseline_context_chars]
    system_prompt = (
        "You are a precise data assistant. Return integer only with no explanation."
    )
    user_prompt = (
        f"Task: {task.query}\n"
        f"Context (possibly truncated):\n{truncated_context}\n"
        "Return the integer answer only."
    )
    started = time.perf_counter()
    text = adapter.call_root(system_prompt=system_prompt, user_prompt=user_prompt)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    parsed = _extract_first_int(text)
    return {
        "method": "baseline_direct_truncated",
        "elapsed_ms": elapsed_ms,
        "raw_answer": text,
        "parsed_answer": parsed,
    }


def run_rlm_direct(
    *, adapter: OpenAICompatibleAdapter, task: BenchmarkTask, args: argparse.Namespace
) -> dict[str, Any]:
    started = time.perf_counter()
    result = RLMRunner(
        adapter=adapter,
        query=task.query,
        context=task.context,
        config=RLMConfig(max_turns=args.max_turns, max_subcalls=args.max_subcalls),
    ).run()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    parsed = _pick_rlm_numeric_answer(result.answer, result.trace)
    return {
        "method": "rlm_direct",
        "elapsed_ms": elapsed_ms,
        "raw_answer": result.answer,
        "parsed_answer": parsed,
        "turns": result.turns,
        "subcalls": result.subcalls,
        "termination_reason": result.termination_reason,
        "trace": result.trace,
    }


def run_rlm_workflow(
    *,
    client: Client,
    model: str,
    task: BenchmarkTask,
    args: argparse.Namespace,
) -> dict[str, Any]:
    workflow_path = Path(__file__).resolve().parent / "rlm_orchestrator.yaml"
    workflow_input = {
        "email_text": task.query,
        "rlm_query": task.query,
        "rlm_context": task.context,
        "rlm_model": model,
        "rlm_max_turns": args.max_turns,
        "rlm_max_subcalls": args.max_subcalls,
    }
    started = time.perf_counter()
    output = client.run_workflow_yaml(
        str(workflow_path),
        workflow_input,
        include_events=False,
        workflow_options={"telemetry": {"enabled": False}},
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    terminal_output = output.get("terminal_output")
    answer_text = ""
    turns = None
    subcalls = None
    termination_reason = None
    if isinstance(terminal_output, dict):
        answer_raw = terminal_output.get("answer")
        answer_text = answer_raw if isinstance(answer_raw, str) else str(answer_raw)
        turns = terminal_output.get("turns")
        subcalls = terminal_output.get("subcalls")
        termination_reason = terminal_output.get("termination_reason")

    parsed = _pick_rlm_numeric_answer(
        answer_text,
        terminal_output.get("trace") if isinstance(terminal_output, dict) else None,
    )
    return {
        "method": "rlm_workflow",
        "elapsed_ms": elapsed_ms,
        "raw_answer": answer_text,
        "parsed_answer": parsed,
        "turns": turns,
        "subcalls": subcalls,
        "termination_reason": termination_reason,
        "workflow_terminal_node": output.get("terminal_node"),
    }


def _is_correct(parsed_answer: int | None, expected: int) -> bool:
    return parsed_answer is not None and parsed_answer == expected


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        method = str(row["method"])
        grouped.setdefault(method, []).append(row)

    summary_methods: dict[str, dict[str, Any]] = {}
    for method, rows in grouped.items():
        accuracy = sum(1 for row in rows if row["correct"]) / max(1, len(rows))
        latencies = [int(row["elapsed_ms"]) for row in rows]
        turns = [int(row["turns"]) for row in rows if isinstance(row.get("turns"), int)]
        subcalls = [
            int(row["subcalls"]) for row in rows if isinstance(row.get("subcalls"), int)
        ]
        summary_methods[method] = {
            "accuracy": accuracy,
            "avg_elapsed_ms": statistics.mean(latencies) if latencies else 0,
            "max_elapsed_ms": max(latencies) if latencies else 0,
            "avg_turns": statistics.mean(turns) if turns else None,
            "avg_subcalls": statistics.mean(subcalls) if subcalls else None,
        }

    return {
        "by_method": summary_methods,
        "rows": results,
    }


def main() -> None:
    args = parse_args()
    provider, api_base, api_key, model_from_env = load_openai_compatible_config()
    model = args.model or model_from_env

    tasks = build_tasks(seed=args.seed, records=args.records)
    adapter = OpenAICompatibleAdapter(api_base=api_base, api_key=api_key, model=model)
    client = Client(provider, api_base=api_base, api_key=api_key)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        baseline = run_baseline_direct(
            adapter=adapter,
            task=task,
            baseline_context_chars=args.baseline_context_chars,
        )
        rlm_direct = run_rlm_direct(adapter=adapter, task=task, args=args)
        rlm_workflow = run_rlm_workflow(
            client=client, model=model, task=task, args=args
        )

        for method_result in (baseline, rlm_direct, rlm_workflow):
            parsed_answer = method_result.get("parsed_answer")
            parsed_int = parsed_answer if isinstance(parsed_answer, int) else None
            row = {
                "task_id": task.task_id,
                "task_description": task.description,
                "expected": task.expected_int,
                **method_result,
                "correct": _is_correct(parsed_int, task.expected_int),
            }
            rows.append(row)

    report = {
        "config": {
            "model": model,
            "seed": args.seed,
            "records": args.records,
            "baseline_context_chars": args.baseline_context_chars,
            "max_turns": args.max_turns,
            "max_subcalls": args.max_subcalls,
        },
        **summarize(rows),
    }

    print(json.dumps(report, indent=2, ensure_ascii=True))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
