from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from simple_agents_py import Client

from rlm_runner import MockAdapter, RLMConfig, RLMRunner, SimpleAgentsAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Recursive Language Model example with SimpleAgents + workflow"
    )
    parser.add_argument(
        "--mode",
        choices=["workflow", "direct"],
        default="workflow",
        help="Run through workflow wrapper or direct runner",
    )
    parser.add_argument(
        "--query",
        default="Summarize the main claim of this context in 2 bullet points.",
        help="RLM query/instruction",
    )
    parser.add_argument(
        "--context-file",
        default=None,
        help="Path to context text file",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Inline context text (used when --context-file is not set)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override (default from WORKFLOW_MODEL or gemini-3-flash)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with deterministic mock adapter (no external model calls)",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-subcalls", type=int, default=24)
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument(
        "--trace-file",
        default=None,
        help="Optional output path for JSON trace file",
    )
    return parser.parse_args()


def load_context(args: argparse.Namespace) -> str:
    if args.context_file is None:
        return args.context
    path = Path(args.context_file)
    if not path.exists():
        raise FileNotFoundError(f"context file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_openai_compatible_config(mock_mode: bool) -> tuple[str, str, str]:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()

    provider = os.getenv("WORKFLOW_PROVIDER", "openai")
    api_base = os.getenv("WORKFLOW_API_BASE") or os.getenv("CUSTOM_API_BASE")
    api_key = os.getenv("WORKFLOW_API_KEY") or os.getenv("CUSTOM_API_KEY")

    if mock_mode:
        return (
            provider,
            api_base or "http://localhost:1/v1",
            api_key or "dummy_api_key_for_mock_mode_12345",
        )

    if not api_base or not api_key:
        raise RuntimeError(
            "Set WORKFLOW_API_BASE and WORKFLOW_API_KEY (or CUSTOM_API_BASE/CUSTOM_API_KEY)."
        )

    return provider, api_base, api_key


def run_direct(args: argparse.Namespace, context_text: str) -> dict[str, Any]:
    model = args.model or os.getenv("WORKFLOW_MODEL") or "gemini-3-flash"

    if args.mock:
        adapter = MockAdapter()
    else:
        provider, api_base, api_key = load_openai_compatible_config(mock_mode=False)
        client = Client(provider, api_base=api_base, api_key=api_key)
        adapter = SimpleAgentsAdapter(client=client, model=model)

    config = RLMConfig(max_turns=args.max_turns, max_subcalls=args.max_subcalls)
    result = RLMRunner(
        adapter=adapter,
        query=args.query,
        context=context_text,
        config=config,
    ).run()
    result_dict = result.to_dict()
    result_dict["mode"] = "direct"
    result_dict["model"] = model
    result_dict["mock_mode"] = args.mock
    return result_dict


def run_workflow(args: argparse.Namespace, context_text: str) -> dict[str, Any]:
    provider, api_base, api_key = load_openai_compatible_config(mock_mode=args.mock)
    client = Client(provider, api_base=api_base, api_key=api_key)

    model = args.model or os.getenv("WORKFLOW_MODEL") or "gemini-3-flash"

    workflow_input = {
        "email_text": args.query,
        "rlm_query": args.query,
        "rlm_context": context_text,
        "rlm_model": model,
        "rlm_mock": args.mock,
        "rlm_max_turns": args.max_turns,
        "rlm_max_subcalls": args.max_subcalls,
        "messages": [
            {
                "role": "system",
                "content": "You are a workflow-driven Recursive Language Model assistant.",
            },
            {"role": "user", "content": args.query},
        ],
    }
    if args.trace_file:
        workflow_input["rlm_trace_path"] = args.trace_file

    workflow_path = Path(__file__).resolve().parent / "rlm_orchestrator.yaml"
    result = client.run_workflow_yaml(
        str(workflow_path),
        workflow_input,
        include_events=args.include_events,
        workflow_options={"telemetry": {"enabled": False}},
    )
    return {
        "mode": "workflow",
        "workflow_path": str(workflow_path),
        "result": result,
    }


def main() -> None:
    args = parse_args()
    context_text = load_context(args)

    output = (
        run_workflow(args, context_text)
        if args.mode == "workflow"
        else run_direct(args, context_text)
    )

    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
