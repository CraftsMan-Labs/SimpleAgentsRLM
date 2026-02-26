# Recursive Language Model (RLM) with SimpleAgents (Python)

Hello folks - this repo shows a practical, from-scratch implementation of a **Recursive Language Model** inspired by:

- RLM paper: https://arxiv.org/abs/2512.24601
- Python package: https://pypi.org/project/simple-agents-py/

The goal is simple: keep long context outside the one-shot prompt, reason with executable steps, and still keep workflow orchestration and traceability.

## What this repo contains

- `rlm_runner.py` - core recursive loop (parse `repl` code, execute, finalize)
- `handlers.py` - workflow custom worker (`RunRlm`) that invokes the runner
- `rlm_orchestrator.yaml` - workflow wrapper around the RLM core
- `run.py` - CLI runner (`direct` or `workflow` mode)
- `benchmark_rlm.py` - benchmark harness (baseline vs RLM)
- `tests/` - smoke + utility tests

## Visual architecture

```text
Query + Long Context
        |
        v
   RLM Root Loop
 (model -> repl code -> execute)
        |
        +--> llm_query(...) sub-call (optional)
        |
        v
 FINAL(...) / FINAL_VAR(...)
        |
        v
   Final answer + trace

Workflow path:
run.py -> run_workflow_yaml -> rlm_orchestrator.yaml -> handlers.py -> RLM runner
```

## Why this matches the RLM idea

As per the paper, this implementation keeps the prompt data as external state (`context` variable), and lets the model work over it through symbolic computation (`repl` code blocks), instead of forcing all reasoning into one token stream.

## Build from scratch (step by step)

### 1) Start a new Python project

```bash
uv init
```

### 2) Add dependencies

In `pyproject.toml`, add:

- `simple-agents-py`
- `python-dotenv`
- `pyyaml`
- `pytest` (dev)

Then install:

```bash
uv sync
uv sync --extra dev
```

### 3) Set up provider env

```bash
cp .env.example .env
```

Set these vars:

- `WORKFLOW_PROVIDER=openai`
- `WORKFLOW_API_BASE=<openai-compatible-base-url>`
- `WORKFLOW_API_KEY=<api-key>`
- `WORKFLOW_MODEL=gemini-3-flash`

### 4) Implement RLM core (`rlm_runner.py`)

Core flow:

1. Build a root prompt with query + context metadata.
2. Ask model for output containing ` ```repl ... ``` ` blocks.
3. Execute those blocks in controlled REPL scope.
4. Detect finalization via `FINAL(...)` or `FINAL_VAR(...)`.
5. Save per-turn traces (`code`, `stdout`, `error`, termination reason).

### 5) Add workflow integration

In `rlm_orchestrator.yaml`:

- Use `custom_worker` node
- Set handler to `RunRlm`

In `handlers.py`:

- Implement `run_rlm(...)`
- Read workflow payload/input
- Call `RLMRunner(...).run()`
- Return structured output (`answer`, `turns`, `subcalls`, `trace`)

### 6) Add CLI (`run.py`)

- `--mode direct`: run runner directly
- `--mode workflow`: run through workflow engine
- support `--query`, `--context`, `--context-file`, and optional trace path

### 7) Add benchmark + tests

- benchmark compares one-shot truncated baseline vs RLM methods
- tests validate parser/utilities and smoke-path behavior

## Running this project

### Install

```bash
uv sync
uv sync --extra dev
```

### Workflow mode (recommended)

```bash
uv run python run.py \
  --mode workflow \
  --query "Summarize this context in one sentence." \
  --context "RLM keeps context external and reasons with code."
```

### Direct mode

```bash
uv run python run.py \
  --mode direct \
  --query "Summarize in 2 bullets" \
  --context "Your long context here"
```

### Tests

```bash
uv run pytest -q
```

## Benchmark: proving behavior

```bash
uv run python benchmark_rlm.py \
  --records 400 \
  --baseline-context-chars 2000 \
  --output benchmark_report.json
```

Check these proof signals:

- `accuracy` by method
- `avg_turns`, `avg_subcalls`
- per-task `trace`, `termination_reason`, `correct`

## Learnings from building this

Here are my real takeaways:

1. **External state is the key difference**  
   Bigger context windows help, but external memory + symbolic steps give better control.

2. **Finalization contract is very important**  
   Models may compute correctly in code but phrase final text loosely. Trace-based checks keep outputs reliable.

3. **Workflow + RLM is a strong combo**  
   Workflow provides orchestration and guardrails; RLM provides flexible decomposition.

4. **Limits are non-negotiable**  
   `max_turns`, `max_subcalls`, and safe REPL scope are essential for production safety.

5. **Measure first, then scale**  
   Get benchmark harness in place early, then improve with async subcalls, deeper recursion, and tougher tasks.

## Next upgrades (recommended)

- async/batched subcalls for latency
- stricter REPL sandboxing
- task suite that forces semantic recursion (not only programmatic counting)
- richer cost/token telemetry in benchmark reports

---

If you want, I can also add a one-page diagram (`ARCHITECTURE.md`) with sequence flow and decision boundaries for publishing or LinkedIn screenshots.
