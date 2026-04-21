# Agent Optimizer Bootstrap Fresh

This repo is a clean restart of the optimizer scaffold. It keeps the generic
Vertex-backed inner-agent structure and the benchmark observability work, while
resetting the benchmark protocol to reduce overfitting pressure.

The main benchmark is now BFCL-primary: most tasks are generated from Berkeley
Function Calling Leaderboard samples, with a tiny synthetic smoke subset kept to
exercise the local harness task kinds.

The main boundaries are:

- Codex is guided by `program.md` in one long-running session.
- Only `inner_agent/**` is mutable during optimizer runs.
- `prepare.py`, `benchmark.py`, and `benchmarks/**` stay fixed.
- `train` is the visible optimization split.
- `dev` is a held-out selection split with aggregate-only visibility.

## Quick Start

```bash
python3 prepare.py
```

Configure a repo-root `.env` file if needed:

```bash
AGENT_PROVIDER=vertex
AGENT_MODEL=gemini-2.5-pro
VERTEX_PROJECT=your-project-id
VERTEX_LOCATION=global
```

Run the visible split:

```bash
uv run benchmark.py --agent-dir inner_agent --split train
```

Run the held-out split:

```bash
uv run benchmark.py --agent-dir inner_agent --split dev
```

Run a single task for train debugging:

```bash
uv run benchmark.py --agent-dir inner_agent --task <task_id>
```

Regenerate the BFCL-derived benchmark tasks:

```bash
python3 scripts/import_bfcl_samples.py
```

## Repo Layout

```text
prepare.py
benchmark.py
program.md
pyproject.toml
results.tsv

inner_agent/
benchmarks/
runs/
scripts/
tests/
```

## What Changed from the Earlier Bootstrap

- The benchmark is now split into `train` and `dev`.
- The main benchmark now uses BFCL-derived function-calling tasks.
- `dev` exposes aggregate metrics only by default.
- `benchmark.py` writes live progress and timing metadata.
- `results.tsv` is intended to track both `train` and `dev` metrics and mark
  the current best checkpoint by `dev` score.

This repo is intended to be the clean baseline that you commit and copy forward
before each new optimizer run.
