# Agent Optimizer Bootstrap

A tiny, coding-agent driven benchmark loop for improving an inner agent.

Inspired by Andrej Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch): keep the repo small,
keep the metric fixed, let coding agents try changes, and keep only what
improves.

## The Three Surfaces

- `prepare.py` checks fixed benchmark data.
- `benchmark.py` runs the fixed train/dev evaluation.
- `inner_agent/` is the agent implementation that experiments edit.

`program.md` is the operating manual for a coding agent.

## Quick Start

```bash
python3 prepare.py
```

Optional `.env`:

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

Debug one visible task:

```bash
uv run benchmark.py --agent-dir inner_agent --task <task_id>
```

## Rule

Optimize held-out `dev` score. Edit only `inner_agent/**`. Simpler wins ties.
