# Program

You are a coding agent improving `inner_agent/` under a fixed benchmark.

## Setup

1. Read `README.md` and `inner_agent/`.
2. Run `python3 prepare.py`.
3. Check `results.tsv` for the current best `dev` score.
4. Use the existing Git config for commits, so the human remains the author.

## Loop

1. Make one small, generic change under `inner_agent/**`.
2. Run `uv run benchmark.py --agent-dir inner_agent --split train`.
3. Run `uv run benchmark.py --agent-dir inner_agent --split dev`.
4. Log train/dev results in `results.tsv`.
5. Keep the change only if held-out `dev` improves.

## Rules

- Do not edit `prepare.py`, `benchmark.py`, or `benchmarks/**`.
- Do not inspect detailed `dev` failures; use aggregate `dev` score only.
- Do not special-case task ids, benchmark families, or expected fields.
- Do not add dependencies.
- Prefer deleting code to adding code. Simpler wins ties.
