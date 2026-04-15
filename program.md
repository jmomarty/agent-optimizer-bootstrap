# Program

You are the outer optimizer for this repo.

## Goal

Improve the inner agent while keeping changes generic and selecting candidates by
held-out `dev` score rather than by the visible `train` benchmark alone.

## Fixed Commands

```bash
python3 prepare.py
```

```bash
uv run benchmark.py --agent-dir inner_agent --split train
```

```bash
uv run benchmark.py --agent-dir inner_agent --split dev
```

```bash
uv run benchmark.py --agent-dir inner_agent --task <task_id>
```

## Operating Rules

1. Edit only files under `inner_agent/**`.
2. Do not change `prepare.py`, `benchmark.py`, or files under `benchmarks/**`.
3. Keep the evaluator and metric stable so candidate runs are comparable.
4. Prefer simple, legible improvements over brittle complexity.
5. Ignore evaluator-only fields such as `expected_answer`, `expected_output`, and `expected_command`.
6. Do not add benchmark-family dispatch or special-casing by `kind`.
7. Do not add handcrafted answer paths keyed to fields like `document`, `record`, or `goal` in order to bypass the model loop.
8. Do not optimize by reproducing benchmark answer formats with handwritten parsers or command constructors.
9. Improvements should come from prompt design, provider usage, loop behavior, tool usage, memory, or other generic response-shaping behavior.

## Conduct Rules

- Prefer changes that improve multiple tasks for one general reason.
- Avoid post-processing that exists only to satisfy benchmark formatting quirks.
- Do not add logic that copies evaluator-specific punctuation or surface forms unless a task explicitly asks for verbatim extraction.
- Prefer prompt, tool, provider, loop, or generic type-shaping changes over task-shaped rewrites.
- If a change can only be justified by reference to a few specific tasks, treat it as suspicious.

## Baseline

Run `python3 prepare.py`, then run both splits once:

1. `uv run benchmark.py --agent-dir inner_agent --split train`
2. `uv run benchmark.py --agent-dir inner_agent --split dev`

If `results.tsv` is empty except for the header, log that pair of runs as the
baseline row with `status=baseline` and `is_best=yes`.

Use this header in `results.tsv`:

```text
run_id	train_score	train_passed	train_failed	dev_score	dev_passed	dev_failed	status	is_best	description
```

## Loop

The loop is:

1. Inspect `runs/current/tasks.jsonl` only after `train` runs.
2. Inspect `results.tsv` to find the current best `dev` score.
3. Form one concrete hypothesis for improving the inner agent.
4. Make a focused change only in `inner_agent/**`.
5. Run `uv run benchmark.py --agent-dir inner_agent --split train`.
6. Run `uv run benchmark.py --agent-dir inner_agent --split dev`.
7. Append one row to `results.tsv` with both the `train` and `dev` metrics.
8. Keep the change only if `dev` improves over the current best `dev` score.
9. If `dev` is equal or worse, discard the change and mark the row `is_best=no`.
10. If `dev` improves, mark the new row `is_best=yes` and update the previous best row to `is_best=no`.
11. Stop after 3 consecutive non-improving `dev` runs.

## Visibility Rules

- `train` is the visible optimization split.
- `dev` is held out for model selection.
- Do not inspect detailed `dev` predictions or per-task `dev` failures.
- Treat `dev` as aggregate-only. Use only the summary metrics printed by the benchmark and stored in `summary.json`.
- Use `uv run benchmark.py --agent-dir inner_agent --task <task_id>` only for `train` debugging.

## Run Monitoring

When a benchmark is running, monitor progress in:

- stdout progress lines from `benchmark.py`
- `runs/current/progress.json`

For `train` runs, `runs/current/tasks.jsonl` is also available.
For `dev` runs, detailed task outputs are intentionally hidden.

Treat `runs/current/progress.json` as the source of truth for live status. A run
is complete when `status` is `completed` and `num_tasks_completed` matches
`num_tasks_total`.
