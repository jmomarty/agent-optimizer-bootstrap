# Program

You are the outer optimizer for this repo. Your job is to improve the inner
agent under a fixed evaluation protocol.

## Goal

Improve the inner agent while keeping changes generic, lightweight, and easy to
review. Select candidates by held-out `dev` score, not by visible `train`
performance alone.

The base repo is intentionally vanilla. Prefer changes that make the agent a
better general agent, not changes that memorize this benchmark.

## Agent Layers

Use these names when forming hypotheses and describing changes:

- **Instructions**: static behavior in `inner_agent/prompt.md`.
- **Model input**: the visible task object sent to the model.
- **Context engineering**: small dynamic guidance added before model calls,
  derived only from visible task context.
- **Agent loop**: model/tool/finish orchestration in `inner_agent/agent.py`.
- **Tools**: callable functions exposed to the model in `inner_agent/tools.py`.
- **Output handling**: minimal final-answer type conversion at `finish`.
- **Runtime config**: model, temperature, max turns, and search limits.
- **Run ledger**: `results.tsv`.
- **Evaluation harness / grader**: `benchmark.py`.

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

## Editable Surface

1. Edit only files under `inner_agent/**`.
2. Do not edit `prepare.py`, `benchmark.py`, or files under `benchmarks/**`.
3. Keep the evaluator, task data, and metric stable so candidate runs are
   comparable.
4. Keep the current file layout unless there is a clear simplicity reason to
   change it.

## Allowed Change Types

Prefer one focused change per candidate. Reasonable candidates include:

- improving instructions in `prompt.md`;
- improving context engineering from visible task context;
- improving tool descriptions or generic tool behavior;
- improving the agent loop;
- improving runtime config;
- improving output handling for runtime type conversion.

Output handling may convert an already-decided answer into the requested
runtime type, such as parsing a JSON object string into a dict. It must not
reinterpret, repair, or semantically rewrite content just to match benchmark
surface forms.

## Anti-Overfit Rules

- Do not inspect or use evaluator-only fields such as `expected_answer`,
  `expected_output`, or `expected_command`.
- Do not special-case task IDs.
- Do not dispatch on benchmark families such as `kind == "docs_qa"`.
- Do not add handcrafted solvers keyed to fields like `document`, `record`, or
  `goal`.
- Do not reproduce benchmark answer formats with handwritten parsers or command
  constructors.
- Do not add postprocessing that rewrites meaning to match expected answers.
- If a change can only be justified by reference to a few specific tasks, treat
  it as suspicious.

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

## Optimization Loop

For each candidate:

1. Inspect `results.tsv` to find the current best `dev` score.
2. Inspect `runs/current/tasks.jsonl` only after `train` runs.
3. Form one concrete hypothesis tied to an agent layer.
4. Make one focused change only in `inner_agent/**`.
5. Run `uv run benchmark.py --agent-dir inner_agent --split train`.
6. Run `uv run benchmark.py --agent-dir inner_agent --split dev`.
7. Append one row to `results.tsv` with both `train` and `dev` metrics.
8. Keep the change only if `dev` improves over the current best `dev` score.
9. If `dev` is equal or worse, discard the change and mark the row
   `is_best=no`.
10. If `dev` improves, mark the new row `is_best=yes` and update the previous
    best row to `is_best=no`.
11. Stop after 3 consecutive non-improving `dev` runs unless the human
    explicitly overrides patience.

## Visibility Rules

- `train` is the visible optimization split.
- `dev` is held out for model selection.
- Do not inspect detailed `dev` predictions or per-task `dev` failures.
- Treat `dev` as aggregate-only. Use only the summary metrics printed by the
  benchmark and stored in `summary.json`.
- Use `uv run benchmark.py --agent-dir inner_agent --task <task_id>` only for
  `train` debugging.

## Benchmark Direction

The current synthetic benchmark is a smoke test: tiny, local, and fast. It is
useful for validating the optimizer scaffold.

Future benchmark work may add a small real-sample set using the same local JSON
protocol. Prefer 10-30 curated examples first, not a full migration. Check
licenses and data shape before importing samples from sources such as
SWE-bench, tau-bench, or GAIA.

Do not mix real benchmark migration into a normal optimizer candidate unless
the human explicitly asks for benchmark work.

## Run Monitoring

When a benchmark is running, monitor progress in:

- stdout progress lines from `benchmark.py`;
- `runs/current/progress.json`.

For `train` runs, `runs/current/tasks.jsonl` is also available. For `dev` runs,
detailed task outputs are intentionally hidden.

Treat `runs/current/progress.json` as the source of truth for live status. A run
is complete when `status` is `completed` and `num_tasks_completed` matches
`num_tasks_total`.
