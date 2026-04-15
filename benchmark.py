from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "benchmarks" / "manifest.json"
RUNS_DIR = ROOT / "runs"


@dataclass
class TaskResult:
    task_id: str
    kind: str
    split: str
    reward: int
    expected: Any
    prediction: Any


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing manifest at {MANIFEST_PATH}. Run `python3 prepare.py` first."
        )
    return load_json(MANIFEST_PATH)


def select_tasks(manifest: dict, task_id: str | None, split: str | None) -> list[dict]:
    selected = []
    for entry in manifest["tasks"]:
        if task_id is not None and entry["id"] != task_id:
            continue
        if split is not None and entry["split"] != split:
            continue
        selected.append(entry)
    return selected


def load_agent(agent_dir: str):
    agent_path = (ROOT / agent_dir).resolve()
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent dir does not exist: {agent_path}")
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module(f"{agent_path.name}.agent")
    return module.Agent(agent_path)


def normalize_string(value: str) -> str:
    return " ".join(value.strip().split())


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def evaluate_prediction(task: dict, prediction: Any) -> int:
    kind = task["kind"]
    if kind == "docs_qa":
        return int(normalize_string(str(prediction)) == normalize_string(task["expected_answer"]))
    if kind == "structured_extract":
        return int(prediction == task["expected_output"])
    if kind == "terminal_command":
        return int(normalize_string(str(prediction)) == normalize_string(task["expected_command"]))
    raise ValueError(f"Unsupported task kind: {kind}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_task_result(path: Path, result: TaskResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def running_score(results: list[TaskResult]) -> float:
    if not results:
        return 0.0
    return sum(item.reward for item in results) / len(results)


def details_visible(selected: list[dict]) -> bool:
    return bool(selected) and all(entry["split"] == "train" for entry in selected)


def progress_task_id(task_id: str | None, *, show_details: bool) -> str | None:
    if not show_details:
        return None
    return task_id


def initialize_run_state(
    *,
    run_id: str,
    agent_dir: str,
    total_tasks: int,
    task_filter: str | None,
    split_filter: str | None,
    show_details: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    current_dir = RUNS_DIR / "current"
    current_dir.mkdir(parents=True, exist_ok=True)

    (current_dir / "run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    tasks_path = current_dir / "tasks.jsonl"
    tasks_path.write_text("", encoding="utf-8")

    started_at = now_utc_iso()
    progress = {
        "run_id": run_id,
        "agent_dir": agent_dir,
        "status": "running",
        "details_visible": show_details,
        "num_tasks_total": total_tasks,
        "num_tasks_completed": 0,
        "num_passed_so_far": 0,
        "num_failed_so_far": 0,
        "current_task_id": None,
        "task_filter": task_filter,
        "split_filter": split_filter,
        "started_at": started_at,
        "updated_at": started_at,
        "finished_at": None,
        "failed_at": None,
    }
    progress_path = current_dir / "progress.json"
    write_json(progress_path, progress)
    return current_dir, tasks_path, progress


def update_progress(
    progress_path: Path,
    progress: dict[str, Any],
    *,
    current_task_id: str | None,
    results: list[TaskResult],
    status: str | None = None,
    error_message: str | None = None,
    finished: bool = False,
) -> None:
    progress["current_task_id"] = current_task_id
    progress["num_tasks_completed"] = len(results)
    progress["num_passed_so_far"] = sum(item.reward for item in results)
    progress["num_failed_so_far"] = len(results) - progress["num_passed_so_far"]
    progress["updated_at"] = now_utc_iso()
    if status is not None:
        progress["status"] = status
    if error_message is not None:
        progress["error_message"] = error_message
    if finished:
        progress["finished_at"] = now_utc_iso()
    if status == "failed":
        progress["failed_at"] = now_utc_iso()
    write_json(progress_path, progress)


def build_summary(
    *,
    run_id: str,
    agent_dir: str,
    task_id: str | None,
    split: str | None,
    results: list[TaskResult],
    started_at: str,
    finished_at: str,
    show_details: bool,
) -> dict[str, Any]:
    mean_score = running_score(results)
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    return {
        "run_id": run_id,
        "agent_dir": agent_dir,
        "details_visible": show_details,
        "num_tasks": len(results),
        "num_passed": sum(item.reward for item in results),
        "num_failed": len(results) - sum(item.reward for item in results),
        "score": mean_score,
        "task_filter": task_id,
        "split_filter": split,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": (finished_dt - started_dt).total_seconds(),
    }


def run(agent_dir: str, task_id: str | None, split: str | None) -> tuple[dict, list[TaskResult]]:
    manifest = load_manifest()
    selected = select_tasks(manifest, task_id=task_id, split=split)
    if not selected:
        raise ValueError("No benchmark tasks matched the request.")

    show_details = details_visible(selected)
    agent = load_agent(agent_dir)
    results: list[TaskResult] = []
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    current_dir, tasks_path, progress = initialize_run_state(
        run_id=run_id,
        agent_dir=agent_dir,
        total_tasks=len(selected),
        task_filter=task_id,
        split_filter=split,
        show_details=show_details,
    )
    progress_path = current_dir / "progress.json"

    print(f"run_id: {run_id}")
    print(f"tasks_total: {len(selected)}")
    print(f"task_filter: {task_id}")
    print(f"split_filter: {split}")
    print(f"details_visible: {show_details}")

    try:
        for index, entry in enumerate(selected, start=1):
            task = load_json(ROOT / entry["path"])
            update_progress(
                progress_path,
                progress,
                current_task_id=progress_task_id(task["id"], show_details=show_details),
                results=results,
            )
            if show_details:
                print(f"[{index}/{len(selected)}] task={task['id']} kind={task['kind']}")
            else:
                print(f"[{index}/{len(selected)}] evaluating held-out task")

            prediction = agent.solve(task)
            reward = evaluate_prediction(task, prediction)
            expected = task.get("expected_answer", task.get("expected_output", task.get("expected_command")))
            result = TaskResult(
                task_id=task["id"],
                kind=task["kind"],
                split=task["split"],
                reward=reward,
                expected=expected,
                prediction=prediction,
            )
            results.append(result)
            if show_details:
                append_task_result(tasks_path, result)
            update_progress(
                progress_path,
                progress,
                current_task_id=None,
                results=results,
            )
            print(
                f"  -> {'PASS' if reward else 'FAIL'} "
                f"passed={sum(item.reward for item in results)}/{len(results)} "
                f"score={running_score(results):.3f}"
            )
    except Exception as exc:
        update_progress(
            progress_path,
            progress,
            current_task_id=progress.get("current_task_id"),
            results=results,
            status="failed",
            error_message=str(exc),
            finished=True,
        )
        raise

    update_progress(
        progress_path,
        progress,
        current_task_id=None,
        results=results,
        status="completed",
        finished=True,
    )
    summary = build_summary(
        run_id=run_id,
        agent_dir=agent_dir,
        task_id=task_id,
        split=split,
        results=results,
        started_at=progress["started_at"],
        finished_at=progress["finished_at"],
        show_details=show_details,
    )
    return summary, results


def write_run_artifacts(summary: dict, results: list[TaskResult]) -> Path:
    history_dir = RUNS_DIR / "history" / summary["run_id"]
    history_dir.mkdir(parents=True, exist_ok=True)
    current_dir = RUNS_DIR / "current"
    current_dir.mkdir(parents=True, exist_ok=True)

    summary_path = history_dir / "summary.json"
    tasks_path = history_dir / "tasks.jsonl"
    summary_json = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(summary_json, encoding="utf-8")

    if summary["details_visible"]:
        with tasks_path.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")
    else:
        tasks_path.write_text("", encoding="utf-8")

    (current_dir / "summary.json").write_text(summary_json, encoding="utf-8")
    (current_dir / "run_id.txt").write_text(summary["run_id"] + "\n", encoding="utf-8")

    progress_path = current_dir / "progress.json"
    if progress_path.exists():
        history_progress = history_dir / "progress.json"
        history_progress.write_text(progress_path.read_text(encoding="utf-8"), encoding="utf-8")

    return history_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed benchmark against an inner agent.")
    parser.add_argument("--agent-dir", required=True, help="Path to the mutable inner agent directory.")
    parser.add_argument("--split", default=None, help="Benchmark split to run, for example train or dev.")
    parser.add_argument("--task", default=None, help="Optional single task id to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, results = run(agent_dir=args.agent_dir, task_id=args.task, split=args.split)
    history_dir = write_run_artifacts(summary, results)

    print(f"run_id: {summary['run_id']}")
    print(f"tasks: {summary['num_tasks']}")
    print(f"passed: {summary['num_passed']}")
    print(f"failed: {summary['num_failed']}")
    print(f"score: {summary['score']:.3f}")
    print(f"history_dir: {history_dir}")


if __name__ == "__main__":
    main()
