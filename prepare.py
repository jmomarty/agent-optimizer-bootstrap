from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARKS_DIR = ROOT / "benchmarks"
RUNS_DIR = ROOT / "runs"
RESULTS_PATH = ROOT / "results.tsv"
RESULTS_HEADER = (
    "run_id\ttrain_score\ttrain_passed\ttrain_failed\tdev_score\tdev_passed\t"
    "dev_failed\tstatus\tis_best\tdescription\n"
)


def main() -> None:
    manifest_path = BENCHMARKS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing benchmark manifest at {manifest_path}.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for task in manifest["tasks"]:
        task_path = ROOT / task["path"]
        if not task_path.exists():
            raise FileNotFoundError(f"Missing benchmark task file: {task_path}")

    (RUNS_DIR / "current").mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "history").mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "current" / ".gitkeep").touch()
    (RUNS_DIR / "history" / ".gitkeep").touch()

    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")

    split_counts = Counter(task["split"] for task in manifest["tasks"])
    kind_counts = Counter(task["kind"] for task in manifest["tasks"])

    print(f"Benchmark ready: {len(manifest['tasks'])} tasks found.")
    print(f"Manifest: {manifest_path}")
    print(f"Splits: {dict(sorted(split_counts.items()))}")
    print(f"Kinds: {dict(sorted(kind_counts.items()))}")


if __name__ == "__main__":
    main()
