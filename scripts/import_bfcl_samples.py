from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
BFCL_DIR = BENCHMARKS_DIR / "bfcl"
MANIFEST_PATH = BENCHMARKS_DIR / "manifest.json"

DATA_URL = (
    "https://huggingface.co/datasets/gorilla-llm/"
    "Berkeley-Function-Calling-Leaderboard/raw/main/BFCL_v3_simple.json"
)
ANSWERS_URL = (
    "https://huggingface.co/datasets/gorilla-llm/"
    "Berkeley-Function-Calling-Leaderboard/raw/main/possible_answer/BFCL_v3_simple.json"
)

NUM_BFCL_TASKS = 80
DEV_EVERY = 5

SMOKE_TASKS = [
    {
        "id": "docs_invoice_contact_01",
        "kind": "docs_qa",
        "path": "benchmarks/docs/invoice_contact_01.json",
        "split": "train",
    },
    {
        "id": "docs_refund_window_01",
        "kind": "docs_qa",
        "path": "benchmarks/docs/refund_window_01.json",
        "split": "train",
    },
    {
        "id": "docs_security_ack_window_01",
        "kind": "docs_qa",
        "path": "benchmarks/docs/security_ack_window_01.json",
        "split": "train",
    },
    {
        "id": "structured_normalize_contact_01",
        "kind": "structured_extract",
        "path": "benchmarks/structured/normalize_contact_01.json",
        "split": "train",
    },
    {
        "id": "structured_normalize_directory_entry_01",
        "kind": "structured_extract",
        "path": "benchmarks/structured/normalize_directory_entry_01.json",
        "split": "train",
    },
    {
        "id": "terminal_find_called_01",
        "kind": "terminal_command",
        "path": "benchmarks/terminal/find_called_01.json",
        "split": "train",
    },
    {
        "id": "terminal_locate_01",
        "kind": "terminal_command",
        "path": "benchmarks/terminal/locate_01.json",
        "split": "train",
    },
    {
        "id": "docs_invoice_contact_09",
        "kind": "docs_qa",
        "path": "benchmarks/docs/invoice_contact_09.json",
        "split": "dev",
    },
    {
        "id": "structured_normalize_contact_09",
        "kind": "structured_extract",
        "path": "benchmarks/structured/normalize_contact_09.json",
        "split": "dev",
    },
    {
        "id": "terminal_locate_08",
        "kind": "terminal_command",
        "path": "benchmarks/terminal/locate_08.json",
        "split": "dev",
    },
]


def load_jsonl(path: Path | None, url: str) -> list[dict[str, Any]]:
    if path is not None:
        text = path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def first_user_message(question: Any) -> str:
    return str(question[0][0]["content"])


def canonical_value(choices: list[Any]) -> Any:
    for choice in choices:
        if choice != "":
            return choice
    return choices[0] if choices else None


def canonical_call(
    answer: dict[str, Any],
    functions: list[dict[str, Any]],
) -> dict[str, Any]:
    function_name, argument_choices = next(iter(answer["ground_truth"][0].items()))
    schema_by_name = {function["name"]: function for function in functions}
    required = set(schema_by_name[function_name]["parameters"].get("required", []))
    arguments: dict[str, Any] = {}
    for name, choices in argument_choices.items():
        if name in required or "" not in choices:
            arguments[name] = canonical_value(choices)
    return {"tool_name": function_name, "arguments": arguments}


def build_task(
    item: dict[str, Any],
    answer: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    task_id = f"bfcl_simple_{index:03d}"
    return {
        "id": task_id,
        "kind": "structured_extract",
        "split": "dev" if (index + 1) % DEV_EVERY == 0 else "train",
        "source": {
            "benchmark": "Berkeley Function Calling Leaderboard",
            "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            "category": "BFCL_v3_simple",
            "original_id": item["id"],
            "license": "Apache-2.0",
            "url": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        },
        "instruction": (
            "Select the single intended function call for the user request. "
            "Return JSON with keys tool_name and arguments. Include only "
            "arguments that should be supplied explicitly."
        ),
        "record": {
            "user_request": first_user_message(item["question"]),
            "available_functions": item["function"],
        },
        "expected_output": canonical_call(answer, item["function"]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a lightweight BFCL sample set.")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--answers-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_jsonl(args.data_file, DATA_URL)
    answers = {item["id"]: item for item in load_jsonl(args.answers_file, ANSWERS_URL)}

    shutil.rmtree(BFCL_DIR, ignore_errors=True)
    BFCL_DIR.mkdir(parents=True)

    manifest_tasks = list(SMOKE_TASKS)
    for index, item in enumerate(data[:NUM_BFCL_TASKS]):
        task = build_task(item, answers[item["id"]], index=index)
        task_path = BFCL_DIR / f"{task['id']}.json"
        write_json(task_path, task)
        manifest_tasks.append(
            {
                "id": task["id"],
                "kind": task["kind"],
                "path": str(task_path.relative_to(ROOT)),
                "split": task["split"],
            }
        )

    write_json(
        MANIFEST_PATH,
        {
            "name": "bfcl-primary-with-smoke",
            "tasks": manifest_tasks,
        },
    )

    train_count = sum(1 for item in manifest_tasks if item["split"] == "train")
    dev_count = len(manifest_tasks) - train_count
    print(f"Wrote {len(manifest_tasks)} manifest tasks: {train_count} train, {dev_count} dev")


if __name__ == "__main__":
    main()
