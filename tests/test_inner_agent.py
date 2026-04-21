from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark
from inner_agent.agent import Agent
from inner_agent.provider import ProviderNotConfiguredError
from inner_agent.tools import resolve_path


ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "inner_agent"


def make_fake_env(responses: list[dict]) -> dict[str, str]:
    return {
        "AGENT_PROVIDER": "fake",
        "AGENT_FAKE_RESPONSES": json.dumps(responses),
    }


class AgentScaffoldTests(unittest.TestCase):
    def test_finish_on_first_turn_returns_answer(self) -> None:
        task = {"id": "t1", "question": "What is the answer?", "expected_answer": "42"}
        env = make_fake_env(
            [{"tool_calls": [{"name": "finish", "arguments": {"answer": "42"}}]}]
        )

        with patch.dict(os.environ, env, clear=False):
            agent = Agent(AGENT_DIR)
            result = agent.solve(task)

        self.assertEqual(result, "42")
        self.assertTrue(agent.last_state.finished)

    def test_search_then_remember_then_finish(self) -> None:
        task = {
            "id": "docs_refund_window",
            "document": (
                "Support hours: Monday-Friday.\n"
                "Refund policy: Customers may request a refund within 30 days of purchase."
            ),
            "question": "How many days after purchase can a customer request a refund?",
            "expected_answer": "30",
        }
        env = make_fake_env(
            [
                {
                    "tool_calls": [
                        {
                            "name": "search_task_text",
                            "arguments": {"query": "refund days purchase"},
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {"name": "remember", "arguments": {"note": "refund window is 30 days"}},
                        {"name": "finish", "arguments": {"answer": "30"}},
                    ]
                },
            ]
        )

        with patch.dict(os.environ, env, clear=False):
            agent = Agent(AGENT_DIR)
            result = agent.solve(task)

        self.assertEqual(result, "30")
        self.assertIn("refund window is 30 days", agent.memory.notes())

    def test_read_task_field_cannot_access_expected_fields(self) -> None:
        task = {
            "id": "t1",
            "record": {"name": "Ada"},
            "expected_answer": "secret",
        }
        env = make_fake_env(
            [
                {
                    "tool_calls": [
                        {"name": "read_task_field", "arguments": {"path": "expected_answer"}}
                    ]
                },
                {
                    "tool_calls": [
                        {"name": "finish", "arguments": {"answer": "done"}}
                    ]
                },
            ]
        )

        with patch.dict(os.environ, env, clear=False):
            agent = Agent(AGENT_DIR)
            agent.solve(task)

        tool_events = [
            event for event in agent.memory.events() if event.kind == "tool_result"
        ]
        read_event = next(
            event for event in tool_events if event.payload["name"] == "read_task_field"
        )
        self.assertEqual(
            read_event.payload["output"]["error"],
            "Field not found in visible task.",
        )

    def test_resolve_path_rejects_invalid_list_indexes(self) -> None:
        payload = {"items": [{"name": "Ada"}]}

        self.assertEqual(resolve_path(payload, "items[0].name"), (True, "Ada"))
        self.assertEqual(resolve_path(payload, "items[name]"), (False, None))
        self.assertEqual(resolve_path(payload, "items[3]"), (False, None))

    def test_unconfigured_provider_raises_clear_error(self) -> None:
        task = {"id": "t1", "question": "Test?"}
        with patch.dict(
            os.environ,
            {
                "AGENT_PROVIDER": "",
                "AGENT_MODEL": "",
                "VERTEX_PROJECT": "",
                "VERTEX_LOCATION": "",
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_CLOUD_LOCATION": "",
            },
            clear=True,
        ):
            agent = Agent(AGENT_DIR)
            with self.assertRaises(ProviderNotConfiguredError):
                agent.solve(task)


class BenchmarkIntegrationTests(unittest.TestCase):
    def test_train_benchmark_writes_live_progress_and_incremental_task_rows(self) -> None:
        tasks_by_name = {
            "task_one.json": {
                "id": "task_one",
                "kind": "docs_qa",
                "split": "train",
                "question": "How many days?",
                "document": "Refund policy: 30 days.",
                "expected_answer": "30",
            },
            "task_two.json": {
                "id": "task_two",
                "kind": "terminal_command",
                "split": "train",
                "goal": "List files",
                "context": {"cwd": "/tmp/demo", "available_commands": ["find"]},
                "expected_command": "find /tmp/demo -type f",
            },
        }
        manifest = {
            "tasks": [
                {"id": "task_one", "kind": "docs_qa", "split": "train", "path": "benchmarks/task_one.json"},
                {
                    "id": "task_two",
                    "kind": "terminal_command",
                    "split": "train",
                    "path": "benchmarks/task_two.json",
                },
            ]
        }

        class InspectingAgent:
            def __init__(self) -> None:
                self.observed_line_count_before_second = None
                self.observed_completed_before_second = None
                self.observed_current_task_before_second = None
                self.calls = 0

            def solve(self, task: dict[str, object]) -> object:
                self.calls += 1
                if self.calls == 2:
                    current_dir = benchmark.RUNS_DIR / "current"
                    self.observed_line_count_before_second = len(
                        (current_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
                    )
                    progress = json.loads(
                        (current_dir / "progress.json").read_text(encoding="utf-8")
                    )
                    self.observed_completed_before_second = progress["num_tasks_completed"]
                    self.observed_current_task_before_second = progress["current_task_id"]

                return task.get(
                    "expected_answer",
                    task.get("expected_output", task.get("expected_command")),
                )

        agent = InspectingAgent()

        def fake_load_json(path: Path) -> dict:
            return tasks_by_name[path.name]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(benchmark, "RUNS_DIR", Path(tmpdir)):
                with patch.object(benchmark, "load_manifest", return_value=manifest):
                    with patch.object(benchmark, "load_json", side_effect=fake_load_json):
                        with patch.object(benchmark, "load_agent", return_value=agent):
                            summary, results = benchmark.run(
                                agent_dir="inner_agent",
                                task_id=None,
                                split="train",
                            )
                            history_dir = benchmark.write_run_artifacts(summary, results)

                            current_dir = Path(tmpdir) / "current"
                            progress = json.loads(
                                (current_dir / "progress.json").read_text(encoding="utf-8")
                            )
                            task_lines = (
                                current_dir / "tasks.jsonl"
                            ).read_text(encoding="utf-8").splitlines()
                            history_tasks_exists = (history_dir / "tasks.jsonl").exists()
                            history_progress_exists = (history_dir / "progress.json").exists()

        self.assertEqual(agent.observed_line_count_before_second, 1)
        self.assertEqual(agent.observed_completed_before_second, 1)
        self.assertEqual(agent.observed_current_task_before_second, "task_two")
        self.assertTrue(summary["details_visible"])
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["num_tasks_total"], 2)
        self.assertEqual(progress["num_tasks_completed"], 2)
        self.assertEqual(len(task_lines), 2)
        self.assertTrue(history_tasks_exists)
        self.assertTrue(history_progress_exists)
        self.assertGreaterEqual(summary["duration_seconds"], 0.0)

    def test_dev_benchmark_hides_task_details(self) -> None:
        tasks_by_name = {
            "task_one.json": {
                "id": "task_one",
                "kind": "docs_qa",
                "split": "dev",
                "question": "How many days?",
                "document": "Refund policy: 30 days.",
                "expected_answer": "30",
            },
            "task_two.json": {
                "id": "task_two",
                "kind": "structured_extract",
                "split": "dev",
                "instruction": "Normalize contact info.",
                "record": {"full_name": "Ada Lovelace", "email": "ada@example.com", "team": "Research"},
                "expected_output": {
                    "full_name": "Ada Lovelace",
                    "email": "ada@example.com",
                    "team": "Research",
                },
            },
        }
        manifest = {
            "tasks": [
                {"id": "task_one", "kind": "docs_qa", "split": "dev", "path": "benchmarks/task_one.json"},
                {
                    "id": "task_two",
                    "kind": "structured_extract",
                    "split": "dev",
                    "path": "benchmarks/task_two.json",
                },
            ]
        }

        class SimpleAgent:
            def solve(self, task: dict[str, object]) -> object:
                return task.get(
                    "expected_answer",
                    task.get("expected_output", task.get("expected_command")),
                )

        def fake_load_json(path: Path) -> dict:
            return tasks_by_name[path.name]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(benchmark, "RUNS_DIR", Path(tmpdir)):
                with patch.object(benchmark, "load_manifest", return_value=manifest):
                    with patch.object(benchmark, "load_json", side_effect=fake_load_json):
                        with patch.object(benchmark, "load_agent", return_value=SimpleAgent()):
                            summary, results = benchmark.run(
                                agent_dir="inner_agent",
                                task_id=None,
                                split="dev",
                            )
                            history_dir = benchmark.write_run_artifacts(summary, results)

                            current_dir = Path(tmpdir) / "current"
                            progress = json.loads(
                                (current_dir / "progress.json").read_text(encoding="utf-8")
                            )
                            task_lines = (
                                current_dir / "tasks.jsonl"
                            ).read_text(encoding="utf-8").splitlines()
                            history_task_lines = (
                                history_dir / "tasks.jsonl"
                            ).read_text(encoding="utf-8").splitlines()

        self.assertFalse(summary["details_visible"])
        self.assertEqual(progress["current_task_id"], None)
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["num_tasks_completed"], 2)
        self.assertEqual(task_lines, [])
        self.assertEqual(history_task_lines, [])

    def test_benchmark_failure_marks_progress_and_keeps_partial_rows(self) -> None:
        tasks_by_name = {
            "task_one.json": {
                "id": "task_one",
                "kind": "docs_qa",
                "split": "train",
                "question": "How many days?",
                "document": "Refund policy: 30 days.",
                "expected_answer": "30",
            },
            "task_two.json": {
                "id": "task_two",
                "kind": "structured_extract",
                "split": "train",
                "instruction": "Normalize contact info.",
                "record": {"name": "Ada Lovelace"},
                "expected_output": {"name": "Ada Lovelace"},
            },
        }
        manifest = {
            "tasks": [
                {"id": "task_one", "kind": "docs_qa", "split": "train", "path": "benchmarks/task_one.json"},
                {
                    "id": "task_two",
                    "kind": "structured_extract",
                    "split": "train",
                    "path": "benchmarks/task_two.json",
                },
            ]
        }

        class FlakyAgent:
            def __init__(self) -> None:
                self.calls = 0

            def solve(self, task: dict[str, object]) -> object:
                self.calls += 1
                if self.calls == 1:
                    return task["expected_answer"]
                raise RuntimeError("boom")

        def fake_load_json(path: Path) -> dict:
            return tasks_by_name[path.name]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(benchmark, "RUNS_DIR", Path(tmpdir)):
                with patch.object(benchmark, "load_manifest", return_value=manifest):
                    with patch.object(benchmark, "load_json", side_effect=fake_load_json):
                        with patch.object(benchmark, "load_agent", return_value=FlakyAgent()):
                            with self.assertRaisesRegex(RuntimeError, "boom"):
                                benchmark.run(
                                    agent_dir="inner_agent",
                                    task_id=None,
                                    split="train",
                                )

                current_dir = Path(tmpdir) / "current"
                progress = json.loads((current_dir / "progress.json").read_text(encoding="utf-8"))
                task_lines = (current_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(progress["status"], "failed")
        self.assertEqual(progress["num_tasks_total"], 2)
        self.assertEqual(progress["num_tasks_completed"], 1)
        self.assertEqual(progress["num_passed_so_far"], 1)
        self.assertEqual(progress["num_failed_so_far"], 0)
        self.assertEqual(progress["current_task_id"], "task_two")
        self.assertEqual(progress["error_message"], "boom")
        self.assertIsNotNone(progress["failed_at"])
        self.assertIsNotNone(progress["finished_at"])
        self.assertEqual(len(task_lines), 1)


if __name__ == "__main__":
    unittest.main()
