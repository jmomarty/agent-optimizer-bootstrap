from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from inner_agent.memory import AgentMemory
from inner_agent.provider import ToolCall
from inner_agent.retrieval import search_task_text


@dataclass(frozen=True)
class ToolOutcome:
    output: dict[str, Any]
    finished: bool = False
    final_answer: Any = None


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_task_field",
                "description": "Read one visible field from the current task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Dot or bracket path into the visible task object.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_task_text",
                "description": "Search visible task text and return the most relevant snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query over visible task text.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Store a short note for later turns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "string", "description": "Short note to remember."}
                    },
                    "required": ["note"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish the task and return the final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "description": "Final answer for the task.",
                        }
                    },
                    "required": ["answer"],
                },
            },
        },
    ]


def dispatch_tool_call(
    call: ToolCall,
    *,
    visible_task: dict[str, Any],
    memory: AgentMemory,
    search_limit: int,
) -> ToolOutcome:
    if call.name == "read_task_field":
        path = str(call.arguments.get("path", ""))
        found, value = resolve_path(visible_task, path)
        if not found:
            return ToolOutcome(output={"path": path, "error": "Field not found in visible task."})
        return ToolOutcome(output={"path": path, "value": value})

    if call.name == "search_task_text":
        query = str(call.arguments.get("query", ""))
        return ToolOutcome(
            output={
                "query": query,
                "matches": search_task_text(visible_task, query, limit=search_limit),
            }
        )

    if call.name == "remember":
        note = str(call.arguments.get("note", "")).strip()
        memory.remember(note)
        return ToolOutcome(output={"stored": note})

    if call.name == "finish":
        return ToolOutcome(
            output={"answer": call.arguments.get("answer")},
            finished=True,
            final_answer=call.arguments.get("answer"),
        )

    return ToolOutcome(output={"error": f"Unknown tool: {call.name}"})


def resolve_path(payload: Any, path: str) -> tuple[bool, Any]:
    if not path:
        return False, None

    current = payload
    normalized_tokens = re.findall(r"[^.\[\]]+", path)

    for token in normalized_tokens:
        if isinstance(current, dict):
            if token not in current:
                return False, None
            current = current[token]
            continue
        if isinstance(current, list):
            if not token.isdigit():
                return False, None
            index = int(token)
            if index >= len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, current
