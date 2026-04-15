from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from inner_agent.config import AgentConfig, load_config
from inner_agent.memory import AgentMemory
from inner_agent.provider import ModelResponse, build_provider
from inner_agent.tools import dispatch_tool_call, tool_specs


class AgentLoopError(RuntimeError):
    """Raised when the baseline agent does not finish within its turn budget."""


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    visible_task: dict[str, Any]


@dataclass
class AgentState:
    turn_count: int = 0
    finished: bool = False
    final_answer: Any = None


class Agent:
    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir
        self.config = load_config(agent_dir)
        self.provider = build_provider(self.config)
        self.memory = AgentMemory()
        self.last_state = AgentState()

    def solve(self, task: dict[str, Any]) -> Any:
        self.memory = AgentMemory()
        self.last_state = AgentState()
        context = self._build_task_context(task)
        messages = self._initial_messages(self.config, context)

        for turn in range(1, self.config.max_turns + 1):
            self.last_state.turn_count = turn
            response = self.provider.generate(
                messages=messages,
                tools=tool_specs(),
                config=self.config,
            )
            self._record_model_response(turn, response)
            messages.append(self._assistant_message(response))

            for call in response.tool_calls:
                outcome = dispatch_tool_call(
                    call,
                    visible_task=context.visible_task,
                    memory=self.memory,
                    search_limit=self.config.search_result_limit,
                )
                self.memory.add(
                    "tool_result",
                    {
                        "turn": turn,
                        "name": call.name,
                        "output": outcome.output,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": json.dumps(outcome.output, sort_keys=True),
                    }
                )
                if outcome.finished:
                    self.last_state.finished = True
                    self.last_state.final_answer = outcome.final_answer
                    return outcome.final_answer

        raise AgentLoopError(
            f"Agent exited without a finish action after {self.config.max_turns} turns."
        )

    def _build_task_context(self, task: dict[str, Any]) -> TaskContext:
        visible_task = {
            key: value
            for key, value in task.items()
            if not key.startswith("expected_")
        }
        return TaskContext(
            task_id=str(task.get("id", "")),
            visible_task=visible_task,
        )

    def _initial_messages(
        self,
        config: AgentConfig,
        context: TaskContext,
    ) -> list[dict[str, Any]]:
        user_prompt = (
            "Visible task object:\n"
            f"{json.dumps(context.visible_task, indent=2, sort_keys=True)}\n\n"
            "Use tools when needed and call finish(answer) when ready."
        )
        return [
            {"role": "system", "content": config.prompt_template},
            {"role": "user", "content": user_prompt},
        ]

    def _record_model_response(self, turn: int, response: ModelResponse) -> None:
        self.memory.add(
            "model_response",
            {
                "turn": turn,
                "content": response.content,
                "tool_calls": [tool_call.to_dict() for tool_call in response.tool_calls],
            },
        )

    def _assistant_message(self, response: ModelResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [tool_call.to_dict() for tool_call in response.tool_calls],
        }
