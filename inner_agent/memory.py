from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryEvent:
    kind: str
    payload: dict[str, Any]


class AgentMemory:
    def __init__(self) -> None:
        self._events: list[MemoryEvent] = []

    def add(self, kind: str, payload: dict[str, Any]) -> None:
        self._events.append(MemoryEvent(kind=kind, payload=payload))

    def remember(self, note: str) -> None:
        self.add("note", {"note": note})

    def events(self) -> list[MemoryEvent]:
        return list(self._events)

    def notes(self) -> list[str]:
        return [event.payload["note"] for event in self._events if event.kind == "note"]
