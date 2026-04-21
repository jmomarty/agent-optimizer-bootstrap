from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model: str
    vertex_project: str | None
    vertex_location: str | None
    max_turns: int
    temperature: float
    search_result_limit: int
    prompt_template: str


def load_prompt(agent_dir: Path) -> str:
    prompt_path = agent_dir / "prompt.md"
    return prompt_path.read_text(encoding="utf-8")


def load_dotenv(root: Path) -> None:
    dotenv_path = root / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_config(agent_dir: Path) -> AgentConfig:
    load_dotenv(agent_dir.parent)
    prompt_template = load_prompt(agent_dir)
    return AgentConfig(
        provider=os.getenv("AGENT_PROVIDER", ""),
        model=os.getenv("AGENT_MODEL", "gemini-2.5-pro"),
        vertex_project=os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT"),
        vertex_location=os.getenv("VERTEX_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or "global",
        max_turns=int(os.getenv("AGENT_MAX_TURNS", "6")),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")),
        search_result_limit=int(os.getenv("AGENT_SEARCH_RESULT_LIMIT", "5")),
        prompt_template=prompt_template,
    )
