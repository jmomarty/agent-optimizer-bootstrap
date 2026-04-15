from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Protocol

from inner_agent.config import AgentConfig


class ProviderNotConfiguredError(RuntimeError):
    """Raised when no usable runtime provider has been configured."""


class ProviderRuntimeError(RuntimeError):
    """Raised when a configured provider cannot execute successfully."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class Provider(Protocol):
    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: AgentConfig,
    ) -> ModelResponse: ...


class UnconfiguredProvider:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: AgentConfig,
    ) -> ModelResponse:
        provider = self.config.provider or "none"
        raise ProviderNotConfiguredError(
            "Agent provider is not configured. "
            f"Current provider setting: {provider!r}. "
            "Vertex wiring is not implemented yet. "
            "Configure a real provider, or use "
            "AGENT_PROVIDER=fake with AGENT_FAKE_RESPONSES for tests."
        )


class VertexProvider:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        if not config.vertex_project:
            raise ProviderNotConfiguredError(
                "AGENT_PROVIDER=vertex requires VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT."
            )
        if not config.vertex_location:
            raise ProviderNotConfiguredError(
                "AGENT_PROVIDER=vertex requires VERTEX_LOCATION or GOOGLE_CLOUD_LOCATION."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderRuntimeError(
                "Vertex provider requires the `google-genai` package. "
                "Install it with `pip install --upgrade google-genai`."
            ) from exc

        self._genai = genai
        self._types = types
        self._client = genai.Client(
            vertexai=True,
            project=config.vertex_project,
            location=config.vertex_location,
            http_options=types.HttpOptions(api_version="v1"),
        )

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: AgentConfig,
    ) -> ModelResponse:
        types = self._types
        generation_config = types.GenerateContentConfig(
            temperature=config.temperature,
            tools=_build_vertex_tools(types, tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
            system_instruction=_system_instruction(messages),
        )
        response = self._client.models.generate_content(
            model=config.model,
            contents=_message_to_vertex_contents(types, messages),
            config=generation_config,
        )
        return ModelResponse(
            content=_response_text(response),
            tool_calls=_response_function_calls(response),
        )


class FakeProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self._cursor = 0

    @classmethod
    def from_env(cls) -> "FakeProvider":
        raw = os.getenv("AGENT_FAKE_RESPONSES")
        if not raw:
            raise ProviderNotConfiguredError(
                "AGENT_PROVIDER is 'fake' but AGENT_FAKE_RESPONSES is missing."
            )
        payload = json.loads(raw)
        responses = [parse_model_response(item) for item in payload]
        return cls(responses)

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: AgentConfig,
    ) -> ModelResponse:
        if self._cursor >= len(self._responses):
            raise RuntimeError("Fake provider ran out of scripted responses.")
        response = self._responses[self._cursor]
        self._cursor += 1
        return response


def parse_model_response(payload: dict[str, Any]) -> ModelResponse:
    tool_calls = [
        ToolCall(name=item["name"], arguments=item.get("arguments", {}))
        for item in payload.get("tool_calls", [])
    ]
    return ModelResponse(content=payload.get("content"), tool_calls=tool_calls)


def build_provider(config: AgentConfig) -> Provider:
    if config.provider.lower() == "vertex":
        return VertexProvider(config)
    if config.provider.lower() == "fake":
        return FakeProvider.from_env()
    return UnconfiguredProvider(config)


def _json_schema_for_tool(tool_spec: dict[str, Any]) -> dict[str, Any]:
    function = tool_spec["function"]
    schema = dict(function["parameters"])
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    return schema


def _tool_name_and_arguments(function_call: Any) -> tuple[str, dict[str, Any]]:
    if hasattr(function_call, "name") and hasattr(function_call, "args"):
        return str(function_call.name), dict(function_call.args or {})

    nested = getattr(function_call, "function_call", None)
    if nested is not None and hasattr(nested, "name") and hasattr(nested, "args"):
        return str(nested.name), dict(nested.args or {})

    if isinstance(function_call, dict):
        if "name" in function_call and "args" in function_call:
            return str(function_call["name"]), dict(function_call.get("args") or {})
        nested = function_call.get("function_call")
        if isinstance(nested, dict):
            return str(nested["name"]), dict(nested.get("args") or {})

    raise ProviderRuntimeError(f"Unsupported function call payload: {function_call!r}")


def _response_text(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    text_parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(str(text))
    if text_parts:
        return "\n".join(text_parts)
    return None


def _response_function_calls(response: Any) -> list[ToolCall]:
    raw_calls = getattr(response, "function_calls", None) or []
    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        name, arguments = _tool_name_and_arguments(raw_call)
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def _build_vertex_tools(types_module: Any, tools: list[dict[str, Any]]) -> list[Any]:
    declarations = []
    for tool_spec in tools:
        function = tool_spec["function"]
        declarations.append(
            types_module.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters_json_schema=_json_schema_for_tool(tool_spec),
            )
        )
    return [types_module.Tool(function_declarations=declarations)]


def _message_to_vertex_contents(types_module: Any, messages: list[dict[str, Any]]) -> list[Any]:
    contents: list[Any] = []
    pending_tool_parts: list[Any] = []

    def flush_tool_parts() -> None:
        nonlocal pending_tool_parts
        if pending_tool_parts:
            contents.append(types_module.Content(role="tool", parts=pending_tool_parts))
            pending_tool_parts = []

    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role != "tool":
            flush_tool_parts()
        if role == "user":
            contents.append(
                types_module.Content(
                    role="user",
                    parts=[types_module.Part.from_text(text=str(message.get("content", "")))],
                )
            )
            continue
        if role == "assistant":
            parts = []
            content = str(message.get("content", "") or "")
            if content:
                parts.append(types_module.Part.from_text(text=content))
            for tool_call in message.get("tool_calls", []):
                parts.append(
                    types_module.Part.from_function_call(
                        name=tool_call["name"],
                        args=tool_call.get("arguments", {}),
                    )
                )
            if parts:
                contents.append(types_module.Content(role="model", parts=parts))
            continue
        if role == "tool":
            pending_tool_parts.append(
                types_module.Part.from_function_response(
                    name=message["name"],
                    response=json.loads(message["content"]),
                )
            )
            continue
        raise ProviderRuntimeError(f"Unsupported message role: {role}")
    flush_tool_parts()
    return contents


def _system_instruction(messages: list[dict[str, Any]]) -> str | None:
    parts = [str(message.get("content", "")) for message in messages if message["role"] == "system"]
    merged = "\n\n".join(part for part in parts if part.strip())
    return merged or None
