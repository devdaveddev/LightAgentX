"""Google Gemini LLM provider implementation."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

from .base import BaseLLM, LLMResponse
from .key_guard import SecureKey


class GeminiLLM(BaseLLM):
    """
    Google Gemini provider via the google-genai SDK.

    Requires: ``pip install google-genai``

    Usage:
        llm = GeminiLLM()                                  # gemini-2.0-flash
        llm = GeminiLLM(model="gemini-2.5-pro-preview-06-05")  # specify model
        llm = GeminiLLM(api_key="AI...")                   # explicit key

        response = llm.chat([{"role": "user", "content": "Hello!"}])
        print(response.content)
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ):
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is required for GeminiLLM. "
                "Install it with: pip install lightagentx[gemini]"
            )

        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Google API key not found. Either pass `api_key=` or set "
                "the GOOGLE_API_KEY environment variable."
            )

        self._api_key = SecureKey(resolved_key, provider="gemini")
        self._client = genai.Client(api_key=self._api_key.unwrap())

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Send messages to Gemini and get a text response."""
        system_instruction, gemini_contents = self._convert_messages(messages)

        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self._client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )

        return LLMResponse(
            content=response.text or "",
            tool_calls=[],
            raw=response,
        )

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """
        Send messages + tool definitions to Gemini.

        Converts OpenAI function-calling schemas to Gemini's format, and
        converts Gemini's function-call responses back to the framework's format.
        """
        system_instruction, gemini_contents = self._convert_messages(messages)

        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction
        if tools:
            config.tools = self._convert_tools(tools)

        response = self._client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )

        content = ""
        parsed_tool_calls = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content += part.text
                elif part.function_call:
                    fc = part.function_call
                    parsed_tool_calls.append({
                        # Unique per call: two parallel calls to the same function
                        # must not share an id (other providers reject duplicates).
                        "id": getattr(fc, "id", None) or f"call_{uuid.uuid4().hex[:16]}",
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            raw=response,
        )

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[genai_types.Content]]:
        """
        Convert OpenAI-style messages to Gemini's Content format.

        Returns (system_instruction, contents).
        """
        system_instruction = ""
        contents: list[genai_types.Content] = []
        # Tool results don't carry the function name, but Gemini matches a
        # function_response to its function_call BY NAME: recover it from the call.
        call_names: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content") or ""

            if role == "system":
                system_instruction = text
                continue

            gemini_role = "model" if role == "assistant" else "user"

            if role == "tool":
                part = genai_types.Part.from_function_response(
                    name=msg.get("name") or call_names.get(msg.get("tool_call_id", ""), "tool"),
                    response={"result": text},
                )
                prev = contents[-1] if contents else None
                if prev is not None and prev.role == "user" and prev.parts and all(
                        p.function_response for p in prev.parts):
                    prev.parts.append(part)  # results of parallel calls share one turn
                else:
                    contents.append(genai_types.Content(role="user", parts=[part]))
            elif role == "assistant" and "tool_calls" in msg:
                parts = []
                if text:
                    parts.append(genai_types.Part.from_text(text=text))
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    call_names[tc.get("id", "")] = func.get("name", "")
                    args_raw = func.get("arguments", "{}")
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    parts.append(genai_types.Part.from_function_call(
                        name=func.get("name", ""),
                        args=args,
                    ))
                contents.append(genai_types.Content(role="model", parts=parts))
            else:
                contents.append(genai_types.Content(
                    role=gemini_role,
                    parts=[genai_types.Part.from_text(text=text)],
                ))

        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[genai_types.Tool]:
        """Convert OpenAI function-calling tool format to Gemini's format."""
        declarations = []
        for t in tools:
            func = t.get("function", {})
            params = func.get("parameters", {})

            declarations.append(genai_types.FunctionDeclaration(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=params if params else None,
            ))

        return [genai_types.Tool(function_declarations=declarations)]
