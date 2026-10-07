"""Validated cloud request payloads and provider-specific sampling policy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def uses_default_sampling(provider: str, model: str) -> bool:
    """Gemini requests omit sampling controls, including OpenRouter routes."""
    return provider == "gemini" or (
        provider == "openrouter" and model.startswith("google/gemini-")
    )


class OpenRouterMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user"]
    content: str


class OpenRouterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str
    messages: list[OpenRouterMessage]
    max_tokens: int
    temperature: float | None = None


class GeminiPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str


class GeminiContent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    parts: list[GeminiPart]
    role: Literal["user"] | None = None


class GeminiGenerationConfig(BaseModel):
    """Omitting thinking config uses the model default on both 2.5 and 3.x."""

    model_config = ConfigDict(extra="forbid", strict=True)

    max_output_tokens: int = Field(serialization_alias="maxOutputTokens")


class GeminiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contents: list[GeminiContent]
    system_instruction: GeminiContent = Field(serialization_alias="systemInstruction")
    generation_config: GeminiGenerationConfig = Field(
        serialization_alias="generationConfig"
    )
