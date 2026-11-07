"""Validated cloud request payloads and provider-specific sampling policy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEEPSEEK_V41_FLASH = "deepseek/deepseek-v4.1-flash"
# Optional reasoning, on by default (OpenRouter catalog: mandatory=false).
# Left on, it spent the whole 16K and 32K output allowance thinking on
# ~60K-175K-token review prompts and returned no review.
QWEN38_FLASH = "qwen/qwen3.8-flash"
_REASONING_DISABLED_MODELS = frozenset({DEEPSEEK_V41_FLASH, QWEN38_FLASH})


def uses_default_sampling(provider: str, model: str) -> bool:
    """Gemini requests omit sampling controls, including OpenRouter routes."""
    return provider == "gemini" or (
        provider == "openrouter" and model.startswith("google/gemini-")
    )


class OpenRouterMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user"]
    content: str


class OpenRouterReasoning(BaseModel):
    """The explicit non-thinking policy used by the V4.1 Flash profile."""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: Literal[False]


def openrouter_reasoning(model: str) -> OpenRouterReasoning | None:
    """Avoid flash models spending their entire review allowance on thinking.

    Applies to DeepSeek V4.1 Flash and Qwen 3.8 Flash (any ``:variant``).
    Only DeepSeek also gets a larger default output allowance.
    """
    if model.partition(":")[0] in _REASONING_DISABLED_MODELS:
        return OpenRouterReasoning(enabled=False)
    return None


def openrouter_default_max_tokens(model: str) -> int | None:
    """Preserve V4.1 Flash's output allowance when no cap was configured."""
    return 65_536 if model.partition(":")[0] == DEEPSEEK_V41_FLASH else None


class OpenRouterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str
    messages: list[OpenRouterMessage]
    max_tokens: int
    temperature: float | None = None
    reasoning: OpenRouterReasoning | None = None


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
