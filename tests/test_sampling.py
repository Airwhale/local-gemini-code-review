"""Sampling migration contracts at the HTTP boundary and in CLI reports."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from code_review import providers
from code_review.cli import _dry_run_report, _run_panel
from code_review.config import ReviewRequest, Settings
from code_review.generation import GeminiGenerationConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-3.8-flash"])
@pytest.mark.parametrize("temperature", [0.3, 1.7])
def test_gemini_request_contract(
    monkeypatch: pytest.MonkeyPatch, model: str, temperature: float
) -> None:
    expected = json.loads((FIXTURES_DIR / "gemini-request.json").read_text())

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(f"/{model}:generateContent")
        assert request.headers["x-goog-api-key"] == "test-key"
        assert json.loads(request.content) == expected
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": "ok"}]}}
                ]
            },
        )

    monkeypatch.setattr(
        providers,
        "_make_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handle)),
    )
    result = providers.call_gemini(
        model=model,
        temperature=temperature,
        max_tokens=100,
        system_prompt="s",
        user_prompt="u",
        api_key="test-key",
    )
    assert result.content == "ok"


def test_mixed_panel_omits_sampling_only_for_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = (
        "google/gemini-2.5-pro",
        "google/gemini-3.8-flash",
        "google/gemini-2.5-pro:online",
        "anthropic/claude-sonnet-4.5",
        "google/gemma-3-27b-it",
    )

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        expected = {
            "model": body["model"],
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            "max_tokens": 100,
        }
        if body["model"] in models[-2:]:
            expected["temperature"] = 1.7
        assert body == expected
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
        )

    monkeypatch.setattr(
        providers,
        "_make_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handle)),
    )
    settings = Settings(
        provider="openrouter",
        model=models[0],
        models=models,
        temperature=1.7,
        max_tokens=100,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
        api_key="test-key",
        referer="test",
        title="test",
    )
    results, failures = _run_panel(settings, ReviewRequest("s", "u", "diff", 1))
    assert not failures
    assert tuple(results) == models
    assert all(result.content == "ok" for result in results.values())


@pytest.mark.parametrize(
    ("provider", "models", "expected"),
    [
        (
            "gemini",
            ("gemini-3.8-flash",),
            "model default (Gemini; configured 0.3 ignored)",
        ),
        (
            "openrouter",
            ("google/gemini-2.5-pro",),
            "model default (Gemini; configured 0.3 ignored)",
        ),
        (
            "openrouter",
            ("google/gemini-2.5-pro", "google/gemini-3.8-flash"),
            "model default (Gemini; configured 0.3 ignored)",
        ),
        (
            "openrouter",
            ("google/gemini-2.5-pro", "anthropic/claude-sonnet-4.5"),
            "0.3 (Gemini uses model default)",
        ),
        ("openrouter", ("anthropic/claude-sonnet-4.5",), "0.3"),
        ("ollama", ("qwen3-coder:30b",), "0.3"),
    ],
)
def test_report_shows_effective_sampling(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    models: tuple[str, ...],
    expected: str,
) -> None:
    monkeypatch.setattr("code_review.cli._estimate_cost_line", lambda *args: None)
    settings = Settings(
        provider=provider,
        model=models[0],
        models=models if len(models) > 1 else None,
        temperature=0.3,
        max_tokens=100,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    report = _dry_run_report(settings, ReviewRequest("s", "u", "diff", 1))
    assert f"temperature:       {expected}\n" in report


@pytest.mark.parametrize(
    "field", ["temperature", "topP", "topK", "thinkingBudget", "thinking_budget"]
)
def test_gemini_config_rejects_deprecated_controls(field: str) -> None:
    with pytest.raises(ValidationError):
        GeminiGenerationConfig.model_validate({"max_output_tokens": 100, field: 1})
