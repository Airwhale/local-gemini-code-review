"""Sampling migration contracts at the HTTP boundary and in CLI reports."""

import argparse
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from code_review import cli, providers
from code_review.cli import _dry_run_report, _run_panel
from code_review.config import ReviewRequest, Settings
from code_review.errors import ContextOverflow
from code_review.generation import GeminiGenerationConfig, OpenRouterReasoning

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
    "max_tokens,defaulted", [(16000, False), (32768, False), (16000, True)]
)
def test_deepseek_reasoning_in_mixed_panel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    max_tokens: int,
    defaulted: bool,
) -> None:
    models = (
        "anthropic/claude-sonnet-5.5",
        "google/gemini-4-argon",
        "deepseek/deepseek-v4.1-flash",
        "deepseek/deepseek-v4.1-flash:nitro",
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4.1-pro",
    )

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        expected_limit = 65536 if defaulted and model in models[2:4] else max_tokens
        assert body["max_tokens"] == expected_limit
        if model in models[2:4]:
            assert body["reasoning"] == {"effort": "low"}
        else:
            assert "reasoning" not in body
        if model == models[1]:
            assert "temperature" not in body
        else:
            assert body["temperature"] == 0.3
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
        temperature=0.3,
        max_tokens=max_tokens,
        max_tokens_defaulted=defaulted,
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
    stderr = capsys.readouterr().err
    deepseek_limit = 65536 if defaulted else max_tokens
    assert f"effort=low max_tokens={deepseek_limit}" in stderr
    assert stderr.count("[reasoning]") == 2


def test_deepseek_empty_reasoning_response_is_not_a_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["reasoning"] == {"effort": "low"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": None, "reasoning": "internal reasoning"},
                    }
                ],
                "usage": {
                    "completion_tokens": 16000,
                    "completion_tokens_details": {"reasoning_tokens": 16000},
                },
            },
        )

    monkeypatch.setattr(
        providers,
        "_make_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handle)),
    )
    with pytest.raises(ContextOverflow, match="before producing content"):
        providers.call_openrouter(
            model="deepseek/deepseek-v4.1-flash",
            system_prompt="s",
            user_prompt="u",
            api_key="test-key",
            referer="test",
            title="test",
            temperature=0.3,
            max_tokens=16000,
        )


def test_dry_run_shows_deepseek_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("code_review.cli._estimate_cost_line", lambda *args: None)
    models = ("deepseek/deepseek-v4.1-flash", "google/gemini-4-argon")
    settings = Settings(
        provider="openrouter",
        model=models[0],
        models=models,
        temperature=0.3,
        max_tokens=16000,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    report = _dry_run_report(settings, ReviewRequest("s", "u", "diff", 1))
    assert "reasoning:         deepseek/deepseek-v4.1-flash: low" in report
    assert "max_tokens:        16000" in report


def test_reasoning_schema_rejects_unsupported_controls() -> None:
    with pytest.raises(ValidationError):
        OpenRouterReasoning.model_validate({"effort": "medium"})
    with pytest.raises(ValidationError):
        OpenRouterReasoning.model_validate({"effort": "low", "max_tokens": 4096})


def test_defaulted_panel_limits_match_cost_and_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from code_review.cli import _estimate_cost_line

    models = (
        "anthropic/claude-sonnet-5.5",
        "google/gemini-4-argon",
        "deepseek/deepseek-v4.1-flash",
    )
    settings = Settings(
        provider="openrouter",
        model=models[0],
        models=models,
        temperature=0.3,
        max_tokens=16000,
        max_tokens_defaulted=True,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    calls: list[tuple[str, int]] = []

    def price(
        provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        calls.append((model, completion_tokens))
        return completion_tokens / 1_000_000

    monkeypatch.setattr("code_review.cli.estimate_cost_usd", price)
    estimate = _estimate_cost_line(settings, 4000)
    assert calls == [(models[0], 16000), (models[1], 16000), (models[2], 65536)]
    assert estimate is not None and "65,536" in estimate
    report = _dry_run_report(settings, ReviewRequest("s", "u", "diff", 1))
    assert f"{models[0]}=16000" in report
    assert f"{models[1]}=16000" in report
    assert f"{models[2]}=65536" in report


@pytest.mark.parametrize("defaulted", [False, True])
def test_reference_guard_uses_resolved_deepseek_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    defaulted: bool,
) -> None:
    file = tmp_path / "x.py"
    file.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_read_diff_source",
        lambda args: "diff --git a/x.py b/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+x = 1\n",
    )
    monkeypatch.setattr(cli, "changed_file_paths", lambda args: [file])
    monkeypatch.setattr(cli, "model_context_limit", lambda *args: 40000)
    options = argparse.Namespace(
        codebase=False,
        include=[],
        exclude=[],
        full_files=None,
        diff_file=None,
        pr=None,
        staged=False,
    )
    settings = Settings(
        provider="openrouter",
        model="deepseek/deepseek-v4.1-flash",
        temperature=0.3,
        max_tokens=16000,
        max_tokens_defaulted=defaulted,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    request = cli._build_request(options, settings)
    assert ("<REFERENCE_FILES>" in request.user_prompt) is not defaulted


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
