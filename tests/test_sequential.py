"""Tests for --sequential panel execution: scheduling (concurrent default
vs one-at-a-time), prompt independence, aggregation equivalence, failure
handling, exit precedence, flag validation and dry-run reporting."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import code_review.cli as cli
from code_review.config import ReviewRequest, Settings, _resolve_settings
from code_review.errors import (
    ConfigError,
    ContextOverflow,
    RateLimit,
    ReviewError,
    TransportError,
)
from code_review.panel import _panel_max_workers, panel_execution_mode
from code_review.providers import CallResult

FIXTURES = Path(__file__).parent / "fixtures"
CURRENT = (FIXTURES / "reporting-current.md").read_text(encoding="utf-8")
SECOND = (FIXTURES / "reporting-second-model.md").read_text(encoding="utf-8")
MODELS = ("qwen/qwen3.8-flash", "z-ai/glm-5.3")
REQUEST = ReviewRequest("SYSTEM", "USER", "diff", 4)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        provider="openrouter",
        model=MODELS[0],
        models=MODELS,
        temperature=0.3,
        max_tokens=100,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
        api_key="test-key",
        referer="https://example.invalid",
        title="test",
    )
    base.update(overrides)
    return Settings(**base)


class TestScheduling:
    def test_concurrent_default_overlaps_calls(self, monkeypatch: pytest.MonkeyPatch):
        # Both calls must be in flight at once to pass the barrier; a serial
        # schedule would time out the first wait and break it.
        barrier = threading.Barrier(len(MODELS), timeout=10)

        def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
            barrier.wait()
            return CallResult(CURRENT)

        monkeypatch.setattr(cli, "_execute_call", fake)
        results, failures = cli._run_panel(_settings(), REQUEST)
        assert failures == []
        assert list(results) == list(MODELS)

    def test_sequential_preserves_order_and_never_overlaps(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        models = ("m-1", "m-2", "m-3", "m-4")
        lock = threading.Lock()
        active = 0
        peak = 0
        events: list[str] = []

        def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                events.append(f"start {model}")
            time.sleep(0.02)
            with lock:
                active -= 1
                events.append(f"end {model}")
            return CallResult(CURRENT)

        monkeypatch.setattr(cli, "_execute_call", fake)
        results, failures = cli._run_panel(
            _settings(models=models, model=models[0], sequential=True), REQUEST
        )
        assert peak == 1
        assert events == [e for m in models for e in (f"start {m}", f"end {m}")]
        assert list(results) == list(models)
        assert failures == []
        assert "--sequential: 4 models run one at a time" in capsys.readouterr().err

    @pytest.mark.parametrize("sequential", [False, True])
    def test_every_model_gets_the_same_independent_prompt(
        self, monkeypatch: pytest.MonkeyPatch, sequential: bool
    ):
        seen: dict[str, tuple[str, str]] = {}

        def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
            seen[model] = (system, user)
            return CallResult(f"output of {model}")

        monkeypatch.setattr(cli, "_execute_call", fake)
        cli._run_panel(_settings(sequential=sequential), REQUEST)
        # No model's output is fed into a later model's prompt.
        assert seen == {m: ("SYSTEM", "USER") for m in MODELS}

    def test_worker_policy(self):
        assert _panel_max_workers("openrouter", 3) == 3
        assert _panel_max_workers("openrouter", 8) == 4
        assert _panel_max_workers("openrouter", 3, sequential=True) == 1
        assert _panel_max_workers("claude-code", 2) == 2
        # Ollama stays serial with or without the flag.
        assert _panel_max_workers("ollama", 3) == 1
        assert _panel_max_workers("ollama", 3, sequential=True) == 1

    def test_execution_mode_text(self):
        assert panel_execution_mode("openrouter", 2, False) == (
            "concurrent (up to 2 models at once)"
        )
        assert panel_execution_mode("openrouter", 2, True).startswith(
            "sequential (--sequential)"
        )
        assert panel_execution_mode("ollama", 2, False).startswith(
            "sequential (ollama default)"
        )

    def test_ollama_panel_message_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.setattr(
            cli, "_execute_call", lambda *a: CallResult("# Change summary: x\n")
        )
        cli._run_panel(
            _settings(
                provider="ollama",
                models=("a:1", "b:2"),
                model="a:1",
                ollama_host="http://localhost:11434",
                ollama_timeout=1.0,
            ),
            REQUEST,
        )
        assert "[panel] ollama models run sequentially" in capsys.readouterr().err


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    execute: Any,
) -> None:
    monkeypatch.setattr("sys.argv", ["code-review", "--no-project-config"])
    monkeypatch.setattr(cli, "_resolve_settings", lambda *args: settings)
    monkeypatch.setattr(cli, "_build_requests", lambda *args: [REQUEST])
    monkeypatch.setattr(cli, "_estimate_cost_line", lambda *args: None)
    monkeypatch.setattr(cli, "_execute_call", execute)
    cli.main()


class TestAggregationEquivalence:
    @pytest.mark.parametrize("output_format", ["json", "markdown"])
    def test_same_report_in_both_modes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        output_format: str,
    ):
        outputs = {MODELS[0]: CURRENT, MODELS[1]: SECOND}
        usage = {MODELS[0]: (100, 10), MODELS[1]: (200, 20)}

        def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
            # The first model finishes LAST when concurrent, so a report
            # keyed by completion order would differ between modes.
            if model == MODELS[0] and not settings.sequential:
                time.sleep(0.05)
            prompt, completion = usage[model]
            return CallResult(
                outputs[model], prompt_tokens=prompt, completion_tokens=completion
            )

        reports = {}
        for sequential in (False, True):
            _run_main(
                monkeypatch,
                _settings(format=output_format, sequential=sequential),
                fake,
            )
            reports[sequential] = capsys.readouterr().out
        assert reports[False] == reports[True]
        if output_format == "json":
            report = json.loads(reports[True])
            assert report["models"] == list(MODELS)
            assert report["usage"] == {"prompt_tokens": 300, "completion_tokens": 30}
            shared = [f for f in report["findings"] if len(f["found_by"]) == 2]
            assert shared and shared[0]["found_by"] == list(MODELS)
            assert [e["model"] for e in report["per_model"]] == list(MODELS)
        else:
            assert reports[True].startswith("# Panel review (2/2 models)")
            assert "Found by: qwen/qwen3.8-flash, z-ai/glm-5.3" in reports[True]


class TestFailures:
    @pytest.mark.parametrize("sequential", [False, True])
    def test_one_model_failure_keeps_the_panel(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        sequential: bool,
    ):
        def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
            if model == MODELS[0]:
                raise RateLimit("429 synthetic")
            return CallResult(SECOND)

        _run_main(monkeypatch, _settings(format="json", sequential=sequential), fake)
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        failed = report["per_model"][0]
        assert failed["model"] == MODELS[0]
        assert failed["error"]["category"] == "RATE_LIMIT"
        assert report["per_model"][1]["error"] is None
        assert f"WARN: [panel] {MODELS[0]} failed: RATE_LIMIT [exit 11]" in (
            captured.err
        )

    @pytest.mark.parametrize("sequential", [False, True])
    def test_sequential_continues_after_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, sequential: bool
    ):
        called: list[str] = []

        def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
            called.append(model)
            if model == "m-1":
                raise TransportError("down")
            return CallResult(CURRENT)

        monkeypatch.setattr(cli, "_execute_call", fake)
        models = ("m-1", "m-2", "m-3")
        results, failures = cli._run_panel(
            _settings(models=models, model="m-1", sequential=sequential), REQUEST
        )
        assert sorted(called) == list(models)
        assert list(results) == ["m-2", "m-3"]
        assert [m for m, _ in failures] == ["m-1"]

    @pytest.mark.parametrize("sequential", [False, True])
    def test_all_failed_exit_precedence(
        self, monkeypatch: pytest.MonkeyPatch, sequential: bool
    ):
        errors: dict[str, ReviewError] = {
            MODELS[0]: TransportError("net"),
            MODELS[1]: ContextOverflow("too big"),
        }

        def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
            raise errors[model]

        with pytest.raises(ContextOverflow):
            _run_main(monkeypatch, _settings(sequential=sequential), fake)

    def test_unexpected_exception_still_surfaces(self, monkeypatch: pytest.MonkeyPatch):
        def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
            raise RuntimeError("bug")

        monkeypatch.setattr(cli, "_execute_call", fake)
        with pytest.raises(RuntimeError):
            cli._run_panel(_settings(sequential=True), REQUEST)


def _args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        base=None,
        pr=None,
        staged=False,
        codebase=False,
        include=[],
        exclude=[],
        provider="openrouter",
        model=None,
        models=None,
        ollama_host=None,
        temperature=None,
        max_tokens=None,
        retries=None,
        min_severity="LOW",
        min_found_by=None,
        context=None,
        no_context=False,
        output=None,
        format=None,
        baseline=None,
        dry_run=False,
        chunk=False,
        sequential=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestSettings:
    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        monkeypatch.delenv("CODE_REVIEW_PROVIDER", raising=False)

    def test_default_is_concurrent(self):
        assert _resolve_settings(_args(models=",".join(MODELS))).sequential is False

    def test_sequential_with_models(self):
        settings = _resolve_settings(_args(models=",".join(MODELS), sequential=True))
        assert settings.sequential is True
        assert settings.models == MODELS

    def test_sequential_without_models_rejected(self):
        with pytest.raises(ConfigError, match="--sequential needs a --models panel"):
            _resolve_settings(_args(sequential=True))

    def test_sequential_with_single_model_rejected(self):
        with pytest.raises(ConfigError, match="--sequential"):
            _resolve_settings(_args(model="z-ai/glm-5.3", sequential=True))

    def test_sequential_with_project_config_panel(self):
        settings = _resolve_settings(_args(sequential=True), {"models": list(MODELS)})
        assert settings.sequential is True and settings.models == MODELS

    def test_defaults_unchanged(self):
        settings = _resolve_settings(_args())
        assert settings.provider == "openrouter"
        assert settings.model == "google/gemini-2.5-pro"


class TestCli:
    def _prepare(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
    ) -> None:
        diff = tmp_path / "d.patch"
        diff.write_text("diff --git a/f.py b/f.py\n+x = 1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        monkeypatch.setattr(cli, "_estimate_cost_line", lambda *args: None)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.delenv("CODE_REVIEW_PROVIDER", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "code-review",
                "--diff-file",
                str(diff),
                "--dry-run",
                "--no-project-config",
                *extra,
            ],
        )

    def _dry_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        extra: list[str],
    ) -> tuple[str, str]:
        self._prepare(tmp_path, monkeypatch, extra)
        cli.main()
        return capsys.readouterr()

    def test_dry_run_concurrent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        out, _err = self._dry_run(
            tmp_path, monkeypatch, capsys, ["--models", ",".join(MODELS)]
        )
        assert "models:            qwen/qwen3.8-flash, z-ai/glm-5.3 (panel)" in out
        assert "execution:         concurrent (up to 2 models at once)" in out

    def test_dry_run_sequential(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        out, _err = self._dry_run(
            tmp_path,
            monkeypatch,
            capsys,
            ["--models", ",".join(MODELS), "--sequential"],
        )
        assert (
            "execution:         sequential (--sequential): one model at a time, "
            "in the order given"
        ) in out

    def test_sequential_without_models_is_config_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        self._prepare(tmp_path, monkeypatch, ["--sequential"])
        with pytest.raises(SystemExit) as exc_info:
            cli._entrypoint()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert err.startswith("ERROR: CONFIG [exit 2]")
        assert "--sequential needs a --models panel" in err

    def test_sequential_with_chunk_is_config_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        # A real panel, so the rejection comes from the chunk+panel rule
        # rather than from --sequential lacking --models.
        with pytest.raises(ConfigError, match="--chunk and --models"):
            self._dry_run(
                tmp_path,
                monkeypatch,
                capsys,
                ["--models", ",".join(MODELS), "--sequential", "--chunk"],
            )


def test_settings_field_defaults_to_concurrent() -> None:
    fields = {f.name: f for f in dataclasses.fields(Settings)}
    assert fields["sequential"].default is False


def test_interrupted_concurrent_panel_cancels_running_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import code_review.claude_code as cc

    signalled: list[bool] = []
    real_cancel = cli.cancel_running_calls

    def spy_cancel() -> None:
        signalled.append(True)
        real_cancel()

    def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
        if model == MODELS[0]:
            raise KeyboardInterrupt
        return CallResult(CURRENT)

    monkeypatch.setattr(cli, "_execute_call", fake)
    monkeypatch.setattr(cli, "cancel_running_calls", spy_cancel)
    try:
        with pytest.raises(KeyboardInterrupt):
            cli._run_panel(_settings(), REQUEST)
        # In-flight calls were told to stop...
        assert signalled == [True]
        # ...and the signal does not outlive the panel that raised it.
        assert not cc._CANCEL.is_set()
    finally:
        cc.reset_cancellation()


@pytest.mark.parametrize("sequential", [False, True])
def test_stale_cancellation_does_not_leak_into_the_next_panel(
    monkeypatch: pytest.MonkeyPatch, sequential: bool
) -> None:
    import code_review.claude_code as cc

    cc.cancel_running_calls()  # e.g. left behind by an earlier caller
    seen: list[bool] = []

    def fake(settings: Settings, system: str, user: str, model: str) -> CallResult:
        seen.append(cc._CANCEL.is_set())
        return CallResult(CURRENT)

    monkeypatch.setattr(cli, "_execute_call", fake)
    try:
        cli._run_panel(_settings(sequential=sequential), REQUEST)
    finally:
        cc.reset_cancellation()
    assert seen == [False, False]
