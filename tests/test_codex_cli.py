"""Tests for --provider codex: argv/env construction, event-stream
interpretation, tool-use enforcement, bounded process handling, settings
resolution and end-to-end CLI runs against a fake Codex executable."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import code_review.cli as cli
import code_review.codex_cli as cx
import code_review.config as config
from code_review.claude_code import ProcessOutcome
from code_review.config import Settings, _resolve_settings
from code_review.errors import (
    ConfigError,
    ContextOverflow,
    ProviderHiccup,
    RateLimit,
    ReviewError,
    TransportError,
)
from code_review.providers import CallResult

FAKE_SOURCE = Path(__file__).parent / "fixtures" / "fake_codex.py"
MODEL = "gpt-6.1-sol"


@pytest.fixture
def fake(tmp_path: Path) -> tuple[str, ...]:
    target_dir = tmp_path / "fake codex dir"
    target_dir.mkdir()
    target = target_dir / "fake codex.py"
    shutil.copyfile(FAKE_SOURCE, target)
    return (sys.executable, str(target))


@pytest.fixture
def temp_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "temp parent with spaces"
    parent.mkdir()
    return parent


def _call(
    fake: tuple[str, ...],
    temp_parent: Path,
    scenario: str,
    *,
    record: Path | None = None,
    timeout: float = 60.0,
    user_prompt: str = "review this diff",
    **extra_env: str,
) -> CallResult:
    env = dict(os.environ)
    env["FAKE_CODEX_SCENARIO"] = scenario
    if record is not None:
        env["FAKE_CODEX_RECORD"] = str(record)
    env.update(extra_env)
    return cx.call_codex(
        system_prompt="SYSTEM PROMPT",
        user_prompt=user_prompt,
        model=MODEL,
        command=fake,
        timeout=timeout,
        environ=env,
        temp_parent=temp_parent,
    )


class TestArgv:
    def test_locked_down_flags(self, tmp_path: Path):
        argv = cx.build_codex_argv(
            ("codex",),
            model=MODEL,
            workdir=tmp_path / "w",
            instructions_file=tmp_path / "i.md",
        )
        assert argv[:2] == ["codex", "exec"]
        assert argv[-1] == "-"  # prompt from stdin
        for flag in (
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
        ):
            assert flag in argv
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert argv[argv.index("--model") + 1] == MODEL
        overrides = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
        assert "approval_policy='never'" in overrides
        assert "forced_login_method='chatgpt'" in overrides
        assert "project_doc_max_bytes=0" in overrides
        assert any(o.startswith("model_instructions_file='") for o in overrides)
        disabled = [argv[i + 1] for i, a in enumerate(argv) if a == "--disable"]
        assert set(disabled) == set(cx.DISABLED_FEATURES)
        assert "shell_tool" in disabled and "code_mode_host" in disabled
        for forbidden in (
            "--dangerously-bypass-approvals-and-sandbox",
            "--full-auto",
            "--add-dir",
            "--oss",
            "--profile",
        ):
            assert forbidden not in argv
        # No double quotes anywhere: npm's codex.cmd runs through cmd.exe.
        assert not any('"' in a for a in argv)

    def test_batch_launcher_accepts_the_argv(self, tmp_path: Path):
        argv = cx.build_codex_argv(
            ("C:/npm/codex.cmd",),
            model=MODEL,
            workdir=tmp_path / "dir with spaces",
            instructions_file=tmp_path / "dir with spaces" / "i.md",
        )
        cx._check_batch_shim(argv)  # must not raise

    @pytest.mark.parametrize(
        "name", ["plain", "o'brien", "with space", "dollar$sign", "smile😀"]
    )
    def test_instructions_path_round_trips_as_toml(self, tmp_path: Path, name: str):
        import tomllib

        path = tmp_path / name / "i.md"
        argv = cx.build_codex_argv(
            ("codex",), model=MODEL, workdir=tmp_path, instructions_file=path
        )
        (override,) = [
            argv[i + 1]
            for i, a in enumerate(argv)
            if a == "-c" and argv[i + 1].startswith("model_instructions_file=")
        ]
        parsed = tomllib.loads(override)
        assert parsed["model_instructions_file"] == str(path)

    def test_quoted_path_needs_native_executable(self, tmp_path: Path):
        argv = cx.build_codex_argv(
            ("C:/npm/codex.cmd",),
            model=MODEL,
            workdir=tmp_path,
            instructions_file=tmp_path / "o'brien" / "i.md",
        )
        with pytest.raises(ConfigError, match="batch script"):
            cx._check_batch_shim(argv)

    def test_launcher_error_names_codex(
        self, temp_parent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # A temp parent with a quote forces the escaped TOML form, which the
        # .cmd check refuses; the message must point at Codex, not Claude.
        quoted = tmp_path / "o'brien"
        quoted.mkdir()
        with pytest.raises(ConfigError) as exc_info:
            cx.call_codex(
                system_prompt="s",
                user_prompt="u",
                model=MODEL,
                command=(str(tmp_path / "codex.cmd"),),
                timeout=5,
                temp_parent=quoted,
            )
        message = str(exc_info.value)
        assert "Codex launcher" in message
        assert "$CODE_REVIEW_CODEX_BIN" in message and "codex.exe" in message
        assert "CLAUDE" not in message

    def test_prompt_on_stdin_instructions_in_file(
        self, fake: tuple[str, ...], temp_parent: Path, tmp_path: Path
    ):
        record = tmp_path / "rec.json"
        payload = "UNIQUE-PAYLOAD café → 日本語\r\nline2"
        _call(fake, temp_parent, "success", record=record, user_prompt=payload)
        seen = json.loads(record.read_text(encoding="utf-8"))
        assert seen["stdin"] == payload
        assert "UNIQUE-PAYLOAD" not in " ".join(seen["argv"])
        assert seen["instructions"] == "SYSTEM PROMPT"
        # The working directory is empty: the instructions file lives beside it.
        assert seen["cwd_entries"] == []
        assert Path(seen["cwd"]).parent.parent == temp_parent
        assert list(temp_parent.iterdir()) == []


class TestEnvironment:
    def test_api_keys_and_secrets_removed(self):
        env, stripped = cx.build_child_env(
            {
                "PATH": "p",
                "openai_api_key": "sk-x",
                "CODEX_API_KEY": "c",
                "ANTHROPIC_API_KEY": "a",
                "OPENROUTER_API_KEY": "o",
                "CODEX_HOME": "h",
            }
        )
        assert env == {"PATH": "p", "CODEX_HOME": "h"}
        assert stripped == ["CODEX_API_KEY", "OPENAI_API_KEY"]

    def test_child_never_sees_api_keys(
        self,
        fake: tuple[str, ...],
        temp_parent: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        record = tmp_path / "rec.json"
        _call(
            fake,
            temp_parent,
            "success",
            record=record,
            OPENAI_API_KEY="sk-proj-SYNTHETICVALUE000000000000",
        )
        seen = json.loads(record.read_text(encoding="utf-8"))
        assert "OPENAI_API_KEY" not in seen["env_keys"]
        err = capsys.readouterr().err
        assert "OPENAI_API_KEY" in err and "SYNTHETICVALUE" not in err


class TestValidation:
    @pytest.mark.parametrize("name", ["gpt-6.1-sol", "gpt-5.5", "gpt-6-luna"])
    def test_slugs_accepted(self, name: str):
        assert cx.validate_codex_model(name) == name

    def test_latest_rejected(self):
        with pytest.raises(ConfigError, match="floating"):
            cx.validate_codex_model("gpt-daybreak-blue-latest")

    @pytest.mark.parametrize("name", ["openai/gpt-5", "GPT 6", ""])
    def test_non_slugs_rejected(self, name: str):
        with pytest.raises(ConfigError, match="Codex model slug"):
            cx.validate_codex_model(name)

    def test_timeout_bounds(self):
        assert cx.resolve_codex_timeout(None, "d") == cx.DEFAULT_CODEX_TIMEOUT
        assert cx.resolve_codex_timeout("60", "$X") == 60.0
        for bad in ("0", "-1", "inf", "nan", "1e9", "abc"):
            with pytest.raises(ConfigError):
                cx.resolve_codex_timeout(bad, "$X")

    def test_command_never_from_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        repo = tmp_path / "reviewed repo"
        repo.mkdir()
        name = "codex.exe" if sys.platform == "win32" else "codex"
        (repo / name).write_bytes(b"")
        (repo / name).chmod(0o755)
        real_dir = tmp_path / "real bin"
        real_dir.mkdir()
        (real_dir / name).write_bytes(b"")
        (real_dir / name).chmod(0o755)
        monkeypatch.chdir(repo)
        (found,) = cx.resolve_codex_command(
            {"PATH": os.pathsep.join([".", str(real_dir)])}
        )
        assert os.path.normcase(found) == os.path.normcase(str(real_dir / name))
        with pytest.raises(ConfigError, match="absolute path"):
            cx.resolve_codex_command({"CODE_REVIEW_CODEX_BIN": name})
        with pytest.raises(ConfigError, match="not found on PATH"):
            cx.resolve_codex_command({"PATH": "."})


class TestProcess:
    def test_success(self, fake: tuple[str, ...], temp_parent: Path):
        result = _call(fake, temp_parent, "success")
        assert "Empty input crashes parser." in result.content
        assert (result.prompt_tokens, result.completion_tokens) == (1200, 77)

    def test_non_json_lines_ignored(self, fake: tuple[str, ...], temp_parent: Path):
        assert "Empty input" in _call(fake, temp_parent, "extra_lines").content

    def test_diagnostic_items_are_not_tools(
        self,
        fake: tuple[str, ...],
        temp_parent: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        result = _call(fake, temp_parent, "diagnostics")
        assert "Empty input" in result.content
        err = capsys.readouterr().err
        # Unexpected diagnostics surface once each; the code-mode notice this
        # configuration causes on purpose stays quiet.
        assert err.count("WARN: codex reported: Codex is ignoring") == 1
        assert "Code Mode" not in err

    def test_tool_start_stops_the_run_immediately(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        started = time.monotonic()
        with pytest.raises(ReviewError, match="tool call") as exc_info:
            _call(fake, temp_parent, "tool_then_sleep")
        assert "command_execution" in str(exc_info.value)
        # Stopped at the event, not after the fake's 60s sleep.
        assert time.monotonic() - started < 30
        assert list(temp_parent.iterdir()) == []

    def test_completed_tool_item_discards_review(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        with pytest.raises(ReviewError, match="file_change"):
            _call(fake, temp_parent, "tool_completed")

    @pytest.mark.parametrize(
        ("scenario", "error_type", "fragment"),
        [
            ("turn_failed_auth", ConfigError, "codex login"),
            ("error_quota", RateLimit, "usage or rate limit"),
            ("turn_failed_context", ContextOverflow, "too large"),
            ("unexpected_argument", ConfigError, "rejected a required option"),
            ("no_turn_completed", ProviderHiccup, "turn.completed"),
            ("malformed", ProviderHiccup, "turn.completed"),
            ("empty_message", ProviderHiccup, "without a final message"),
            # Found by the real gpt-6.1-sol review of this PR: a malformed
            # final message must not let earlier commentary pass as the review.
            ("malformed_final", ProviderHiccup, "malformed event"),
        ],
    )
    def test_failures_are_typed(
        self,
        fake: tuple[str, ...],
        temp_parent: Path,
        scenario: str,
        error_type: type[ReviewError],
        fragment: str,
    ):
        with pytest.raises(error_type) as exc_info:
            _call(fake, temp_parent, scenario)
        assert type(exc_info.value) is error_type
        assert fragment in str(exc_info.value)
        assert exc_info.value.provider == "codex"

    def test_error_text_redacted(self, fake: tuple[str, ...], temp_parent: Path):
        with pytest.raises(ReviewError) as exc_info:
            _call(fake, temp_parent, "secret_in_error")
        rendered = f"{exc_info.value} {exc_info.value.detail}"
        assert "SYNTHETICSECRET" not in rendered

    def test_timeout(self, fake: tuple[str, ...], temp_parent: Path):
        started = time.monotonic()
        with pytest.raises(TransportError, match="did not finish within 1s"):
            _call(fake, temp_parent, "sleep", timeout=1.0)
        assert time.monotonic() - started < 30

    def test_missing_executable(self, temp_parent: Path, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            cx.call_codex(
                system_prompt="s",
                user_prompt="u",
                model=MODEL,
                command=(str(tmp_path / "no such" / "codex.exe"),),
                timeout=5,
                temp_parent=temp_parent,
            )

    def test_oversized_output(self, monkeypatch: pytest.MonkeyPatch):
        outcome = ProcessOutcome(
            returncode=None, stdout=b"x", stderr=b"", stdout_overflow=True
        )
        with pytest.raises(ContextOverflow):
            cx.interpret_outcome(outcome, model=MODEL, timeout=1)

    def test_cancelled(self):
        outcome = ProcessOutcome(
            returncode=None, stdout=b"", stderr=b"", cancelled=True
        )
        with pytest.raises(ReviewError, match="cancelled") as exc_info:
            cx.interpret_outcome(outcome, model=MODEL, timeout=1)
        assert type(exc_info.value) is ReviewError

    def test_cancelled_call_is_not_retried(self, monkeypatch: pytest.MonkeyPatch):
        from code_review.providers import _call_with_retries

        calls = 0
        outcome = ProcessOutcome(
            returncode=None, stdout=b"", stderr=b"", cancelled=True
        )

        def call() -> CallResult:
            nonlocal calls
            calls += 1
            return cx.interpret_outcome(outcome, model=MODEL, timeout=1)

        monkeypatch.setattr("code_review.providers.time.sleep", lambda s: None)
        with pytest.raises(ReviewError, match="cancelled"):
            _call_with_retries(call, label="codex", retries=3, auto_retry=False)
        assert calls == 1

    @pytest.mark.parametrize(
        ("message", "error_type"),
        [
            # Whole status codes only; digits inside other numbers don't count.
            ("context_length_exceeded: input is 1429012 tokens", ContextOverflow),
            ("request 4015029 failed: unknown failure", ReviewError),
            ("forced_login_method is set to chatgpt", ReviewError),
            ("upstream returned 5034 bytes of junk", ReviewError),
            ("HTTP 429 Too Many Requests", RateLimit),
            ("401 Unauthorized", ConfigError),
            ("please run `codex login`", ConfigError),
            ("HTTP 503 from upstream", TransportError),
        ],
    )
    def test_status_codes_match_whole_numbers(
        self, message: str, error_type: type[ReviewError]
    ):
        err = cx.classify_failure(message, model=MODEL)
        assert type(err) is error_type

    def test_recovered_error_event_does_not_fail_the_turn(
        self,
        fake: tuple[str, ...],
        temp_parent: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        result = _call(fake, temp_parent, "transient_error")
        assert "Empty input" in result.content
        assert (
            "WARN: codex reported (recovered): Reconnecting" in capsys.readouterr().err
        )

    def test_error_after_last_completed_turn_still_fails(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        with pytest.raises(RateLimit):
            _call(fake, temp_parent, "error_after_turn")


def _args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        base=None,
        pr=None,
        staged=False,
        codebase=False,
        include=[],
        exclude=[],
        provider="codex",
        model=None,
        models=None,
        ollama_host=None,
        temperature=None,
        max_tokens=None,
        retries=None,
        min_severity=None,
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
    def _isolate(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(config, "resolve_codex_command", lambda: ("codex-fake",))
        for name in (
            "CODE_REVIEW_PROVIDER",
            "CODE_REVIEW_CODEX_MODEL",
            "CODE_REVIEW_CODEX_TIMEOUT",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_defaults(self):
        settings = _resolve_settings(_args())
        assert settings.provider == "codex"
        assert settings.model == "gpt-6.1-sol"
        assert settings.codex_command == ("codex-fake",)
        assert settings.codex_timeout == cx.DEFAULT_CODEX_TIMEOUT
        assert settings.api_key is None

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CODE_REVIEW_CODEX_MODEL", "gpt-5.5")
        monkeypatch.setenv("CODE_REVIEW_CODEX_TIMEOUT", "120")
        settings = _resolve_settings(_args())
        assert (settings.model, settings.codex_timeout) == ("gpt-5.5", 120.0)

    def test_project_config_cannot_select_codex(self):
        with pytest.raises(ConfigError, match="'codex' cannot be selected"):
            _resolve_settings(_args(provider=None), {"provider": "codex"})

    def test_project_config_quota_keys_ignored(
        self, capsys: pytest.CaptureFixture[str]
    ):
        settings = _resolve_settings(
            _args(),
            {"models": ["gpt-6.1-sol", "gpt-5.5"], "retries": 9, "format": "json"},
        )
        assert settings.models is None and settings.retries == 0
        assert settings.format == "json"
        assert "--provider codex ignores .code-review.toml keys models, retries" in (
            capsys.readouterr().err
        )

    def test_sequential_codex_panel(self):
        settings = _resolve_settings(
            _args(models="gpt-6.1-sol,gpt-5.5", sequential=True)
        )
        assert settings.models == ("gpt-6.1-sol", "gpt-5.5")
        assert settings.sequential is True


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        provider="codex",
        model=MODEL,
        temperature=0.3,
        max_tokens=100,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
        codex_command=("codex-fake",),
        codex_timeout=30.0,
    )
    base.update(overrides)
    return Settings(**base)


def test_execute_call_has_no_automatic_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fail(**kwargs: Any) -> CallResult:
        calls.append(kwargs)
        raise TransportError("timeout")

    monkeypatch.setattr(cli, "call_codex", fail)
    monkeypatch.setattr("code_review.providers.time.sleep", lambda s: None)
    with pytest.raises(TransportError):
        cli._execute_call(_settings(), "sys", "user", MODEL)
    assert len(calls) == 1
    assert calls[0]["command"] == ("codex-fake",)


class TestCliEndToEnd:
    @pytest.fixture
    def run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: Any):
        diff = tmp_path / "change.patch"
        diff.write_text(
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,2 +1,3 @@\n def parse(items):\n+    first = items[0]\n"
            "     return first\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        monkeypatch.setattr(config, "resolve_codex_command", lambda: fake)
        monkeypatch.delenv("CODE_REVIEW_PROVIDER", raising=False)
        monkeypatch.delenv("CODE_REVIEW_CODEX_MODEL", raising=False)

        def _run(scenario: str, *extra: str) -> None:
            monkeypatch.setenv("FAKE_CODEX_SCENARIO", scenario)
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "code-review",
                    "--provider",
                    "codex",
                    "--diff-file",
                    str(diff),
                    "--no-project-config",
                    *extra,
                ],
            )
            cli._entrypoint()

        return _run

    def test_json_review(self, run: Any, capsys: pytest.CaptureFixture[str]):
        run("success", "--format", "json")
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert (report["provider"], report["model"]) == ("codex", MODEL)
        assert [f["title"] for f in report["findings"]] == [
            "Empty input crashes parser."
        ]
        assert "TIP:" not in captured.err

    def test_auth_exit_code(self, run: Any, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit) as exc_info:
            run("turn_failed_auth")
        assert exc_info.value.code == 2
        assert "\nERROR: CONFIG [exit 2]\n" in capsys.readouterr().err

    def test_tool_use_exit_code(self, run: Any, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit) as exc_info:
            run("tool_completed")
        assert exc_info.value.code == 1
        assert "tool call" in capsys.readouterr().err

    def test_dry_run(self, run: Any, capsys: pytest.CaptureFixture[str]):
        run("success", "--dry-run")
        out = capsys.readouterr().out
        assert "provider:          codex" in out
        assert f"model:             {MODEL}" in out
        assert "codex_executable:" in out
        assert "any tool call stops the run" in out
        assert "temperature:       not sent" in out

    def test_list_models(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.setattr(sys, "argv", ["code-review", "--list-models"])
        cli.main()
        assert "--provider codex   (default: gpt-6.1-sol)" in capsys.readouterr().out
