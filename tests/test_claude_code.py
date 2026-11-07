"""Tests for --provider claude-code: argv/env construction, envelope
validation, failure classification, bounded process handling, settings
resolution and end-to-end CLI runs against a fake Claude executable."""

from __future__ import annotations

import _thread
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import code_review.claude_code as cc
import code_review.cli as cli
import code_review.config as config
from code_review.config import ReviewRequest, Settings, _resolve_settings
from code_review.errors import (
    ConfigError,
    ContextOverflow,
    ProviderHiccup,
    RateLimit,
    ReviewError,
    SafetyRefusal,
    TransportError,
)
from code_review.providers import CallResult

FAKE_SOURCE = Path(__file__).parent / "fixtures" / "fake_claude.py"
MODEL = "claude-opus-5-5"


@pytest.fixture
def fake(tmp_path: Path) -> tuple[str, ...]:
    """The fake executable, copied under a directory containing spaces."""
    target_dir = tmp_path / "fake claude dir"
    target_dir.mkdir()
    target = target_dir / "fake claude.py"
    shutil.copyfile(FAKE_SOURCE, target)
    return (sys.executable, str(target))


@pytest.fixture
def temp_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "temp parent with spaces"
    parent.mkdir()
    return parent


def _env(scenario: str, record: Path | None = None, **extra: str) -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["FAKE_CLAUDE_SCENARIO"] = scenario
    if record is not None:
        env["FAKE_CLAUDE_RECORD"] = str(record)
    env.update(extra)
    return env


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
    return cc.call_claude_code(
        system_prompt="SYSTEM PROMPT",
        user_prompt=user_prompt,
        model=MODEL,
        command=fake,
        timeout=timeout,
        environ=_env(scenario, record, **extra_env),
        temp_parent=temp_parent,
    )


class TestArgv:
    def test_locked_down_flags(self, tmp_path: Path):
        argv = cc.build_claude_argv(
            ("claude",), model=MODEL, system_prompt_file=tmp_path / "p.md"
        )
        assert argv[:2] == ["claude", "--print"]
        joined = " ".join(argv)
        assert "--output-format json" in joined
        assert f"--model {MODEL}" in joined
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--disallowedTools") + 1] == "mcp__*"
        for flag in (
            "--strict-mcp-config",
            "--safe-mode",
            "--disable-slash-commands",
            "--no-session-persistence",
        ):
            assert flag in argv
        assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
        assert argv[argv.index("--permission-prompts") + 1] == "none"
        # Never: bare (skips subscription auth), permission bypasses, or
        # anything that would add tools / directories.
        for forbidden in (
            "--bare",
            "--dangerously-skip-permissions",
            "--allow-dangerously-skip-permissions",
            "--allowedTools",
            "--add-dir",
            "--mcp-config",
            "--fallback-model",
        ):
            assert forbidden not in argv

    def test_prompt_is_not_on_the_command_line(
        self, fake: tuple[str, ...], temp_parent: Path, tmp_path: Path
    ):
        record = tmp_path / "rec.json"
        _call(fake, temp_parent, "success", record=record, user_prompt="UNIQUE-PAYLOAD")
        seen = json.loads(record.read_text(encoding="utf-8"))
        assert "UNIQUE-PAYLOAD" not in " ".join(seen["argv"])
        assert seen["stdin"] == "UNIQUE-PAYLOAD"
        assert seen["system_prompt"] == "SYSTEM PROMPT"


class TestEnvironment:
    def test_paid_backends_and_runner_secrets_removed(self):
        env, stripped = cc.build_child_env(
            {
                "PATH": "p",
                "anthropic_api_key": "sk-ant-x",
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "ANTHROPIC_AUTH_TOKEN": "",
                "OPENROUTER_API_KEY": "or-key",
                "GEMINI_API_KEY": "g-key",
                "HOME": "h",
            }
        )
        assert env == {"PATH": "p", "HOME": "h"}
        # Empty values are dropped silently; set ones are reported by name.
        assert stripped == ["ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK"]

    def test_child_never_sees_credentials(
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
            ANTHROPIC_API_KEY="sk-ant-api03-SYNTHETICVALUE0000",
            OPENROUTER_API_KEY="sk-or-SYNTHETICVALUE0000",
        )
        seen = json.loads(record.read_text(encoding="utf-8"))
        assert "ANTHROPIC_API_KEY" not in seen["env_keys"]
        assert "OPENROUTER_API_KEY" not in seen["env_keys"]
        err = capsys.readouterr().err
        assert "ANTHROPIC_API_KEY" in err  # named, so the user knows
        assert "SYNTHETICVALUE" not in err  # but never the value

    def test_redact(self):
        text = (
            "key sk-ant-api03-abcdefghijkl token=abcdef123456 "
            "Authorization: Bearer abcdefghijklmnop "
            "jwt eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.c2lnbmF0dXJlMTIz"
        )
        out = cc.redact(text, ["explicit-secret-value"])
        assert "sk-ant-api03" not in out
        assert "abcdef123456" not in out
        assert "abcdefghijklmnop" not in out
        assert "eyJhbGci" not in out
        assert cc.redact("has explicit-secret-value", ["explicit-secret-value"]) == (
            "has [REDACTED]"
        )


class TestModelValidation:
    @pytest.mark.parametrize(
        "name", ["sonnet", "opus", "fable", "claude-sonnet-latest"]
    )
    def test_floating_aliases_rejected(self, name: str):
        with pytest.raises(ConfigError, match="floating"):
            cc.validate_claude_model(name)

    @pytest.mark.parametrize("name", ["anthropic/claude-sonnet-4.5", "gpt-5", ""])
    def test_non_native_ids_rejected(self, name: str):
        with pytest.raises(ConfigError, match="native Claude model ID"):
            cc.validate_claude_model(name)

    @pytest.mark.parametrize(
        "name",
        ["claude-sonnet-5-5", "claude-opus-5-5[1m]", "claude-haiku-4-5-20251001"],
    )
    def test_pinned_ids_accepted(self, name: str):
        assert cc.validate_claude_model(name) == name


def _make_executable(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = "claude.exe" if sys.platform == "win32" else "claude"
    target = directory / name
    target.write_bytes(b"")
    target.chmod(0o755)
    return target


class TestResolveCommandSafety:
    def test_current_directory_is_never_searched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # A reviewed repo that ships its own `claude` must not get executed,
        # even when PATH contains "." or an empty entry.
        repo = tmp_path / "reviewed repo"
        _make_executable(repo)
        real = _make_executable(tmp_path / "real bin")
        monkeypatch.chdir(repo)
        monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
        path_value = os.pathsep.join([".", "", str(real.parent)])
        (found,) = cc.resolve_claude_command({"PATH": path_value})
        assert os.path.normcase(found) == os.path.normcase(str(real))

    def test_relative_entries_alone_find_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        repo = tmp_path / "reviewed repo"
        _make_executable(repo / "bin")
        _make_executable(repo)
        monkeypatch.chdir(repo)
        with pytest.raises(ConfigError, match="not found on PATH"):
            cc.resolve_claude_command({"PATH": os.pathsep.join([".", "bin"])})

    def test_explicit_relative_path_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _make_executable(tmp_path)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigError, match="absolute path"):
            cc.resolve_claude_command(
                {
                    "CODE_REVIEW_CLAUDE_BIN": "claude.exe"
                    if os.name == "nt"
                    else "claude"
                }
            )


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill is Windows-only")
class TestTaskkillResolution:
    def test_system_taskkill_is_absolute_in_system_directory(self):
        found = cc._system_taskkill()
        assert found is not None
        path = Path(found)
        assert path.is_absolute() and path.name.lower() == "taskkill.exe"
        assert path.parent.name.lower() == "system32"

    def test_cleanup_ignores_taskkill_planted_in_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        repo = tmp_path / "reviewed repo"
        repo.mkdir()
        (repo / "taskkill.exe").write_bytes(b"planted")
        monkeypatch.chdir(repo)
        launched: list[list[str]] = []
        real_run = subprocess.run

        def spy(argv: list[str], **kwargs: Any) -> Any:
            launched.append(list(argv))
            return real_run(argv, **kwargs)

        monkeypatch.setattr(cc.subprocess, "run", spy)
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            cc._kill_tree(proc)
            assert proc.wait(timeout=15) is not None
        finally:
            if proc.poll() is None:
                proc.kill()
        assert launched, "taskkill was not invoked"
        exe = Path(launched[0][0])
        assert exe.is_absolute()
        assert exe.parent != repo
        assert os.path.normcase(str(exe)) == os.path.normcase(
            str(cc._system_taskkill())
        )


class TestResolveCommand:
    def test_explicit_path(self, tmp_path: Path):
        exe = tmp_path / "my claude" / "claude.exe"
        exe.parent.mkdir()
        exe.write_bytes(b"")
        assert cc.resolve_claude_command({"CODE_REVIEW_CLAUDE_BIN": str(exe)}) == (
            str(exe),
        )

    def test_explicit_missing(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="CODE_REVIEW_CLAUDE_BIN"):
            cc.resolve_claude_command(
                {"CODE_REVIEW_CLAUDE_BIN": str(tmp_path / "nope.exe")}
            )

    def test_not_on_path(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found on PATH"):
            cc.resolve_claude_command({"PATH": str(tmp_path)})

    def test_timeout_parsing(self):
        assert cc.resolve_claude_timeout(None, "default") == cc.DEFAULT_CLAUDE_TIMEOUT
        assert cc.resolve_claude_timeout("30", "$X") == 30.0
        assert cc.resolve_claude_timeout("86400", "$X") == 86400.0
        # inf / huge values would overflow Windows' millisecond wait.
        for bad in ("abc", "0", "-5", "nan", "inf", "1e7"):
            with pytest.raises(ConfigError):
                cc.resolve_claude_timeout(bad, "$X")

    def test_batch_shim_rejects_metacharacters(self):
        cc._check_batch_shim(["C:/x/claude.exe", "a&b"])  # native exe: fine
        cc._check_batch_shim(["C:/x/claude.cmd", "plain", "C:/dir with spaces/p.md"])
        with pytest.raises(ConfigError, match="batch script"):
            cc._check_batch_shim(["C:/x/claude.cmd", "C:/a&b/p.md"])


class TestEndToEndProcess:
    def test_success(self, fake: tuple[str, ...], temp_parent: Path, tmp_path: Path):
        record = tmp_path / "rec.json"
        result = _call(fake, temp_parent, "success", record=record)
        assert "Empty input crashes parser." in result.content
        assert result.prompt_tokens == 135  # input + cache creation + cache read
        assert result.completion_tokens == 42
        assert result.truncated is False
        seen = json.loads(record.read_text(encoding="utf-8"))
        # Ran in a private directory under the requested parent, cleaned up.
        assert Path(seen["cwd"]).parent == temp_parent
        assert not Path(seen["cwd"]).exists()
        assert list(temp_parent.iterdir()) == []

    def test_unicode_stdin_round_trips(
        self, fake: tuple[str, ...], temp_parent: Path, tmp_path: Path
    ):
        record = tmp_path / "rec.json"
        payload = "diff: café → naïve 日本語 \u2014 emoji \U0001f600\r\nline2"
        _call(fake, temp_parent, "success", record=record, user_prompt=payload)
        seen = json.loads(record.read_text(encoding="utf-8"))
        assert seen["stdin"] == payload

    def test_reviewed_repo_is_not_the_working_directory(
        self,
        fake: tuple[str, ...],
        temp_parent: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        repo = tmp_path / "reviewed repo"
        repo.mkdir()
        (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.chdir(repo)
        before = {p.name: p.read_bytes() for p in repo.iterdir()}
        _call(fake, temp_parent, "mutate")
        after = {p.name: p.read_bytes() for p in repo.iterdir()}
        assert after == before
        assert list(temp_parent.iterdir()) == []  # the stray write was discarded

    def test_stray_line_before_envelope_tolerated(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        assert "Empty input" in _call(fake, temp_parent, "warning_line").content

    def test_stray_line_after_envelope_tolerated(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        assert "Empty input" in _call(fake, temp_parent, "trailing_line").content

    def test_timeout_message_states_the_real_bounds(self):
        with pytest.raises(ConfigError, match="greater than 0 and at most 86400"):
            cc.resolve_claude_timeout("0", "$X")

    def test_truncated_review_warns(
        self,
        fake: tuple[str, ...],
        temp_parent: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        result = _call(fake, temp_parent, "truncated_review")
        assert result.truncated is True
        assert "truncated" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("scenario", "error_type", "fragment"),
        [
            ("auth_error", ConfigError, "not signed in"),
            ("auth_error_exit0", ConfigError, "not signed in"),
            ("quota", RateLimit, "usage or rate limit"),
            ("billing", ConfigError, "billing or credit"),
            ("error_subtype", TransportError, "Claude API request failed"),
            ("unknown_option", ConfigError, "rejected a required option"),
            ("crash", ReviewError, "something went badly wrong"),
            ("malformed", ProviderHiccup, "no valid JSON"),
            ("truncated_json", ProviderHiccup, "no valid JSON"),
            ("empty", ProviderHiccup, "no valid JSON"),
            ("bad_types", ProviderHiccup, "no valid JSON"),
            ("missing_result", ProviderHiccup, "no result text"),
            ("empty_max_tokens", ContextOverflow, "output limit"),
            ("refusal", SafetyRefusal, "declined"),
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
        assert exc_info.value.provider == "claude-code"
        assert list(temp_parent.iterdir()) == []

    def test_error_text_is_redacted(self, fake: tuple[str, ...], temp_parent: Path):
        with pytest.raises(ConfigError) as exc_info:
            _call(fake, temp_parent, "secret_in_error")
        rendered = f"{exc_info.value} {exc_info.value.detail}"
        assert "SYNTHETICSECRET" not in rendered
        assert "[REDACTED]" in rendered

    def test_plain_stdout_error_is_redacted(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        with pytest.raises(ConfigError) as exc_info:
            _call(fake, temp_parent, "plain_secret_stdout")
        rendered = f"{exc_info.value} {exc_info.value.detail}"
        assert "SYNTHETICSECRET" not in rendered
        assert "[REDACTED]" in rendered

    def test_api_error_reported_as_success_is_not_a_review(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        with pytest.raises(TransportError):
            _call(fake, temp_parent, "api_error_as_success")

    def test_user_settings_api_key_warns_without_value(
        self,
        fake: tuple[str, ...],
        temp_parent: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        config_dir = tmp_path / "claude config"
        config_dir.mkdir()
        (config_dir / "settings.json").write_text(
            json.dumps(
                {
                    "apiKeyHelper": "/synthetic/helper.sh",
                    "env": {"ANTHROPIC_API_KEY": "sk-ant-api03-SYNTHETICVALUE00"},
                }
            ),
            encoding="utf-8",
        )
        _call(fake, temp_parent, "success", CLAUDE_CONFIG_DIR=str(config_dir))
        err = capsys.readouterr().err
        assert "apiKeyHelper, env.ANTHROPIC_API_KEY" in err
        assert "SYNTHETICVALUE" not in err
        assert "/synthetic/helper.sh" not in err

    def test_cancellation_stops_the_process(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        timer = threading.Timer(0.5, cc.cancel_running_calls)
        timer.start()
        started = time.monotonic()
        try:
            with pytest.raises(TransportError, match="cancelled"):
                _call(fake, temp_parent, "sleep", timeout=60.0)
        finally:
            timer.cancel()
            cc.reset_cancellation()
        assert time.monotonic() - started < 30

    def test_ctrl_c_interrupts_the_wait(self, fake: tuple[str, ...], temp_parent: Path):
        # KeyboardInterrupt in the main thread must arrive promptly (sliced
        # waits) and still stop the process and remove the temp directory.
        timer = threading.Timer(0.5, _thread.interrupt_main)
        timer.start()
        started = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                _call(fake, temp_parent, "sleep", timeout=60.0)
        finally:
            timer.cancel()
        assert time.monotonic() - started < 30

    def test_lingering_grandchild_does_not_hang(
        self, fake: tuple[str, ...], temp_parent: Path
    ):
        started = time.monotonic()
        result = _call(fake, temp_parent, "grandchild", timeout=60.0)
        assert "Empty input crashes parser." in result.content
        # Bounded by the reader-join timeout, not the helper's 30s sleep.
        assert time.monotonic() - started < 25

    def test_timeout_stops_the_process(self, fake: tuple[str, ...], temp_parent: Path):
        started = time.monotonic()
        with pytest.raises(TransportError, match="did not finish within 1s"):
            _call(fake, temp_parent, "sleep", timeout=1.0)
        assert time.monotonic() - started < 30
        assert list(temp_parent.iterdir()) == []

    def test_oversized_output_is_capped(
        self,
        fake: tuple[str, ...],
        temp_parent: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(cc, "MAX_STDOUT_BYTES", 64 * 1024)
        started = time.monotonic()
        with pytest.raises(ContextOverflow, match="exceeded"):
            _call(fake, temp_parent, "flood", timeout=60.0)
        # Stopped at the cap, not left running until the 60s sleep ends.
        assert time.monotonic() - started < 30

    def test_missing_executable(self, temp_parent: Path, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            cc.call_claude_code(
                system_prompt="s",
                user_prompt="u",
                model=MODEL,
                command=(str(tmp_path / "does not exist" / "claude.exe"),),
                timeout=5,
                temp_parent=temp_parent,
            )

    def test_oversized_stdin_rejected_before_launch(
        self, temp_parent: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(cc, "MAX_STDIN_BYTES", 10)
        with pytest.raises(ContextOverflow, match="stdin"):
            cc.call_claude_code(
                system_prompt="s",
                user_prompt="x" * 11,
                model=MODEL,
                command=("never-run",),
                timeout=5,
                temp_parent=temp_parent,
            )

    def test_empty_command(self):
        with pytest.raises(ConfigError):
            cc.call_claude_code(
                system_prompt="s", user_prompt="u", model=MODEL, command=(), timeout=1
            )


class TestInterpret:
    def _outcome(self, payload: Any, returncode: int = 0) -> cc.ProcessOutcome:
        return cc.ProcessOutcome(
            returncode=returncode, stdout=json.dumps(payload).encode(), stderr=b""
        )

    def test_zero_exit_with_error_envelope_is_failure(self):
        payload = {"type": "result", "subtype": "success", "is_error": True}
        payload["result"] = "API Error: 500 internal server error"
        with pytest.raises(TransportError):
            cc.interpret_outcome(self._outcome(payload), model=MODEL, timeout=1)

    def test_nonzero_exit_with_success_envelope_is_failure(self):
        payload = {"type": "result", "subtype": "success", "is_error": False}
        # Review text that happens to contain error-like phrases must not
        # drive the classification; only stderr may.
        payload["result"] = "### L1: [HIGH] Authentication bypass; rate limit missing"
        with pytest.raises(ReviewError) as exc_info:
            cc.interpret_outcome(self._outcome(payload, 1), model=MODEL, timeout=1)
        assert type(exc_info.value) is ReviewError
        assert "exit status 1" in str(exc_info.value)

    def test_wrong_type_tag_rejected(self):
        payload = {"type": "assistant", "subtype": "success", "is_error": False}
        with pytest.raises(ProviderHiccup):
            cc.interpret_outcome(self._outcome(payload), model=MODEL, timeout=1)

    def test_model_mismatch_and_denials_warn(self, capsys: pytest.CaptureFixture[str]):
        payload = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "# Change summary: x\n\nNo issues found.",
            "modelUsage": {"claude-other-model": {}},
            "permission_denials": [{"tool_name": "Edit"}],
            "usage": {"input_tokens": True, "output_tokens": "7"},
        }
        result = cc.interpret_outcome(self._outcome(payload), model=MODEL, timeout=1)
        # Malformed usage metadata degrades to "unknown", never fails.
        assert result.prompt_tokens is None and result.completion_tokens is None
        err = capsys.readouterr().err
        assert "denied 1 tool request" in err
        assert "not the requested model" in err

    @pytest.mark.parametrize(
        ("message", "status", "error_type"),
        [
            ("anything", 401, ConfigError),
            ("anything", 429, RateLimit),
            ("anything", 413, ContextOverflow),
            ("anything", 404, ConfigError),
            ("anything", 529, TransportError),
            ("Prompt is too long", None, ContextOverflow),
            ("model_not_found: claude-x", None, ConfigError),
            ("You've hit your limit · resets 3pm", None, RateLimit),
            ("Request took too long to start", None, ReviewError),
        ],
    )
    def test_classification(
        self, message: str, status: int | None, error_type: type[ReviewError]
    ):
        err = cc.classify_failure(message, model=MODEL, status=status)
        assert type(err) is error_type


def _args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        base=None,
        pr=None,
        staged=False,
        codebase=False,
        include=[],
        exclude=[],
        provider="claude-code",
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
    def _isolate(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(config, "resolve_claude_command", lambda: ("claude-fake",))
        for name in (
            "CODE_REVIEW_PROVIDER",
            "CODE_REVIEW_CLAUDE_MODEL",
            "CODE_REVIEW_CLAUDE_TIMEOUT",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_defaults_to_pinned_opus(self):
        settings = _resolve_settings(_args())
        assert settings.provider == "claude-code"
        assert settings.model == "claude-opus-5-5"
        assert settings.claude_command == ("claude-fake",)
        assert settings.claude_timeout == cc.DEFAULT_CLAUDE_TIMEOUT
        assert settings.api_key is None

    def test_env_model_and_timeout(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CODE_REVIEW_CLAUDE_MODEL", "claude-opus-5-5")
        monkeypatch.setenv("CODE_REVIEW_CLAUDE_TIMEOUT", "120")
        settings = _resolve_settings(_args())
        assert settings.model == "claude-opus-5-5"
        assert settings.claude_timeout == 120.0

    def test_floating_alias_rejected(self):
        with pytest.raises(ConfigError, match="floating"):
            _resolve_settings(_args(model="sonnet"))

    def test_panel_models_validated(self):
        with pytest.raises(ConfigError):
            _resolve_settings(_args(models="claude-sonnet-5-5,opus"))

    def test_project_config_cannot_select_claude_code(self):
        with pytest.raises(ConfigError, match="cannot be selected"):
            _resolve_settings(_args(provider=None), {"provider": "claude-code"})

    def test_env_can_select_claude_code(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CODE_REVIEW_PROVIDER", "claude-code")
        assert _resolve_settings(_args(provider=None)).provider == "claude-code"

    def test_project_config_cannot_set_quota_keys(
        self, capsys: pytest.CaptureFixture[str]
    ):
        settings = _resolve_settings(
            _args(min_severity=None),
            {
                "models": ["claude-opus-5-5", "claude-sonnet-5-5"],
                "model": "claude-opus-5-5[1m]",
                "retries": 10,
                "min_severity": "HIGH",
            },
        )
        assert settings.models is None
        assert settings.model == "claude-opus-5-5"
        assert settings.retries == 0
        assert settings.min_severity == "HIGH"  # review-shaping keys still apply
        assert "ignores .code-review.toml keys model, models, retries" in (
            capsys.readouterr().err
        )

    def test_operator_retries_still_apply(self):
        assert _resolve_settings(_args(retries=2)).retries == 2


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        provider="claude-code",
        model=MODEL,
        temperature=0.3,
        max_tokens=100,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
        claude_command=("claude-fake",),
        claude_timeout=30.0,
    )
    base.update(overrides)
    return Settings(**base)


class TestExecuteCall:
    def test_no_automatic_retry(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[dict[str, Any]] = []

        def fail(**kwargs: Any) -> CallResult:
            calls.append(kwargs)
            raise TransportError("timeout")

        monkeypatch.setattr(cli, "call_claude_code", fail)
        monkeypatch.setattr("code_review.providers.time.sleep", lambda s: None)
        with pytest.raises(TransportError):
            cli._execute_call(_settings(), "sys", "user", MODEL)
        assert len(calls) == 1
        assert calls[0]["command"] == ("claude-fake",)
        assert calls[0]["timeout"] == 30.0

        calls.clear()
        with pytest.raises(TransportError):
            cli._execute_call(_settings(retries=1), "sys", "user", MODEL)
        assert len(calls) == 2  # explicit --retries still applies


class TestCliEndToEnd:
    @pytest.fixture
    def run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake: tuple[str, ...],
    ):
        diff = tmp_path / "change.patch"
        diff.write_text(
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,2 +1,3 @@\n def parse(items):\n+    first = items[0]\n"
            "     return first\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        monkeypatch.setattr(config, "resolve_claude_command", lambda: fake)
        monkeypatch.delenv("CODE_REVIEW_PROVIDER", raising=False)
        monkeypatch.delenv("CODE_REVIEW_CLAUDE_MODEL", raising=False)

        def _run(scenario: str, *extra: str) -> None:
            monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", scenario)
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "code-review",
                    "--provider",
                    "claude-code",
                    "--model",
                    MODEL,
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
        assert report["provider"] == "claude-code"
        assert report["model"] == MODEL
        assert [f["title"] for f in report["findings"]] == [
            "Empty input crashes parser."
        ]
        assert report["usage"] == {"prompt_tokens": 135, "completion_tokens": 42}
        assert "[usage] prompt=135 completion=42" in captured.err
        assert "TIP:" not in captured.err

    def test_auth_failure_exit_code(self, run: Any, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit) as exc_info:
            run("auth_error")
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "\nERROR: CONFIG [exit 2]\n" in err
        assert "never logs in" in err

    def test_quota_exit_code(self, run: Any, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit) as exc_info:
            run("quota")
        assert exc_info.value.code == 11
        assert "ERROR: RATE_LIMIT [exit 11]" in capsys.readouterr().err

    def test_dry_run_reports_isolation(
        self, run: Any, capsys: pytest.CaptureFixture[str]
    ):
        run("success", "--dry-run")
        out = capsys.readouterr().out
        assert "provider:          claude-code" in out
        assert f"model:             {MODEL}" in out
        assert "execution:         single model call" in out
        assert "claude_executable:" in out
        assert "no tools" in out
        assert "temperature:       not sent" in out
        assert "est_cost" not in out

    def test_list_models_shows_default(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.setattr(sys, "argv", ["code-review", "--list-models"])
        cli.main()
        assert "--provider claude-code   (default: claude-opus-5-5)" in (
            capsys.readouterr().out
        )


def test_review_request_shape_is_shared() -> None:
    # The claude-code path consumes the same ReviewRequest as every other
    # provider; nothing provider-specific leaks into the request type.
    request = ReviewRequest("s", "u", "diff", 1)
    assert (request.system_prompt, request.user_prompt) == ("s", "u")
