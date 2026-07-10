"""Unit tests for the pure logic in review.py.

The project's selling point is a *contract* -- typed exit codes, a stable
stderr format, per-provider alias resolution -- and nearly all of that
logic is pure. These tests feed the classifiers canned inputs (HTTP
statuses + bodies, argparse namespaces, paths) and assert the typed
outcomes, so the contract can't drift silently as providers are added.

No network, no git, no Ollama server required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import code_review.cli as review
from code_review.cli import (
    BUILTIN_CODEBASE_EXCLUDES,
    INJECTION_GUARD,
    CallResult,
    ConfigError,
    ContextOverflow,
    ProviderHiccup,
    RateLimit,
    ReviewError,
    ReviewRequest,
    SafetyRefusal,
    Settings,
    TransportError,
    _apply_context,
    _call_with_retries,
    _classify_http_error,
    _dry_run_report,
    _format_size,
    _format_usage_line,
    _glob_match,
    _min_severity_instruction,
    _normalize_ollama_host,
    _number_lines,
    _ollama_prompt_guard,
    _parse_retry_after,
    _resolve_model,
    _usage_int,
    _write_output_file,
)


def _ns(provider: str, model: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(provider=provider, model=model)


# ---------------------------------------------------------------------------
# _classify_http_error
# ---------------------------------------------------------------------------


class TestClassifyHttpError:
    def _classify(self, status: int, body: str, **kwargs) -> ReviewError:
        return _classify_http_error(
            status, body, model="m", provider="p", **kwargs
        )

    def test_429_is_rate_limit(self):
        assert isinstance(self._classify(429, "slow down"), RateLimit)

    def test_429_includes_retry_after_hint(self):
        err = self._classify(429, "slow down", retry_after="30")
        assert "Retry-After: 30s" in str(err)

    def test_429_retry_after_http_date_gets_no_seconds_suffix(self):
        # Retry-After may be an HTTP-date; appending "s" would mangle it.
        err = self._classify(
            429, "slow down", retry_after="Wed, 21 Oct 2015 07:28:00 GMT"
        )
        assert "Retry-After: Wed, 21 Oct 2015 07:28:00 GMT" in str(err)
        assert "GMTs" not in str(err)

    def test_5xx_is_transport(self):
        assert isinstance(self._classify(503, "service unavailable"), TransportError)

    def test_5xx_with_overflow_phrase_stays_transport(self):
        # A provider-side failure page mentioning "token limit" must NOT
        # be misclassified as a do-not-retry CONTEXT_OVERFLOW.
        err = self._classify(500, "internal error: token limit tracker crashed")
        assert isinstance(err, TransportError)

    def test_413_is_context_overflow(self):
        assert isinstance(self._classify(413, "payload too large"), ContextOverflow)

    @pytest.mark.parametrize(
        "phrase",
        ["context_length", "too long", "exceeds the maximum", "token limit"],
    )
    def test_4xx_overflow_phrases(self, phrase: str):
        err = self._classify(400, f"error: input {phrase} for this model")
        assert isinstance(err, ContextOverflow)

    def test_unrecognized_4xx_falls_through_to_generic(self):
        err = self._classify(401, "invalid api key")
        assert type(err) is ReviewError
        assert err.exit_code == 1


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_explicit_slug_passes_through(self):
        assert _resolve_model(_ns("openrouter", "vendor/some-model")) == "vendor/some-model"

    def test_openrouter_alias_resolves(self):
        assert _resolve_model(_ns("openrouter", "claude")) == "anthropic/claude-sonnet-4.5"

    def test_ollama_alias_resolves(self):
        assert _resolve_model(_ns("ollama", "local")) == "qwen3-coder:30b"

    def test_cross_provider_alias_raises_config_error(self):
        with pytest.raises(ConfigError) as exc_info:
            _resolve_model(_ns("gemini", "claude"))
        assert "--provider openrouter" in str(exc_info.value)

    def test_ollama_alias_with_openrouter_raises(self):
        with pytest.raises(ConfigError) as exc_info:
            _resolve_model(_ns("openrouter", "local"))
        assert "--provider ollama" in str(exc_info.value)

    def test_provider_defaults(self, monkeypatch: pytest.MonkeyPatch):
        for var in ("OPENROUTER_MODEL", "GEMINI_MODEL", "OLLAMA_MODEL"):
            monkeypatch.delenv(var, raising=False)
        assert _resolve_model(_ns("openrouter")) == "google/gemini-2.5-pro"
        assert _resolve_model(_ns("gemini")) == "gemini-2.5-pro"
        assert _resolve_model(_ns("ollama")) == "qwen3-coder:30b"

    def test_env_var_alias_resolves_too(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OLLAMA_MODEL", "local")
        assert _resolve_model(_ns("ollama")) == "qwen3-coder:30b"


# ---------------------------------------------------------------------------
# _normalize_ollama_host
# ---------------------------------------------------------------------------


class TestNormalizeOllamaHost:
    def test_schemeless_host_port_gets_http(self):
        # Ollama's own $OLLAMA_HOST convention.
        assert _normalize_ollama_host("0.0.0.0:11434") == "http://0.0.0.0:11434"

    def test_full_url_unchanged(self):
        assert _normalize_ollama_host("http://localhost:11434") == "http://localhost:11434"

    def test_https_preserved(self):
        assert _normalize_ollama_host("https://ollama.example.com") == "https://ollama.example.com"

    def test_trailing_slash_stripped(self):
        assert _normalize_ollama_host("http://localhost:11434/") == "http://localhost:11434"

    def test_whitespace_stripped(self):
        assert _normalize_ollama_host(" localhost:11434 ") == "http://localhost:11434"

    @pytest.mark.parametrize("host", ["", "   "])
    def test_empty_host_raises_config_error(self, host: str):
        # Without the check, "" would normalize to the invalid URL "http:".
        with pytest.raises(ConfigError) as exc_info:
            _normalize_ollama_host(host)
        assert "OLLAMA_HOST" in str(exc_info.value)

    @pytest.mark.parametrize("host", ["http://", "http://:11434"])
    def test_hostname_less_url_raises_config_error(self, host: str):
        # A scheme with no hostname would fail later as an untyped httpx
        # URL error instead of a CONFIG error.
        with pytest.raises(ConfigError) as exc_info:
            _normalize_ollama_host(host)
        assert "hostname" in str(exc_info.value)

    def test_ipv6_host_accepted(self):
        assert _normalize_ollama_host("http://[::1]:11434") == "http://[::1]:11434"


# ---------------------------------------------------------------------------
# _ollama_prompt_guard
# ---------------------------------------------------------------------------


class TestOllamaPromptGuard:
    def test_small_prompt_passes(self):
        _ollama_prompt_guard(1000, 4096, model="m")  # no raise

    def test_oversized_prompt_raises_context_overflow(self):
        # 100K chars ~ 25K tokens >> a 4096-token window.
        with pytest.raises(ContextOverflow) as exc_info:
            _ollama_prompt_guard(100_000, 4096, model="m")
        # The runner requests the window per call now; the fix is the
        # env var, not a server restart.
        assert "$OLLAMA_NUM_CTX" in str(exc_info.value)

    def test_boundary_just_under_window_passes(self):
        # 4095 tokens' worth of chars against a 4096 window.
        _ollama_prompt_guard(4095 * review.OLLAMA_CHARS_PER_TOKEN, 4096, model="m")

    def test_unenforced_oversize_warns_instead_of_raising(
        self, capsys: pytest.CaptureFixture
    ):
        # Window unknown (env unset, model not loaded): a hard error
        # could reject a valid run on a 32K/256K machine, so the guard
        # only warns.
        _ollama_prompt_guard(100_000, 4096, model="m", enforced=False)
        err = capsys.readouterr().err
        assert err.startswith("WARN:")
        assert "OLLAMA_NUM_CTX" in err
        assert "model `m`" in err

    def test_unenforced_small_prompt_stays_silent(
        self, capsys: pytest.CaptureFixture
    ):
        _ollama_prompt_guard(1000, 4096, model="m", enforced=False)
        assert capsys.readouterr().err == ""


class TestMatchLoadedContext:
    PS_DATA = {
        "models": [
            {"name": "qwen3-coder:30b", "model": "qwen3-coder:30b", "context_length": 32768},
            {"name": "qwen3-coder-next:latest", "model": "qwen3-coder-next:latest", "context_length": 262144},
        ]
    }

    def test_exact_tagged_match(self):
        assert review._match_loaded_context(self.PS_DATA, "qwen3-coder:30b") == 32768

    def test_untagged_model_matches_latest(self):
        # Ollama normalizes untagged names to :latest when loading.
        assert review._match_loaded_context(self.PS_DATA, "qwen3-coder-next") == 262144

    def test_not_loaded_returns_none(self):
        assert review._match_loaded_context(self.PS_DATA, "llama3:8b") is None

    def test_missing_context_field_returns_none(self):
        # Older Ollama servers predate context_length in /api/ps.
        data = {"models": [{"name": "m:latest", "model": "m:latest"}]}
        assert review._match_loaded_context(data, "m:latest") is None

    def test_empty_or_malformed_data_returns_none(self):
        assert review._match_loaded_context({}, "m") is None
        assert review._match_loaded_context({"models": ["junk"]}, "m") is None

    @pytest.mark.parametrize("data", [None, [], "nope", 42])
    def test_top_level_non_dict_returns_none(self, data: object):
        # A misbehaving proxy can return a top-level list/string; the
        # probe must degrade to None, not AttributeError.
        assert review._match_loaded_context(data, "m") is None


class TestOllamaPostVerify:
    HOST = "http://localhost:11434"

    def test_no_usage_field_skips_silently(self, capsys: pytest.CaptureFixture):
        review._ollama_post_verify(None, 4096, host=self.HOST, model="m")
        assert capsys.readouterr().err == ""

    def test_below_margin_passes(self):
        review._ollama_post_verify(4000, 4096, host=self.HOST, model="m")

    def test_at_margin_raises(self):
        with pytest.raises(ContextOverflow) as exc_info:
            review._ollama_post_verify(4014, 4096, host=self.HOST, model="m")
        assert "truncated server-side" in str(exc_info.value)

    def test_unknown_window_reprobes(self, monkeypatch: pytest.MonkeyPatch):
        # Window wasn't sent (tier 3); the model is loaded post-call, so
        # /api/ps supplies the real window for verification.
        monkeypatch.setattr(
            review, "_detect_ollama_num_ctx", lambda host, model: 4096
        )
        with pytest.raises(ContextOverflow):
            review._ollama_post_verify(4090, None, host=self.HOST, model="m")

    def test_unknown_window_reprobe_fails_warns_and_skips(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        monkeypatch.setattr(
            review, "_detect_ollama_num_ctx", lambda host, model: None
        )
        review._ollama_post_verify(50_000, None, host=self.HOST, model="m")
        err = capsys.readouterr().err
        assert err.startswith("WARN:")
        assert "OLLAMA_NUM_CTX" in err


# ---------------------------------------------------------------------------
# _glob_match + builtin excludes
# ---------------------------------------------------------------------------


class TestGlobMatch:
    def test_extension_matches_at_any_depth(self):
        assert _glob_match(Path("a/b/c.py"), ["*.py"])

    def test_basename_pattern_matches_nested(self):
        assert _glob_match(Path("backend/tests/test_api.py"), ["test_*.py"])

    def test_top_level_dist_excluded(self):
        # Regression: `*/dist/*` alone cannot match a repo-root dist/.
        assert _glob_match(Path("dist/bundle.js"), BUILTIN_CODEBASE_EXCLUDES)

    def test_nested_dist_excluded(self):
        assert _glob_match(Path("pkg/dist/bundle.js"), BUILTIN_CODEBASE_EXCLUDES)

    def test_top_level_build_excluded(self):
        assert _glob_match(Path("build/out.o"), BUILTIN_CODEBASE_EXCLUDES)

    def test_lock_files_excluded(self):
        assert _glob_match(Path("uv.lock"), BUILTIN_CODEBASE_EXCLUDES)
        assert _glob_match(Path("frontend/package-lock.json"), BUILTIN_CODEBASE_EXCLUDES)

    def test_source_files_not_excluded(self):
        assert not _glob_match(Path("src/main.py"), BUILTIN_CODEBASE_EXCLUDES)
        assert not _glob_match(Path("distributed/worker.py"), BUILTIN_CODEBASE_EXCLUDES)


# ---------------------------------------------------------------------------
# _number_lines
# ---------------------------------------------------------------------------


class TestNumberLines:
    def test_numbers_are_one_indexed_and_transcribable(self):
        out = _number_lines("alpha\nbeta")
        assert out.splitlines()[0].strip().startswith("1:")
        assert out.splitlines()[1].strip().startswith("2:")

    def test_trailing_newline_preserved(self):
        assert _number_lines("a\n").endswith("\n")
        assert not _number_lines("a").endswith("\n")

    def test_empty_content_unchanged(self):
        assert _number_lines("") == ""


# ---------------------------------------------------------------------------
# _format_size
# ---------------------------------------------------------------------------


class TestFormatSize:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [(0, "0 B"), (999, "999 B"), (1000, "1 KB"), (100_000, "100 KB"), (1_500_000, "1.5 MB")],
    )
    def test_units(self, n: int, expected: str):
        assert _format_size(n) == expected


# ---------------------------------------------------------------------------
# Error-model contract invariants
# ---------------------------------------------------------------------------


class TestErrorModelContract:
    def test_exit_codes_match_readme_table(self):
        # These values are documented in README "Error model"; changing
        # one is a breaking change for LLM callers and must be deliberate.
        assert ConfigError.exit_code == 2
        assert review.SafetyRefusal.exit_code == 10
        assert RateLimit.exit_code == 11
        assert ContextOverflow.exit_code == 12
        assert review.ProviderHiccup.exit_code == 13
        assert TransportError.exit_code == 14
        assert ReviewError.exit_code == 1

    def test_print_error_emits_parseable_prefix(self, capsys: pytest.CaptureFixture):
        review._print_error(RateLimit("x", model="m", provider="p"))
        err = capsys.readouterr().err
        assert err.startswith("ERROR: RATE_LIMIT [exit 11]\n")


# ---------------------------------------------------------------------------
# _parse_retry_after + retry_after_seconds attribute
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    def test_delta_seconds(self):
        assert _parse_retry_after("30") == 30.0

    def test_delta_seconds_with_whitespace(self):
        assert _parse_retry_after("  120  ") == 120.0

    def test_http_date_in_future(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        future = datetime.now(timezone.utc) + timedelta(seconds=120)
        parsed = _parse_retry_after(format_datetime(future, usegmt=True))
        assert parsed is not None
        assert 0.0 < parsed <= 121.0

    def test_http_date_in_past_clamps_to_zero(self):
        assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0

    def test_garbage_returns_none(self):
        assert _parse_retry_after("soon-ish") is None

    @pytest.mark.parametrize(
        "value",
        [
            "Wed, 32 Dec 999999999999999999 07:28:00 GMT",  # OverflowError-shaped
            "Wed,",  # truncated structure
            ", , , ,",
        ],
    )
    def test_pathological_dates_return_none_not_raise(self, value: str):
        # codex finding: the email parser raises more than ValueError on
        # pathological inputs; any escape would turn a 429 into UNKNOWN.
        assert _parse_retry_after(value) is None

    def test_classify_sets_seconds_for_digits(self):
        err = _classify_http_error(
            429, "slow down", model="m", provider="p", retry_after="30"
        )
        assert isinstance(err, RateLimit)
        assert err.retry_after_seconds == 30.0

    def test_classify_sets_none_for_garbage_header(self):
        err = _classify_http_error(
            429, "slow down", model="m", provider="p", retry_after="soon-ish"
        )
        assert isinstance(err, RateLimit)
        assert err.retry_after_seconds is None

    def test_classify_no_header_leaves_none(self):
        err = _classify_http_error(429, "slow down", model="m", provider="p")
        assert isinstance(err, RateLimit)
        assert err.retry_after_seconds is None


# ---------------------------------------------------------------------------
# _call_with_retries
# ---------------------------------------------------------------------------


class _FlakyCall:
    """Callable raising the given exceptions in order, then returning a value."""

    def __init__(self, failures: list[Exception], value: str = "ok"):
        self.failures = list(failures)
        self.value = value
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.value


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(review.time, "sleep", recorded.append)
    return recorded


class TestCallWithRetries:
    def test_default_retries_transient_once(self, sleeps: list[float]):
        call = _FlakyCall([ProviderHiccup("x")])
        assert _call_with_retries(call, label="t", retries=0) == "ok"
        assert call.calls == 2
        assert sleeps == [2.0]

    def test_default_gives_up_after_second_transient(self, sleeps: list[float]):
        call = _FlakyCall([TransportError("x"), TransportError("y")])
        with pytest.raises(TransportError):
            _call_with_retries(call, label="t", retries=0)
        assert call.calls == 2
        assert sleeps == [2.0]

    def test_default_never_retries_rate_limit(self, sleeps: list[float]):
        call = _FlakyCall([RateLimit("x")])
        with pytest.raises(RateLimit):
            _call_with_retries(call, label="t", retries=0)
        assert call.calls == 1
        assert sleeps == []

    def test_extra_retries_backoff_sequence(self, sleeps: list[float]):
        call = _FlakyCall([TransportError("a"), TransportError("b"), TransportError("c")])
        assert _call_with_retries(call, label="t", retries=2) == "ok"
        assert call.calls == 4
        assert sleeps == [2.0, 4.0, 8.0]

    def test_rate_limit_sleeps_retry_after(self, sleeps: list[float]):
        err = RateLimit("x")
        err.retry_after_seconds = 5.0
        call = _FlakyCall([err])
        assert _call_with_retries(call, label="t", retries=1) == "ok"
        assert sleeps == [5.0]

    def test_rate_limit_defaults_to_60s_without_header(self, sleeps: list[float]):
        call = _FlakyCall([RateLimit("x")])
        assert _call_with_retries(call, label="t", retries=1) == "ok"
        assert sleeps == [60.0]

    def test_rate_limit_sleep_clamped_with_warn(
        self, sleeps: list[float], capsys: pytest.CaptureFixture
    ):
        err = RateLimit("x")
        err.retry_after_seconds = 86400.0
        call = _FlakyCall([err])
        assert _call_with_retries(call, label="t", retries=1) == "ok"
        assert sleeps == [review.MAX_RETRY_SLEEP]
        assert "WARN:" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "exc",
        [ConfigError("x"), SafetyRefusal("x"), ContextOverflow("x")],
    )
    def test_never_retried_categories(self, sleeps: list[float], exc: ReviewError):
        call = _FlakyCall([exc])
        with pytest.raises(type(exc)):
            _call_with_retries(call, label="t", retries=5)
        assert call.calls == 1
        assert sleeps == []


# ---------------------------------------------------------------------------
# Injection guard, severity filter, usage line, output file
# ---------------------------------------------------------------------------


class TestInjectionGuard:
    def test_default_context_carries_guard(self):
        out = _apply_context("PROMPT", review.DEFAULT_CONTEXT)
        assert INJECTION_GUARD in out
        assert out.index(INJECTION_GUARD) < out.index("</CONTEXT_FOR_REVIEWER>")

    def test_custom_context_carries_guard_too(self):
        out = _apply_context("PROMPT", "my project context")
        assert "my project context" in out
        assert INJECTION_GUARD in out

    def test_no_context_disables_guard(self):
        assert _apply_context("PROMPT", None) == "PROMPT"


class TestMinSeverityInstruction:
    def test_low_is_empty(self):
        assert _min_severity_instruction("LOW") == ""

    def test_high_lists_kept_levels(self):
        text = _min_severity_instruction("HIGH")
        assert "<SEVERITY_FILTER>" in text
        assert "HIGH or higher" in text
        assert "HIGH, CRITICAL" in text

    def test_critical_keeps_only_critical(self):
        text = _min_severity_instruction("CRITICAL")
        assert "CRITICAL or higher (CRITICAL)" in text


class TestFormatUsageLine:
    def test_both_counts(self):
        line = _format_usage_line(
            CallResult("x", prompt_tokens=48210, completion_tokens=1533),
            "openrouter",
            "google/gemini-2.5-pro",
        )
        assert line == (
            "[usage] prompt=48,210 completion=1,533 total=49,743 tokens "
            "(openrouter/google/gemini-2.5-pro)"
        )

    def test_no_usage_returns_none(self):
        assert _format_usage_line(CallResult("x"), "p", "m") is None

    def test_partial_usage_skips_total(self):
        line = _format_usage_line(
            CallResult("x", prompt_tokens=100), "p", "m"
        )
        assert line is not None
        assert "completion=?" in line
        assert "total" not in line


class TestUsageInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(42, 42), (0, 0), (True, None), ("42", None), (None, None), (4.2, None)],
    )
    def test_coercion(self, value: object, expected: int | None):
        assert _usage_int(value) == expected


class TestWriteOutputFile:
    def test_writes_utf8_lf(self, tmp_path):
        target = tmp_path / "review.md"
        _write_output_file("café\nline2\n", str(target))
        raw = target.read_bytes()
        assert raw == "café\nline2\n".encode("utf-8")
        assert b"\r" not in raw

    def test_unwritable_path_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError):
            _write_output_file("x", str(tmp_path))  # a directory, not a file


# ---------------------------------------------------------------------------
# Dry-run report
# ---------------------------------------------------------------------------


def _settings(**overrides) -> Settings:
    base = dict(
        provider="openrouter",
        model="google/gemini-2.5-pro",
        temperature=0.3,
        max_tokens=16000,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    base.update(overrides)
    return Settings(**base)


class TestDryRunReport:
    def test_diff_mode_fields(self):
        request = ReviewRequest(
            system_prompt="S" * 100,
            user_prompt="U" * 300,
            mode="diff",
            payload_chars=250,
        )
        report = _dry_run_report(_settings(), request)
        assert "DRY RUN" in report
        assert "provider:          openrouter" in report
        assert "mode:              diff" in report
        assert "est_prompt_tokens: ~100" in report
        assert "files:" not in report

    def test_codebase_mode_lists_files(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n", encoding="utf-8")
        request = ReviewRequest(
            system_prompt="S",
            user_prompt="U",
            mode="codebase",
            payload_chars=6,
            files=[f],
        )
        report = _dry_run_report(_settings(), request)
        assert "files:             1" in report
        assert "a.py" in report

    def test_ollama_window_line(self):
        request = ReviewRequest("S", "U", "diff", 1)
        report = _dry_run_report(
            _settings(provider="ollama"), request,
            ollama_window="4,096 tokens (advisory-default)",
        )
        assert "ollama_window:     4,096 tokens (advisory-default)" in report
