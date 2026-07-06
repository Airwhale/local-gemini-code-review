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

import review
from review import (
    BUILTIN_CODEBASE_EXCLUDES,
    ConfigError,
    ContextOverflow,
    RateLimit,
    ReviewError,
    TransportError,
    _classify_http_error,
    _format_size,
    _glob_match,
    _normalize_ollama_host,
    _number_lines,
    _ollama_prompt_guard,
    _resolve_model,
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
        assert "OLLAMA_CONTEXT_LENGTH" in str(exc_info.value)

    def test_boundary_just_under_window_passes(self):
        # 4095 tokens' worth of chars against a 4096 window.
        _ollama_prompt_guard(4095 * review.OLLAMA_CHARS_PER_TOKEN, 4096, model="m")


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
