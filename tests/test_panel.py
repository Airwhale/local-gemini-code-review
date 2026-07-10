"""Tests for the multi-model panel (--models): clustering, exit
semantics, worker policy, envelope/markdown rendering, and settings
resolution rules."""

from __future__ import annotations

import argparse

import pytest

import review
from review import (
    CallResult,
    ConfigError,
    ContextOverflow,
    Finding,
    ParsedReview,
    RateLimit,
    ReviewError,
    TransportError,
    _panel_exit_error,
    _panel_max_workers,
    _resolve_settings,
    build_panel_envelope,
    merge_panel_findings,
    panel_findings_match,
    render_panel_markdown,
)


def _finding(
    title: str,
    file: str = "a.py",
    line: int | None = 10,
    severity: str = "HIGH",
) -> Finding:
    return Finding(
        file=file, line=line, severity=severity,
        title=title, body=f"body of {title}", suggestion=None,
    )


def _parsed(findings: list[Finding], summary: str = "s") -> ParsedReview:
    return ParsedReview(
        summary=summary, findings=findings, clean=not findings,
        parse_ok=True, problems=[],
    )


class TestPanelFindingsMatch:
    def test_same_title_same_place_matches(self):
        assert panel_findings_match(_finding("t"), _finding("t", line=15))

    def test_reworded_title_same_location_same_severity_matches(self):
        assert panel_findings_match(
            _finding("retry loop lacks sleep"),
            _finding("missing backoff between retries", line=12),
        )

    def test_reworded_title_different_severity_no_match(self):
        # Cross-model consensus is a high-precision signal: severity
        # disagreement on the same hunk stays two findings.
        assert not panel_findings_match(
            _finding("retry loop lacks sleep"),
            _finding("missing backoff between retries", severity="LOW"),
        )

    def test_different_file_no_match(self):
        assert not panel_findings_match(
            _finding("t"), _finding("t2", file="b.py")
        )


class TestMergePanelFindings:
    def test_consensus_cluster(self):
        merged = merge_panel_findings({
            "model-a": _parsed([_finding("issue x")]),
            "model-b": _parsed([_finding("issue x reworded", line=14)]),
        })
        assert len(merged) == 1
        assert merged[0].found_by == ["model-a", "model-b"]
        # Representative comes from the first model in CLI order.
        assert merged[0].finding.title == "issue x"

    def test_disjoint_findings_stay_separate(self):
        merged = merge_panel_findings({
            "model-a": _parsed([_finding("issue x")]),
            "model-b": _parsed([_finding("issue y", file="b.py")]),
        })
        assert len(merged) == 2

    def test_same_model_cannot_vouch_twice(self):
        # Two similar findings from ONE model must not merge into a
        # fake-consensus cluster.
        merged = merge_panel_findings({
            "model-a": _parsed([_finding("issue x"), _finding("issue x again", line=12)]),
        })
        assert all(c.found_by == ["model-a"] for c in merged)
        assert len(merged) == 2

    def test_ordering_consensus_then_severity(self):
        merged = merge_panel_findings({
            "a": _parsed([
                _finding("lonely critical", file="z.py", severity="CRITICAL"),
                _finding("shared medium", file="m.py", severity="MEDIUM"),
            ]),
            "b": _parsed([_finding("shared medium reworded", file="m.py", line=12, severity="MEDIUM")]),
        })
        # Consensus (2 models) outranks severity (CRITICAL, 1 model).
        assert merged[0].found_by == ["a", "b"]
        assert merged[1].finding.severity == "CRITICAL"


class TestPanelWorkers:
    def test_ollama_sequential(self):
        assert _panel_max_workers("ollama", 3) == 1

    def test_cloud_capped_at_four(self):
        assert _panel_max_workers("openrouter", 8) == 4
        assert _panel_max_workers("gemini", 2) == 2


class TestPanelExitError:
    def test_precedence_pinned(self):
        # This ordering is documented API (README error model).
        assert review._CATEGORY_PRECEDENCE == (
            "CONFIG",
            "SAFETY_REFUSAL",
            "CONTEXT_OVERFLOW",
            "RATE_LIMIT",
            "PROVIDER_HICCUP",
            "TRANSPORT",
            "UNKNOWN",
        )

    def test_non_retryable_beats_retryable(self):
        failures = [
            ("m1", TransportError("net down")),
            ("m2", ContextOverflow("too big")),
        ]
        assert isinstance(_panel_exit_error(failures), ContextOverflow)

    def test_cli_order_breaks_ties(self):
        first, second = RateLimit("first"), RateLimit("second")
        assert _panel_exit_error([("m1", first), ("m2", second)]) is first

    def test_unknown_category_sorts_last(self):
        generic = ReviewError("weird")
        rate = RateLimit("429")
        assert _panel_exit_error([("m1", generic), ("m2", rate)]) is rate


class TestPanelRendering:
    def _fixture(self):
        parsed_by_model = {
            "model-a": _parsed([_finding("issue x")]),
            "model-b": _parsed([], summary=None),
        }
        parsed_by_model["model-b"].clean = True
        results = {
            "model-a": CallResult("raw a", prompt_tokens=100, completion_tokens=10),
            "model-b": CallResult("raw b", prompt_tokens=200, completion_tokens=20),
        }
        raw = {"model-a": "raw a", "model-b": "raw b"}
        failures = [("model-c", RateLimit("429"))]
        merged = merge_panel_findings(parsed_by_model)
        return parsed_by_model, results, raw, failures, merged

    def test_markdown_shape(self):
        parsed, _results, raw, failures, merged = self._fixture()
        text = render_panel_markdown(merged, parsed, raw, failures, n_models=3)
        assert text.startswith("# Panel review (2/3 models)")
        assert "Found by: model-a" in text
        assert "`model-b`: clean -- no issues found" in text
        assert "`model-c`: FAILED -- RATE_LIMIT" in text
        assert "## Appendix: model-a" in text
        assert "raw b" in text

    def test_envelope_shape(self):
        parsed, results, raw, failures, merged = self._fixture()
        envelope = build_panel_envelope(
            mode="diff",
            provider="openrouter",
            temperature=0.3,
            models=("model-a", "model-b", "model-c"),
            merged=merged,
            parsed_by_model=parsed,
            results_by_model=results,
            raw_by_model=raw,
            failures=failures,
        )
        assert envelope["model"] is None
        assert envelope["models"] == ["model-a", "model-b", "model-c"]
        assert envelope["usage"] == {"prompt_tokens": 300, "completion_tokens": 30}
        assert [f["found_by"] for f in envelope["findings"]] == [["model-a"]]
        per_model = {e["model"]: e for e in envelope["per_model"]}
        assert per_model["model-c"]["error"]["category"] == "RATE_LIMIT"
        assert per_model["model-c"]["error"]["exit_code"] == 11
        assert per_model["model-a"]["error"] is None
        assert envelope["parse_ok"] is True

    def test_parse_failure_embeds_raw_in_per_model(self):
        bad = ParsedReview(
            summary=None, findings=[], clean=False, parse_ok=False, problems=["x"],
        )
        envelope = build_panel_envelope(
            mode="diff",
            provider="openrouter",
            temperature=0.3,
            models=("m",),
            merged=[],
            parsed_by_model={"m": bad},
            results_by_model={"m": CallResult("garbage output")},
            raw_by_model={"m": "garbage output"},
            failures=[],
        )
        assert envelope["parse_ok"] is False
        assert envelope["per_model"][0]["raw"] == "garbage output"


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        base=None, pr=None, staged=False, codebase=False,
        include=[], exclude=[],
        provider="openrouter", model=None, models=None,
        ollama_host=None, temperature=None, max_tokens=None,
        retries=None, min_severity="LOW", context=None, no_context=False,
        output=None, format=None, baseline=None, dry_run=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestPanelSettings:
    @pytest.fixture(autouse=True)
    def _api_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    def test_models_resolve_aliases_preflight(self):
        settings = _resolve_settings(_args(models="pro, claude"))
        assert settings.models == (
            "google/gemini-2.5-pro",
            "anthropic/claude-sonnet-4.5",
        )
        assert settings.model == "google/gemini-2.5-pro"

    def test_models_exclusive_with_model(self):
        with pytest.raises(ConfigError):
            _resolve_settings(_args(models="pro,claude", model="flash"))

    def test_models_needs_two(self):
        with pytest.raises(ConfigError):
            _resolve_settings(_args(models="pro"))

    def test_models_rejects_duplicates_after_aliasing(self):
        # `pro` and the raw slug resolve to the same model.
        with pytest.raises(ConfigError):
            _resolve_settings(_args(models="pro,google/gemini-2.5-pro"))

    def test_models_rejects_wrong_provider_alias(self):
        with pytest.raises(ConfigError) as exc_info:
            _resolve_settings(_args(models="pro,local"))
        assert "--provider ollama" in str(exc_info.value)

    def test_models_exclusive_with_baseline(self):
        with pytest.raises(ConfigError):
            _resolve_settings(_args(models="pro,claude", baseline="r.json"))
