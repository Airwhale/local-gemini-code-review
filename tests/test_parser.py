"""Tests for the markdown findings parser (--format json / --baseline).

Two kinds of inputs: hand-written canned strings exercising specific
grammar tolerances, and three frozen real-world outputs under
tests/fixtures/ from the 3-model dogfood run on PR #2's diff (claude
clean-form; gemini-pro ``### L21: [HIGH]`` shape; deepseek
``### L+117:`` / ``### L+0:`` shape with ``Suggested change:`` +
```` ```diff ```` fences). Real outputs are the regression corpus for a
parser whose whole job is tolerating real-model drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from code_review.cli import (
    CallResult,
    ConfigError,
    Finding,
    ParsedReview,
    build_json_envelope,
    diff_against_baseline,
    enforce_min_severity,
    filter_baseline_findings,
    finding_fingerprint,
    findings_match,
    load_baseline,
    normalize_title,
    parse_review_markdown,
    parse_review_markdown_safe,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


WELL_FORMED = """\
# Change summary: Adds a retry policy to the HTTP client.

## File: src/client.py
### L42: [HIGH] Retry loop never sleeps between attempts.

The loop calls the endpoint back-to-back, hammering a rate-limited API.

Suggested change:
```
    for attempt in range(retries):
        time.sleep(2 ** attempt)
        response = call()
```

### L108: [MEDIUM] Timeout constant duplicated from config.py.
Also seen at src/config.py L12.

## File: src/config.py
### L12: [LOW] Magic number lacks a unit suffix in its name.
Rename TIMEOUT to TIMEOUT_SECONDS.
"""


class TestWellFormed:
    def test_summary_and_counts(self):
        parsed = parse_review_markdown(WELL_FORMED)
        assert parsed.parse_ok
        assert not parsed.clean
        assert parsed.summary == "Adds a retry policy to the HTTP client."
        assert len(parsed.findings) == 3

    def test_first_finding_fields(self):
        finding = parse_review_markdown(WELL_FORMED).findings[0]
        assert finding.file == "src/client.py"
        assert finding.line == 42
        assert finding.severity == "HIGH"
        assert finding.title == "Retry loop never sleeps between attempts."
        assert "hammering a rate-limited API" in finding.body
        assert finding.suggestion is not None
        assert "time.sleep(2 ** attempt)" in finding.suggestion
        assert "Suggested change:" not in finding.body

    def test_file_boundaries(self):
        parsed = parse_review_markdown(WELL_FORMED)
        assert [f.file for f in parsed.findings] == [
            "src/client.py",
            "src/client.py",
            "src/config.py",
        ]

    def test_finding_without_suggestion(self):
        finding = parse_review_markdown(WELL_FORMED).findings[1]
        assert finding.suggestion is None
        assert "Also seen at" in finding.body


class TestCleanOutputs:
    @pytest.mark.parametrize(
        "text",
        [
            "# Change summary: A tidy refactor.\n\nNo issues found. "
            "Code looks clean and ready to merge.\n",
            "# Codebase review summary: Small, well-kept project.\n"
            "No issues found. Code looks clean.\n",
        ],
    )
    def test_clean_variants(self, text: str):
        parsed = parse_review_markdown(text)
        assert parsed.clean
        assert parsed.parse_ok
        assert parsed.findings == []


class TestHeadingTolerance:
    def test_plus_prefixed_line_number(self):
        # Real deepseek shape: diff marker transcribed into the heading.
        text = "## File: a.py\n### L+117: [MEDIUM] Something drifted.\nBody.\n"
        finding = parse_review_markdown(text).findings[0]
        assert finding.line == 117
        assert finding.severity == "MEDIUM"

    def test_plus_zero_maps_to_none(self):
        text = "## File: a.py\n### L+0: [LOW] File-level nit.\nBody.\n"
        parsed = parse_review_markdown(text)
        assert parsed.findings[0].line is None
        assert any("line 0" in p for p in parsed.problems)
        assert parsed.parse_ok  # severity present -> still readable

    def test_range_takes_first_number(self):
        text = "## File: a.py\n### L10-L20: [HIGH] Spans a block.\nBody.\n"
        assert parse_review_markdown(text).findings[0].line == 10

    @pytest.mark.parametrize("token", ["[high]", "HIGH", "[ HIGH ]"])
    def test_severity_variants(self, token: str):
        text = f"## File: a.py\n### L5: {token} Title here.\nBody.\n"
        assert parse_review_markdown(text).findings[0].severity == "HIGH"

    def test_missing_severity_is_unknown(self):
        text = "## File: a.py\n### L5: Title without a tag.\nBody.\n"
        parsed = parse_review_markdown(text)
        assert parsed.findings[0].severity == "UNKNOWN"
        assert parsed.parse_ok  # line present -> still readable

    def test_severity_words_inside_prose_do_not_match(self):
        # codex finding: "Below" contains "low" and used to parse as
        # severity LOW with a corrupted title.
        text = "## File: a.py\n### L5: Below threshold is ignored.\nBody.\n"
        finding = parse_review_markdown(text).findings[0]
        assert finding.severity == "UNKNOWN"
        assert finding.title == "Below threshold is ignored."

    @pytest.mark.parametrize(
        "title",
        [
            "Higher-order function misuse.",
            "Use lowercase constants throughout.",
            "The following code is problematic.",  # contains 'low' twice
        ],
    )
    def test_embedded_severity_words_leave_title_intact(self, title: str):
        text = f"## File: a.py\n### L5: [MEDIUM] {title}\nBody.\n"
        finding = parse_review_markdown(text).findings[0]
        assert finding.severity == "MEDIUM"
        assert finding.title == title

    def test_orphan_finding_before_any_file(self):
        text = "# Change summary: x.\n### L5: [LOW] Orphaned.\nBody.\n"
        parsed = parse_review_markdown(text)
        assert parsed.findings[0].file is None
        assert any("before any" in p for p in parsed.problems)

    def test_windows_path_normalized(self):
        text = "## File: src\\sub\\a.py\n### L5: [LOW] T.\nB.\n"
        assert parse_review_markdown(text).findings[0].file == "src/sub/a.py"


class TestFences:
    def test_heading_inside_fence_not_split(self):
        text = (
            "# Change summary: x.\n\n"
            "## File: a.py\n"
            "### L5: [HIGH] Fence quotes markdown.\n"
            "Body before.\n"
            "```\n"
            "### L1: [HIGH] this is fenced content, not a finding\n"
            "## File: not-a-real-file\n"
            "```\n"
            "Body after.\n"
        )
        parsed = parse_review_markdown(text)
        assert len(parsed.findings) == 1
        assert "fenced content" in parsed.findings[0].body
        assert parsed.findings[0].suggestion is None  # untagged mid fence

    def test_bold_suggested_change_leadin(self):
        text = (
            "## File: a.py\n"
            "### L5: [HIGH] T.\n"
            "Body.\n\n"
            "**Suggested change:**\n"
            "```python\n"
            "x = 2\n"
            "```\n"
        )
        finding = parse_review_markdown(text).findings[0]
        assert finding.suggestion == "x = 2"

    def test_trailing_diff_fence_without_leadin_is_suggestion(self):
        text = "## File: a.py\n### L5: [HIGH] T.\nBody.\n```diff\n-old\n+new\n```\n"
        finding = parse_review_markdown(text).findings[0]
        assert finding.suggestion == "-old\n+new"

    def test_mid_body_diff_fence_stays_in_body(self):
        text = (
            "## File: a.py\n"
            "### L5: [HIGH] T.\n"
            "The hunk in question:\n"
            "```diff\n"
            "-old\n"
            "```\n"
            "And that is why it breaks.\n"
        )
        finding = parse_review_markdown(text).findings[0]
        assert finding.suggestion is None
        assert "-old" in finding.body


class TestParseFailureModes:
    def test_garbage_falls_back(self):
        parsed = parse_review_markdown("complete nonsense, no structure at all")
        assert not parsed.parse_ok
        assert parsed.findings == []

    def test_missing_summary_is_tolerated(self):
        text = "## File: a.py\n### L5: [LOW] T.\nB.\n"
        parsed = parse_review_markdown(text)
        assert parsed.parse_ok
        assert parsed.summary is None
        assert any("no summary" in p for p in parsed.problems)

    def test_unreadable_heading_forces_parse_not_ok(self):
        text = "# Change summary: x.\n## File: a.py\n### just some words\nB.\n"
        parsed = parse_review_markdown(text)
        assert not parsed.parse_ok

    def test_safe_wrapper_never_raises(self, monkeypatch: pytest.MonkeyPatch):
        from code_review import cli as review_module

        def _boom(_text: str):
            raise RuntimeError("parser bug")

        monkeypatch.setattr(review_module, "parse_review_markdown", _boom)
        parsed = parse_review_markdown_safe("anything")
        assert not parsed.parse_ok
        assert any("parser crashed" in p for p in parsed.problems)


class TestDogfoodFixtures:
    def test_claude_clean(self):
        parsed = parse_review_markdown(_fixture("dogfood-claude.md"))
        assert parsed.clean
        assert parsed.parse_ok
        assert parsed.summary is not None

    def test_gemini_pro_findings(self):
        parsed = parse_review_markdown(_fixture("dogfood-pro.md"))
        assert parsed.parse_ok
        assert len(parsed.findings) == 3
        severities = [f.severity for f in parsed.findings]
        assert severities == ["HIGH", "MEDIUM", "LOW"]
        assert parsed.findings[0].file == ".github/workflows/ci.yml"
        assert parsed.findings[0].line == 21
        assert all(f.suggestion for f in parsed.findings)

    def test_deepseek_plus_headings(self):
        parsed = parse_review_markdown(_fixture("dogfood-deepseek.md"))
        assert parsed.parse_ok
        # 28 findings across 8 files -- frozen count for this fixture.
        # (Fun fact: the parser corrected the human tally; the session
        # that triaged this output by eye counted 19.)
        assert len(parsed.findings) == 28
        # The L+117 heading parses to a real line, L+0 to None.
        lines = [f.line for f in parsed.findings]
        assert 117 in lines
        assert None in lines
        assert any(f.suggestion for f in parsed.findings)


class TestFingerprints:
    def _finding(self, **overrides) -> Finding:
        base: dict[str, Any] = dict(
            file="a.py",
            line=42,
            severity="HIGH",
            title="Retry loop never sleeps.",
            body="",
            suggestion=None,
        )
        base.update(overrides)
        return Finding(**base)

    def test_normalize_title(self):
        assert normalize_title("  Retry `loop`  never sleeps. ") == (
            "retry loop never sleeps"
        )

    def test_fingerprint_stable_across_line_drift(self):
        a, b = self._finding(line=42), self._finding(line=48)
        assert finding_fingerprint(a) == finding_fingerprint(b)
        assert findings_match(a, b)

    def test_line_drift_beyond_window_no_match(self):
        assert not findings_match(self._finding(line=42), self._finding(line=60))

    def test_none_line_matches_on_fingerprint(self):
        assert findings_match(self._finding(line=None), self._finding(line=42))

    def test_different_severity_changes_fingerprint(self):
        assert finding_fingerprint(self._finding()) != finding_fingerprint(
            self._finding(severity="LOW")
        )


class TestSummaryOnlyOutput:
    def test_summary_only_is_parse_failure(self):
        # The template mandates findings or the literal clean phrase; a
        # bare summary means the model drifted (bullets, truncation).
        # parse_ok=True here would let --baseline resolve everything and
        # agents treat the run as clean.
        text = (
            "# Change summary: Adds a retry policy.\n\n"
            "Some prose about the change, but no finding headings and no "
            "clean marker.\n"
        )
        parsed = parse_review_markdown(text)
        assert parsed.findings == []
        assert parsed.clean is False
        assert parsed.parse_ok is False
        assert any("no finding headings" in p for p in parsed.problems)


class TestMinSeverityEnforcement:
    """The <SEVERITY_FILTER> prompt block is a request; JSON envelopes
    and panel reports enforce the floor post-parse via these helpers."""

    def _finding(self, severity: str) -> Finding:
        return Finding(
            file="a.py",
            line=10,
            severity=severity,
            title=f"{severity} finding",
            body="",
            suggestion=None,
        )

    def _parsed(self, severities: list[str]) -> ParsedReview:
        return ParsedReview(
            summary="s",
            findings=[self._finding(s) for s in severities],
            clean=False,
            parse_ok=True,
            problems=[],
        )

    def test_drops_below_floor(self):
        parsed = self._parsed(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        filtered = enforce_min_severity(parsed, "HIGH")
        assert [f.severity for f in filtered.findings] == ["HIGH", "CRITICAL"]

    def test_unknown_severity_always_kept(self):
        # Hiding a finding the parser couldn't rate would be worse than
        # showing it.
        parsed = self._parsed(["LOW", "UNKNOWN"])
        filtered = enforce_min_severity(parsed, "CRITICAL")
        assert [f.severity for f in filtered.findings] == ["UNKNOWN"]

    def test_low_floor_is_noop_same_object(self):
        parsed = self._parsed(["LOW"])
        assert enforce_min_severity(parsed, "LOW") is parsed

    def test_nothing_dropped_returns_same_object(self):
        parsed = self._parsed(["HIGH", "CRITICAL"])
        assert enforce_min_severity(parsed, "HIGH") is parsed

    def test_clean_and_parse_ok_untouched(self):
        # `clean` reports what the model said; an all-filtered review is
        # "not clean, but nothing at your floor", not "clean".
        parsed = self._parsed(["LOW"])
        filtered = enforce_min_severity(parsed, "HIGH")
        assert filtered.findings == []
        assert filtered.clean is False
        assert filtered.parse_ok is True

    def test_baseline_entries_filtered_too(self):
        # Below-floor baseline entries must not surface as "resolved"
        # when they were merely filtered, not fixed.
        doc = {
            "findings": [
                {"severity": "LOW", "fingerprint": "abc", "file": "a.py"},
                {"severity": "HIGH", "fingerprint": "def", "file": "a.py"},
            ]
        }
        filtered = filter_baseline_findings(doc, "HIGH")
        assert [e["severity"] for e in filtered["findings"]] == ["HIGH"]
        # Original document is not mutated.
        assert len(doc["findings"]) == 2

    def test_baseline_malformed_entries_kept(self):
        doc = {"findings": [{"fingerprint": "abc"}, "not-a-dict"]}
        filtered = filter_baseline_findings(doc, "HIGH")
        assert filtered["findings"] == [{"fingerprint": "abc"}, "not-a-dict"]


class TestBaseline:
    def _doc(self, findings: list[Finding]) -> dict:
        return build_json_envelope(
            mode="diff",
            provider="openrouter",
            model="m",
            temperature=0.3,
            parsed=parse_review_markdown_safe(WELL_FORMED).__class__(
                summary="s",
                findings=findings,
                clean=False,
                parse_ok=True,
                problems=[],
            ),
            result=CallResult("raw"),
            raw_markdown="raw",
        )

    def _finding(self, title: str, line: int | None = 10) -> Finding:
        return Finding(
            file="a.py",
            line=line,
            severity="HIGH",
            title=title,
            body="",
            suggestion=None,
        )

    def test_new_persisting_resolved(self):
        old = [self._finding("issue one"), self._finding("issue two", line=50)]
        doc = self._doc(old)
        current = [self._finding("issue one", line=14), self._finding("issue three")]
        statuses, resolved = diff_against_baseline(current, doc)
        assert statuses == ["persisting", "new"]
        assert len(resolved) == 1
        assert "issue two" in resolved[0]["title"]

    def test_reworded_title_still_persists(self):
        # Empirical case from back-to-back flash runs: same file, same
        # line, same issue, completely reworded title. Pass 2 (location)
        # must carry the match that the fingerprint misses.
        doc = self._doc([self._finding("Add a comment explaining the default value")])
        current = [self._finding("Add a comment to clarify its purpose", line=12)]
        statuses, resolved = diff_against_baseline(current, doc)
        assert statuses == ["persisting"]
        assert resolved == []

    def test_severity_rerate_still_persists(self):
        old = self._finding("The comment is misleading")
        doc = self._doc([old])
        rerated = Finding(
            file=old.file,
            line=old.line,
            severity="MEDIUM",
            title="The comment should be clarified",
            body="",
            suggestion=None,
        )
        statuses, _resolved = diff_against_baseline([rerated], doc)
        assert statuses == ["persisting"]

    def test_different_file_never_persists(self):
        doc = self._doc([self._finding("issue")])
        moved = Finding(
            file="other.py",
            line=10,
            severity="HIGH",
            title="issue",
            body="",
            suggestion=None,
        )
        # Identical title but different file: strong pass fails on the
        # fingerprint (file is hashed), relaxed pass fails on location.
        statuses, resolved = diff_against_baseline([moved], doc)
        assert statuses == ["new"]
        assert len(resolved) == 1

    def test_lineless_baseline_entry_does_not_wildcard_match(self):
        # A line-less baseline entry must not vouch for an unrelated
        # finding in the same file via the relaxed pass -- with title
        # and severity already ignored there, a missing line matching
        # anything would collapse the whole file into one bucket.
        doc = self._doc([self._finding("Old general file-level concern", line=None)])
        current = [self._finding("Completely different specific issue", line=10)]
        statuses, resolved = diff_against_baseline(current, doc)
        assert statuses == ["new"]
        assert len(resolved) == 1

    def test_non_list_findings_in_baseline_is_config_error(self, tmp_path):
        # A hand-mangled findings value would TypeError deep inside
        # diff_against_baseline -> UNKNOWN; must be typed CONFIG before
        # any tokens are spent.
        import json as _json

        bad = tmp_path / "baseline.json"
        bad.write_text(
            _json.dumps({"schema_version": 1, "findings": 5}), encoding="utf-8"
        )
        with pytest.raises(ConfigError) as exc_info:
            load_baseline(str(bad))
        assert "non-list" in str(exc_info.value)

    def test_lineless_finding_still_persists_via_fingerprint(self):
        # Pass 1 (fingerprint) remains the path for line-less findings:
        # same file+severity+title matches regardless of missing lines.
        doc = self._doc([self._finding("The helper lacks a docstring", line=None)])
        current = [self._finding("The helper lacks a docstring", line=None)]
        statuses, resolved = diff_against_baseline(current, doc)
        assert statuses == ["persisting"]
        assert resolved == []

    def test_one_baseline_entry_vouches_once(self):
        doc = self._doc([self._finding("dup issue")])
        current = [self._finding("dup issue"), self._finding("dup issue")]
        statuses, resolved = diff_against_baseline(current, doc)
        assert statuses == ["persisting", "new"]
        assert resolved == []

    def test_load_baseline_rejects_garbage(self, tmp_path):
        bad = tmp_path / "b.json"
        bad.write_text("not json", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_baseline(str(bad))

    def test_load_baseline_rejects_wrong_schema(self, tmp_path):
        bad = tmp_path / "b.json"
        bad.write_text('{"schema_version": 99}', encoding="utf-8")
        with pytest.raises(ConfigError):
            load_baseline(str(bad))

    def test_load_baseline_missing_file(self, tmp_path):
        with pytest.raises(ConfigError):
            load_baseline(str(tmp_path / "nope.json"))


class TestEnvelope:
    def test_shape_and_parse_ok_true_omits_raw(self):
        parsed = parse_review_markdown(WELL_FORMED)
        envelope = build_json_envelope(
            mode="diff",
            provider="openrouter",
            model="m",
            temperature=0.3,
            parsed=parsed,
            result=CallResult("x", prompt_tokens=10, completion_tokens=5),
            raw_markdown=WELL_FORMED,
        )
        assert envelope["schema_version"] == 1
        assert envelope["parse_ok"] is True
        assert "raw" not in envelope
        assert envelope["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
        assert len(envelope["findings"]) == 3
        assert all("fingerprint" in f for f in envelope["findings"])
        assert all("status" not in f for f in envelope["findings"])

    def test_parse_failure_embeds_raw(self):
        parsed = parse_review_markdown_safe("garbage")
        envelope = build_json_envelope(
            mode="diff",
            provider="p",
            model="m",
            temperature=0.3,
            parsed=parsed,
            result=CallResult("garbage"),
            raw_markdown="garbage",
        )
        assert envelope["parse_ok"] is False
        assert envelope["raw"] == "garbage"
        assert envelope["usage"] is None

    def test_statuses_attach_to_findings(self):
        parsed = parse_review_markdown(WELL_FORMED)
        envelope = build_json_envelope(
            mode="diff",
            provider="p",
            model="m",
            temperature=0.3,
            parsed=parsed,
            result=CallResult("x"),
            raw_markdown=WELL_FORMED,
            statuses=["new", "persisting", "new"],
            resolved=[],
        )
        assert [f["status"] for f in envelope["findings"]] == [
            "new",
            "persisting",
            "new",
        ]
        assert envelope["resolved"] == []
