"""Unit tests for the evidence-discipline rules and base-drift warning.

These cover the fork's answer to the tool's dominant false-finding source:
in diff mode the model sees hunks and nothing else, so it invents the rest
of the file ("X is undefined", "add the missing docstring", a suggestion
byte-identical to the code it 'fixes'). Two mechanisms fight that -- the
prompt rule here, and auto-attached full files in cli -- and both have a
sharp edge worth pinning:

* the full-files rule must NOT claim the file is hidden (it isn't, and that
  framing would suppress correct whole-file findings), and
* the rule must survive --no-context, which exists to disable the *safety*
  wrapper, not the review instructions.

No network, no git, no Ollama server required.
"""

from __future__ import annotations

import code_review.sources as sources
from code_review.parser import Finding, annotate_in_hunk, hunk_ranges
from code_review.prompts import (
    DIFF_EVIDENCE_RULE,
    FULL_FILES_EVIDENCE_RULE,
    build_diff_prompts,
)

_DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-a = 1\n+a = 2\n"

# Two hunks in one file: post-image lines 10-13 and 101-102.
_TWO_HUNK_DIFF = (
    "diff --git a/src/x.py b/src/x.py\n"
    "--- a/src/x.py\n"
    "+++ b/src/x.py\n"
    "@@ -10,3 +10,4 @@\n a\n+b\n c\n d\n"
    "@@ -100,2 +101,2 @@\n-e\n+f\n"
)


def _f(path: str | None, line: int | None) -> Finding:
    return Finding(path, line, "HIGH", "t", "b", None)


class TestHunkRanges:
    def test_maps_post_image_ranges_per_file(self) -> None:
        assert hunk_ranges(_TWO_HUNK_DIFF) == {"src/x.py": [(10, 13), (101, 102)]}

    def test_single_line_hunk_without_count(self) -> None:
        # `@@ -1 +1 @@` (no comma) means a one-line hunk.
        diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        assert hunk_ranges(diff) == {"a.py": [(1, 1)]}

    def test_deleted_file_has_no_post_image(self) -> None:
        diff = (
            "diff --git a/gone.py b/gone.py\n--- a/gone.py\n"
            "+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n"
        )
        assert hunk_ranges(diff) == {}

    def test_pure_deletion_hunk_contributes_no_range(self) -> None:
        # `+n,0` -- nothing on the RIGHT side to anchor a comment to.
        diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -5,2 +4,0 @@\n-x\n-y\n"
        assert hunk_ranges(diff) == {"a.py": []}


class TestAnnotateInHunk:
    def test_inside_and_outside_hunks(self) -> None:
        inside, outside = _f("src/x.py", 11), _f("src/x.py", 500)
        annotate_in_hunk([inside, outside], hunk_ranges(_TWO_HUNK_DIFF))
        assert inside.in_hunk is True
        assert outside.in_hunk is False

    def test_boundaries_are_inclusive(self) -> None:
        first, last = _f("src/x.py", 10), _f("src/x.py", 13)
        annotate_in_hunk([first, last], hunk_ranges(_TWO_HUNK_DIFF))
        assert first.in_hunk is True
        assert last.in_hunk is True

    def test_suffix_path_resolves(self) -> None:
        # Models cite paths inconsistently; a bare name still resolves when
        # exactly one diff path ends with it.
        f = _f("x.py", 101)
        annotate_in_hunk([f], hunk_ranges(_TWO_HUNK_DIFF))
        assert f.in_hunk is True

    def test_ambiguous_suffix_stays_unknown(self) -> None:
        # Two files share a basename -- guessing would be worse than None.
        diff = _TWO_HUNK_DIFF + (
            "diff --git a/other/x.py b/other/x.py\n"
            "--- a/other/x.py\n+++ b/other/x.py\n@@ -1,1 +1,1 @@\n-q\n+r\n"
        )
        f = _f("x.py", 11)
        annotate_in_hunk([f], hunk_ranges(diff))
        assert f.in_hunk is None

    def test_unknown_is_none_not_false(self) -> None:
        # None means "can't say"; False means "outside the diff". A caller
        # automating suggestions must treat those differently.
        no_line, no_match = _f("src/x.py", None), _f("nowhere/z.py", 5)
        annotate_in_hunk([no_line, no_match], hunk_ranges(_TWO_HUNK_DIFF))
        assert no_line.in_hunk is None
        assert no_match.in_hunk is None

    def test_empty_ranges_leave_findings_untouched(self) -> None:
        f = _f("src/x.py", 11)
        annotate_in_hunk([f], {})
        assert f.in_hunk is None


class TestDiffEvidenceRule:
    def test_hunks_only_tells_the_model_the_file_is_hidden(self) -> None:
        _, user = build_diff_prompts(_DIFF, "ctx")
        assert DIFF_EVIDENCE_RULE in user
        assert "HIDDEN from you" in user

    def test_full_files_does_not_claim_the_file_is_hidden(self) -> None:
        # The load-bearing assertion: with reference bodies attached the
        # file is visible, so the hunks-only framing would be a lie and
        # would talk the model out of legitimate whole-file findings.
        _, user = build_diff_prompts(_DIFF, "ctx", full_files=True)
        assert FULL_FILES_EVIDENCE_RULE in user
        assert DIFF_EVIDENCE_RULE not in user
        assert "HIDDEN from you" not in user

    def test_full_files_still_flags_the_repo_wide_blind_spot(self) -> None:
        # Changed files are attached; unchanged callers elsewhere are not.
        _, user = build_diff_prompts(_DIFF, "ctx", full_files=True)
        assert "were NOT changed are still hidden" in user

    def test_both_modes_demand_a_no_op_check(self) -> None:
        # Kills the single most common observed noise class: a "fix" that
        # is identical to the code already on screen.
        for full in (False, True):
            _, user = build_diff_prompts(_DIFF, "ctx", full_files=full)
            assert "identical to what is already there" in user

    def test_both_modes_offer_needs_verification(self) -> None:
        for full in (False, True):
            _, user = build_diff_prompts(_DIFF, "ctx", full_files=full)
            assert "NEEDS-VERIFICATION" in user

    def test_no_context_keeps_the_rule_but_drops_the_safety_wrapper(self) -> None:
        # --no-context is the escape hatch for wrapper-triggered refusals.
        # It must not also silently disable the review instructions.
        _, user = build_diff_prompts(_DIFF, None)
        assert "Evidence discipline" in user
        assert "<CONTEXT_FOR_REVIEWER>" not in user


class TestBaseAheadWarning:
    def test_warns_when_base_has_commits_head_lacks(self, monkeypatch, capsys) -> None:
        # `git diff <base>` is two-dot, so base-only commits appear as
        # removals and get reported as intentional deletions/reverts.
        monkeypatch.setattr(sources, "_run_git", lambda args: "3\n")
        sources._warn_if_base_ahead("origin/main")
        err = capsys.readouterr().err
        assert "3 commit(s) not in HEAD" in err
        assert "two-dot" in err

    def test_silent_when_base_is_an_ancestor(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sources, "_run_git", lambda args: "0\n")
        sources._warn_if_base_ahead("origin/main")
        assert capsys.readouterr().err == ""

    def test_unresolvable_ref_is_not_fatal(self, monkeypatch, capsys) -> None:
        # Best-effort: the real error surfaces from the diff call that
        # follows, with a better message than this probe could give.
        def _boom(args: list[str]) -> str:
            raise sources.ConfigError("bad ref")

        monkeypatch.setattr(sources, "_run_git", _boom)
        sources._warn_if_base_ahead("nope")
        assert capsys.readouterr().err == ""

    def test_non_numeric_output_is_ignored(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sources, "_run_git", lambda args: "not-a-number\n")
        sources._warn_if_base_ahead("origin/main")
        assert capsys.readouterr().err == ""
