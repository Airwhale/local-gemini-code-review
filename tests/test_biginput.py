"""Tests for the big-input modes: --full-files reference bundling and
--chunk splitting/packing/budgeting/envelope."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import code_review.cli as review
from code_review.cli import (
    CallResult,
    ConfigError,
    ContextOverflow,
    ParsedReview,
    Settings,
    _chunk_budget,
    _pack_contiguous,
    _validate_flag_combos,
    build_chunked_envelope,
    build_reference_section,
    changed_file_paths,
    partition_codebase,
    partition_diffs,
    split_diff_by_file,
)

MULTI_FILE_DIFF = (
    "diff --git a/one.py b/one.py\n"
    "index 111..222 100644\n"
    "--- a/one.py\n"
    "+++ b/one.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-old\n"
    "+new\n"
    "diff --git a/two.py b/two.py\n"
    "index 333..444 100644\n"
    "--- a/two.py\n"
    "+++ b/two.py\n"
    "@@ -5,1 +5,2 @@\n"
    " ctx\n"
    "+added\n"
    "diff --git a/three.py b/three.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/three.py\n"
    "@@ -0,0 +1,1 @@\n"
    "+content\n"
)


class TestSplitDiffByFile:
    def test_lossless_rejoin(self):
        parts = split_diff_by_file(MULTI_FILE_DIFF)
        assert len(parts) == 3
        assert "".join(parts) == MULTI_FILE_DIFF

    def test_each_part_starts_at_file_anchor(self):
        parts = split_diff_by_file(MULTI_FILE_DIFF)
        assert all(p.startswith("diff --git ") for p in parts)

    def test_preamble_attaches_to_first_part(self):
        diff = "some preamble\n" + MULTI_FILE_DIFF
        parts = split_diff_by_file(diff)
        assert parts[0].startswith("some preamble\n")
        assert "".join(parts) == diff

    def test_no_anchor_returns_whole(self):
        assert split_diff_by_file("just text") == ["just text"]

    def test_empty_diff(self):
        assert split_diff_by_file("") == []


class TestPacking:
    def test_next_fit_preserves_order(self):
        chunks = _pack_contiguous([4, 4, 4, 4], budget=8)
        assert chunks == [[0, 1], [2, 3]]

    def test_single_chunk_when_under_budget(self):
        assert _pack_contiguous([1, 2, 3], budget=100) == [[0, 1, 2]]

    def test_partition_diffs_oversize_names_offender(self):
        big = "diff --git a/huge b/huge\n" + "x" * 100
        with pytest.raises(ContextOverflow) as exc_info:
            partition_diffs([big], budget=50)
        assert "diff --git a/huge b/huge" in str(exc_info.value)

    def test_partition_diffs_packs(self):
        parts = split_diff_by_file(MULTI_FILE_DIFF)
        sizes = [len(p) for p in parts]
        budget = sizes[0] + sizes[1]
        chunks = partition_diffs(parts, budget=budget)
        assert len(chunks) == 2
        assert "".join(chunks) == MULTI_FILE_DIFF

    def test_partition_codebase_by_bundled_size(self, tmp_path: Path):
        files = []
        for name in ("a.py", "b.py", "c.py"):
            f = tmp_path / name
            f.write_text("x = 1\n" * 50, encoding="utf-8")
            files.append(f)
        one_bundle = len(review.bundle_codebase([files[0]]))
        chunks = partition_codebase(files, budget=one_bundle)
        assert [len(c) for c in chunks] == [1, 1, 1]

    def test_partition_codebase_oversize_file(self, tmp_path: Path):
        f = tmp_path / "big.py"
        f.write_text("x" * 1000, encoding="utf-8")
        with pytest.raises(ContextOverflow) as exc_info:
            partition_codebase([f], budget=100)
        assert "big.py" in str(exc_info.value)


class TestReferenceSection:
    def test_shape(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text("def f():\n    return 1\n", encoding="utf-8")
        section = build_reference_section([f])
        assert section.startswith("\n\n<REFERENCE_FILES>")
        assert section.rstrip().endswith("</REFERENCE_FILES>")
        assert f"======== FILE: {f.as_posix()} ========" in section
        assert "REFERENCE ONLY" in section
        assert "1: def f():" in section  # line-numbered

    def test_empty_paths(self):
        assert build_reference_section([]) == ""


class TestChangedFilePaths:
    def _capture(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake_run_git(args: list[str]) -> str:
            calls.append(args)
            return "one.py\ntwo.py\n"

        monkeypatch.setattr(review, "_run_git", fake_run_git)
        return calls

    def test_staged(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._capture(monkeypatch)
        args = argparse.Namespace(pr=None, staged=True, base=None)
        assert changed_file_paths(args) == [Path("one.py"), Path("two.py")]
        assert calls == [["git", "diff", "--cached", "--name-only"]]

    def test_base(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._capture(monkeypatch)
        args = argparse.Namespace(pr=None, staged=False, base="main")
        changed_file_paths(args)
        assert calls == [["git", "diff", "--name-only", "main"]]

    def test_default_merge_base(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._capture(monkeypatch)
        args = argparse.Namespace(pr=None, staged=False, base=None)
        changed_file_paths(args)
        assert calls == [["git", "diff", "--name-only", "--merge-base", "origin/HEAD"]]


class TestGhRepoPin:
    """Bare `gh pr N` resolves through gh's default-repo logic, which on
    forks often points at UPSTREAM -- --repo must pin every gh call, and
    the PR URL announce must make the resolution visible."""

    class _Result:
        stdout = "out\n"
        stderr = ""

    def _capture(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            return self._Result()

        monkeypatch.setattr(review.subprocess, "run", fake_run)
        return calls

    def test_repo_appended_to_every_gh_call(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._capture(monkeypatch)
        review.pr_diff(7, "owner/name")
        review.pr_changed_files(7, "owner/name")
        review.pr_head_sha(7, "owner/name")
        review.pr_url(7, "owner/name")
        assert len(calls) == 4
        for args in calls:
            assert args[:2] == ["gh", "pr"]
            assert args[-2:] == ["--repo", "owner/name"]

    def test_without_repo_command_stays_bare(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._capture(monkeypatch)
        review.pr_diff(7)
        assert "--repo" not in calls[0]

    def test_pr_url_queries_the_url_field(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._capture(monkeypatch)
        assert review.pr_url(7) == "out"
        assert "--json" in calls[0]
        assert "url" in calls[0]

    def test_read_diff_source_announces_pr_url(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        monkeypatch.setattr(
            review, "pr_url", lambda pr, repo=None: "https://github.com/o/n/pull/7"
        )
        monkeypatch.setattr(review, "pr_diff", lambda pr, repo=None: "diff text")
        args = argparse.Namespace(diff_file=None, pr=7, repo=None)
        assert review._read_diff_source(args) == "diff text"
        err = capsys.readouterr().err
        assert "[gh] reviewing PR #7: https://github.com/o/n/pull/7" in err


class TestGuardPrFullFiles:
    """--full-files with --pr must not silently pair a GitHub-sourced
    diff with file bodies from an unrelated local checkout."""

    HEAD = "a" * 40
    OTHER = "b" * 40

    def _patch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        pr_head: str,
        local_head: str,
        dirty: bool = False,
    ) -> None:
        monkeypatch.setattr(review, "pr_head_sha", lambda pr, repo=None: pr_head)

        def fake_run_git(args: list[str]) -> str:
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return local_head + "\n"
            if args[:3] == ["git", "status", "--porcelain"]:
                return " M code_review/cli.py\n" if dirty else ""
            raise AssertionError(f"unexpected git call: {args}")

        monkeypatch.setattr(review, "_run_git", fake_run_git)

    def test_head_mismatch_is_config_error(self, monkeypatch: pytest.MonkeyPatch):
        self._patch(monkeypatch, pr_head=self.HEAD, local_head=self.OTHER)
        with pytest.raises(ConfigError) as exc_info:
            review._guard_pr_full_files(3)
        assert "gh pr checkout 3" in str(exc_info.value)

    def test_mismatch_hint_carries_the_repo_pin(self, monkeypatch: pytest.MonkeyPatch):
        # A bare `gh pr checkout N` in the hint would re-create the
        # wrong-repo footgun --repo exists to avoid.
        self._patch(monkeypatch, pr_head=self.HEAD, local_head=self.OTHER)
        with pytest.raises(ConfigError) as exc_info:
            review._guard_pr_full_files(3, "owner/name")
        assert "gh pr checkout 3 --repo owner/name" in str(exc_info.value)

    def test_matching_clean_tree_passes_silently(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        self._patch(monkeypatch, pr_head=self.HEAD, local_head=self.HEAD)
        review._guard_pr_full_files(3)
        assert capsys.readouterr().err == ""

    def test_matching_dirty_tree_warns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        self._patch(monkeypatch, pr_head=self.HEAD, local_head=self.HEAD, dirty=True)
        review._guard_pr_full_files(3)
        err = capsys.readouterr().err
        assert err.startswith("WARN:")
        assert "uncommitted changes" in err


def _combo_args(**overrides) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        chunk=False,
        models=None,
        full_files=False,
        baseline=None,
        codebase=False,
        diff_file=None,
        pr=None,
        repo=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestFlagCombos:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(chunk=True, models="a,b"),
            dict(chunk=True, full_files=True),
            dict(chunk=True, baseline="r.json"),
            dict(full_files=True, codebase=True),
            dict(full_files=True, diff_file="d.patch"),
            dict(repo="owner/name"),  # --repo only applies to --pr
        ],
    )
    def test_rejected_pairs(self, kwargs: dict):
        with pytest.raises(ConfigError):
            _validate_flag_combos(_combo_args(**kwargs))

    def test_valid_combos_pass(self):
        _validate_flag_combos(_combo_args(chunk=True))
        _validate_flag_combos(_combo_args(full_files=True))
        _validate_flag_combos(_combo_args(pr=7, repo="owner/name"))
        _validate_flag_combos(_combo_args())


def _settings(**overrides) -> Settings:
    base: dict[str, Any] = dict(
        provider="openrouter",
        model="m",
        temperature=0.3,
        max_tokens=16000,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    base.update(overrides)
    return Settings(**base)


class TestChunkBudget:
    def test_cloud_uses_bundle_cap(self):
        budget, note = _chunk_budget(_settings())
        assert budget == review.MAX_BUNDLE_CHARS
        assert note is None

    def test_ollama_enforced_window(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            review, "_resolve_ollama_window", lambda h, m, e: (32768, True, "env")
        )
        budget, note = _chunk_budget(
            _settings(provider="ollama", ollama_host="http://x", ollama_timeout=1.0)
        )
        assert note is None
        # window*4 minus measured prompt overhead; must be positive and
        # below the raw window size.
        assert 0 < budget < 32768 * review.OLLAMA_CHARS_PER_TOKEN

    def test_ollama_unknown_window_warns(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            review,
            "_resolve_ollama_window",
            lambda h, m, e: (review.DEFAULT_OLLAMA_NUM_CTX, False, "advisory-default"),
        )
        budget, note = _chunk_budget(
            _settings(provider="ollama", ollama_host="http://x", ollama_timeout=1.0)
        )
        assert note is not None
        assert "OLLAMA_NUM_CTX" in note

    def test_ollama_overhead_exceeds_window(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            review, "_resolve_ollama_window", lambda h, m, e: (128, True, "env")
        )
        with pytest.raises(ContextOverflow):
            _chunk_budget(
                _settings(provider="ollama", ollama_host="http://x", ollama_timeout=1.0)
            )


class TestChunkedEnvelope:
    def test_shape(self):
        parsed_ok = ParsedReview(
            summary="s",
            clean=False,
            parse_ok=True,
            problems=[],
            findings=[
                review.Finding(
                    file="a.py",
                    line=1,
                    severity="HIGH",
                    title="t",
                    body="b",
                    suggestion=None,
                )
            ],
        )
        parsed_bad = ParsedReview(
            summary=None,
            clean=False,
            parse_ok=False,
            problems=["garbled"],
            findings=[],
        )
        envelope = build_chunked_envelope(
            mode="codebase",
            provider="openrouter",
            model="m",
            temperature=0.3,
            chunk_data=[
                ("2 file(s)", parsed_ok, CallResult("raw1", 100, 10), "raw1"),
                ("3 file(s)", parsed_bad, CallResult("raw2", 200, 20), "raw2"),
            ],
        )
        assert envelope["chunks"] == 2
        assert envelope["parse_ok"] is False  # one chunk failed to parse
        assert envelope["findings"][0]["chunk"] == 1
        assert envelope["usage"] == {"prompt_tokens": 300, "completion_tokens": 30}
        assert "raw" not in envelope["per_chunk"][0]
        assert envelope["per_chunk"][1]["raw"] == "raw2"
        assert envelope["problems"] == ["chunk 2: garbled"]
