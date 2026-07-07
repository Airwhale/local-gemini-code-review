"""Tests for the eval harness's scoring logic (evals/run.py).

The harness itself shells the CLI and spends tokens, so only the pure
``score()`` function is unit-tested; it's loaded by path because
``evals/`` is a script directory, not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "evals_run", Path(__file__).parent.parent / "evals" / "run.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_RUN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN)
score = _RUN.score


def _envelope(findings: list[dict]) -> dict:
    return {"findings": findings}


def _finding(file: str = "a.py", line: int | None = 5, body: str = "") -> dict:
    return {"file": file, "line": line, "title": "t", "body": body}


BUG = {"file": "a.py", "line_range": [4, 8], "keywords": ["remainder"]}


class TestScore:
    def test_file_keyword_and_range_match(self):
        env = _envelope([_finding(line=5, body="drops the remainder")])
        assert score(env, {"bug": [BUG]}) == (1, 1, 0)

    def test_wrong_line_is_noise_not_caught(self):
        # Right file, right keyword, but the finding points somewhere
        # else in the file -- must not count as catching the planted bug.
        env = _envelope([_finding(line=40, body="drops the remainder")])
        assert score(env, {"bug": [BUG]}) == (0, 1, 1)

    def test_line_less_finding_skips_the_range_check(self):
        # Models legitimately omit line numbers sometimes; punishing that
        # would measure formatting, not recall.
        env = _envelope([_finding(line=None, body="drops the remainder")])
        assert score(env, {"bug": [BUG]}) == (1, 1, 0)

    def test_bug_without_range_matches_on_file_and_keyword(self):
        bug = {"file": "a.py", "keywords": ["remainder"]}
        env = _envelope([_finding(line=999, body="drops the remainder")])
        assert score(env, {"bug": [bug]}) == (1, 1, 0)

    def test_wrong_file_never_matches(self):
        env = _envelope([_finding(file="b.py", line=5, body="remainder")])
        assert score(env, {"bug": [BUG]}) == (0, 1, 1)

    def test_clean_fixture_counts_everything_as_noise(self):
        # The clean-refactor control: no [[bug]] entries -> recall 0/0,
        # every finding is noise.
        env = _envelope([_finding(), _finding(line=9)])
        assert score(env, {}) == (0, 0, 2)

    def test_one_finding_cannot_vouch_for_two_bugs(self):
        bug2 = {"file": "a.py", "line_range": [4, 8], "keywords": ["remainder"]}
        env = _envelope([_finding(line=5, body="drops the remainder")])
        assert score(env, {"bug": [BUG, bug2]}) == (1, 2, 0)
