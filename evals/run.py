#!/usr/bin/env python3
"""Eval harness: run the reviewer against planted-bug fixtures and score it.

Turns temperature / model tuning from anecdote into data. Each fixture
under ``evals/fixtures/<name>/`` is a ``diff.patch`` with known planted
bugs described in ``expected.toml`` (``[[bug]]`` entries with a target
file, an approximate line range, and a keyword list). A planted bug
counts as CAUGHT when any finding names the right file and mentions any
keyword in its title+body; findings matching no planted bug count as
noise (potential false positives -- some may be legitimate extra
findings, so read them before drawing conclusions). A fixture with no
``[[bug]]`` entries is a clean-diff control: recall reads 0/0 and every
finding is noise -- the hallucination rate on clean code, which the
bug fixtures can't measure.

Costs real tokens: the harness prints the planned call count and asks
for confirmation unless --yes is passed.

    uv run evals/run.py --model flash
    uv run evals/run.py --model pro --model claude --temperature 0.3 --temperature 0.5
    uv run evals/run.py --fixture off-by-one --model flash --yes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

EVALS_DIR = Path(__file__).parent
REPO_ROOT = EVALS_DIR.parent
FIXTURES_DIR = EVALS_DIR / "fixtures"


def discover_fixtures(only: list[str]) -> list[Path]:
    fixtures = sorted(
        d
        for d in FIXTURES_DIR.iterdir()
        if (d / "diff.patch").is_file() and (d / "expected.toml").is_file()
    )
    if only:
        wanted = set(only)
        fixtures = [f for f in fixtures if f.name in wanted]
        missing = wanted - {f.name for f in fixtures}
        if missing:
            sys.exit(f"unknown fixture(s): {', '.join(sorted(missing))}")
    return fixtures


def run_review(fixture: Path, model: str, temperature: float) -> dict:
    """Invoke the runner on a fixture diff; return the JSON envelope."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "review.py"),
        "--diff-file",
        str(fixture / "diff.patch"),
        "--format",
        "json",
        "--model",
        model,
        "--temperature",
        str(temperature),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"review exited {result.returncode} for {fixture.name}:\n"
            f"{result.stderr.strip()[-500:]}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        # Exit 0 with non-JSON stdout is a runner bug, but surface it as
        # an ERR row with the payload tail instead of a raw traceback.
        raise RuntimeError(
            f"review exited 0 for {fixture.name} but stdout was not JSON "
            f"({exc}); tail:\n{result.stdout.strip()[-500:]}"
        ) from exc


def score(envelope: dict, expected: dict) -> tuple[int, int, int]:
    """Return (bugs_caught, bugs_total, noise_findings).

    A finding matches a planted bug when it names the right file,
    mentions any keyword, and -- when the fixture declares a
    ``line_range`` and the finding carries a line number -- points
    inside that range. Line-less findings skip the range check (models
    legitimately omit lines sometimes; punishing that would measure
    formatting, not recall). If models cite defensible nearby lines,
    widen the fixture's range rather than adding slack here.
    """
    findings = envelope.get("findings") or []
    bugs = expected.get("bug") or []
    matched_findings: set[int] = set()
    caught = 0
    for bug in bugs:
        line_range = bug.get("line_range")
        for idx, finding in enumerate(findings):
            if idx in matched_findings:
                continue
            if finding.get("file") != bug["file"]:
                continue
            line = finding.get("line")
            if (
                isinstance(line_range, list)
                and len(line_range) == 2
                and isinstance(line, int)
                and not (line_range[0] <= line <= line_range[1])
            ):
                continue
            # `or ""`: a present-but-None title/body would render as the
            # string "None" and false-match a "none" keyword.
            haystack = (
                f"{finding.get('title') or ''} {finding.get('body') or ''}".lower()
            )
            if any(keyword.lower() in haystack for keyword in bug["keywords"]):
                caught += 1
                matched_findings.add(idx)
                break
    noise = len(findings) - len(matched_findings)
    return caught, len(bugs), noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="SLUG",
        help="Model slug/alias; repeatable. Default: flash (cheap).",
    )
    parser.add_argument(
        "--temperature",
        action="append",
        type=float,
        default=[],
        metavar="T",
        help="Temperature; repeatable. Default: 0.3.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        metavar="NAME",
        help="Run only this fixture; repeatable. Default: all.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the cost confirmation prompt.",
    )
    args = parser.parse_args()

    models = args.model or ["flash"]
    temperatures = args.temperature or [0.3]
    fixtures = discover_fixtures(args.fixture)
    if not fixtures:
        sys.exit("no fixtures found")

    total_calls = len(fixtures) * len(models) * len(temperatures)
    print(
        f"{len(fixtures)} fixture(s) x {len(models)} model(s) x "
        f"{len(temperatures)} temperature(s) = {total_calls} paid API call(s)"
    )
    if not args.yes:
        answer = input("proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            sys.exit("aborted")

    rows = []
    for model in models:
        for temperature in temperatures:
            for fixture in fixtures:
                expected = tomllib.loads(
                    (fixture / "expected.toml").read_text(encoding="utf-8-sig")
                )
                try:
                    envelope = run_review(fixture, model, temperature)
                except RuntimeError as exc:
                    print(f"ERROR {fixture.name} [{model} T={temperature}]: {exc}")
                    rows.append((model, temperature, fixture.name, "ERR", "-", "-"))
                    continue
                caught, total, noise = score(envelope, expected)
                rows.append(
                    (
                        model,
                        temperature,
                        fixture.name,
                        f"{caught}/{total}",
                        str(noise),
                        "ok" if envelope.get("parse_ok") else "parse_fail",
                    )
                )

    print()
    header = ("model", "T", "fixture", "recall", "noise", "parse")
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(6)]
    for row in [header, *rows]:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

    caught_total = sum(int(r[3].split("/")[0]) for r in rows if r[3] not in ("ERR",))
    bugs_total = sum(int(r[3].split("/")[1]) for r in rows if r[3] not in ("ERR",))
    errored = sum(1 for r in rows if r[3] == "ERR")
    if bugs_total:
        # Say what the denominator covers: ERR rows never ran, so an
        # unqualified "3/3" after a failed fixture would read as a
        # perfect sweep it isn't.
        qualifier = (
            f" (completed runs only; {errored} errored run(s) excluded)"
            if errored
            else ""
        )
        print(f"\noverall recall: {caught_total}/{bugs_total}{qualifier}")

    # A failed runner invocation must fail the harness (and the manual
    # Evals workflow gating on it) -- an all-ERR table exiting 0 would
    # read as a green run that scored nothing.
    if errored:
        sys.exit(f"{errored} of {len(rows)} run(s) errored (marked ERR above)")


if __name__ == "__main__":
    main()
