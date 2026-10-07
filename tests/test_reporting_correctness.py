"""CLI regressions for incomplete baselines and false panel agreement."""

import dataclasses
import json
from pathlib import Path

import jsonschema
import pytest

from code_review import cli
from code_review.config import ReviewRequest, Settings
from code_review.parser import build_json_envelope, parse_review_markdown
from code_review.providers import CallResult

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
CURRENT = (FIXTURES_DIR / "reporting-current.md").read_text(encoding="utf-8")
BASELINE = (FIXTURES_DIR / "reporting-baseline.md").read_text(encoding="utf-8")
SECOND_MODEL = (FIXTURES_DIR / "reporting-second-model.md").read_text(encoding="utf-8")
SCHEMA = json.loads(
    (PROJECT_ROOT / "docs" / "schema" / "review-envelope.schema.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        provider="openrouter",
        model="model-a",
        temperature=0.3,
        max_tokens=100,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )


def _run_cli(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr("sys.argv", ["code-review", "--no-project-config"])
    monkeypatch.setattr(cli, "_resolve_settings", lambda *args: settings)
    monkeypatch.setattr(
        cli, "_build_requests", lambda *args: [ReviewRequest("s", "u", "diff", 1)]
    )
    monkeypatch.setattr(cli, "_estimate_cost_line", lambda *args: None)
    cli.main()


@pytest.mark.parametrize("truncated", [True, False])
@pytest.mark.parametrize("output_format", ["json", "markdown"])
def test_baseline_resolution_requires_complete_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    truncated: bool,
    output_format: str,
) -> None:
    baseline = build_json_envelope(
        mode="diff",
        provider="openrouter",
        model="model-a",
        temperature=0.3,
        parsed=parse_review_markdown(BASELINE),
        result=CallResult(BASELINE),
        raw_markdown=BASELINE,
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_execute_call",
        lambda *args: CallResult(CURRENT, truncated=truncated),
    )
    _run_cli(
        monkeypatch,
        dataclasses.replace(
            settings, baseline=str(baseline_path), format=output_format
        ),
    )
    output = capsys.readouterr()
    assert "1 new, 1 persisting" in output.err
    if truncated:
        assert "resolution unknown (truncated review)" in output.err
        assert "1 resolved" not in output.err
    else:
        assert "1 resolved" in output.err

    if output_format == "markdown":
        assert output.out == CURRENT + "\n"
    else:
        report = json.loads(output.out)
        jsonschema.validate(report, SCHEMA)
        snapshot = {
            "truncated": report["truncated"],
            "statuses": [finding["status"] for finding in report["findings"]],
            "resolved_titles": (
                [finding["title"] for finding in report["resolved"]]
                if "resolved" in report
                else None
            ),
        }
        state = "truncated" if truncated else "complete"
        expected = json.loads(
            (FIXTURES_DIR / f"baseline-{state}.expected.json").read_text(
                encoding="utf-8"
            )
        )
        assert snapshot == expected


def test_envelope_cannot_publish_resolution_from_partial_output() -> None:
    report = build_json_envelope(
        mode="diff",
        provider="openrouter",
        model="model-a",
        temperature=0.3,
        parsed=parse_review_markdown(CURRENT),
        result=CallResult(CURRENT, truncated=True),
        raw_markdown=CURRENT,
        resolved=[{"title": "Unseen finding"}],
    )
    assert "resolved" not in report
    jsonschema.validate(report, SCHEMA)


@pytest.mark.parametrize("output_format", ["json", "markdown"])
def test_panel_threshold_excludes_unrelated_nearby_findings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    output_format: str,
) -> None:
    monkeypatch.setattr(
        cli,
        "_run_panel",
        lambda *args: (
            {"model-a": CallResult(CURRENT), "model-b": CallResult(SECOND_MODEL)},
            [],
        ),
    )
    _run_cli(
        monkeypatch,
        dataclasses.replace(
            settings,
            models=("model-a", "model-b"),
            min_found_by=2,
            format=output_format,
        ),
    )
    output = capsys.readouterr()
    assert "dropped 2 finding(s)" in output.err
    if output_format == "json":
        report = json.loads(output.out)
        jsonschema.validate(report, SCHEMA)
        assert len(report["findings"]) == 1
        assert report["findings"][0]["title"] == "Empty input crashes parser."
        assert report["findings"][0]["found_by"] == ["model-a", "model-b"]
    else:
        assert output.out.count("Found by:") == 1
        # Rejected findings remain inspectable in the raw model appendix.
        assert "SQL injection in user query" in output.out
        assert "File handle leaks on exceptions" in output.out
