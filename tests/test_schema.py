"""Pins docs/schema/review-envelope.schema.json to the envelope builders.

The schema is a published consumer contract; if a builder gains or
reshapes a field, one of these validations must fail so the schema (and
CHANGELOG) get updated in the same PR. All three run shapes are
covered: single-model (with and without baseline / parse failure),
panel (including a failed model), and chunked.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from code_review.cli import (
    CallResult,
    Finding,
    MergedFinding,
    ParsedReview,
    ReviewError,
    build_chunked_envelope,
    build_json_envelope,
    build_panel_envelope,
)

SCHEMA = json.loads(
    (
        Path(__file__).parent.parent / "docs" / "schema" / "review-envelope.schema.json"
    ).read_text(encoding="utf-8")
)


def _validate(envelope: dict) -> None:
    jsonschema.validate(envelope, SCHEMA)


def _finding(line: int | None = 42) -> Finding:
    return Finding(
        file="src/client.py",
        line=line,
        severity="HIGH",
        title="Retry loop never sleeps.",
        body="Hammers the endpoint.",
        suggestion="time.sleep(2 ** attempt)",
    )


def _parsed(findings: list[Finding], parse_ok: bool = True) -> ParsedReview:
    return ParsedReview(
        summary="A summary.",
        findings=findings,
        clean=False,
        parse_ok=parse_ok,
        problems=[] if parse_ok else ["drifted"],
    )


class TestSchemaValidatesBuilders:
    def test_single_model_envelope(self):
        _validate(
            build_json_envelope(
                mode="diff",
                provider="openrouter",
                model="google/gemini-2.5-pro",
                temperature=0.3,
                parsed=_parsed([_finding(), _finding(line=None)]),
                result=CallResult("raw", prompt_tokens=10, completion_tokens=5),
                raw_markdown="raw",
            )
        )

    def test_single_model_with_baseline_fields(self):
        _validate(
            build_json_envelope(
                mode="diff",
                provider="openrouter",
                model="m",
                temperature=0.3,
                parsed=_parsed([_finding()]),
                result=CallResult("raw"),
                raw_markdown="raw",
                statuses=["persisting"],
                resolved=[{"title": "old", "severity": "LOW", "fingerprint": "ab" * 6}],
            )
        )

    def test_single_model_parse_failure_embeds_raw(self):
        envelope = build_json_envelope(
            mode="codebase",
            provider="ollama",
            model="qwen3-coder:30b",
            temperature=0.3,
            parsed=_parsed([], parse_ok=False),
            result=CallResult("garbage"),
            raw_markdown="garbage",
        )
        assert envelope["raw"] == "garbage"
        _validate(envelope)

    def test_panel_envelope_including_a_failure(self):
        ok_parsed = _parsed([_finding()])
        _validate(
            build_panel_envelope(
                mode="diff",
                provider="openrouter",
                temperature=0.3,
                models=("model-a", "model-b"),
                merged=[MergedFinding(finding=_finding(), found_by=["model-a"])],
                parsed_by_model={"model-a": ok_parsed},
                results_by_model={
                    "model-a": CallResult("raw", prompt_tokens=7, completion_tokens=3)
                },
                raw_by_model={"model-a": "raw"},
                failures=[("model-b", ReviewError("boom"))],
            )
        )

    def test_chunked_envelope(self):
        _validate(
            build_chunked_envelope(
                mode="codebase",
                provider="ollama",
                model="qwen3-coder:30b",
                temperature=0.3,
                chunk_data=[
                    ("chunk one", _parsed([_finding()]), CallResult("r1"), "r1"),
                    ("chunk two", _parsed([], parse_ok=False), CallResult("r2"), "r2"),
                ],
            )
        )

    def test_schema_rejects_a_contract_break(self):
        # Sanity that validation has teeth: a wrong-typed core field fails.
        envelope = build_json_envelope(
            mode="diff",
            provider="openrouter",
            model="m",
            temperature=0.3,
            parsed=_parsed([_finding()]),
            result=CallResult("raw"),
            raw_markdown="raw",
        )
        envelope["findings"][0]["fingerprint"] = "not-a-sha1-prefix"
        with pytest.raises(jsonschema.ValidationError):
            _validate(envelope)
