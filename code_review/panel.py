"""Multi-model panel (--models): consensus merge and panel reports.

Runs of the same prompt through several models get merged into one
consensus-annotated report. Motivated by dogfood data: zero cross-model
overlap on real hallucinations, so ``found_by > 1`` is a rare,
high-precision signal. Orchestration (thread pool, per-model calls)
lives in cli; this module is the pure logic.
"""

from __future__ import annotations

import dataclasses

from code_review.errors import ReviewError
from code_review.parser import (
    Finding,
    ParsedReview,
    _location_match,
    finding_fingerprint,
    findings_match,
)
from code_review.prompts import SEVERITY_LEVELS
from code_review.providers import CallResult

_SEVERITY_RANK = {level: idx for idx, level in enumerate(SEVERITY_LEVELS)}

# All-models-failed exit precedence: non-retryable, caller-actionable
# categories dominate retryable ones so an agent pattern-matching the
# single ERROR: block fixes the config/scope problem instead of
# blind-retrying. Documented in README as part of the error model; ties
# break by CLI model order (min() is stable).
_CATEGORY_PRECEDENCE = (
    "CONFIG",
    "SAFETY_REFUSAL",
    "CONTEXT_OVERFLOW",
    "RATE_LIMIT",
    "PROVIDER_HICCUP",
    "TRANSPORT",
    "UNKNOWN",
)


@dataclasses.dataclass
class MergedFinding:
    """One panel cluster: the representative finding + who reported it."""

    finding: Finding
    found_by: list[str]


def panel_findings_match(a: Finding, b: Finding) -> bool:
    """Do two findings from DIFFERENT models describe the same issue?

    Strong fingerprint match, or same location AND same severity. The
    location tier is tighter than the baseline matcher's (which ignores
    severity): consensus is sold as a high-precision signal, and merging
    two different same-hunk findings from different models would
    manufacture false confidence. Under-merging just leaves found_by=1,
    which is the empirical norm anyway.
    """
    if findings_match(a, b):
        return True
    # Location tier: BOTH lines must be known. A line-less finding
    # matching "any line in the same file" would let one vague finding
    # absorb an unrelated specific one and manufacture consensus. (The
    # baseline matcher keeps the looser rule deliberately -- there the
    # cost of a miss is a mislabeled status, not false confidence.)
    if a.line is None or b.line is None:
        return False
    return a.severity == b.severity and _location_match(a.file, a.line, b.file, b.line)


def merge_panel_findings(
    parsed_by_model: dict[str, ParsedReview],
) -> list[MergedFinding]:
    """Greedily cluster findings across models (dict order = CLI order).

    The first model to report an issue provides the representative
    finding (title/body/suggestion); later matches only append to
    ``found_by``. Ordering: consensus first, then severity, then
    location -- the read order a human wants.
    """
    clusters: list[MergedFinding] = []
    for model, parsed in parsed_by_model.items():
        for finding in parsed.findings:
            for cluster in clusters:
                if model not in cluster.found_by and panel_findings_match(
                    cluster.finding, finding
                ):
                    cluster.found_by.append(model)
                    break
            else:
                clusters.append(MergedFinding(finding=finding, found_by=[model]))
    clusters.sort(
        key=lambda c: (
            -len(c.found_by),
            -_SEVERITY_RANK.get(c.finding.severity, -1),
            c.finding.file or "",
            c.finding.line if c.finding.line is not None else 10**9,
        )
    )
    return clusters


def _panel_max_workers(provider: str, n_models: int) -> int:
    """Concurrency for a panel: cloud providers run in parallel (capped);
    ollama runs strictly sequentially -- two local models can't share
    RAM, and interleaved requests just thrash model swaps."""
    if provider == "ollama":
        return 1
    return min(n_models, 4)


def _panel_exit_error(failures: list[tuple[str, ReviewError]]) -> ReviewError:
    """Pick the one typed error to exit with when ALL panel models failed.

    Fixed category precedence (see ``_CATEGORY_PRECEDENCE``); ties break
    by CLI model order because ``min`` is stable.
    """

    def rank(item: tuple[str, ReviewError]) -> int:
        category = item[1].category
        return (
            _CATEGORY_PRECEDENCE.index(category)
            if category in _CATEGORY_PRECEDENCE
            else len(_CATEGORY_PRECEDENCE)
        )

    return min(failures, key=rank)[1]


def render_panel_markdown(
    merged: list[MergedFinding],
    parsed_by_model: dict[str, ParsedReview],
    raw_by_model: dict[str, str],
    failures: list[tuple[str, ReviewError]],
    n_models: int,
) -> str:
    """Fork-generated markdown report for a panel run.

    Merged findings first (consensus-ordered, each with a ``Found by:``
    line), then every model's raw output verbatim in an appendix --
    nothing the models said is lost to the merge.
    """
    lines: list[str] = [
        f"# Panel review ({len(parsed_by_model)}/{n_models} models)",
        "",
    ]

    lines.append("Per-model results:")
    for model, parsed in parsed_by_model.items():
        if parsed.clean:
            note = "clean -- no issues found"
        elif parsed.parse_ok:
            note = (
                f"{len(parsed.findings)} finding(s): {parsed.summary or '(no summary)'}"
            )
        else:
            note = "output could not be parsed (see appendix)"
        lines.append(f"- `{model}`: {note}")
    for model, err in failures:
        lines.append(f"- `{model}`: FAILED -- {err.category}: {err}")
    lines.append("")

    if merged:
        lines.append("## Merged findings")
        lines.append("")
        for cluster in merged:
            finding = cluster.finding
            location = finding.file or "(no file)"
            if finding.line is not None:
                location += f" L{finding.line}"
            lines.append(f"### {location}: [{finding.severity}] {finding.title}")
            lines.append(f"Found by: {', '.join(cluster.found_by)}")
            if finding.body:
                lines.append("")
                lines.append(finding.body)
            if finding.suggestion:
                lines.append("")
                lines.append("Suggested change:")
                lines.append("```")
                lines.append(finding.suggestion)
                lines.append("```")
            lines.append("")
    else:
        lines.append("No findings from any model.")
        lines.append("")

    for model, raw in raw_by_model.items():
        lines.append(f"## Appendix: {model}")
        lines.append("")
        lines.append(raw.rstrip("\n"))
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def build_panel_envelope(
    *,
    mode: str,
    provider: str,
    temperature: float,
    models: tuple[str, ...],
    merged: list[MergedFinding],
    parsed_by_model: dict[str, ParsedReview],
    results_by_model: dict[str, CallResult],
    raw_by_model: dict[str, str],
    failures: list[tuple[str, ReviewError]],
) -> dict:
    """Assemble the ``--format json`` document for a panel run.

    Same schema_version as the single-model envelope; ``model`` is null
    and ``models`` / ``found_by`` / ``per_model`` carry the panel shape.
    Raw output for a model is embedded in its per_model entry only when
    its parse failed.
    """
    findings_out = []
    for cluster in merged:
        entry = dataclasses.asdict(cluster.finding)
        entry["fingerprint"] = finding_fingerprint(cluster.finding)
        entry["found_by"] = list(cluster.found_by)
        findings_out.append(entry)

    per_model = []
    prompt_total: int | None = None
    completion_total: int | None = None
    for model in models:
        if model in parsed_by_model:
            parsed = parsed_by_model[model]
            result = results_by_model[model]
            entry = {
                "model": model,
                "error": None,
                "parse_ok": parsed.parse_ok,
                "clean": parsed.clean,
                "summary": parsed.summary,
                "findings_count": len(parsed.findings),
                "usage": (
                    {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                    }
                    if result.prompt_tokens is not None
                    or result.completion_tokens is not None
                    else None
                ),
                "truncated": result.truncated,
            }
            if not parsed.parse_ok:
                entry["raw"] = raw_by_model[model]
            if result.prompt_tokens is not None:
                prompt_total = (prompt_total or 0) + result.prompt_tokens
            if result.completion_tokens is not None:
                completion_total = (completion_total or 0) + result.completion_tokens
        else:
            err = next(e for m, e in failures if m == model)
            entry = {
                "model": model,
                "error": {
                    "category": err.category,
                    "exit_code": err.exit_code,
                    "message": str(err),
                },
                "parse_ok": None,
                "clean": None,
                "summary": None,
                "findings_count": 0,
                "usage": None,
                "truncated": None,
            }
        per_model.append(entry)

    return {
        "schema_version": 1,
        "mode": mode,
        "provider": provider,
        "model": None,
        "models": list(models),
        "temperature": temperature,
        "summary": None,
        "clean": bool(parsed_by_model)
        and all(p.clean for p in parsed_by_model.values()),
        "findings": findings_out,
        "usage": (
            {
                "prompt_tokens": prompt_total,
                "completion_tokens": completion_total,
            }
            if prompt_total is not None or completion_total is not None
            else None
        ),
        "truncated": any(r.truncated for r in results_by_model.values()),
        "parse_ok": bool(parsed_by_model)
        and all(p.parse_ok for p in parsed_by_model.values()),
        "problems": [
            f"{model}: {problem}"
            for model, parsed in parsed_by_model.items()
            for problem in parsed.problems
        ],
        "per_model": per_model,
    }
