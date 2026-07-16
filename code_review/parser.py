"""Structured output: the markdown findings parser and JSON envelopes.

The prompts stay byte-identical to upstream; structure is recovered by
PARSING the rigid markdown format locally (line-based state machine,
fence-aware) -- never by prompting for JSON. A parse failure degrades
to ``parse_ok: false`` + raw text; it never destroys a paid-for review.
Also home to finding fingerprints, --baseline round-diffing, and the
--min-severity post-parse enforcement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re

from code_review.chunking import split_diff_by_file
from code_review.errors import ConfigError
from code_review.prompts import SEVERITY_LEVELS
from code_review.providers import CallResult

_SUMMARY_RE = re.compile(r"^#\s+(?:Change|Codebase review)\s+summary:\s*(.*)$", re.I)
_SUMMARY_FALLBACK_RE = re.compile(r"^#\s+.*?summary\s*:?\s*(.*)$", re.I)
_FILE_RE = re.compile(r"^##\s+File:\s*(.+?)\s*$", re.I)
_FINDING_RE = re.compile(r"^###\s+(.*)$")
# Opening fences may carry an info string (```diff); closing fences are
# backticks-only with at least the opening's backtick count.
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,})(\S*)\s*$")
# `\bL\s*[+-]?(\d+)`: accepts L117, L 117, and the diff-anchored L+117 /
# L-42 forms observed in real model output (the upstream skill tells
# models to reference `+`/`-` lines, and some transcribe the marker).
_LINE_TOKEN_RE = re.compile(r"\bL\s*[+-]?(\d+)", re.I)
_BARE_LINE_RE = re.compile(r"^(\d+)\s*:")
# Lookarounds forbid letters on either side so severity words embedded in
# ordinary prose ("Below", "higher", "lowercase") never match; the token
# must stand alone (brackets optional -- some models drop them).
_SEVERITY_RE = re.compile(
    r"\[?\s*(?<![A-Za-z])(CRITICAL|HIGH|MEDIUM|LOW)(?![A-Za-z])\s*\]?", re.I
)
# Lead-in tolerates markdown emphasis: **Suggested change:** etc.
_SUGGESTION_LEADIN_RE = re.compile(
    r"^[*_]{0,2}Suggested (?:change|fix)[*_]{0,2}:?[*_]{0,2}\s*$", re.I
)
_CLEAN_RE = re.compile(r"^No issues found\.", re.M)


@dataclasses.dataclass
class Finding:
    """One review finding recovered from the model's markdown.

    ``in_hunk`` answers "can this be posted as a GitHub suggestion?" --
    True when ``line`` falls inside a changed hunk of ``file`` in the
    reviewed diff, False when it lands outside every hunk, and None when
    the question doesn't apply or can't be answered (codebase mode, no
    line, no diff, or the model's path doesn't match any diff path).
    GitHub rejects a review comment whose line is not in the served diff,
    so a caller automating suggestions needs this to know which findings
    can be one-click and which must go in the review body. Populated by
    ``annotate_in_hunk``; left None by the parser itself, which only ever
    sees the model's markdown and not the diff.
    """

    file: str | None
    line: int | None
    severity: str
    title: str
    body: str
    suggestion: str | None
    in_hunk: bool | None = None


# `+++ b/path` gives the post-image path; `/dev/null` for deletions.
_DIFF_NEW_PATH_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$", re.M)
# `@@ -old,n +new,m @@` -- only the post-image side matters: GitHub anchors
# review comments on RIGHT-side line numbers. The count is optional (`+12`
# means a one-line hunk).
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


def hunk_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Map each changed file to its post-image (RIGHT-side) hunk ranges.

    Ranges are inclusive ``(start, end)`` line numbers in the file as it
    exists after the change -- the same coordinate space GitHub uses for
    review comments, and the same one models are asked to cite.

    Used by ``annotate_in_hunk`` to tell postable findings from ones that
    must go in the review body. Deletions (``+++ /dev/null``) are skipped:
    they have no post-image to comment on.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    for part in split_diff_by_file(diff):
        m = _DIFF_NEW_PATH_RE.search(part)
        if not m:
            continue
        path = m.group(1).strip()
        if path == "/dev/null":
            continue
        spans = ranges.setdefault(path, [])
        for hm in _DIFF_HUNK_RE.finditer(part):
            start = int(hm.group(1))
            count = int(hm.group(2)) if hm.group(2) is not None else 1
            if count <= 0:
                # A pure deletion hunk (`+n,0`) has no post-image lines;
                # there is nothing on the RIGHT side to anchor to.
                continue
            spans.append((start, start + count - 1))
    return ranges


def annotate_in_hunk(
    findings: list[Finding], ranges: dict[str, list[tuple[int, int]]]
) -> None:
    """Set ``Finding.in_hunk`` in place from ``hunk_ranges`` output.

    Leaves ``in_hunk`` as None -- "unknown", not "no" -- whenever the
    question can't be answered honestly: no line cited, or the finding's
    path matches nothing in the diff. Reporting False there would tell a
    caller "not postable" when the truth is "we don't know", and the two
    warrant different handling.

    Path matching is exact-then-suffix: models cite paths inconsistently
    (``src/x.py`` vs ``x.py`` vs ``./src/x.py``), so a bare-name citation
    still resolves as long as exactly one diff path ends with it. An
    ambiguous suffix (two files with the same basename) stays None rather
    than guessing wrong.
    """
    if not ranges:
        return
    for f in findings:
        if f.line is None or not f.file:
            continue
        cited = f.file.strip().lstrip("./")
        spans = ranges.get(cited)
        if spans is None:
            matches = [p for p in ranges if p.endswith("/" + cited) or p == cited]
            if len(matches) != 1:
                continue  # unknown or ambiguous -- leave None
            spans = ranges[matches[0]]
        f.in_hunk = any(start <= f.line <= end for start, end in spans)


@dataclasses.dataclass
class ParsedReview:
    """Result of parsing a review; ``problems`` lists tolerated defects."""

    summary: str | None
    findings: list[Finding]
    clean: bool
    parse_ok: bool
    problems: list[str]


def normalize_title(title: str) -> str:
    """Normalize a finding title for fingerprinting.

    Lowercase; backticks/quotes stripped (models quote identifiers
    inconsistently run-to-run); trailing period dropped; whitespace
    collapsed. Deliberately lossy -- the goal is stability across runs
    at T=0.3, not readability.
    """
    title = title.lower().strip()
    title = title.replace("`", "").replace('"', "").replace("'", "")
    title = title.rstrip(".")
    return " ".join(title.split())


def finding_fingerprint(finding: Finding) -> str:
    """Stable 12-hex-char identity for a finding.

    The line number is deliberately EXCLUDED from the hash: line drift
    between review rounds is expected (code moves), so proximity is
    checked separately in ``findings_match``. Hashing the line would
    make every drifted finding look new.
    """
    key = f"{finding.file or ''}|{finding.severity}|{normalize_title(finding.title)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _fingerprints_match(
    fp_a: str, line_a: int | None, fp_b: str, line_b: int | None
) -> bool:
    """Fingerprint equality plus line proximity (±10).

    Either line being ``None`` matches on fingerprint alone -- real
    models emit line-less findings (observed: deepseek `### L+0:`
    headings), and refusing to match those would mark the same finding
    "new" every round.
    """
    if fp_a != fp_b:
        return False
    if line_a is None or line_b is None:
        return True
    return abs(line_a - line_b) <= 10


def findings_match(a: Finding, b: Finding) -> bool:
    """True when two findings are the same issue (see _fingerprints_match)."""
    return _fingerprints_match(
        finding_fingerprint(a), a.line, finding_fingerprint(b), b.line
    )


def _parse_finding_heading(
    heading: str,
) -> tuple[int | None, str, str, list[str]]:
    """Extract ``(line, severity, title, problems)`` from a ### heading."""
    problems: list[str] = []
    line: int | None = None
    consumed_end = 0

    line_match = _LINE_TOKEN_RE.search(heading)
    if line_match:
        line = int(line_match.group(1))
        consumed_end = line_match.end()
    else:
        bare = _BARE_LINE_RE.match(heading)
        if bare:
            line = int(bare.group(1))
            consumed_end = bare.end()
    if line == 0:
        # Diff-position artifact (observed: `### L+0:`); content lines
        # are 1-indexed, so 0 means "no usable line", not line zero.
        line = None
        problems.append(f"line 0 in heading {heading!r} mapped to None")
    elif line is None:
        problems.append(f"no line number in heading {heading!r}")

    # Search from the end of the line token: the template puts severity
    # after ``L<N>:``, and scanning only the remainder keeps a stray
    # standalone severity word later in a TITLE from being consumed when
    # the real tag was absent.
    severity_match = _SEVERITY_RE.search(heading, consumed_end)
    if severity_match:
        severity = severity_match.group(1).upper()
        consumed_end = max(consumed_end, severity_match.end())
    else:
        severity = "UNKNOWN"
        problems.append(f"no severity in heading {heading!r}")

    title = heading[consumed_end:].strip().lstrip(":]-–— ").strip()
    if not title:
        title = "(untitled)"
        problems.append(f"empty title in heading {heading!r}")

    return line, severity, title, problems


def _extract_suggestion(body_lines: list[str]) -> tuple[str, str | None]:
    """Split a finding body into ``(body, suggestion)``.

    Precedence: a fence preceded by a ``Suggested change:`` lead-in
    always wins. A ``suggestion``/``diff``-tagged fence WITHOUT a
    lead-in qualifies only when it is the LAST block of the body -- a
    tagged fence mid-body is the model quoting the reviewed hunk, not
    proposing a fix, and must stay in the body.
    """

    def _fence_span(start: int) -> tuple[int, int, str] | None:
        """Return (open_idx, close_idx, info) for a fence opening at/after
        ``start`` (skipping blank lines), or None."""
        i = start
        while i < len(body_lines) and not body_lines[i].strip():
            i += 1
        if i >= len(body_lines):
            return None
        opening = _FENCE_RE.match(body_lines[i])
        if not opening:
            return None
        ticks, info = opening.group(1), opening.group(2).lower()
        for j in range(i + 1, len(body_lines)):
            closing = _FENCE_RE.match(body_lines[j])
            if closing and len(closing.group(1)) >= len(ticks) and not closing.group(2):
                return i, j, info
        return None

    # Pass 1: explicit lead-in.
    for idx, line in enumerate(body_lines):
        if _SUGGESTION_LEADIN_RE.match(line):
            span = _fence_span(idx + 1)
            if span is None:
                continue
            open_idx, close_idx, _info = span
            suggestion = "\n".join(body_lines[open_idx + 1 : close_idx])
            remainder = body_lines[:idx] + body_lines[close_idx + 1 :]
            return _join_body(remainder), suggestion or None

    # Pass 2: trailing tagged fence.
    span = None
    scan = 0
    while scan < len(body_lines):
        found = _fence_span(scan)
        if found is None:
            scan += 1
            continue
        span = found
        scan = found[1] + 1
    if span is not None:
        open_idx, close_idx, info = span
        trailing = all(not line.strip() for line in body_lines[close_idx + 1 :])
        if info in {"suggestion", "diff"} and trailing:
            suggestion = "\n".join(body_lines[open_idx + 1 : close_idx])
            remainder = body_lines[:open_idx]
            return _join_body(remainder), suggestion or None

    return _join_body(body_lines), None


def _join_body(lines: list[str]) -> str:
    """Join body lines, trimming leading/trailing blank lines."""
    text = "\n".join(lines)
    return text.strip("\n").strip()


def parse_review_markdown(text: str) -> ParsedReview:
    """Parse the model's review markdown into structured findings.

    Line-based state machine, NOT document-level regex: fenced code
    blocks legally contain ``###`` and ``diff --git`` lines, so headers
    are only recognized outside fences (tracked with the opening fence's
    backtick count; a closing fence needs at least as many backticks and
    no info string).
    """
    problems: list[str] = []
    summary: str | None = None
    findings: list[Finding] = []
    heading_count = 0
    any_unreadable = False

    current_file: str | None = None
    open_heading: tuple[int | None, str, str] | None = None
    open_file: str | None = None
    body_lines: list[str] = []

    in_fence = False
    fence_ticks = 0

    def _close_finding() -> None:
        nonlocal open_heading
        if open_heading is None:
            return
        line, severity, title = open_heading
        body, suggestion = _extract_suggestion(body_lines)
        findings.append(
            Finding(
                file=open_file,
                line=line,
                severity=severity,
                title=title,
                body=body,
                suggestion=suggestion,
            )
        )
        open_heading = None
        body_lines.clear()

    for raw_line in text.splitlines():
        if in_fence:
            closing = _FENCE_RE.match(raw_line)
            if (
                closing
                and len(closing.group(1)) >= fence_ticks
                and not closing.group(2)
            ):
                in_fence = False
            if open_heading is not None:
                body_lines.append(raw_line)
            continue

        fence = _FENCE_RE.match(raw_line)
        if fence:
            in_fence = True
            fence_ticks = len(fence.group(1))
            if open_heading is not None:
                body_lines.append(raw_line)
            continue

        summary_match = _SUMMARY_RE.match(raw_line) or _SUMMARY_FALLBACK_RE.match(
            raw_line
        )
        if summary_match and summary is None:
            _close_finding()
            summary = summary_match.group(1).strip() or None
            continue

        file_match = _FILE_RE.match(raw_line)
        if file_match:
            _close_finding()
            path = file_match.group(1).strip().strip("`\"'")
            # Normalize Windows separators so fingerprints are stable
            # across platforms and across model quoting styles.
            current_file = path.replace("\\", "/")
            continue

        finding_match = _FINDING_RE.match(raw_line)
        if finding_match:
            _close_finding()
            heading_count += 1
            heading = finding_match.group(1).strip()
            line, severity, title, heading_problems = _parse_finding_heading(heading)
            problems.extend(heading_problems)
            if line is None and severity == "UNKNOWN":
                any_unreadable = True
            if current_file is None:
                problems.append(
                    f"finding {heading!r} appeared before any '## File:' header"
                )
            open_heading = (line, severity, title)
            open_file = current_file
            continue

        if open_heading is not None:
            body_lines.append(raw_line)

    _close_finding()

    if summary is None:
        problems.append("no summary heading found")

    clean = not findings and bool(_CLEAN_RE.search(text))
    # Summary-only output (no finding headings, no clean marker) is NOT
    # parse_ok: the template mandates findings or the literal clean
    # phrase, so a bare summary means the model drifted (bullets instead
    # of ### headings, truncation before the first finding). Reporting
    # it as a confident zero-finding review would let --baseline mark
    # everything "resolved" and let agents treat the run as clean.
    parse_ok = clean or bool(findings)
    if summary is not None and heading_count == 0 and not clean and not findings:
        problems.append(
            "summary present but no finding headings and no clean marker; "
            "the findings may be in an unparseable shape (see raw)"
        )
    if any_unreadable:
        parse_ok = False

    return ParsedReview(
        summary=summary,
        findings=findings,
        clean=clean,
        parse_ok=parse_ok,
        problems=problems,
    )


def parse_review_markdown_safe(text: str) -> ParsedReview:
    """``parse_review_markdown`` that never raises.

    A parser bug must not destroy a paid-for review: any unexpected
    exception degrades to ``parse_ok=False`` (the JSON envelope then
    carries the raw markdown) instead of an ``ERROR: UNKNOWN`` exit.
    """
    try:
        return parse_review_markdown(text)
    except Exception as exc:
        return ParsedReview(
            summary=None,
            findings=[],
            clean=False,
            parse_ok=False,
            problems=[f"parser crashed: {type(exc).__name__}: {exc}"],
        )


def load_baseline(path: str) -> dict:
    """Load and validate a ``--baseline`` JSON file (a prior run's
    ``--format json`` output). Failures are the user's config, hence
    ConfigError -- and this runs BEFORE the model call so a bad baseline
    never burns tokens."""
    try:
        # utf-8-sig for the same Windows-editor-BOM reason as
        # _load_project_config; our own --output files never have one.
        with open(path, encoding="utf-8-sig") as fh:
            doc = json.load(fh)
    except OSError as exc:
        raise ConfigError(f"Cannot read --baseline file {path!r}: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"--baseline file {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise ConfigError(
            f"--baseline file {path!r} is not a schema_version=1 review "
            "envelope (expected the output of a --format json run)."
        )
    findings = doc.get("findings")
    if findings is not None and not isinstance(findings, list):
        # A hand-mangled findings value would TypeError deep inside
        # diff_against_baseline -> UNKNOWN; catch it here, pre-spend.
        raise ConfigError(
            f"--baseline file {path!r} has a non-list `findings` value "
            f"({type(findings).__name__}); expected the findings array "
            "of a --format json envelope."
        )
    return doc


def _location_match(
    file_a: str | None, line_a: int | None, file_b: str | None, line_b: int | None
) -> bool:
    """Same file and lines within ±10 (a missing line matches any line)."""
    if file_a != file_b:
        return False
    if line_a is None or line_b is None:
        return True
    return abs(line_a - line_b) <= 10


def diff_against_baseline(
    current: list[Finding], baseline_doc: dict
) -> tuple[list[str], list[dict]]:
    """Compare current findings against a prior run's envelope.

    Returns ``(statuses, resolved)``: one ``"new"``/``"persisting"``
    status per current finding, plus the baseline entries nothing
    matched -- informational "resolved since last round".

    Two greedy passes, each consuming baseline entries so one entry
    can't vouch for two current findings:

    1. **Strong**: fingerprint (file+severity+normalized title) equal,
       lines within ±10.
    2. **Relaxed**: same file, lines within ±10, and BOTH sides must
       carry a real line number -- title and severity
       ignored. Empirically required: on back-to-back identical runs at
       T=0.3 the model rewords every title ("add a comment explaining
       the default" vs "add a comment to clarify its purpose") and even
       re-rates severities (HIGH -> MEDIUM on the same function), so a
       title-hashed fingerprint alone matched 0 of 8 genuinely repeated
       findings. Location survives rewording; it carries the match.

    Still heuristic: two *different* findings within 10 lines of each
    other in the same file can cross-match in pass 2. Greedy consumption
    bounds the damage to one mislabeled status per collision.
    """
    baseline_entries = [
        entry
        for entry in baseline_doc.get("findings") or []
        if isinstance(entry, dict) and isinstance(entry.get("fingerprint"), str)
    ]

    def _entry_line(entry: dict) -> int | None:
        line = entry.get("line")
        return line if isinstance(line, int) else None

    def _entry_file(entry: dict) -> str | None:
        file = entry.get("file")
        return file if isinstance(file, str) else None

    unmatched = list(baseline_entries)
    statuses: list[str | None] = [None] * len(current)

    # Pass 1: strong fingerprint match.
    for idx, finding in enumerate(current):
        fp = finding_fingerprint(finding)
        for entry_idx, entry in enumerate(unmatched):
            if _fingerprints_match(
                fp, finding.line, entry["fingerprint"], _entry_line(entry)
            ):
                unmatched.pop(entry_idx)
                statuses[idx] = "persisting"
                break

    # Pass 2: relaxed location match for whatever pass 1 left over.
    # Both sides must carry a REAL line here: _location_match treats a
    # missing line as matching anything, which is right for the
    # fingerprint-backed pass but would let one line-less baseline entry
    # vouch for any unrelated finding in the same file when title and
    # severity are already being ignored. Line-less findings can still
    # persist via pass 1's fingerprint.
    for idx, finding in enumerate(current):
        if statuses[idx] is not None or finding.line is None:
            continue
        for entry_idx, entry in enumerate(unmatched):
            if _entry_line(entry) is None:
                continue
            if _location_match(
                finding.file, finding.line, _entry_file(entry), _entry_line(entry)
            ):
                unmatched.pop(entry_idx)
                statuses[idx] = "persisting"
                break

    return [s or "new" for s in statuses], unmatched


def _severity_at_or_above(severity: str, floor: str) -> bool:
    """True when ``severity`` meets the floor.

    Severities the parser couldn't rate (``UNKNOWN``) always pass:
    hiding a finding whose severity we don't know would be worse than
    showing it.
    """
    if severity not in SEVERITY_LEVELS:
        return True
    return SEVERITY_LEVELS.index(severity) >= SEVERITY_LEVELS.index(floor)


def enforce_min_severity(parsed: ParsedReview, floor: str) -> ParsedReview:
    """Drop parsed findings below the ``--min-severity`` floor.

    The ``<SEVERITY_FILTER>`` prompt appendix ASKS the model not to
    report below-floor findings, but a prompt is a request, not a
    guarantee. Wherever the runner synthesizes structured findings
    (--format json envelopes, panel reports) the floor is enforced here
    after parsing, making the flag a hard contract for agent callers.
    Verbatim markdown output is deliberately left untouched -- filtering
    it would require rewriting the model's own text.

    ``clean`` is not recomputed: it reports what the model said, and
    "not clean, but nothing at or above your floor" is exactly what an
    empty ``findings`` list next to ``clean: false`` means.
    """
    if floor == "LOW":
        return parsed
    kept = [f for f in parsed.findings if _severity_at_or_above(f.severity, floor)]
    if len(kept) == len(parsed.findings):
        return parsed
    return dataclasses.replace(parsed, findings=kept)


def filter_baseline_findings(baseline_doc: dict, floor: str) -> dict:
    """Apply the severity floor to a baseline document's findings.

    Without this, running with a higher floor than the baseline round
    would report every below-floor baseline entry as ``resolved`` when
    it was merely filtered, not fixed. Entries whose severity is
    missing or malformed are kept, mirroring ``_severity_at_or_above``.
    """
    if floor == "LOW":
        return baseline_doc
    entries = baseline_doc.get("findings") or []
    kept = [
        entry
        for entry in entries
        if not isinstance(entry, dict)
        or not isinstance(entry.get("severity"), str)
        or _severity_at_or_above(entry["severity"], floor)
    ]
    filtered = dict(baseline_doc)
    filtered["findings"] = kept
    return filtered


def build_json_envelope(
    *,
    mode: str,
    provider: str,
    model: str,
    temperature: float,
    parsed: ParsedReview,
    result: CallResult,
    raw_markdown: str,
    statuses: list[str] | None = None,
    resolved: list[dict] | None = None,
) -> dict:
    """Assemble the ``--format json`` stdout document (schema_version 1).

    ``raw`` is embedded only when parsing failed, so a caller always has
    the full model output one way or the other. Exit code stays 0 on
    parse failure -- exit codes describe transport/config outcomes, not
    model formatting; agents branch on ``parse_ok``.
    """
    findings_out = []
    for idx, finding in enumerate(parsed.findings):
        entry = dataclasses.asdict(finding)
        entry["fingerprint"] = finding_fingerprint(finding)
        if statuses is not None:
            entry["status"] = statuses[idx]
        findings_out.append(entry)
    envelope: dict = {
        "schema_version": 1,
        "mode": mode,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "summary": parsed.summary,
        "clean": parsed.clean,
        "findings": findings_out,
        "usage": (
            {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
            if result.prompt_tokens is not None or result.completion_tokens is not None
            else None
        ),
        "truncated": result.truncated,
        "parse_ok": parsed.parse_ok,
        "problems": parsed.problems,
    }
    if resolved is not None:
        envelope["resolved"] = resolved
    if not parsed.parse_ok:
        envelope["raw"] = raw_markdown
    return envelope


def build_chunked_envelope(
    *,
    mode: str,
    provider: str,
    model: str,
    temperature: float,
    chunk_data: list[tuple[str, ParsedReview, CallResult, str]],
) -> dict:
    """Assemble the ``--format json`` document for a ``--chunk`` run.

    ``chunk_data`` is ``(label, parsed, result, raw)`` per chunk in
    execution order. Findings are concatenated (chunks are disjoint
    content, so no dedup is needed) with a ``chunk`` index added; raw
    output embeds per-chunk only when that chunk's parse failed.
    """
    findings_out = []
    per_chunk = []
    prompt_total: int | None = None
    completion_total: int | None = None
    for idx, (label, parsed, result, raw) in enumerate(chunk_data, start=1):
        for finding in parsed.findings:
            entry = dataclasses.asdict(finding)
            entry["fingerprint"] = finding_fingerprint(finding)
            entry["chunk"] = idx
            findings_out.append(entry)
        chunk_entry = {
            "chunk": idx,
            "label": label,
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
            chunk_entry["raw"] = raw
        per_chunk.append(chunk_entry)
        if result.prompt_tokens is not None:
            prompt_total = (prompt_total or 0) + result.prompt_tokens
        if result.completion_tokens is not None:
            completion_total = (completion_total or 0) + result.completion_tokens

    return {
        "schema_version": 1,
        "mode": mode,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "chunks": len(chunk_data),
        "summary": None,
        "clean": all(parsed.clean for _, parsed, _r, _raw in chunk_data),
        "findings": findings_out,
        "usage": (
            {
                "prompt_tokens": prompt_total,
                "completion_tokens": completion_total,
            }
            if prompt_total is not None or completion_total is not None
            else None
        ),
        "truncated": any(r.truncated for _, _p, r, _raw in chunk_data),
        "parse_ok": all(parsed.parse_ok for _, parsed, _r, _raw in chunk_data),
        "problems": [
            f"chunk {idx}: {problem}"
            for idx, (_label, parsed, _r, _raw) in enumerate(chunk_data, start=1)
            for problem in parsed.problems
        ],
        "per_chunk": per_chunk,
    }


# ---------------------------------------------------------------------------
# Multi-model panel (--models)
# ---------------------------------------------------------------------------
#
# Runs the same prompt through several models and merges the parsed
# findings into one consensus-annotated report. Motivated by the PR #2
# dogfood run: on the same real diff one model returned clean, one
# returned only hallucinations, and one had a single real bug in 28
# findings -- cross-model agreement is the strongest cheap filter for
# plausible-but-wrong findings. Empirically ``found_by`` of 1 is the
# norm; consensus (>1) is a rare, high-precision signal.

# Severity rank for merged-finding ordering; UNKNOWN sorts below LOW.
