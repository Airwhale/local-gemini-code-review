# Evidence required for review conclusions

Status: Accepted, 2026-10-06.

## Context

A truncated response can still contain parseable findings. Treating absent
baseline findings as resolved then mistakes incomplete output for evidence
of a fix. Similarly, shared location and severity do not establish that two
models identified the same issue.

## Decision

- Keep `new` and `persisting` statuses for parseable findings in truncated
  responses, but omit `resolved` from JSON and report resolution as unknown
  on stderr. Enforce the omission in the envelope builder as well as the CLI.
- Require the existing fingerprint match for panel consensus: file,
  severity, and normalized title must agree; when both findings carry line
  numbers, they must be within 10 lines. A missing line remains compatible
  only when the fingerprint matches.
- Keep baseline's separately documented relaxed matching policy unchanged.
- Test these conclusions through the CLI with fixed model responses and
  validate JSON output against the published schema.

## Consequences

Schema version 1 is retained because `resolved` is already optional.
Consumers must use `truncated` to distinguish incomplete output from a
complete review. Complete baseline runs retain their previous behavior.

Panels favor precision over recall: differently worded titles for the same
issue may remain separate. The Markdown appendix retains every model's
raw response for inspection, including findings excluded by the consensus
threshold. JSON retains per-model counts and embeds raw output only when
that model's response could not be parsed.
