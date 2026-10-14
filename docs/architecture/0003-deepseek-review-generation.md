# DeepSeek V4.1 Flash review generation

Status: accepted. Date: 2026-10-13. Updated: 2026-10-14.

## Context

DeepSeek V4.1 Flash defaults to high reasoning effort. Repeated review calls
exhausted 16,000 or 32,768 output tokens without producing a visible review.
Reasoning and review text share the completion ceiling; extra reasoning can
consume the whole request without yielding findings.

A live full-diff test with low effort and a 65,536-token ceiling completed with
45,910 completion tokens, valid structured findings, and no truncation. Low
effort alone did not finish the same review at 16,000 tokens.

Further full-diff reviews exhausted all 65,536 tokens at low effort, both with
and without full-file attachments. Other runs completed under the same settings.
Low effort reduces thinking intensity but does not reserve tokens for a review.
The larger allowance therefore failed to provide consistent completion. The
model supports non-thinking mode, and OpenRouter exposes the explicit
`reasoning.enabled=false` switch.

## Decision

- On OpenRouter, request `reasoning.enabled=false` for the exact
  `deepseek/deepseek-v4.1-flash` model family, including colon-suffixed IDs.
  This replaces the initial low-effort policy. Do not combine disabled thinking
  with an effort or reasoning budget that could re-enable it. Validate the
  request object with Pydantic v2. Other models receive no new reasoning
  configuration, preserving the Gemini generation decision.
- Retain that model's total completion allowance of 65,536 only when no CLI,
  environment, or project cap was provided. Keep other models at their existing
  defaults, including in mixed panels. Never silently raise an explicit cap.
- Centralize per-model allowance resolution. Requests, cost estimates,
  reference-context guards, progress messages, and dry-run reports use the same
  value. Cost estimates sum each model's actual allowance and display the
  largest per-model ceiling for panels.
- Keep standalone API callers' supplied limits exact. Keep provider selection,
  account policies, credentials, output schemas, and exit codes unchanged.
- Never count reasoning-only output as a review. After failure at the recommended
  profile, narrow or chunk the input instead of repeatedly escalating budgets.
  Do not automatically retry, increase the cap, or switch models after output
  exhaustion. A partial panel must identify failed slots explicitly.

## Consequences

The same model now generates reviews without its optional thinking phase.
Finding quality may differ from thinking mode; completion alone is not a quality
guarantee. Validate behavioral changes using planted-bug and clean-control evals,
and check individual findings against source. Dry runs and live progress make
the disabled setting explicit, allowing agents to detect an outdated runner.

## Validation

Offline contracts cover mixed panels, explicit CLI/env/project limits,
model-specific cost ceilings and reports, legacy models, suffixed IDs, disabled
thinking payloads, unchanged settings for other models, and reasoning-only error
classification without automatic retries. The generation module remains subject
to 100% statement coverage in CI. Live validation uses public review inputs and
keeps reports outside the repository.

On 2026-10-14, non-thinking mode completed all four eval fixtures with successful
parsing: all three planted bugs were found and the clean refactor had no findings.
The planted-bug fixtures also produced five additional findings under the eval
scorer. This small check establishes completion and basic recall, not parity
with thinking-mode review quality.

The same 66,559-character public diff used for the earlier low-effort validation
also completed with thinking disabled: 1,912 completion tokens, successful
parsing, and no truncation. Reports remain outside version control.

## References

- [DeepSeek thinking controls](https://api-docs.deepseek.com/guides/thinking_mode/)
- [OpenRouter reasoning and output ceilings](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
