# DeepSeek V4.1 Flash review generation

Status: accepted. Date: 2026-10-13.

## Context

DeepSeek V4.1 Flash defaults to high reasoning effort. Repeated review calls
exhausted 16,000 or 32,768 output tokens without producing a visible review.
Reasoning and review text share the completion ceiling; extra reasoning can
consume the whole request without yielding findings.

A live full-diff test with low effort and a 65,536-token ceiling completed with
45,910 completion tokens, valid structured findings, and no truncation. Low
effort alone did not finish the same review at 16,000 tokens.

## Decision

- On OpenRouter, request `reasoning.effort=low` for the exact
  `deepseek/deepseek-v4.1-flash` model family, including colon-suffixed IDs.
  Validate the request object with Pydantic v2. Other models receive no new
  reasoning configuration, preserving the Gemini generation decision.
- Default that model's total completion allowance to 65,536 only when no CLI,
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
  A partial panel must identify failed slots explicitly.

## Validation

Offline contracts cover mixed panels, explicit CLI/env/project limits,
model-specific cost ceilings and reports, legacy models, suffixed IDs, and
reasoning-only error classification. The generation module remains subject to
100% statement coverage in CI. Live validation uses public review inputs and
keeps reports outside the repository.

## References

- [DeepSeek thinking controls](https://api-docs.deepseek.com/guides/thinking_mode/)
- [OpenRouter reasoning and output ceilings](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
