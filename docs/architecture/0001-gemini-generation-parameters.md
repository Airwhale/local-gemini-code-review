# Gemini generation parameters

Status: Accepted, 2026-10-06.

## Context

Google is removing custom sampling controls from upcoming Gemini models.
This runner sent `temperature` through both its direct Gemini adapter and
OpenRouter. It has never sent a thinking budget, top-p, or top-k.

## Decision

- Omit sampling parameters for the direct Gemini provider and OpenRouter
  model IDs beginning with `google/gemini-`, including suffixed IDs.
- Omit thinking configuration to use the selected model's default. This
  also avoids sending `thinkingLevel` to Gemini 2.5 on `generateContent`,
  where that field is unsupported. A future explicit thinking option must
  validate the selected model's supported levels.
- Keep the supported `generateContent` endpoint. Use `gemini-3.8-flash` as
  the direct provider default because Gemini 2.5 access is restricted for
  new users. Keep the working OpenRouter default and aliases.
- Validate the affected cloud request payloads with Pydantic v2 and reject
  unknown fields. Serialize optional values with `exclude_none=True` so
  omitted controls never become explicit JSON nulls.
- Keep the existing temperature argument, CLI/config settings, and numeric
  JSON envelope field for compatibility. Reports identify when Gemini uses
  defaults. Other providers' temperature behavior remains available.

## Consequences

Gemini 2.5 reviews now also use default sampling, so review findings can
change. Temperature sweeps are meaningful only for non-Gemini models.
The envelope's `temperature` records configuration, not an effective Gemini
value. HTTP contract tests cover direct requests and mixed OpenRouter panels.

## References

- [Google migration checklist](https://ai.google.dev/gemini-api/docs/latest-model)
- [Thinking controls for generateContent](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
- [Interactions API status](https://ai.google.dev/gemini-api/docs/interactions-overview)
