# Architecture

A one-page map of how a review request flows through the runner, the design decisions that shaped it, and where things live. Written for contributors — usage lives in the [README](../README.md), day-to-day operations in the [runbook](./llm-code-review-runbook.md).

## Data flow

```
            CLI flags ──────────────┐
   process env + .env files ────────┤  trusted
                                    ├──> _resolve_settings() ──> frozen Settings
 .code-review.toml (REVIEWED repo, ─┘  untrusted: capped keys,      (validated before any
  upward walk, load announced)         WARN-and-drop, skippable      git or network I/O)
                                       via --no-project-config

  source modes (mutually exclusive)
  --base / --staged / (default merge-base) ──> git diff -U5 ─┐
  --pr N ──> gh pr diff ─────────────────────────────────────┤──> diff text
  --diff-file PATH|- ──> file/stdin ─────────────────────────┘      │
  --codebase ──> git ls-files + filters ──> numbered file bundle    │
                                                                    ▼
  prompt assembly (_build_request)
    upstream prompts, byte-identical (SKILL.md + commands/*.toml via _prompt_root)
    + fork-owned RUNTIME appendices, never edits to upstream files:
        <CONTEXT_FOR_REVIEWER>  safety context + prompt-injection guard
        <REFERENCE_FILES>       --full-files changed-file bodies (context only)
        <SEVERITY_FILTER>       --min-severity request to the model
    --chunk splits at file boundaries BEFORE this step (one request per chunk)
                                                                    ▼
  _execute_call  (also the thread target for --models panels)
    provider dispatch ──> call_openrouter | call_gemini | call_ollama
    every HTTP client built by _make_client()   ← single seam; wire tests
    (and the Ollama /api/ps probe)                mock ALL transport here
    retries: _call_with_retries — built-in single 2s retry on hiccup/transport;
             --retries N adds backoff + Retry-After-honoring 429 retries (clamped 300s)
    HTTP errors: _classify_http_error ──> typed ReviewError subclass ──> exit code
    ollama only: 3-tier window resolution ($OLLAMA_NUM_CTX → /api/ps → advisory),
                 pre-flight size guard, post-call prompt_eval_count truncation verify
                                                                    ▼
  output
    markdown (default): model text verbatim on stdout
    --format json: parse_review_markdown (line-based state machine; fence-aware;
      crash-proof — parse failure ⇒ parse_ok:false + raw embedded, still exit 0)
      ──> enforce_min_severity ──> envelope (schema_version 1)
    --baseline: two-pass fingerprint+location diff ──> new/persisting/resolved
    --models:  consensus merge (found_by) + per-model appendix/errors
    --chunk:   per-chunk banners (markdown) or one concatenated envelope (json)
    --output PATH: exact stdout bytes also written to a file
```

## Design decisions

**Deterministic primitive, not an agent.** One request per model per chunk; no tool-calling, no retry-with-reworded-prompt, no auto-fix. Model output is printed or parsed — never executed. Complex behavior (iterate-until-clean loops, panels of panels) belongs in the *caller*, which is usually an LLM agent following the runbook. See the README's Non-goals.

**Parse, don't prompt.** Structured output comes from parsing the rigid upstream markdown format locally, never from asking the model for JSON. The upstream prompts stay byte-identical (clean upstream merges, unchanged model behavior), and a model that drifts from the format degrades to `parse_ok: false` + raw text instead of a hard failure — a paid-for review is never destroyed by a formatting quirk.

**The error contract is public API.** Exit codes (0/1/2/10–14/130) and the `ERROR: <CATEGORY> [exit <N>]` stderr first line are what agent callers branch on, so they're pinned by tests (`TestErrorModelContract`) and documented as tables. Two invariants follow: no new stderr line may ever start with `ERROR:` (informational lines use the documented prefixes: `WARN:`, `[usage]`, `[retry]`, `[ollama]`, `[config]`, `[baseline]`, `[panel …]`, `[chunk …]`, `skip`), and any classification change is a documented, changelogged event.

**Trust boundaries.** CLI/env/user-`.env` are trusted; the reviewed repo is not — its code gets an injection guard, and its `.code-review.toml` is capped to review-shaping keys (no credentials, no prompt context, no Ollama endpoint). Details in [SECURITY.md](../SECURITY.md).

**Fail fast, before spending.** Settings resolve and validate before any git or network activity; panel aliases resolve pre-flight; baselines parse before the model call; the Ollama window guard runs pre-flight where the window is known. Typed `CONFIG` errors (exit 2) mean "fix and re-run", never "retry".

**Windows is a first-class target.** Explicit `encoding="utf-8"` on every file/subprocess boundary (cp1252 mojibake in diffs produced *phantom review findings* in early testing), `utf-8-sig` where Windows editors write BOMs, stdout reconfigured to UTF-8. CI runs the suite on windows-latest for this reason.

**`cli.py` is the stable import path.** The runner was a single module through the M1–M7 stack (kept whole so refactors wouldn't invalidate open PRs) and was split afterward. `code_review/cli.py` re-exports the full surface, so `from code_review.cli import X` keeps working for tests, the `review.py` shim, and any downstream scripts; the console entry point stays `code_review.cli:_entrypoint`. Tests monkeypatch the *owning* module (patching a re-export in `cli` doesn't affect callers inside the sibling modules).

## Code map

The import graph is acyclic, top to bottom:

| Path | What it is |
|---|---|
| `code_review/errors.py` | The typed error model: exception hierarchy → exit codes, and the `ERROR:` stderr block renderer. No project imports. |
| `code_review/prompts.py` | Prompt assets and assembly: upstream skill/command loading (`_prompt_root` tiers), the safety-context + injection-guard wrapper, severity ladder + `--min-severity` appendix, payload caps, file bundling/numbering, the `<REFERENCE_FILES>` block. |
| `code_review/providers.py` | The wire layer: `_make_client` seam, the three `call_*` functions, HTTP-error classification, retry policy, model aliases/defaults, and the Ollama window machinery (probe → guard → post-call verify). |
| `code_review/sources.py` | What gets reviewed: git/gh subprocess plumbing (repo-pinned PR calls + the `[gh]` announce), `--diff-file`/stdin, codebase gathering/filtering, subdirectory path re-basing. |
| `code_review/parser.py` | Structured output: the markdown findings parser (fence-aware state machine), fingerprints, `--baseline` diffing, severity enforcement, and the single/chunked JSON envelopes. |
| `code_review/panel.py` | Panel pure logic: consensus matching/merge, the panel markdown report and envelope, all-failed exit precedence. |
| `code_review/chunking.py` | `--chunk`: lossless file-boundary splitting, next-fit packing, and the per-chunk budget math (bundle cap / enforced Ollama window). |
| `code_review/config.py` | Settings resolution: `.env` layering, the untrusted `.code-review.toml` (capped keys), `_layered` precedence, model resolution, the frozen `Settings`/`ReviewRequest` shapes. |
| `code_review/cli.py` | Orchestration: argparse, `_build_request(s)`, `_execute_call` (provider dispatch + per-call window resolution), panel/chunk run loops, `main()`/`_entrypoint()` — plus the compatibility facade re-exporting the sibling modules' surface. |
| `review.py` | 11-line back-compat shim so `uv run review.py` keeps working from a checkout. Not an API surface. |
| `code_review/__init__.py` | `__version__` from package metadata. |
| `skills/`, `commands/` | Prompt assets at the repo root (upstream files byte-identical; `code-review-codebase` is fork-added). Force-included into the wheel as package data. |
| `tests/` | Offline suite: `test_review.py` (pure logic + error contract), `test_parser.py` (+ frozen real-model outputs under `tests/fixtures/`), `test_panel.py`, `test_biginput.py`, `test_config.py`, `test_wire.py` (canned HTTP through `_make_client`). |
| `evals/` | Paid, manual scoring harness: planted-bug fixtures + a clean-diff control, recall/noise table. |
| `.github/workflows/` | `ci.yml` (lint + type-check + tests on ubuntu/windows + wheel-content check), `evals.yml` (manual dispatch, spends tokens). |

## Testing strategy

Four layers, cheapest first:

1. **Pure-logic unit tests** pin the contracts: error classification, alias resolution, parser grammar, fingerprint matching, config precedence, flag exclusions.
2. **Wire tests** (`httpx.MockTransport` through the `_make_client` seam) cover every provider's response shapes — safety blocks, truncation, 429/Retry-After, auth failures, malformed JSON — with zero network.
3. **Frozen real-model fixtures** keep the parser honest against actual output drift (three models' verbatim reviews of a real PR, including the pathological heading shapes that motivated the grammar).
4. **Evals** (paid, manual) measure *behavior*: recall on planted bugs, noise on clean code. Run them for any change to prompts, temperature defaults, or models — tuning debates get settled with the table, not anecdotes.
