# Changelog

Notable user-facing changes, newest first. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); the version is the wheel version in `pyproject.toml`. The exit-code table and `ERROR:` stderr contract are public API — breaking changes to them will always be called out here explicitly.

## [0.2.0] — unreleased (PR stack #3–#9)

The M1–M7 roadmap: the single-file runner became an installable, testable, agent-friendly tool.

### Added

- `uv tool install git+…` packaging: `code-review` console command, prompt assets shipped inside the wheel, layered `.env` resolution (`$CODE_REVIEW_ENV` → per-user config dir → checkout root). `review.py` remains as a back-compat shim.
- `--format json`: the model's markdown is parsed locally into a structured findings envelope (`schema_version: 1`) — prompts unchanged, never prompted for JSON. Parse failure embeds the raw output and still exits 0 (`parse_ok: false`).
- `--baseline <prior.json>`: findings labeled `new`/`persisting` plus a `resolved` list, via two-pass fingerprint + location matching (titles reword between runs; location carries the match).
- `--models a,b,c` panels: same review through several models, consensus-merged (`found_by`), concurrent on cloud / sequential on Ollama, documented exit precedence when all models fail. Motivated by dogfood data: zero cross-model overlap on real hallucinations.
- `--full-files` (changed files' full content as reference context) and `--chunk` (file-boundary splitting for over-budget payloads, fail-fast, `[chunk i/n]` progress).
- `--diff-file PATH|-`: review a prepared unified diff (pipelines, replay, the eval harness).
- `--dry-run`, `--output`, `--retries` (Retry-After-aware, clamped), `--min-severity`, `--no-project-config`, `--version`, `[usage]` token stderr line.
- `--repo owner/name` pins the repository for `--pr` (bare PR numbers resolve through gh's default-repo logic, which on forks often points at upstream); every `--pr` run announces the resolved PR URL on stderr (`[gh] reviewing PR #N: <url>`).
- Per-project `.code-review.toml` (upward walk, announced on load, unknown keys dropped with a WARN).
- Ollama native `/api/chat` with a context-window truncation guard: 3-tier window resolution (`$OLLAMA_NUM_CTX` → `/api/ps` detection → advisory), per-request `num_ctx`, and a post-call `prompt_eval_count` verify that caught a real silent truncation in live testing.
- Quality infra: 286-test suite including `httpx.MockTransport` wire tests and frozen real-model parser fixtures; ruff + mypy (source *and* tests/evals) gating CI on ubuntu + windows; wheel-content verification; an eval harness with planted-bug fixtures **and a clean-diff control** scoring recall and hallucination noise.
- Docs: rewritten README (contract tables for agent callers), operations runbook, architecture overview, SECURITY.md threat model, issue/PR templates.

### Changed

- **Ollama endpoint moved** from the OpenAI-compat path to native `/api/chat`; usage comes from `prompt_eval_count`/`eval_count`.
- `--min-severity` is now **enforced post-parse** in JSON envelopes, panel reports, and chunked envelopes (it was prompt-level only, which the model could ignore). Verbatim markdown output remains best-effort.
- HTTP 401/403 from providers are now typed `CONFIG` errors naming the key env var (previously fell through to `UNKNOWN [exit 1]`); OpenRouter moderation-flagged 403s map to `SAFETY_REFUSAL`.
- `--full-files --pr` now **requires** the local HEAD to be the PR head (typed `CONFIG` error suggesting `gh pr checkout N`); previously a WARN allowed silently mismatched reference content.

### Security

- `.code-review.toml` (which lives in the possibly-untrusted *reviewed* repo) can no longer set the prompt `context` (self-review injection), `ollama_host` (diff exfiltration), or `ollama_num_ctx`/`ollama_timeout`; `--no-project-config` skips the file entirely.
- Prompt-injection guard appended to every request: instructions inside reviewed code are data, and manipulation attempts get flagged.
- Provider `Retry-After` sleep values clamped to 300s.

### Fixed

- Severity regex no longer matches substrings ("Below" → `LOW`); panel consensus no longer merges line-less findings by location; `$CODE_REVIEW_PROVIDER` values are validated like CLI ones; `Retry-After` parsing survives overflow/malformed headers; non-object provider JSON is typed `PROVIDER_HICCUP` — including *nested* malformed shapes (non-dict choices/message/candidates/content, non-string content), which previously escaped as raw `AttributeError` → `UNKNOWN`; panel `--dry-run` probes every model's window; eval harness exits non-zero when any run errors and its scoring honors each fixture's `line_range` (right-file/right-keyword/wrong-location no longer counts as caught); missing/partial prompt-asset dirs and non-string command `prompt` values are typed `CONFIG` instead of raw tracebacks; BOM-prefixed config files parse on Windows.

## [0.1.0] — 2026-07-06 (PR #2)

Fork baseline hardening: the standalone three-provider runner (OpenRouter / Gemini API / local Ollama) with the typed error model (exit codes 0/1/2/10–14, `ERROR: <CATEGORY> [exit <N>]` stderr contract), safety-context wrapper, Windows-first UTF-8 handling, the first test suite, and the LLM-agent runbook. Consolidated the fork's early branches into `main`.

## Fork point

Forked from [gemini-cli-extensions/code-review](https://github.com/gemini-cli-extensions/code-review) (Apache-2.0). The upstream prompt files ship byte-identical; see the README's *Fork provenance* section for the full delta.
