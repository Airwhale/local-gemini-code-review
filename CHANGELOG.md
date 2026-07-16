# Changelog

Notable user-facing changes, newest first. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); the version is the wheel version in `pyproject.toml`. The exit-code table and `ERROR:` stderr contract are public API — breaking changes to them will always be called out here explicitly.

## [Unreleased]

Precision pass, aimed at the tool's dominant failure mode — a model asserting things about code it cannot see. In diff mode it sees hunks plus ~3 lines of context and fills the rest in from imagination: "`X` is undefined" when `X` is defined 100 lines up, "add the missing docstring" when one exists off-hunk, a suggested fix byte-identical to the code it claims to repair. Observed across ~20 real review runs.

The changes fall into three groups: **remove the blind spot** (auto full-file context, and an evidence-discipline rule for when the files don't fit), **make findings actionable** (`in_hunk`, `needs_verification`, `--min-found-by`, a pre-call cost ceiling), and **remove known traps** (base-drift warning, diff-timestamp parsing, panel discoverability, `--list-models`, a call heartbeat).

### Added

- **`--min-found-by N`** (panel mode): drop merged findings fewer than N models reported. The tool already knew cross-model agreement was "the strongest cheap filter for plausible-but-wrong findings" — this makes it a flag instead of a post-processing chore. Applied after merging, to both markdown and JSON, with a stderr count of what was dropped; the per-model raw appendix still shows everything. Requires `--models` (a consensus count for one model is meaningless — asking is a typed `CONFIG` error). Env `CODE_REVIEW_MIN_FOUND_BY`; also a `.code-review.toml` key.
- **Pre-call cost ceiling**: a `[cost] est ~$0.17 -- ceiling: prompt ~7,388 tok + completion <= 16,000 tok` line next to "Reviewing…", and `est_cost` in `--dry-run`. Honest by construction: the prompt side is estimated (chars/4) and the completion side is *bounded* by `--max-tokens`, so it's a ceiling, not a prediction; panels sum every model. Prices come from OpenRouter's live feed (cached a day in the user config dir) rather than a hardcoded table that would silently go stale — and when pricing can't be sourced (Gemini has no unauthenticated feed, unknown slug, feed unreachable) the line is **omitted rather than guessed**, matching the existing "never estimate usage" rule. Ollama reports `$0.00 (local)`. This matters more now that auto full-file context makes payloads several times larger than the bare diff.
- **`needs_verification` on every finding**: the evidence rule asks models to prefix a title `NEEDS-VERIFICATION:` when a finding rests on unseen code; the parser now lifts that into a boolean and strips the marker from `title`. Callers triage on a field instead of string-matching titles, and fingerprints stay stable whether or not the model hedged (so `--baseline` doesn't report a hedged re-run as a new finding). Additive to the schema.
- **`--list-models`**: print model aliases + per-provider defaults and exit. Deliberately offline — the friction was "what do I type", which is local data, not "what are OpenRouter's 300 models".
- **Elapsed-time heartbeat** on long calls (`... still waiting (45s elapsed)`), so a slow model isn't indistinguishable from a hang. TTY-only: piped stderr stays byte-for-byte identical, because it's a documented contract for agent callers.
- **Evidence-discipline rule in diff prompts.** The model is now told what it *cannot* see, that absence from a hunk is not absence from the file, to prefix `NEEDS-VERIFICATION:` on claims that depend on unshown code, and — the cheapest high-yield rule — to drop any suggestion identical to the code already on screen. The rule flips when `--full-files` attaches the bodies (claiming a visible file is hidden would suppress correct findings), and rides *outside* the safety wrapper so `--no-context` can't strip it.
- **`in_hunk` on every finding** (`--format json`, diff modes): `true` = inside a changed hunk (postable as a one-click GitHub ```suggestion```), `false` = outside every hunk (must go in the review body — GitHub 422s the alternative), `null` = unknown/not applicable (codebase mode, no line, or an unmatched/ambiguous path). `null` means "verify", not "no". Additive to the envelope schema; existing consumers are unaffected.
- **Base-drift warning for `--base`.** `git diff <base>` is two-dot, so once the base moves ahead, its commits appear in the diff as *removals* and get reported as intentional deletions/reverts. The runner now counts `HEAD..<base>` and warns, with both escapes (rebase, or diff the merge-base). Warn rather than silently switch to three-dot: three-dot hides drift, and drift is what bites at merge time.
- **Panel discoverability**: single-model runs print a one-line `TIP:` pointing at `--models`, since cross-model agreement is the cheapest precision filter the tool has and it lived behind the less obvious flag. Silence with `CODE_REVIEW_NO_TIPS=1`.

### Fixed

- **`--min-found-by` was a silent no-op.** It was parsed and validated but never passed into the `Settings` constructor, so the panel filter read the dataclass default (1) and dropped nothing — the flag did exactly what not passing it did. Found by dogfooding this tool on its own diff.
- **A consensus floor from `$CODE_REVIEW_MIN_FOUND_BY` / `.code-review.toml` silently filtered nothing.** The CLI-flag combo was rejected in `_validate_flag_combos`, but that only sees argv, so an env/TOML value with no panel slipped past — the exact "you think a filter applied but it didn't" failure the flag exists to prevent. Now validated where `models` resolves, so every layer is covered.
- **Auto full-file context could hard-fail a model with a small window.** The 700K-char cap is global and sized for Gemini (1M tokens) / Claude (200K); `deepseek-chat-v3.1` is 163,840, so a payload cleared the cap and still blew the window — HTTP 400 on a review that would have succeeded hunks-only, i.e. auto breaking something the user never opted into. Auto now checks the model's published context window (from the same OpenRouter feed the cost estimate uses) and declines with a `NOTE:` instead of failing. Unknown window → no guard, attach and let the provider decide; explicit `--full-files` still errors rather than degrade silently.
- **OpenRouter's context-length 400 was classified `UNKNOWN` (exit 1) instead of `CONTEXT_OVERFLOW` (exit 12).** Its wording ("This endpoint's maximum context length is …") matched none of the phrase list, so callers got "escalate if unclear" for what is really "the scope is wrong, not the call".
- `in_hunk` path extraction now tolerates unified diffs carrying timestamps (`+++ b/x.py\t2026-07-16 10:00:00`), which GNU `diff -u` emits and `--diff-file` explicitly accepts. Previously the path captured the timestamp, matched nothing, and silently degraded every finding in that file to `in_hunk: null`.

### Changed

- **`--full-files` is now on by default (auto).** Hunk-only context is the single largest source of false findings, so changed files are attached whenever the payload fits under the 700K cap. Three states: *auto* (default — attach if it fits, otherwise fall back to hunks-only with a `NOTE:`), `--full-files` (strict — typed error if it doesn't fit, i.e. today's behavior), `--no-full-files` (off). Auto is best-effort by design: it costs tokens, so it declines rather than fails a review you never opted into — including when git can't resolve changed files (shallow clones, detached CI checkouts) or `--pr` isn't checked out at the PR head. `--diff-file` is excluded from auto entirely: a handed-in diff has no verifiable relationship to the local tree, so attaching local bodies could pair the review with unrelated content.
- **Module split**: the single-file `code_review/cli.py` (~5,000 lines) is now nine focused modules (`errors`, `prompts`, `providers`, `sources`, `parser`, `panel`, `chunking`, `config`, plus a slimmed orchestrating `cli`). No behavior change; `from code_review.cli import X` keeps working (cli re-exports the full surface) and the console entry point is unchanged. See the code map in `docs/architecture.md`.

## [0.2.0] — 2026-07-07

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
- Quality infra: an offline test suite (several hundred tests: `httpx.MockTransport` wire tests, frozen real-model parser fixtures, contract pins); ruff + mypy (source *and* tests/evals) gating CI on ubuntu + windows; wheel-content verification; an eval harness with planted-bug fixtures **and a clean-diff control** scoring recall and hallucination noise.
- A formal JSON Schema for the `--format json` envelope (`docs/schema/review-envelope.schema.json`) covering single-model, panel, and chunked documents — pinned to the envelope builders by tests so it can't drift from what the tool emits.
- Docs: rewritten README (contract tables for agent callers, a sample-output teaser in Quick start, a Troubleshooting symptom table), operations runbook, architecture overview, SECURITY.md threat model, issue/PR templates, and a documented release/tagging procedure for reproducible pinned installs.

### Changed

- **Ollama endpoint moved** from the OpenAI-compat path to native `/api/chat`; usage comes from `prompt_eval_count`/`eval_count`.
- **Summary-only model output is now a parse failure** (`parse_ok: false` + raw embedded, still exit 0): the template mandates findings or the literal clean phrase, so a bare summary means the model drifted — reporting it as a confident zero-finding review let `--baseline` resolve everything and agents treat the run as clean.
- **Baseline relaxed matching requires real line numbers on both sides** — a line-less baseline entry no longer vouches for any same-file finding (line-less findings still persist via the fingerprint pass).
- Gemini usage now counts `thoughtsTokenCount` in completion tokens — thinking models bill reasoning separately, and the `[usage]` line was understating cost.
- `--min-severity` is now **enforced post-parse** in JSON envelopes, panel reports, and chunked envelopes (it was prompt-level only, which the model could ignore). Verbatim markdown output remains best-effort.
- HTTP 401/403 from providers are now typed `CONFIG` errors naming the key env var (previously fell through to `UNKNOWN [exit 1]`); OpenRouter moderation-flagged 403s map to `SAFETY_REFUSAL`.
- `--full-files --pr` now **requires** the local HEAD to be the PR head (typed `CONFIG` error suggesting `gh pr checkout N`); previously a WARN allowed silently mismatched reference content.

### Security

- `.code-review.toml` (which lives in the possibly-untrusted *reviewed* repo) can no longer set the prompt `context` (self-review injection), `ollama_host` (diff exfiltration), or `ollama_num_ctx`/`ollama_timeout`; `--no-project-config` skips the file entirely.
- Prompt-injection guard appended to every request: instructions inside reviewed code are data, and manipulation attempts get flagged.
- Provider `Retry-After` sleep values clamped to 300s.

### Fixed

- CLI `--model` now overrides a `.code-review.toml` panel (documented CLI > config precedence) instead of crashing as "mutually exclusive"; config-sourced panels can no longer bypass the `--chunk` exclusion; `--full-files` works from repo subdirectories (git's root-relative paths are re-based onto CWD — previously every reference file silently stat-failed); `--staged --full-files` warns when unstaged edits make the working tree diverge from the reviewed index; `--chunk --codebase` budgets measure the codebase prompt overhead (not the smaller diff one); `--diff-file -` on a terminal fails fast instead of hanging on stdin; the `/api/ps` probe tolerates a non-list `models` value; eval scoring no longer renders None titles as `"None"`, wraps non-JSON stdout as an ERR row, and qualifies overall recall when runs errored.
- Severity regex no longer matches substrings ("Below" → `LOW`); panel consensus no longer merges line-less findings by location; `$CODE_REVIEW_PROVIDER` values are validated like CLI ones; `Retry-After` parsing survives overflow/malformed headers; non-object provider JSON is typed `PROVIDER_HICCUP` — including *nested* malformed shapes (non-list `choices`/`candidates` containers, non-dict choice/message/candidate/content, non-string content), which previously escaped as raw `AttributeError`/`KeyError` → `UNKNOWN`; panel `--dry-run` probes every model's window; eval harness exits non-zero when any run errors and its scoring honors each fixture's `line_range` (right-file/right-keyword/wrong-location no longer counts as caught); missing/partial prompt-asset dirs and non-string command `prompt` values are typed `CONFIG` instead of raw tracebacks; BOM-prefixed config files parse on Windows.

## [0.1.0] — 2026-07-06 (PR #2)

Fork baseline hardening: the standalone three-provider runner (OpenRouter / Gemini API / local Ollama) with the typed error model (exit codes 0/1/2/10–14, `ERROR: <CATEGORY> [exit <N>]` stderr contract), safety-context wrapper, Windows-first UTF-8 handling, the first test suite, and the LLM-agent runbook. Consolidated the fork's early branches into `main`.

## Fork point

Forked from [gemini-cli-extensions/code-review](https://github.com/gemini-cli-extensions/code-review) (Apache-2.0). The upstream prompt files ship byte-identical; see the README's *Fork provenance* section for the full delta.
