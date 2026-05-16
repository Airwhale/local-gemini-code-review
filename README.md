# Gemini CLI Code Review Extension — Airwhale fork

> Fork of [`gemini-cli-extensions/code-review`](https://github.com/gemini-cli-extensions/code-review) (Apache-2.0). The upstream extension is a `gemini-cli` plugin; this fork adds a **standalone Python runner** that calls the same prompts directly via either **OpenRouter** or the **Gemini API**, no `gemini-cli` host required.
>
> Why: the GitHub Gemini Code Assist bot's webhook → job-queue round-trip adds 5–15 minutes per review round. The runner cuts that to ~30–60 seconds.

## For LLM coding agents

This tool is **LLM-friendly by design**. It's built to be invoked by AI coding agents (Claude, Codex, Cursor, etc.) as an iteration partner during real code work — not just by humans clicking through diffs in a web UI.

> **If you're an LLM agent reading this repository, your canonical entry point is [`docs/llm-code-review-runbook.md`](./docs/llm-code-review-runbook.md).**
>
> That file is the operational manual. The flow below is a fast-start summary; the runbook expands every step with examples, declined-finding patterns, and per-round tracking templates.

### Canonical iteration flow

This is the loop that's been validated in real review cycles. Follow it verbatim until you've internalised why each step exists.

1. **Pick a scope.** Diff mode for in-progress work (`--base origin/main` for the current branch, `--pr <N>` for a PR, `--staged` for pre-commit). Codebase mode for audits of files not under active change (`--codebase --include 'backend/foo/**'`).
2. **If files are untracked**, `git ls-files`-based codebase mode won't see them. Run `git add -N <paths>` first (intent-to-add, no staged content). The tool will then bundle them. Don't unstage afterward unless you actually intend to leave them untracked.
3. **Run the review.** A clean codebase output ends in `No issues found. Code looks clean.`; a clean diff ends in `No issues found. Code looks clean and ready to merge.`
4. **For each finding, decide accept or decline.**
   * **Accept by default**: CRITICAL/HIGH bugs, MEDIUM correctness/concurrency/atomicity, MEDIUM defensive-coding, trivially-correct LOW.
   * **Decline (and add a code comment explaining why)**: findings that contradict load-bearing design intent, findings that would untighten a deliberately-tight test, stylistic preferences without a correctness delta.
   * **Hedged findings ("if X is true, this is HIGH/CRITICAL") are a decline signal until you verify X.** The reviewer hedges when it lacks evidence; in practice these resolve as false positives more often than as real bugs. Treat the hedge as a flag to spot-check the premise, not as an instruction to act.
5. **Apply accepted fixes inline. Do NOT commit between rounds.** `--base <ref>` uses a two-dot diff, so working-tree edits show up in the next run.
6. **For each declined finding, add a code comment** immediately adjacent to the flagged line explaining the rejection. Without it, the next round re-flags the same finding. This is the central operational rule that makes the loop converge.
7. **Re-run.** Repeat steps 3–6.
8. **Stop conditions** (in priority order):
   * Output is clean ("No issues found...") → done.
   * A round produces only hallucinated findings (wrong line numbers, contradictory suggestions, or claims that don't match the code) → done; the model has run out of substantive material.
   * You've hit 4 rounds → done; remaining findings are usually noise or out of scope.
9. **Run tests + build.** Commit. Push.

### What "hallucination" looks like in practice

The model occasionally fabricates a finding it then partially refutes — e.g., *"This call passes the wrong type at L772. A better fix would be to pass the list of vectors directly"* — when the existing code at the actual (non-L772) line already does exactly the "better fix." When you see a finding's own suggested fix match the existing code, decline without action.

Pre-fix to v1.1, line numbers in codebase mode drifted 5–150 lines depending on file size. As of [b124501](https://github.com/Airwhale/local-gemini-code-review/commit/b124501) the bundle pre-numbers every line (`cat -n` style) and the model transcribes the prefix instead of counting; codebase-mode line numbers are now exact. If you see drift again, that's a regression — file an issue.

### Per-round ledger

Keep one row per finding so the final commit message or PR comment can summarise the cycle:

```
| Round | File | Line | Severity | Finding                          | Action              |
|-------|------|------|----------|----------------------------------|---------------------|
| 1     | a.py | 808  | LOW      | Custom markdown parser fragile   | Declined w/ comment |
| 1     | b.py | 111  | MEDIUM   | Missing type hint on `result`    | Applied             |
| 2     | a.py | 313  | MEDIUM   | Silent LLMClient fallback        | Applied             |
| 3     | -    | -    | -        | Vector sum duplication           | Declined w/ comment |
| 4     | -    | -    | -        | Hallucination (line mismatch)    | No action           |
```

This is the artifact a human reviewer reads to understand what changed and why. The runbook's "Per-round tracking" section has the long form.

The rest of this README is the general-audience documentation (humans, contributors, anyone evaluating the fork). The runbook is the agent-targeted documentation, and it is the file to read first if your job is to use the tool, not understand its provenance.

## What this fork adds vs upstream

| File | New / modified | Purpose |
|---|---|---|
| `review.py` | **new** | Standalone runner. Loads upstream skill + command prompts for diff review, adds fork-specific skill + command for whole-codebase review, POSTs to OpenRouter or the Gemini API. Includes a curated alias table so `--model claude` expands to the full OpenRouter slug. |
| `skills/code-review-codebase/SKILL.md` | **new** | Fork-specific whole-codebase review skill. Same persona / severity rubric as the upstream `code-review-commons` skill, but Critical Constraints adapted to permit comments on any line of any file in the bundle (the upstream "comment only on `+`/`-` lines" rule forbids commenting on whole-file content). |
| `commands/codebase-review.toml` | **new** | Fork-specific command for `--codebase` mode. Defines the bundle delimiter and per-file-findings output shape. |
| `pyproject.toml` | **new** | `uv`-managed deps (`httpx`, `python-dotenv`). |
| `.env.example` | **new** | Documents `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, and optional model / provider overrides. |
| `.gitignore` | **new** | Protects `.env` and the `uv` virtualenv from leaking. |
| `docs/llm-code-review-runbook.md` | **new** | Operational runbook for using the tool as an LLM iteration partner. |
| `README.md` | **modified** | This file — documents the fork's runner alongside the upstream CLI mode. |
| `skills/code-review-commons/SKILL.md` | unchanged | Upstream system prompt (loaded verbatim for diff review). |
| `commands/code-review.toml` | unchanged | Upstream diff-review user-prompt template (loaded verbatim). |
| `commands/pr-code-review.toml` | unchanged | Upstream PR-review command (not used by the runner). |
| `gemini-extension.json`, `GEMINI.md`, `LICENSE` | unchanged | Upstream metadata. |

The fork keeps the upstream skill / command files **byte-for-byte identical** so upstream improvements rebase cleanly:

```bash
git fetch upstream
git checkout main && git merge upstream/main   # picks up upstream prompt changes
```

## Quick start

```bash
# One-time: clone and configure
git clone https://github.com/Airwhale/local-gemini-code-review
cd local-gemini-code-review
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY=... or GEMINI_API_KEY=... (or both)

# From the runner directory, against the CWD:
cd /path/to/local-gemini-code-review
uv run review.py --base origin/main         # iterative review of current branch
uv run review.py --pr 6                     # review a specific PR
uv run review.py --codebase                 # whole tracked codebase (filtered)
uv run review.py --base origin/main --model claude   # use Claude instead of Gemini

# Or invoke from any project directory by pointing at the runner:
cd /path/to/my-project
uv run --project /path/to/local-gemini-code-review /path/to/local-gemini-code-review/review.py
```

`uv` resolves deps on first run. No global `pip install` required.

## Provider selection

The runner supports two transport paths to the same `gemini-2.5-pro` model:

```bash
uv run review.py                        # default: openrouter
uv run review.py --provider openrouter  # explicit
uv run review.py --provider gemini      # direct Gemini API
```

| Provider | Endpoint | Required env var | Default model | Notes |
|---|---|---|---|---|
| `openrouter` (default) | `openrouter.ai/api/v1/chat/completions` | `OPENROUTER_API_KEY` | `google/gemini-2.5-pro` | OpenAI-compatible chat-completions wire format. One bill for many providers. |
| `gemini` | `generativelanguage.googleapis.com/v1beta/models/...` | `GEMINI_API_KEY` | `gemini-2.5-pro` | Google AI Studio's `generateContent` endpoint. Slightly lower latency (one less hop). |

**Known gotcha for the Gemini API path:** the free tier has zero per-day quota for `gemini-2.5-pro` as of the time of writing. You'll see HTTP 429 immediately. Workarounds:

1. `--model gemini-2.5-flash` — the free tier does allow flash, and it's ~3× faster anyway. Quality drops a bit but the review structure is still solid.
2. Upgrade to a paid Google AI Studio plan if you want pro via the direct API.
3. Use `--provider openrouter` — OpenRouter has its own pro quota and bills you directly.

Set a default per-environment via `$CODE_REVIEW_PROVIDER=gemini` in your `.env` so you don't have to pass `--provider` every invocation.

### Model aliases

The `--model` flag accepts any OpenRouter slug, but a small curated alias table collapses the most common reviewers to a short name. Aliases work **only** under `--provider openrouter`; the Gemini API direct path takes bare Gemini model names. Passing an alias with `--provider gemini` errors out with a clear message.

| Alias | Resolves to | Notes |
|---|---|---|
| `pro` / `gemini-pro` | `google/gemini-2.5-pro` | Current default. |
| `flash` / `gemini-flash` | `google/gemini-2.5-flash` | ~3× faster than pro, some quality loss. |
| `claude` / `claude-sonnet` | `anthropic/claude-sonnet-4.5` | Great as a second-opinion reviewer alongside a Gemini round. |
| `claude-opus` | `anthropic/claude-opus-4.5` | Larger model; slower and pricier. |
| `gpt` | `openai/gpt-5` | Useful for an independent third opinion. |
| `gpt-mini` | `openai/gpt-5-mini` | Cheaper / faster GPT. |
| `deepseek` | `deepseek/deepseek-chat-v3.1` | Cheap, surprisingly strong on code review. |

Raw slugs still pass through unchanged, so anything OpenRouter serves — including newer models that haven't earned an alias yet — works via `--model <vendor>/<model>`.

## Modes

| Flag | What it does |
|---|---|
| *(none)* | Diff: current branch vs `origin/HEAD` merge-base — matches the upstream `gemini-cli` `/code-review` shape. |
| `--base main` | Diff: current branch vs an explicit base ref (**includes uncommitted changes**, so iterative-review loops work without committing each pass). |
| `--pr <N>` | Diff: pulls a GitHub PR diff via `gh pr diff` (requires `gh auth login`). |
| `--staged` | Diff: staged changes only — good for pre-commit reviews. |
| `--codebase` | Whole codebase: bundles tracked files (via `git ls-files`) and reviews them all. Narrow with `--include` / `--exclude` glob flags. |

`--model <slug>` overrides the default model. Use `google/gemini-2.5-flash` / `flash` (OpenRouter) or `gemini-2.5-flash` (Gemini API) for ~3× faster reviews with some quality loss.

Two more runtime knobs:

- `--temperature <float>` (default `0.5`, env `CODE_REVIEW_TEMPERATURE`): sampling randomness. Higher = more findings per call, more hallucinations. Lower = tighter, fewer findings.
- `--max-tokens <int>` (default `16000`, env `CODE_REVIEW_MAX_TOKENS`): ceiling on output. Not a target — you pay only for what's emitted. Default avoids mid-finding truncation on thorough reviews.

The defaults shifted from `0.2` / `~8K` after empirical iteration: `0.2` produced 1–2 findings per round on diffs that plausibly contained more, requiring 5–7 rounds to converge. `0.5` typically produces 3–5 findings per round and converges in 3–5 rounds.

### Whole-codebase mode (`--codebase`)

For "audit this repo I just inherited" or "find bugs in code none of us touched in this PR," the diff modes don't help. `--codebase` bundles every tracked file (filtered) into a single payload and reviews them as a whole.

```bash
uv run review.py --codebase                              # everything tracked, minus built-in noise filters
uv run review.py --codebase --include 'backend/**/*.py'  # narrow to a directory + extension
uv run review.py --codebase --exclude '**/test_*'        # widen then narrow
uv run review.py --codebase --model claude               # use Claude as the codebase reviewer
```

File selection pipeline:

1. `git ls-files` → all tracked files (so `.gitignore` already filters `node_modules`, `.venv`, build artifacts).
2. Apply user `--include` globs if any.
3. Apply user `--exclude` globs.
4. Apply built-in defensive excludes: lock files, minified output, common binary extensions (`*.png`, `*.svg`, `*.woff`, etc.), `*/dist/*`, `*/build/*`.
5. Drop individual files larger than 100 KB (logged on stderr — they're usually data fixtures or vendored blobs).

A bundle cap (700,000 chars, ~175 K tokens at the standard 4-chars-per-token estimate) is enforced pre-flight; if the bundle is too large the runner exits with the 10 largest files in the current selection so you can target `--exclude` flags effectively rather than paying for a request that would fail mid-flight on the smaller-context models. The cap is conservative against both Gemini 2.5 Pro (1 M-token context) and Claude Sonnet 4.5 (200 K-token context), so the same selection works regardless of which model you target with `--model`.

Output is the same severity-tagged per-file findings shape as diff mode (see the **Output format** section below — the diff-mode template applies, with the per-file section anchoring to the bundle's `======== FILE: <path> ========` delimiters instead of a diff). The **architectural-summary output shape** (high-level "patterns / structure / smells" section preceding the per-file findings) is an explicit TODO; the trade-offs (hallucination risk on architectural takes, less actionable output, token-budget contention) are documented in the runbook under "Future modes."

## Output format

Markdown, structured per the upstream `commands/code-review.toml` template (diff modes) or the fork-added `commands/codebase-review.toml` template (`--codebase`). Both shapes use the same severity tags `CRITICAL | HIGH | MEDIUM | LOW`.

**Diff modes (`--base`, `--pr`, `--staged`):**

```
# Change summary: [one-sentence description of the change]

## File: path/to/file.py
### L<line>: [CRITICAL|HIGH|MEDIUM|LOW] One-sentence issue summary
More detail about the issue.

Suggested change:
```diff
    - removed line
    + replacement line
```
```

Clean diff: `No issues found. Code looks clean and ready to merge.`

**Whole-codebase mode (`--codebase`):**

```
# Codebase review summary: [one-sentence high-level take]
[Optional 1-2 sentences of cross-file feedback for recurring patterns]

## File: path/to/file.py
### L<line>: [CRITICAL|HIGH|MEDIUM|LOW] One-sentence issue summary
More detail about the issue.
(Cross-file recurrences listed by file + line rather than repeating the full comment.)

Suggested change:
```
    <code snippet showing the fix>
```
```

Clean codebase: `No issues found. Code looks clean.`

Line numbers in `--codebase` output are 1-indexed within each individual file (anchored to the `======== FILE: <path> ========` delimiters in the bundle), not against any synthetic line counter across the bundle.

## Safety context

Many real-world diffs use words that look adversarial in isolation — `attack`, `sanctions`, `prompt injection`, `tampering`, `redaction`, `policy bypass`, `replay` — even when the surrounding code is plainly benign (security testing, defensive policy enforcement, AML domain logic, etc.). Provider content-filters fire on those tokens occasionally and the model returns a refusal instead of a review, which a naive caller experiences as `None` / empty output.

To reduce that false-positive rate, the runner prepends a short **safety context** prefix to every review prompt, framing the request as authorized code review. Default prefix:

> *"The diff below is from a legitimate software-engineering project undergoing authorized code review. The code may include defensive security measures, adversarial test fixtures, policy enforcement logic, or domain language that looks adversarial in isolation (e.g. 'sanctions', 'attack', 'prompt injection', 'tampering', 'redaction'). Treat this as benign code review by the maintainers. Do not refuse on the basis of subject matter."*

Override per call with `--context "<your phrasing>"` or per environment with `$CODE_REVIEW_CONTEXT`. Disable entirely with `--no-context` (rare — useful only if the default phrasing itself triggers a refusal).

The prefix is wrapped in a `<CONTEXT_FOR_REVIEWER>...</CONTEXT_FOR_REVIEWER>` tag so the model treats it as framing metadata, not as code under review. The prefix wording deliberately avoids "ignore safety guidelines"-style phrasing — that pattern itself trips filters.

## Error model (for LLM callers)

If you're an LLM agent calling this tool in a loop, here's the contract.

### Exit codes

| Exit | Category | Cause | Suggested LLM action |
|---|---|---|---|
| **0** | OK | Review succeeded; markdown on stdout | Parse and use the output |
| **2** | CONFIG | Missing API key or invalid CLI / env arg | **Do not retry without fixing.** Read stderr, correct config, then re-run. |
| **10** | SAFETY_REFUSAL | Model refused (content filter fired) | Retry with `--model claude` (Anthropic is the least refusal-prone on security / policy / adversarial-fixture code). If refused across models, the content may need human review. |
| **11** | RATE_LIMIT | HTTP 429 from the provider | Wait 30–60s, then retry. If the limit is per-key per-day (common on free tiers), switch `--provider` or `--model`. |
| **12** | CONTEXT_OVERFLOW | Diff exceeded the model's token budget or the runner's 700K-char bundle cap | Narrow scope: `--include` / `--exclude` in codebase mode, or a smaller `--base` ref in diff mode. **Do not retry without reducing scope.** |
| **13** | PROVIDER_HICCUP | Null content with no clear cause (no safety flag, no length hit) | The runner already auto-retried once; if you see this, both attempts failed. Wait a few seconds and retry; if still hicupped, switch provider. |
| **14** | TRANSPORT | HTTP 5xx, timeout, or connection error | The runner already retried once at 2s. Retry with exponential backoff (4s, 8s); escalate if 3 retries fail. |
| **1** | UNKNOWN | Catchall (unexpected exception, non-JSON response, etc.) | Read stderr for the surviving exception; escalate if unclear. |

### Stderr format

Stable single-line prefix for machine parsing:

```
ERROR: <CATEGORY> [exit <N>]
```

Followed by free-form lines for human readability:

```
ERROR: SAFETY_REFUSAL [exit 10]
Reason: Model refused with finish_reason='safety'
Model: google/gemini-2.5-pro
Provider: openrouter
Suggested: Retry with a different model: ``--model claude`` is the most refusal-resistant...
Detail: {"choices":[{"message":{"content":null,...}}],...}
```

An agent can `grep -oE '^ERROR: [A-Z_]+'` the stderr to extract the category cheaply.

### Auto-retry behavior

The runner auto-retries **once** on:

- `PROVIDER_HICCUP` — usually recovers on the second call
- `TRANSPORT` — HTTP 5xx / network timeout

Other categories surface immediately:

- `SAFETY_REFUSAL` — a second call with the same model + prompt almost always reproduces the refusal; better to switch model than burn tokens
- `RATE_LIMIT` — retry without waiting just makes it worse
- `CONTEXT_OVERFLOW` — the scope is wrong, not the call
- `CONFIG` — fix the config

### Decision tree for LLM callers

```
review.py exited:
├── 0   → review succeeded; markdown on stdout
├── 2   → check config; do NOT retry without changes
├── 10  → retry with --model claude; if still 10, escalate to human
├── 11  → wait 60s and retry; if still 11, switch --provider
├── 12  → narrow scope (--include/--exclude or smaller --base); do NOT retry without scope change
├── 13  → runner already retried once; retry yourself once after a few seconds; if still 13, switch provider
├── 14  → exponential backoff retry (4s, 8s, 16s); escalate after 3 attempts
└── 1   → read stderr; escalate
```

## Operational runbook

The full workflow — iteration loop, accept/decline heuristics, the decline-comment contract, known gotchas, per-round tracking — lives in [`docs/llm-code-review-runbook.md`](./docs/llm-code-review-runbook.md). Same content works for human developers and LLM agents alike.

## Why this exists

The GitHub Gemini Code Assist bot is excellent at finding real concurrency, security, and correctness bugs — but it lives behind a GitHub webhook that calls a Google job queue, and the wall-time latency makes iterative cycles painful when you want a review every few commits. Running the same prompts locally via OpenRouter or the Gemini API cuts the loop from minutes to seconds and works offline from GitHub.

In practice this turns the workflow from "push → wait → fix → repeat" into "stage → review locally → fix → stage → review locally → commit when clean," with the GitHub bot reserved as a final-mile verification pass instead of an iteration partner. The Apache-2.0 license on the upstream extension permits this kind of derivative work.

A 10-iteration test against a ~5K-line PR caught 3 HIGH-severity correctness bugs (infinite session-creation loop, header-overwrite latent bug, Map-collision UX bug) plus several MEDIUM-severity defensive-coding issues — all in ~10 minutes wall-time. The same review through the GitHub bot would have taken ~50 minutes for the equivalent rounds.

## Upstream sync

When upstream ships changes to the skill or command prompts:

```bash
git fetch upstream
git checkout main && git merge upstream/main
```

Because the runner doesn't touch `skills/` or `commands/`, conflicts are unlikely.

---

## Original upstream README

The Code Review extension is an open-source Gemini CLI extension, built to enhance your repository's code quality. The extension adds a new command to Gemini CLI that analyzes code changes to identify a variety of code quality issues.

This extension is brought to you by the authors of the [Gemini Code Assist GitHub App](https://github.com/apps/gemini-code-assist), which provides code reviews directly in your GitHub pull requests.

### Installation (upstream CLI mode)

If you want to install the upstream extension into `gemini-cli` (separate from the standalone runner this fork ships), point at the **upstream repository**, not this fork:

```bash
gemini extensions install https://github.com/gemini-cli-extensions/code-review
```

If you do not yet have Gemini CLI installed, or if the installed version is older than 0.4.0, see [Gemini CLI installation instructions](https://github.com/google-gemini/gemini-cli?tab=readme-ov-file#-installation).

### Use the extension (upstream CLI mode)

The Code Review extension adds the `/code-review` command to Gemini CLI which analyzes code changes on your current branch for quality issues. See the [upstream docs](https://github.com/gemini-cli-extensions/code-review) for `/pr-code-review` and MCP setup.

## Legal

- License: [Apache License 2.0](./LICENSE) (inherited from upstream)
- Upstream: <https://github.com/gemini-cli-extensions/code-review>
