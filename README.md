# Gemini CLI Code Review Extension — Airwhale fork

> Fork of [`gemini-cli-extensions/code-review`](https://github.com/gemini-cli-extensions/code-review) (Apache-2.0). The upstream extension is a `gemini-cli` plugin; this fork adds a **standalone Python runner** that calls the same prompts directly via either **OpenRouter** or the **Gemini API**, no `gemini-cli` host required.
>
> Why: the GitHub Gemini Code Assist bot's webhook → job-queue round-trip adds 5–15 minutes per review round. The runner cuts that to ~30–60 seconds.

## For LLM coding agents

This tool is **LLM-friendly by design**. It's built to be invoked by AI coding agents (Claude, Codex, Cursor, etc.) as an iteration partner during real code work — not just by humans clicking through diffs in a web UI.

> **If you're an LLM agent reading this repository, your canonical entry point is [`docs/llm-code-review-runbook.md`](./docs/llm-code-review-runbook.md).**
>
> That file is the operational manual — exact invocation patterns, accept / decline heuristics, the decline-comment contract that makes the iteration loop converge instead of churning on the same finding every pass, the known gotchas, and the per-round tracking template. It is written to be self-contained: you can pick it up cold and start producing clean review cycles without any other context.

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

# Review the current branch of whatever project you're in:
cd /path/to/my-project
uv run --project /path/to/local-gemini-code-review /path/to/local-gemini-code-review/review.py

# Or invoke from the runner directory against an external CWD:
cd /path/to/local-gemini-code-review
uv run review.py --pr 6
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

A bundle cap (700 KB ≈ 175 K tokens) is enforced pre-flight; if the bundle is too large the runner exits with the 10 largest files in the current selection so you can target `--exclude` flags effectively rather than paying for a request that would fail mid-flight on the smaller-context models.

Output is the same severity-tagged per-file findings shape as diff mode. The **architectural-summary output shape** (high-level "patterns / structure / smells" section preceding the per-file findings) is an explicit TODO; the trade-offs (hallucination risk on architectural takes, less actionable output, token-budget contention) are documented in the runbook under "Future modes."

## Output format

Markdown, structured per the upstream `commands/code-review.toml` template:

```
# Change summary: [one-sentence description]

## File: path/to/file.py
### L<line>: [CRITICAL|HIGH|MEDIUM|LOW] One-sentence issue summary
More detail about the issue.

Suggested change:
```diff
    - removed line
    + replacement line
```
```

When the diff is clean: `No issues found. Code looks clean and ready to merge.`

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
