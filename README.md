# Gemini CLI Code Review Extension — OpenRouter fork

This is an Airwhale fork of [`gemini-cli-extensions/code-review`](https://github.com/gemini-cli-extensions/code-review). The upstream extension targets Google's `gemini-cli` host and is licensed Apache-2.0.

**What this fork adds:** a thin Python runner (`review.py`) that loads the upstream `skills/code-review-commons/SKILL.md` and `commands/code-review.toml` prompts unchanged and POSTs them to OpenRouter with a Gemini model. That gives the same review behavior as the GitHub `/gemini review` bot at ~30–60s per round instead of 5–15 minutes, because the request bypasses GitHub's webhook → Google job queue.

The upstream skill/command files are kept verbatim so future upstream improvements rebase cleanly. All OpenRouter-specific code lives in `review.py`, `pyproject.toml`, `.env.example`, and `.gitignore`.

## Quick start

```bash
# One-time: configure the runner
cd code-review
cp .env.example .env
# edit .env to set OPENROUTER_API_KEY=sk-or-...

# Review the current branch in any project (uses merge-base against origin/HEAD):
cd /path/to/my-project
uv run --project /path/to/code-review /path/to/code-review/review.py

# Or, more ergonomically, from the code-review dir against another repo:
cd /path/to/code-review
uv run review.py --pr 6   # reviews PR #6 of whichever repo CWD points at
```

`uv` resolves the deps (`httpx`, `python-dotenv`) on first run. No global pip install.

## Modes

| Flag | What it diffs |
|---|---|
| *(none)* | Current branch vs `origin/HEAD` merge-base — matches the upstream `gemini-cli` `/code-review` shape |
| `--base main` | Current branch vs an explicit base ref |
| `--pr <N>` | Pulls a GitHub PR diff via `gh pr diff` (requires `gh auth login`) |
| `--staged` | Staged changes only — good for pre-commit reviews |

`--model google/gemini-2.5-flash` swaps to the faster model (~10–20s) with some quality loss. Default is `google/gemini-2.5-pro` to match what the GitHub bot serves.

## Output format

Markdown, structured per the upstream `commands/code-review.toml` template:

```
# Change summary: [one-sentence description]

## File: path/to/file.py
### L<line>: [CRITICAL|HIGH|MEDIUM|LOW] One-sentence issue summary
More detail about the issue.

Suggested change:
```
    - removed line
    + replacement line
```
```

When the diff is clean, you get a single line: `No issues found. Code looks clean and ready to merge.`

## Why this exists

The GitHub Gemini Code Assist bot's wall-time latency (webhook → job queue) makes iterative cycles painful when you want a code review every few commits. Running the same prompts locally via OpenRouter cuts the loop to seconds and works offline from GitHub. The Apache-2.0 license on the upstream extension permits this kind of derivative work.

## Upstream sync

When upstream ships changes to the skill/command prompts:

```bash
git fetch upstream
git checkout main && git merge upstream/main   # updates the prompts
git checkout openrouter && git rebase main     # rebases the runner
```

Because the runner doesn't touch `skills/` or `commands/`, conflicts are unlikely.

---

## Original upstream README

The Code Review extension is an open-source Gemini CLI extension, built to enhance your repository's code quality. The extension adds a new command to Gemini CLI that analyzes code changes to identify a variety of code quality issues.

This extension is brought to you by the authors of the [Gemini Code Assist GitHub App](https://github.com/apps/gemini-code-assist), which provides code reviews directly in your GitHub pull requests.

### Installation (upstream CLI mode)

Install the Code Review extension by running the following command from your terminal *(requires Gemini CLI v0.4.0 or newer)*:

```bash
gemini extensions install https://github.com/Airwhale/code-review
```

If you do not yet have Gemini CLI installed, or if the installed version is older than 0.4.0, see [Gemini CLI installation instructions](https://github.com/google-gemini/gemini-cli?tab=readme-ov-file#-installation).

### Use the extension (upstream CLI mode)

The Code Review extension adds the `/code-review` command to Gemini CLI which analyzes code changes on your current branch for quality issues. See the [upstream docs](https://github.com/gemini-cli-extensions/code-review) for `/pr-code-review` and MCP setup.

## Legal

- License: [Apache License 2.0](./LICENSE) (inherited from upstream)
