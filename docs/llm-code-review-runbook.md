# LLM code-review runbook

Operational guide for using this fork's `review.py` as an iteration partner — written for coding agents (Claude, Codex, etc.) and humans who want the same workflow. Assumes the runner is already installed and `.env` is configured at the repo root.

The runner is a thin Python wrapper around the upstream `gemini-cli-extensions/code-review` skill + command prompts. It POSTs them to either OpenRouter or the Gemini API and prints structured-markdown findings. Same model, same prompt, same output shape as the GitHub `/gemini review` bot — without the 5–15 min webhook → job-queue wait.

---

## Where the tool lives

```
Repository:       https://github.com/Airwhale/code-review
Branch:           main
Entry point:      review.py at the repo root
Dependencies:     uv-managed (httpx, python-dotenv) -- first run installs them
Config:           .env at the runner's repo root (NOT in the project being reviewed)
```

Required keys in `.env` (set whichever provider you'll use; missing keys fail fast with a clear error):
- `OPENROUTER_API_KEY` — for the default OpenRouter provider
- `GEMINI_API_KEY` — for the Gemini API direct provider

Optional overrides:
- `CODE_REVIEW_PROVIDER` — `openrouter` (default) or `gemini`
- `OPENROUTER_MODEL` — defaults to `google/gemini-2.5-pro`
- `GEMINI_MODEL` — defaults to `gemini-2.5-pro`

---

## Invocation shape

From any project directory, point at the runner via `uv run --project`:

```bash
cd /path/to/your-project
uv run --project /path/to/code-review /path/to/code-review/review.py --base origin/main
```

Or run directly from the runner directory against an external CWD:

```bash
cd /path/to/code-review
uv run review.py --pr 6
```

The runner reads `.env` from its own directory, not from CWD — configure it once at the runner location and invoke from any project folder.

### Four diff modes (mutually exclusive)

| Flag | Diff scope | Use when |
|---|---|---|
| *(none)* | merge-base against `origin/HEAD` | quick check on current branch |
| `--base origin/main` | **two-dot** diff vs ref, **includes working-tree** | **iterating before commit** — uncommitted edits show up |
| `--pr <N>` | `gh pr diff N` (requires `gh auth login`) | reviewing an existing PR |
| `--staged` | staged-only | pre-commit hook style |

### Two providers (swap with `--provider`)

| Provider | Env key required | Default model | Notes |
|---|---|---|---|
| `openrouter` (default) | `OPENROUTER_API_KEY` | `google/gemini-2.5-pro` | Recommended for iterative work. OpenRouter has reliable quota for pro. |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.5-pro` | Direct to Google AI Studio. The **free tier has zero quota for pro**; use `--model gemini-2.5-flash` if you only have a free key. |

`--model gemini-2.5-flash` (OpenRouter: `--model google/gemini-2.5-flash`) trades some quality for ~3× speed. Use it during heavy iteration when fast turnaround matters; switch to pro for the final pass.

---

## The iteration loop

```
1. Make edits in the target repo (fix a bug, build a feature, etc.).
2. Run:  uv run --project <runner> <runner>/review.py --base origin/main
3. Read the output. It is structured markdown with severity tags:
   CRITICAL > HIGH > MEDIUM > LOW.
4. For each finding, decide: accept or decline.
5. Apply accepted fixes inline. Do NOT commit yet.
6. Re-run step 2.
7. Repeat until output is:
      "No issues found. Code looks clean and ready to merge."
8. Run tests and a build. Commit. Push.
```

**Critical: do not commit between rounds.** The `--base origin/main` mode includes working-tree changes (two-dot diff `git diff -U5 <base>`, not three-dot `<base>...HEAD`). Re-running picks up your in-progress fixes immediately. Committing each round produces noisy history; the reviewer is happy reviewing your uncommitted edits.

---

## Accept / decline heuristics

**Accept by default:**

- **CRITICAL / HIGH** unless the finding is factually wrong (rare in practice). Real correctness or security bugs.
- **MEDIUM** about correctness, concurrency, atomicity, latent bugs, schema or type safety.
- **MEDIUM** about defensive coding — wider exception catches, header normalization, partial-key cache invalidation, etc. These are cheap and ship hardening.
- **LOW** when the suggestion is trivially right: consolidating duplicated patterns, fixing typos, tightening assertions, improving comment accuracy.

**Decline (and add a code comment explaining why):**

- Findings that contradict load-bearing design intent already encoded in the code. Example: an `/events` vs `/timeline` endpoint pair that looks redundant but is deliberately split for a planned SSE migration. The reviewer doesn't know your roadmap unless your code comments tell it.
- Findings that would untighten a deliberately-tight test. Example: a hardcoded expected count that the reviewer wants you to compute dynamically from the fixture — that turns the test into a tautology (`what the code reads == what the code reports`) and removes the regression-catch.
- Stylistic preferences that don't change correctness, when the existing form has a defensible rationale. Example: `maxsize=8` vs `maxsize=1` on an `lru_cache` where the 8 leaves headroom for test fixtures that load alternate paths.

**The decline contract:** every declined finding gets a code comment explaining the rationale, immediately adjacent to the code the reviewer flagged. Without the comment, the next iteration's model will surface the same finding again. The comment is a contract with future review rounds — it's how you teach the reviewer about decisions it can't infer from the diff alone.

---

## Known gotchas

1. **Transient `None` output.** OpenRouter occasionally returns an empty completion (content filter, provider hiccup, rate-limit interstitial). The runner surfaces this as a literal `None`. Just rerun the same command — the second call has always worked in practice. Don't debug it.

2. **Free-tier 429 on Gemini direct.** `--provider gemini --model gemini-2.5-pro` requires a paid Google AI Studio plan; the free tier returns HTTP 429 immediately. Either use `--provider openrouter` (preferred) or `--model gemini-2.5-flash --provider gemini`.

3. **Two-dot diff vs three-dot.** `--base <ref>` uses `git diff -U5 <ref>` (two-dot). Three-dot (`<ref>...HEAD`) would show only committed changes, miss working-tree edits, and the reviewer would keep re-flagging the same issues. The two-dot semantics is intentional for the iterative-review workflow.

4. **Comment-as-defense.** If you decline a finding without commenting, expect the same finding on the next round. Comments are the canonical way to suppress repeat findings.

5. **Diff size shapes round count.** Reviews on ~300K-char diffs (large feature PRs) cost more tokens and surface more findings per round. Smaller, focused diffs converge to clean faster. Typical observed round counts:
   - Small PR (~25K chars, single feature): 3 rounds to clean
   - Medium PR (~50K chars): 4–6 rounds
   - Large PR (~300K chars, multi-file feature): 8–12 rounds

6. **`tee` to a file.** When the output is large, pipe to `tee /tmp/review.md` so you can re-read findings without re-invoking the tool. Saves context budget on subsequent steps.

---

## When to also call `/gemini review` on GitHub

Treat the local tool as the **iteration partner** and the GitHub `/gemini review` bot as the **final-mile verifier**. They use the same model and similar prompts; the GitHub bot's only advantage is independence ("a third party reviewed this, not your own prompt-following loop").

Bring in the GitHub bot when:

- The diff touches concurrency, locks, signing, replay, differential privacy ledgers, policy boundaries, or auth surfaces — anywhere a missed bug has outsized blast radius.
- The PR is large enough that you want a credibility signal beyond "I ran the same model locally."
- You're about to merge to `main` and want one independent confirmation.

For most PRs, three to five clean local rounds is sufficient and saves 30–45 minutes per PR vs the webhook round-trip.

---

## Per-round tracking

Keep a one-line ledger per round so the final commit message or PR comment can summarize the cycle:

```
Iter N: <severity> <one-line-description> -> applied | declined-with-comment | clean
```

After the cycle, the ledger becomes the table in the commit body or PR comment. Example from PR #7 in the parent project:

```
| Round | Finding                                            | Action                |
|-------|----------------------------------------------------|-----------------------|
| 1     | MED cache duplication between modules              | Applied               |
| 1     | LOW lru_cache maxsize nit                          | Declined w/ comment   |
| 2     | MED CWD-relative path resolution                   | Applied               |
| 3     | None -- "No issues found. Code looks clean..."     | Ready                 |
```

This is the artifact a human reviewer reads to understand what changed and why. Do not skip it.

---

## One-liner reference

The command shape used most often during iterative work:

```bash
uv run --project /path/to/code-review /path/to/code-review/review.py --base origin/main 2>&1 | tee /tmp/review.md
```

Tail the file in another shell, or pipe to `head -80` if you only want the first few findings.

---

## Provenance

This fork (`Airwhale/code-review`) keeps the upstream `gemini-cli-extensions/code-review` skill and command prompts byte-identical so upstream improvements rebase cleanly. The only fork additions are `review.py`, `pyproject.toml`, `.env.example`, `.gitignore`, and the rewritten root `README.md`. See the root README for the full list of fork modifications and how to sync against upstream.
