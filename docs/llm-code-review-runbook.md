# LLM code-review runbook

Operational guide for using `review.py` as an iteration partner during code work. Same workflow whether you're a human developer or a coding agent (Claude, Codex, etc.).

The runner is a thin Python wrapper around the upstream `gemini-cli-extensions/code-review` skill + command prompts. It POSTs them to OpenRouter, the Gemini API, or a local Ollama server (offline / no API key / no token cost), and prints structured-markdown findings — same prompts, same output shape as the GitHub `/gemini review` bot, without the 5–15 min webhook → job-queue wait.

---

## Setup

```
Repository:    https://github.com/Airwhale/local-gemini-code-review
Entry point:   review.py at the repo root
Dependencies:  uv-managed -- first run installs them
Config:        .env at the runner's repo root (NOT at the project being reviewed)
```

Set the key for whichever cloud provider you'll use; missing keys fail fast with a clear error. The local provider needs no key — just a running Ollama server.

- `OPENROUTER_API_KEY` — default provider
- `GEMINI_API_KEY` — direct Google AI Studio path
- *(none)* — local Ollama. Requires `ollama serve` running and at least one model pulled.

Optional environment variables:

- `CODE_REVIEW_PROVIDER` — set provider default (e.g. `ollama`)
- `OPENROUTER_MODEL` / `GEMINI_MODEL` / `OLLAMA_MODEL` — override the per-provider default model
- `OLLAMA_HOST` — Ollama server URL (default `http://localhost:11434`). Override for non-default ports, remote Ollama, or WSL distros without localhost mirroring.
- `OLLAMA_TIMEOUT` — HTTP timeout for Ollama calls in seconds (default `1800`, i.e. 30 minutes — accommodates CPU cold-starts and thorough reviews).
- `OLLAMA_NUM_CTX` — the context window (tokens) the runner REQUESTS per call via the native endpoint's `options.num_ctx`. Usually unset: the runner reads a loaded model's real window via `/api/ps`, requests that, and hard-enforces it pre-flight (`CONTEXT_OVERFLOW`, exit 12, because Ollama silently truncates oversized prompts instead of erroring). When the window is unknown, `num_ctx` is omitted (the server keeps its VRAM-tier default), the guard warns, and `prompt_eval_count` is verified after the call as a hard backstop. Set explicitly to pin the window everywhere — requested per call, no server restart, RAM permitting.

---

## Invocation

From any project directory:

```bash
cd /path/to/your-project
uv run --project /path/to/local-gemini-code-review /path/to/local-gemini-code-review/review.py --base origin/main
```

Or from the runner directory against an external CWD:

```bash
cd /path/to/local-gemini-code-review
uv run review.py --pr 42
```

The runner reads `.env` from its own directory — configure once, invoke from anywhere.

### Source modes (mutually exclusive)

| Flag                 | Source scope                                                   | Use when                                |
|----------------------|----------------------------------------------------------------|-----------------------------------------|
| *(none)*             | diff: merge-base against `origin/HEAD`                         | quick check on current branch           |
| `--base origin/main` | diff: two-dot diff vs ref, **includes working tree**           | iterating before commit                 |
| `--pr <N>`           | diff: `gh pr diff N` (requires `gh auth login`)                | reviewing an existing PR                |
| `--staged`           | diff: staged-only                                              | pre-commit hook style                   |
| `--codebase`         | whole codebase: `git ls-files` bundled (filtered); see below   | auditing an unfamiliar repo             |

### Providers

| `--provider`    | Env key required        | Default model            | Notes                                                                                   |
|-----------------|-------------------------|--------------------------|-----------------------------------------------------------------------------------------|
| `openrouter` *  | `OPENROUTER_API_KEY`    | `google/gemini-2.5-pro`  | Default. Reliable quota; recommended for iterative work.                                |
| `gemini`        | `GEMINI_API_KEY`        | `gemini-2.5-pro`         | Direct to Google AI Studio. **Free tier has zero quota for pro** — use flash if free.   |
| `ollama`        | *(none — local)*        | `qwen3-coder:30b`        | Offline / no API key / no token cost. CPU inference is slower (1–5 min per review typical). Different failure mode than cloud — see "Local vs cloud" below. |

\* default

### Local vs cloud — pick deliberately



**Use cloud when:** you want fast, structured triage; iterating against a small diff; converging in fewer rounds matters more than zero false positives.

**Use Ollama (local) when:** you're offline; the code is sensitive enough you don't want it leaving the machine; you want a free sanity check; the cloud providers are rate-limited / down; you want a second-opinion pass after a cloud review (different blind spots catch different things).

**For high-stakes PRs**, run *both* and reconcile.  You should also probably consider using other models on Openrouter:  Kimi, Sonnet, and ChatGPT are all good options and will likely all pick up different bugs.

### Setting up Ollama (when `--provider ollama` is right)

1. Install [Ollama](https://ollama.com/download). If Smart App Control / Application Control blocks the installer on Windows, install in WSL2 — the runner reaches the WSL server via localhost mirroring without extra config.
2. Pull a model. Default expected by the runner:

    ```bash
    ollama pull qwen3-coder:30b
    ```

    `qwen3-coder:30b` is a 30B MoE with ~3.3B active parameters — the quality/speed sweet spot on CPU because active-param count drives inference speed, not total. Use the `local-pro` alias (`qwen3-coder-next`) for a larger MoE if disk + quality budget allows.

3. Verify the server is reachable from the runner's host:

    ```bash
    curl http://localhost:11434/api/tags
    ```

    Should return a JSON list including the model you pulled.

4. Run the review:

    ```bash
    uv run review.py --provider ollama --base origin/main
    ```

The runner's Ollama-specific errors are surgical: a connection refusal raises `ConfigError` (exit 2) suggesting `ollama serve`; a 404 on the model raises `ConfigError` (exit 2) with the exact `ollama pull <model>` command to run. No retries on these — they're configuration problems the caller has to fix.

### Model selection

`--model <slug>` overrides the default. Aliases are **scoped per provider** — each one only resolves under its declared `--provider`. Passing an alias to the wrong provider raises a typed `CONFIG` error naming the right one rather than sending an invalid model name to the upstream API.

**OpenRouter aliases** (use with `--provider openrouter`, the default):

| Alias | Resolves to | Notes |
|---|---|---|
| `pro` / `gemini-pro` | `google/gemini-2.5-pro` | Current default. |
| `flash` / `gemini-flash` | `google/gemini-2.5-flash` | ~3× faster than `pro` with some quality loss — use during heavy iteration, switch to `pro` for the final pass. |
| `claude` / `claude-sonnet` | `anthropic/claude-sonnet-4.5` | Great as a second-opinion reviewer alongside a Gemini round. |
| `claude-opus` | `anthropic/claude-opus-4.5` | Larger model; slower and pricier. |
| `gpt` | `openai/gpt-5` | Independent third opinion. |
| `gpt-mini` | `openai/gpt-5-mini` | Cheaper / faster GPT. |
| `deepseek` | `deepseek/deepseek-chat-v3.1` | Cheap, surprisingly strong on code review. |

**Ollama aliases** (use with `--provider ollama`):

| Alias | Resolves to | Notes |
|---|---|---|
| `local` | `qwen3-coder:30b` | Current Ollama default. 30B MoE with ~3.3B active params — the recommended quality/speed balance on CPU. |
| `local-pro` | `qwen3-coder-next` | 80B/3B MoE. Higher quality at the cost of ~52 GB download and slightly slower active path. |

**The `gemini` (direct-API) provider has no aliases** — it takes bare Gemini model names only (e.g. `gemini-2.5-pro`, `gemini-2.5-flash`). Raw provider-native slugs always pass through unchanged, so anything OpenRouter serves or anything pulled into Ollama — including newer models without aliases yet — works via `--model <slug>`.

### Tuning sampling: `--temperature` and `--max-tokens`

Two runtime knobs control how much the model says and how exploratory it is:

| Flag | Default | What it controls | When to change |
|---|---|---|---|
| `--temperature <float>` | `0.3` (env: `CODE_REVIEW_TEMPERATURE`) | Sampling randomness. Higher = more exploration, more findings per call, more hallucinations. Lower = tighter, more conservative, fewer findings. | Drop to `0.2` for security-critical PRs where decline-comment overhead is expensive. Raise to `0.5–0.7` for first-pass audits where you want maximum coverage and can afford the false-positive rate. |
| `--max-tokens <int>` | `16000` (env: `CODE_REVIEW_MAX_TOKENS`) | Ceiling on output token count. Not a target — you pay only for what the model actually emits. Default ensures a thorough review isn't truncated mid-finding. | Rarely needs changing. Drop to `4000` if you genuinely want short, focused output (e.g. CI-step where only critical findings matter). Raise if you see truncated `Suggested change:` blocks in very large reviews. |
| `--min-severity <LEVEL>` | `LOW` (no filter) | Prompt-level severity floor: `MEDIUM`/`HIGH`/`CRITICAL` drop lower-severity findings from the output entirely. | `HIGH` for fast pre-commit gates; leave at `LOW` for the thorough pre-PR pass. |
| `--retries <N>` | `0` (env: `CODE_REVIEW_RETRIES`) | Extra retry attempts beyond the built-in single transient retry; `N > 0` also retries RATE_LIMIT honoring `Retry-After` (clamped 300s). | Set `2–3` for unattended runs; keep `0` when your agent loop manages its own backoff. |

The temperature default has been retuned twice based on empirical observation:

- **`0.2`** (original): too conservative — 1–2 findings per round on diffs that plausibly contained more, requiring 5–7 rounds to converge.
- **`0.5`** (raised in response to the above): Prone to hallucinations. more findings per round (3–5 typical), but during cross-model integration testing `google/gemini-2.5-pro` produced a HIGH-severity finding that referenced a CLI flag (`--timeout`) and quoted "help text" that did not exist in the codebase. The proposed fix would have crashed the runner with `AttributeError: 'Namespace' object has no attribute 'timeout'`. Confident, well-formatted, and a hallucination.
- **`0.3`** (current): tight enough to cut the hallucination rate, loose enough to keep "more findings than 0.2." If your project shows different behavior, retune; the constant in `review.py` documents the history so the next maintainer can see the evidence.

### Whole-codebase mode (`--codebase`)

For situations that diff review can't help with — auditing a repo you just inherited, finding bugs in code none of your PRs have touched — `--codebase` bundles every tracked file (filtered) and reviews them as a single payload.

```bash
uv run review.py --codebase                              # all tracked files, minus built-in noise
uv run review.py --codebase --include 'backend/**/*.py'  # narrow to a directory + extension
uv run review.py --codebase --exclude '**/test_*'        # widen then narrow
uv run review.py --codebase --model claude               # use Claude as the codebase reviewer
```

Selection pipeline: `git ls-files` → user `--include` (if any) → user `--exclude` → built-in defensive excludes (lock files, minified bundles, binary asset extensions, `*/dist/*`, `*/build/*`) → drop individual files >100 KB (logged on stderr).

Pre-flight bundle cap is 700 K chars (~175 K tokens — conservative against both Gemini 2.5 Pro's 1 M context and Claude Sonnet 4.5's 200 K context). If the bundle exceeds the cap, the runner exits with the 10 largest files in the current selection so you can target `--exclude` effectively rather than paying for a request that would fail mid-flight.

Output is the same severity-tagged per-file findings format as diff mode — same accept/decline heuristics apply.

---

## Iteration loop

```
1. Edit the target repo (fix a bug, build a feature, etc.).
2. (If reviewing untracked files in codebase mode: `git add -N <paths>`
   first so `git ls-files` sees them. No staged content; just makes
   them visible to the bundler.)
3. Run:  uv run --project <runner> <runner>/review.py --base origin/main
4. Read the structured-markdown output. Findings are tagged
   CRITICAL > HIGH > MEDIUM > LOW.
5. For each finding, decide: accept or decline.
6. Apply accepted fixes inline. Do NOT commit yet.
7. Re-run step 3.
8. Repeat until ONE of:
   - Output is "No issues found. Code looks clean..."
   - A round produces only hallucinated findings (decline-only round)
   - You have hit 4 rounds and remaining findings are stylistic noise
9. Run tests + build. Commit. Push.
```

**Do not commit between rounds.** `--base origin/main` uses a two-dot diff (`git diff -U5 <base>`), which includes working-tree edits. Re-running picks up in-progress fixes immediately. Committing every round produces noisy history; the reviewer is happy reviewing uncommitted edits.

**The 4-round cap is a pragmatic ceiling, not a hard rule.** Empirically, real bugs surface in rounds 1–3 and round 4 is usually either clean or "all hallucinations." If round 4 is still producing substantive findings, the diff is too large — split it.

**Codebase mode requires tracked files.** The runner uses `git ls-files` for safety (so it doesn't accidentally review `.venv/`, build artefacts, etc.). For new code that hasn't been committed yet, `git add -N <paths>` marks files as intent-to-add — they appear in `git ls-files` but their content stays unstaged. This is the cleanest way to get a review on work-in-progress without committing prematurely.

---

## Accept / decline heuristics

**Accept by default:**

- **CRITICAL / HIGH** unless the finding is factually wrong (rare). Real correctness or security bugs.
- **MEDIUM** about correctness, concurrency, atomicity, latent bugs, schema or type safety.
- **MEDIUM** about defensive coding — wider exception catches, header normalization, partial-key cache invalidation. Cheap hardening.
- **LOW** when trivially right: consolidating duplicates, fixing typos, tightening assertions, correcting inaccurate comments.

**Decline (and add a code comment explaining why):**

- Findings that contradict load-bearing design intent already encoded. *Shape:* a pair of API endpoints that look redundant but are deliberately split for a planned future migration. The reviewer doesn't know your roadmap unless your code comments tell it.
- Findings that would untighten a deliberately-tight test. *Shape:* a hardcoded expected count the reviewer wants computed dynamically from the fixture — that turns the test into a tautology (*what the code reads == what the code reports*) and removes the regression-catch.
- Stylistic preferences that don't change correctness when the existing form has a defensible rationale. *Shape:* a cache `maxsize` chosen with deliberate headroom for test fixtures rather than the obvious-singleton value.
- **Hedged findings ("if X is true, this is HIGH/CRITICAL").** The reviewer hedges when it lacks evidence to fully evaluate a hypothesis. In practice these resolve as false positives more often than as real bugs. *Shape:* "If `approved_body_hash` doesn't exclude `route_approval`, this is CRITICAL" — verify the premise (`approved_body_hash` does exclude it via an explicit `APPROVED_BODY_HASH_EXCLUDES` constant) before acting. Treat the hedge as a flag to spot-check the premise, not as an instruction.
- **Self-refuting findings.** When a finding's own "better fix" matches the existing code, the reviewer hallucinated a bug it then walked back. *Shape:* "Wrong type at L772 — a better fix would be to pass the list of vectors directly" when the actual line (which may not even be L772) already does exactly that. Decline without action; no code comment needed because the next round won't re-flag the same hallucination consistently.

---

## The decline contract

**Every declined finding gets a code comment immediately adjacent to the flagged code**, explaining why the suggestion was rejected. Without the comment, the next iteration's model will surface the same finding again. The comment is a contract with future review rounds — it's how you teach the reviewer about decisions it cannot infer from the diff alone.

This is the central operational rule. Without it, the loop churns on the same findings indefinitely. With it, the loop converges to clean in a small number of rounds.

---

## Error model

The runner exits with **typed exit codes** so an LLM caller can react differently to each failure mode without parsing prose. The full table + decision tree lives in the README's "Error model (for LLM callers)" section; the short version:

| Exit | Category | Quick action |
|---|---|---|
| 0 | OK | use stdout |
| 2 | CONFIG | fix env / CLI flag; don't retry |
| 10 | SAFETY_REFUSAL | retry with `--model claude` |
| 11 | RATE_LIMIT | wait 60s; switch provider if persistent |
| 12 | CONTEXT_OVERFLOW | narrow scope; don't retry as-is (if max_tokens was hit before any content: raise `--max-tokens`; if the Ollama guard/post-verify fired: raise `$OLLAMA_NUM_CTX` — requested per call, RAM permitting) |
| 13 | PROVIDER_HICCUP | retry; runner already auto-retried once |
| 14 | TRANSPORT | exponential backoff; escalate after 3 |
| 1 | UNKNOWN | read stderr; escalate |

Stderr always starts with a stable line `ERROR: <CATEGORY> [exit <N>]` followed by free-form human-readable detail. An agent can grep that prefix to extract the category without parsing. Informational stderr lines never start with `ERROR:` — they use the fixed prefixes `Reviewing`, `WARN:`, `[usage]`, `[retry]`, `[ollama]`, and `skip` (full table in the README's "Stderr format" section).

The runner **auto-retries once** on `PROVIDER_HICCUP` and `TRANSPORT` before exiting; other categories surface immediately because retrying without changes doesn't help. Pass `--retries N` (env `CODE_REVIEW_RETRIES`) to grant N additional attempts with exponential backoff — that also enables `RATE_LIMIT` retries honoring the provider's `Retry-After` (clamped to 300s). `CONFIG` / `SAFETY_REFUSAL` / `CONTEXT_OVERFLOW` are never retried. An agent that manages its own retry loop should keep `--retries 0` (the default) to stay in control of timing.

Two agent-facing conveniences: `--dry-run` resolves everything and prints prompt sizes / est. tokens / the file list without spending tokens (read-only git/gh subprocesses and the Ollama `/api/ps` probe still run, so exit-12 parity holds), and `--output <path>` writes the review to a file alongside stdout. `--min-severity HIGH` narrows a round to the findings worth acting on first.

**For high-stakes reviews, run a panel.** `--models pro,claude,deepseek --format json` fans the same review across models and merges findings with `found_by` consensus annotations. A finding two vendors agree on is nearly always real; a singleton is where hallucinations live (see "Local vs cloud"). Panel exit contract: ≥1 model succeeded → exit 0 with per-model failures as `WARN: [panel] …` lines and `per_model[].error` entries; all failed → one typed error chosen by `CONFIG > SAFETY_REFUSAL > CONTEXT_OVERFLOW > RATE_LIMIT > PROVIDER_HICCUP > TRANSPORT > UNKNOWN`.

**Reach for `--full-files` when hunk context isn't enough.** Diff review sees ±5 lines around each change; `--full-files` adds the changed files' complete content as reference so the model can catch changes that are locally fine but wrong against code elsewhere in the file. **Reach for `--chunk` when the payload can't fit** (huge diffs, whole codebases through a small local window): sequential per-chunk reviews, fail-fast (exit 0 only if every chunk succeeded), at the documented cost of cross-chunk blindness.

**Prefer `--format json` when consuming findings programmatically.** It parses the markdown into `{summary, findings[{file, line, severity, title, body, suggestion, fingerprint}], parse_ok, …}` locally (prompts unchanged); on parse failure it exits 0 with `parse_ok: false` and the raw markdown embedded — branch on the field, not the exit code. Chain rounds with `--output round1.json` then `--baseline round1.json`: findings come back tagged `new`/`persisting` plus a `resolved` list, which replaces "re-read the whole review and remember what you declined" with set arithmetic. Matching is heuristic (location carries reworded titles; ±10-line window) — treat statuses as strong hints.

## Safety context

The runner prepends a short framing prefix to every review request, wrapped in `<CONTEXT_FOR_REVIEWER>...</CONTEXT_FOR_REVIEWER>`, to reduce false-positive content-filter refusals on diffs that contain words like `attack`, `sanctions`, `prompt injection`, `tampering`, `redaction` (common in security testing, AML compliance, policy enforcement code).

Override per call with `--context "<your phrasing>"` or per environment with `$CODE_REVIEW_CONTEXT`. Disable with `--no-context` (rarely useful — the default already drops refusal rate significantly).

See the README's "Safety context" section for the default phrasing.

## Known gotchas

1. **Free-tier 429 on Gemini direct.** `--provider gemini --model gemini-2.5-pro` requires a paid Google AI Studio plan; the free tier returns HTTP 429 immediately (`RATE_LIMIT`, exit 11). Either use `--provider openrouter` (preferred) or `--model gemini-2.5-flash`.

2. **Codebase mode line numbers used to drift.** Before commit `b124501`, the bundle had no per-line anchors and the model estimated line positions from visual context, drifting 5–150 lines depending on file size. As of `b124501` every content line is pre-numbered (`cat -n` style) and the model transcribes the prefix instead of counting. If you ever see drift again on a current build, that's a regression worth investigating — the prompt or bundle format may have been changed in a way that broke the contract.

3. **Two-dot vs three-dot diff.** `--base <ref>` uses two-dot (`git diff -U5 <ref>`) so working-tree changes show up. Three-dot (`<ref>...HEAD`) would show only committed changes and the reviewer would keep re-flagging the same issues. The two-dot semantics is intentional for the iteration workflow.

4. **Diff size shapes round count.** Larger diffs surface more findings per round and take more rounds to converge. Rough observed shapes:
   - Small PR (~25K-char diff, single feature): 3–4 rounds
   - Medium PR (~50K chars): 4–6 rounds
   - Large PR (~300K chars): 8–12 rounds

5. **`tee` to a file.** When output is large, pipe to `tee /tmp/review.md` so you can re-read findings without re-invoking the tool. Saves context budget on subsequent steps.

6. **Codebase-mode bundle cap.** `--codebase` enforces a 700 K-char (~175 K-token) pre-flight cap on the concatenated bundle. If you hit it, the runner exits with `CONTEXT_OVERFLOW` (exit 12) and lists the 10 largest files in the current selection — use those to target `--exclude` flags. Common offenders: vendored fixture JSON, committed schema dumps, test data files.

7. **Ollama cold-start latency.** The first request after starting `ollama serve` (or after a model has been idle for a few minutes) takes 10–60 s extra while the model loads into RAM. Subsequent calls within Ollama's keep-alive window are fast. If you see no output for ~30 s on the first call, that's normal — don't assume the runner hung. `OLLAMA_TIMEOUT` (default 1800 s) covers worst-case CPU cold-start plus a thorough review; lower it if you want faster failure.

8. **Ollama "model not pulled" returns a `CONFIG` error (exit 2), not a 404.** The runner intercepts the 404 from the local server and surfaces it as a typed `CONFIG` error with the exact `ollama pull <model>` command to run. Don't retry without pulling first — retry without changes hits the same error.

9. **Truncated-but-nonempty output warns on stderr.** If the model hits the `max_tokens` ceiling but still returned content, the runner prints the partial review (exit 0) plus a `WARN: ... truncated at max_tokens` line on stderr. If you're parsing findings programmatically, check stderr for that warning before treating the list as complete.

10. **Ollama silently truncates oversized prompts — the runner guards against it twice.** Ollama generates from whatever fragment of the prompt fits `num_ctx` instead of erroring, which would yield a plausible-looking review of a fraction of the diff. The runner resolves the window per call and *requests* it via the native endpoint's `options.num_ctx`: `$OLLAMA_NUM_CTX` if set (hard pre-flight exit 12 on likely overflow) → the loaded model's actual window from `/api/ps`, requested back unchanged (hard pre-flight exit 12; covers every round after the first in an iterative loop) → window unknown: `num_ctx` omitted so the server keeps its VRAM-tier default (4K/32K/256K), the guard **warns only**, and after the call the runner checks `prompt_eval_count` against the (now-detectable) window — if the prompt filled it, the output is **discarded with a hard exit 12** rather than returned as a bogus review. Note the post-check can false-negative (KV-cache reuse undercounts `prompt_eval_count` on repeated prefixes) but not false-positive. If you see the WARN or exit 12, set `$OLLAMA_NUM_CTX` to your actual window (requested per call — no server restart, RAM permitting).

11. **Ollama review depth is lower than cloud.** Local models, especially on CPU, tend to under-report findings versus a cloud reviewer on the same diff. This is the inverse of the cloud-hallucinates problem documented under "Local vs cloud." A clean "no issues" from Ollama is **not** equivalent in confidence to a clean review from `claude` or `pro` — treat it as a sanity check, not a final verdict, when stakes are high.

---

## Future modes (TODO)

Two extensions deferred to v2; both have explicit design notes in this section so a future maintainer or LLM agent picking up the project knows the trade-offs already considered.

### Architectural-summary output shape (`--summary`)

A proposed flag that would prepend a high-level "patterns / structure / smells" section to the per-file findings produced by `--codebase`. Useful as orientation on a new-to-you codebase ("here's what this repo is, and here's the shape of its problems").

**Why deferred:** the per-file findings shape that ships today is safer because:

1. **Lower hallucination risk.** Line-level findings either match the code or they don't — directly verifiable. Architectural takes require deep context the model doesn't reliably have, and the model will produce one regardless of whether the codebase actually has the problem it describes.
2. **More actionable.** Per-file findings translate directly to edits. Architectural observations require human translation, and often the "fix" isn't a code change but a design decision.
3. **Cheaper token-wise.** A serious architectural section costs 1–2 K output tokens that would otherwise go to concrete findings.

Implementation sketch when the time comes: a separate `--summary` flag (codebase-mode-only) plus a tweak to `commands/codebase-review.toml` to ask for a leading architectural section. The skill prompt likely needs an extra severity bucket (`OBSERVATION`?) or explicit framing that the architectural section is exempt from the "demonstrable bug or improvement" requirement that gates the per-file findings.

### Per-file iteration

If a codebase legitimately exceeds the bundle cap and `--include` filtering would lose too much context, an alternative is one model call per file, aggregating findings client-side. Cheaper per call but more total calls; loses cross-file context the bundle approach preserves. Not in scope until a real user hits the cap on a project they can't reasonably narrow.

---

## When to also call `/gemini review` on GitHub (or run a second provider)

Treat the runner as the **iteration partner** and a second reviewer as the **final-mile verifier**. Options for the verifier:

- **GitHub `/gemini review` bot** — same model family as the default, similar prompts; only advantage is independence ("a third party reviewed this, not my own prompt-following loop").
- **A different runner provider on the same diff** — cheap and immediate. Cloud and local have different blind spots; running both is the closest thing to a structured "second opinion" you can get without paying for it.

Bring in a verifier when:

- The diff touches concurrency, locks, signing/replay, differential privacy ledgers, policy boundaries, or auth surfaces — anywhere a missed bug has outsized blast radius.
- The PR is large enough that you want a credibility signal beyond "I ran one model locally."
- You're about to merge to a protected branch and want one independent confirmation.

**Concrete pattern:** when the cloud reviewer says "no issues," re-run with `--provider ollama` (or vice versa). When they disagree, the disagreement itself is signal — verify the specific finding against the actual code. When they agree on "clean," that's a stronger green light than either alone.

For most PRs, three to five clean local rounds is sufficient and saves 30–45 minutes per PR vs the webhook round-trip.

---

## Per-round tracking

Keep a one-line ledger per round so the final commit message or PR comment can summarize the cycle:

```
Iter N: <severity> <one-line-description> -> applied | declined-with-comment | clean
```

After the cycle, the ledger becomes a table in the commit body or PR comment:

```
| Round | Finding                                            | Action                |
|-------|----------------------------------------------------|-----------------------|
| 1     | MED: cache duplicated across modules               | Applied               |
| 1     | LOW: lru_cache maxsize stylistic nit               | Declined w/ comment   |
| 2     | MED: CWD-relative path resolution is fragile       | Applied               |
| 3     | None -- "No issues found. Code looks clean..."     | Ready                 |
```

This is the artifact a human reviewer reads to understand what changed and why. Don't skip it.

---

## One-liner reference

The command shape used most often during iterative work:

```bash
uv run --project /path/to/local-gemini-code-review /path/to/local-gemini-code-review/review.py --base origin/main 2>&1 | tee /tmp/review.md
```

Tail the file in another shell, or pipe to `head -80` if you only want the top findings.

---

## Provenance

This fork keeps the upstream `gemini-cli-extensions/code-review` skill and command prompts byte-identical so upstream improvements rebase cleanly. Fork additions: `review.py` (three-provider runner: OpenRouter / Gemini API / local Ollama), `pyproject.toml`, `.env.example`, `.gitignore`, this runbook, and a rewritten root `README.md`. See the root README for the full list of fork modifications and how to sync against upstream.
