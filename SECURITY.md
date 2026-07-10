# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: [Report a vulnerability](https://github.com/Airwhale/local-gemini-code-review/security/advisories/new). Please don't open a public issue for anything exploitable. This is a maintained-in-spare-time fork — reports are handled best-effort, and the supported version is the tip of `main`.

## Threat model

This tool sends source code to an LLM provider and prints the response. Knowing exactly what it trusts, what it doesn't, and where your code travels is most of the security story.

### Where your code goes

| Provider | Your diff/codebase is sent to | Notes |
|---|---|---|
| `openrouter` | openrouter.ai, then the model vendor it routes to | Subject to OpenRouter's and the vendor's data policies |
| `gemini` | Google AI Studio (`generativelanguage.googleapis.com`) | Subject to Google's data policies |
| `ollama` | Your own machine (`localhost` by default) | Code never leaves the machine — use this for sensitive repos |

There is no other data egress: no telemetry, no crash reporting, no update checks.

### Trusted inputs

- **CLI flags and process environment** — fully trusted, including `.env` files loaded from *your* config locations (`$CODE_REVIEW_ENV`, the per-user config dir, the runner checkout). API keys come **only** from here.
- **Prompt assets** — shipped inside the installed wheel (or the runner checkout). `$CODE_REVIEW_PROMPT_DIR` can override them; setting it is an operator decision.

### Untrusted inputs

- **The code under review.** Diffs and file contents may contain adversarial text ("ignore previous instructions…"). The runner appends an injection guard to every request telling the model that instructions inside reviewed code are data to review, never directives — and to flag manipulation attempts. This is mitigation, not proof: treat model output accordingly (see below).
- **`.code-review.toml` in the reviewed repo.** A PR branch can add or edit this file, so its capabilities are deliberately capped:
  - Loading one is always announced on stderr (`[config] loaded <path>`).
  - It can never set **API keys**, the **prompt `context`** (that would let a repo instruct its own reviewer), or **`ollama_host` / `ollama_num_ctx` / `ollama_timeout`** (a hostile host URL would receive the full diff).
  - `--no-project-config` ignores the file entirely — use it when auditing untrusted checkouts, since even `exclude` can hide files from a `--codebase` review.
- **Model output.** It is printed (and optionally parsed into JSON) — never executed, never fed to a shell, never used for tool calls. The runner is a deterministic one-request primitive by design (see README Non-goals). If *you* pipe the output into something that acts on it (an agent loop, an auto-fixer), the trust decision is yours; findings can be hallucinated, and suggested-change diffs must be reviewed like any other untrusted patch.

### Subprocesses and network

- Local subprocesses are limited to read-only `git` (diffs, file listings) and `gh` (PR diffs/metadata). Nothing from the model or from reviewed repos is passed to a shell.
- HTTP requests go only to the selected provider endpoint (plus the local Ollama `/api/ps` probe). Retry-After sleep values from providers are clamped (300s max) so a hostile header can't stall a pipeline indefinitely.

## Hardening tips for CI

- Store the provider key as a repository/organization **secret**; grant the workflow only the permissions it needs (`contents: read`, plus `pull-requests: write` if posting comments).
- Prefer `--provider ollama` on self-hosted runners when the code must not leave your infrastructure.
- Pass `--no-project-config` when the workflow can run against fork PRs or otherwise untrusted branches.
- Remember GitHub does not expose secrets to `pull_request` runs from forks — a fork-PR review workflow needs `pull_request_target` with careful checkout handling, which is beyond this project's scope.
