# Codex CLI provider

Status: accepted. Date: 2026-11-07.

## Context

`--provider claude-code` (ADR 0004) runs a review through a locally
installed agent CLI on the user's existing subscription sign-in. The same
workflow is wanted for OpenAI models through the Codex CLI and a ChatGPT
sign-in, without API-key billing.

Claude Code can be made tool-free with `--tools ""`. Codex CLI 0.167 has no
equivalent. Probes of a real `codex exec` run with every available tool
feature disabled showed the model still sees `apply_patch`, sub-agent
(`collaboration.*`) and MCP-resource tools, plus a JavaScript `exec` tool.
Codex reports that tool as failing closed once `code_mode_host` is
disabled. The user's global `~/.codex/AGENTS.md` and the skills list still
load. `codex debug prompt-input` does not apply `--disable`, so it cannot
serve as evidence either way; the probes asked a real run directly.

## Decision

- Add opt-in `--provider codex`, defaulting to `gpt-6.1-sol` (the
  top-priority coding model in the catalog a ChatGPT sign-in offers).
  `-latest` slugs and `vendor/model` slugs are rejected.
- Run `codex exec --json --ephemeral --ignore-user-config --ignore-rules
  --skip-git-repo-check --sandbox read-only --cd <empty temp dir>` with
  `approval_policy='never'`, `forced_login_method='chatgpt'`,
  `project_doc_max_bytes=0`, `history.persistence='none'`,
  `skills.bundled.enabled=false`, `web_search='disabled'`,
  `model_instructions_file='<review system prompt>'` and every tool
  feature the CLI lets us disable. The payload goes on stdin (`-`).
- Strip `OPENAI_API_KEY`, `CODEX_API_KEY`, `AZURE_OPENAI_API_KEY` and other
  tools' credentials from the child environment. Never log in, never copy
  or relocate credentials (so `CODEX_HOME` is left alone, which is why the
  global `AGENTS.md` cannot be excluded).
- Treat tool use as a failed review: the shared bounded runner gained a
  per-line stdout hook, and the Codex provider stops the process tree as
  soon as an `item.*` event names any item other than `agent_message`,
  `reasoning` or `error`. Codex uses `error` items for its own diagnostics,
  such as ignored settings. The run then fails with exit 1 and its output
  is discarded. Together with the read-only sandbox and denied approvals,
  a prompt-injected tool call cannot write or use the network, and its
  result never becomes a review.
- Validate event lines with Pydantic v2. Success needs exit 0, a
  `turn.completed` event and a non-empty final `agent_message`.
  `turn.failed` / `error` events and nonzero exits map to the existing
  exit codes with the same rules as Claude Code.
- `-c` overrides use single-quoted TOML literal strings, so npm's
  `codex.cmd` launcher works on Windows under the existing batch-launcher
  argument check (no double quotes reach `cmd.exe`).
- Reuse the Claude Code provider's executable lookup (absolute PATH
  entries only), bounded runner, cancellation, redaction, timeout bounds,
  quota rules (no automatic retry; project config cannot select the
  provider or set `model`/`models`/`retries`) and dry-run reporting.

## Consequences

- Isolation is weaker than Claude Code's and is documented as such. The
  enforcement is detect-and-stop, not prevention: a tool call can begin
  before the runner stops it.
- The user's global Codex instructions can influence review style.
- Feature names are version-specific. A Codex CLI that does not know a
  disabled feature or option fails as `CONFIG` with an update hint; new
  tool-providing features in later versions would not be disabled
  automatically, but any resulting tool call still fails the review.
- Codex reports no output-limit stop reason, so `truncated` is always
  `false` for this provider.
- Verified against the real Codex CLI 0.167.0 with a ChatGPT sign-in: a
  synthetic review returned the expected finding with no tool events.
