# Sequential panels and the Claude Code provider

Status: accepted. Date: 2026-11-07.

## Context

A common agent workflow uses two inexpensive models for regular review
rounds (for example `qwen/qwen3.8-flash` and `z-ai/glm-5.3` on OpenRouter).
The coding agent verifies their findings, fixes confirmed bugs and reruns
checks. A separate final review of the fixed code then uses a stronger
Claude model (Opus 5.5 by default; Sonnet 5.5 for a faster, lighter pass).
The final reviewer should run on the user's existing Claude Code sign-in
rather than a separately billed API key.

Panels already ran models concurrently (Ollama sequentially). Some users want
the regular reviewers called one at a time, for steadier rate-limit behavior
and easier-to-follow logs, without changing what the panel reports.

## Decision

### `--sequential`

- Opt-in flag for `--models` panels. Concurrent execution stays the default;
  Ollama stays sequential regardless.
- Models run in CLI order on the calling thread. Each receives the identical
  `ReviewRequest`; no model's output reaches another model's prompt.
- Results, failures, merging, `found_by`, usage and exit precedence go through
  the existing code unchanged, so both modes produce the same report.
- `--dry-run` prints the execution mode, derived from the same worker policy
  `_run_panel` uses.
- `--sequential` without a panel is a `CONFIG` error. A `models` list from
  `.code-review.toml` counts as a panel.

### `--provider claude-code`

- Opt-in provider that runs the installed `claude` executable in `--print`
  mode with `--output-format json`. The default model is the native, pinned
  ID `claude-opus-5-5` (changed from `claude-sonnet-5-5` during review:
  the final pass runs once per change, so the stronger model is worth the
  extra quota). Floating CLI aliases (`sonnet`, `opus`, ...) and
  non-native IDs are rejected.
- Read-only by construction: `--tools ""`, `--disallowedTools "mcp__*"`,
  `--strict-mcp-config`, `--safe-mode`, `--disable-slash-commands`,
  `--permission-mode dontAsk`, `--permission-prompts none` and
  `--no-session-persistence`. `--system-prompt-file` replaces the default
  agent prompt. The process runs in a fresh, empty temporary directory, so
  the reviewed repository's `.claude/`, `.mcp.json` and `CLAUDE.md` are not in
  its working tree.
- `--bare` is not used because it skips subscription/OAuth sign-in.
  `--dangerously-skip-permissions` is never used.
- Authentication is whatever the CLI already has. The runner never logs in,
  switches accounts or reads credential stores. To prevent a silent fallback
  to pay-per-token billing, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and
  the Bedrock/Vertex/Foundry selectors are removed from the child environment
  (named in a `NOTE:`, values never printed). The runner's own provider and
  forge credentials are removed too. Claude Code's user settings can still
  configure an `apiKeyHelper` or an API key in their `env` block, which the
  runner cannot override; it warns by setting name instead.
- The executable comes only from absolute `PATH` entries or an absolute
  `$CODE_REVIEW_CLAUDE_BIN`. Windows' default lookup searches the current
  directory first, which is usually the reviewed repository, so a planted
  `claude.exe` would otherwise run.
- The review prompt goes through stdin; arguments are an argv list with
  `shell=False`. `.cmd`/`.bat` launchers are refused when an argument
  contains `cmd.exe` metacharacters.
- The process is bounded: a wall-clock timeout (default 900 s, at most one
  day), an 8 MB stdout cap, a 256 KB stderr collection cap, and process-tree
  termination on timeout, overflow, Ctrl-C or panel cancellation. Waits are
  sliced so interrupts arrive promptly on Windows. Pipes are read with
  unbuffered reads and abandoned after a grace period if a leftover helper
  keeps them open, so cleanup cannot hang.
- The outer envelope is validated with Pydantic v2. Success requires exit 0,
  `type == "result"`, `subtype == "success"`, `is_error == false` and
  non-empty `result` text. The result text then goes through the existing
  markdown parser. A real signed-out CLI was observed returning
  `subtype: "success"` with `is_error: true` and exit 1, so no single signal
  is trusted alone.
- Failures map onto the existing exit codes: sign-in, billing, unknown model
  and unsupported-option failures are `CONFIG`; usage limits are
  `RATE_LIMIT`; oversized prompts or output are `CONTEXT_OVERFLOW`; empty,
  malformed or truncated JSON is `PROVIDER_HICCUP`; timeouts and API overload
  are `TRANSPORT`; refusals are `SAFETY_REFUSAL`. Error text is scrubbed of
  credential-shaped strings.
- No automatic retry, because each attempt spends subscription quota and can
  run for minutes. `--retries N` still opts in.
- The provider can be selected only by the CLI flag or environment. The
  reviewed repository's `.code-review.toml` cannot select it. Under this
  provider the file's `model`, `models` and `retries` keys are ignored,
  because they decide how many subscription-billed calls run and which model
  is billed. The executable and timeout are environment-only.
- The final Claude review is a separate invocation, not an automatic third panel
  member. It is meant to run after the coding agent has fixed the regular
  reviewers' findings.

## Consequences

- Existing providers, defaults, output schemas and exit codes are unchanged.
- Temperature and max-token settings are not sent to Claude Code; the
  dry-run report says so. No cost estimate is shown for this provider.
- `--safe-mode` still applies admin-managed policy settings, and Claude Code
  user-level settings outside the disabled customizations still load.
- Tests use a fake Claude executable (`tests/fixtures/fake_claude.py`) run
  through the real subprocess path, including paths with spaces, Unicode
  stdin, timeouts and output overflow.
