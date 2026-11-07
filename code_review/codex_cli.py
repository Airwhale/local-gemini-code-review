"""OpenAI Codex CLI provider: one locked-down ``codex exec`` review call.

``--provider codex`` is the OpenAI counterpart of ``--provider
claude-code``: it sends the review prompts to the locally installed Codex
CLI in non-interactive ``exec`` mode, using the CLI's existing ChatGPT
sign-in. The runner never logs in, switches accounts or reads credential
stores, and forces ChatGPT sign-in so a review never bills an API key.

Isolation is weaker than Claude Code's and the gap is handled explicitly.
Codex CLI 0.167 has no switch that removes every tool: with all the
feature flags below disabled, the model can still see ``apply_patch``,
sub-agent and MCP-resource tools (the JavaScript ``exec`` tool is listed
but fails closed once ``code_mode_host`` is disabled), and the user's
global ``AGENTS.md`` is still loaded. So the call is constrained in depth:

- ``--sandbox read-only`` and ``approval_policy='never'``: tool commands
  cannot write files or reach the network, and anything needing approval
  is denied rather than prompted.
- ``--ignore-user-config`` / ``--ignore-rules`` / ``--ephemeral``: no
  user ``config.toml`` (MCP servers, profiles, providers), no exec-policy
  rules, nothing persisted.
- Every tool-providing feature that 0.167 lets us switch off is off.
- ``model_instructions_file`` replaces Codex's built-in agent
  instructions with the review system prompt.
- The process runs in a fresh, empty temporary directory.
- The runner watches the ``--json`` event stream and stops the process
  the moment an item other than a message or reasoning starts (a command,
  file change, MCP call, web search...), then fails the review: output
  from a run that used tools is never reported as a review.

The review payload goes through stdin; arguments are an argv list with
``shell=False``. ``-c`` overrides use single-quoted TOML literal strings
so they survive npm's ``codex.cmd`` launcher on Windows (cmd.exe mangles
double quotes; ``_check_batch_shim`` enforces that).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from code_review.claude_code import (
    _AUTH_PHRASES,
    _BILLING_PHRASES,
    _CONTEXT_PHRASES,
    _MODEL_PHRASES,
    _OPTION_PHRASES,
    _QUOTA_PHRASES,
    _RUNNER_SECRET_ENV,
    MAX_STDERR_BYTES,
    MAX_STDIN_BYTES,
    MAX_STDOUT_BYTES,
    ProcessOutcome,
    _check_batch_shim,
    _find_on_path,
    _has,
    redact,
    run_bounded,
)
from code_review.errors import (
    ConfigError,
    ContextOverflow,
    ProviderHiccup,
    RateLimit,
    ReviewError,
    TransportError,
)
from code_review.providers import CallResult, _usage_int

PROVIDER = "codex"
DEFAULT_CODEX_TIMEOUT = 900.0
MAX_CODEX_TIMEOUT = 86_400.0

# Feature flags verified to exist in Codex CLI 0.167; each removes a tool
# or a source of extra context. Unknown names would be rejected by the CLI
# (reported as a CONFIG error naming the version requirement).
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "view_image",
    "apps",
    "plugins",
    "multi_agent",
    "image_generation",
    "browser_use",
    "in_app_browser",
    "computer_use",
    "hooks",
    "goals",
    "memories",
    "code_mode_host",
    "sleep_tool",
    "tool_suggest",
    "skill_search",
)

# Removed from the child environment. API keys would let Codex bill
# pay-per-token instead of the ChatGPT plan (``forced_login_method`` also
# pins ChatGPT sign-in); the rest are other tools' credentials Codex never
# needs.
_PAID_BACKEND_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY", "AZURE_OPENAI_API_KEY")
_OTHER_SECRET_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

# Event items that are part of a plain answer, plus ``error`` items, which
# are Codex's own diagnostics (ignored settings, disabled features), not
# actions. Anything else is a tool.
_ALLOWED_ITEM_TYPES = frozenset({"agent_message", "reasoning", "error"})
# Diagnostics this configuration causes on purpose; not worth a WARN line.
_EXPECTED_DIAGNOSTICS = ("Code Mode is unavailable because code-mode host is disabled",)
_MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class CodexItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    type: str
    text: str | None = None
    message: str | None = None  # ``error`` (diagnostic) items


class CodexEvent(BaseModel):
    """One ``codex exec --json`` event line (outer shape only)."""

    model_config = ConfigDict(extra="ignore")

    type: str
    item: CodexItem | None = None
    usage: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# Configuration helpers (used by config._resolve_settings)
# ---------------------------------------------------------------------------


def validate_codex_model(model: str) -> str:
    """Require a concrete Codex model slug (e.g. ``gpt-6.1-sol``)."""
    if model.endswith("-latest"):
        raise ConfigError(
            f"Model {model!r} is a floating alias. --provider codex needs a "
            "concrete model slug so the reviewer is pinned, e.g. --model "
            "gpt-6.1-sol."
        )
    if not _MODEL_RE.fullmatch(model):
        raise ConfigError(
            f"Model {model!r} is not a Codex model slug. --provider codex takes "
            "slugs like gpt-6.1-sol (not OpenRouter slugs such as openai/gpt-5); "
            "`codex debug models` lists the ones your sign-in offers."
        )
    return model


def resolve_codex_command(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Locate Codex: absolute ``$CODE_REVIEW_CODEX_BIN`` or absolute PATH entries.

    Never resolves through the current directory (the reviewed repo).
    """
    env = os.environ if environ is None else environ
    explicit = (env.get("CODE_REVIEW_CODEX_BIN") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise ConfigError(
                f"$CODE_REVIEW_CODEX_BIN={explicit!r} must be an absolute path "
                "(a relative path would resolve inside the reviewed repository)."
            )
        if not path.is_file():
            raise ConfigError(
                f"$CODE_REVIEW_CODEX_BIN={explicit!r} does not exist or is not a "
                "file. Point it at the Codex executable, or unset it to search PATH."
            )
        return (str(path),)
    found = _find_on_path("codex", env.get("PATH"), env.get("PATHEXT"))
    if found is None:
        raise ConfigError(
            "Codex CLI `codex` not found on PATH (relative PATH entries and the "
            "current directory are never searched). Install it with `npm install "
            "-g @openai/codex`, sign in yourself with `codex login`, or set "
            "$CODE_REVIEW_CODEX_BIN to its absolute path."
        )
    return (found,)


def resolve_codex_timeout(raw: str | None, source: str) -> float:
    if raw is None:
        return DEFAULT_CODEX_TIMEOUT
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"codex_timeout {raw!r} (from {source}) is not a valid number (seconds)."
        ) from exc
    if not 0 < timeout <= MAX_CODEX_TIMEOUT:
        raise ConfigError(
            f"codex_timeout={timeout} (from {source}) must be greater than 0 "
            f"and at most {MAX_CODEX_TIMEOUT:.0f} seconds."
        )
    return timeout


# ---------------------------------------------------------------------------
# Process construction
# ---------------------------------------------------------------------------


def _toml_string(value: str) -> str:
    """Encode ``value`` as a TOML string for a ``-c key=value`` override.

    A single-quoted literal string when possible: it needs no escaping and
    no double quotes, so it survives npm's ``codex.cmd`` launcher. A value
    containing ``'`` or a line break cannot be a literal string, so it is
    sent as an escaped basic (double-quoted) string instead; that works with
    the native executable, and ``_check_batch_shim`` refuses it for a
    ``.cmd`` launcher with a pointer to the native one.
    """
    if not any(ch in value for ch in "'\r\n"):
        return f"'{value}'"
    # JSON string escapes are a subset of TOML basic-string escapes;
    # ensure_ascii=False keeps astral characters out of \\u surrogate pairs.
    return json.dumps(value, ensure_ascii=False)


def build_codex_argv(
    command: Sequence[str], *, model: str, workdir: Path, instructions_file: Path
) -> list[str]:
    """The full argument vector for one review call (no shell involved)."""
    overrides = {
        "approval_policy": "'never'",
        "forced_login_method": "'chatgpt'",
        "project_doc_max_bytes": "0",
        "history.persistence": "'none'",
        "skills.bundled.enabled": "false",
        "web_search": "'disabled'",
        "hide_agent_reasoning": "true",
        "model_instructions_file": _toml_string(str(instructions_file)),
    }
    argv = [
        *command,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--sandbox",
        "read-only",
        "--cd",
        str(workdir),
        "--model",
        model,
    ]
    for key, value in overrides.items():
        argv += ["-c", f"{key}={value}"]
    for feature in DISABLED_FEATURES:
        argv += ["--disable", feature]
    argv.append("-")  # read the prompt from stdin
    return argv


def build_child_env(parent: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """Copy ``parent`` minus API-key billing variables and other secrets."""
    paid = {name.upper() for name in _PAID_BACKEND_ENV}
    drop = {name.upper() for name in (*_RUNNER_SECRET_ENV, *_OTHER_SECRET_ENV)}
    env: dict[str, str] = {}
    stripped: list[str] = []
    for key, value in parent.items():
        upper = key.upper()
        if upper in paid:
            if value:
                stripped.append(upper)
            continue
        if upper in drop:
            continue
        env[key] = value
    return env, sorted(set(stripped))


def _secret_values(parent: Mapping[str, str]) -> list[str]:
    names = {
        n.upper() for n in (*_PAID_BACKEND_ENV, *_RUNNER_SECRET_ENV, *_OTHER_SECRET_ENV)
    }
    return [v for k, v in parent.items() if k.upper() in names and len(v) >= 8]


def _tool_event(line: bytes) -> bool:
    """True when a stdout event line shows a non-message item starting."""
    try:
        data = json.loads(line)
    except ValueError:
        return False
    if not isinstance(data, dict) or not str(data.get("type", "")).startswith("item."):
        return False
    item = data.get("item")
    return isinstance(item, dict) and item.get("type") not in _ALLOWED_ITEM_TYPES


# ---------------------------------------------------------------------------
# Result interpretation
# ---------------------------------------------------------------------------

_AUTH_HINT = (
    "Codex is not signed in, or its sign-in expired or was rejected. Sign in "
    "yourself with `codex login` (ChatGPT), then rerun. This tool never logs "
    "in or reads credentials, and it forces ChatGPT sign-in rather than "
    "API-key billing."
)
_QUOTA_HINT = (
    "Codex reported a usage or rate limit. Wait for the limit to reset and "
    "rerun; this tool never buys credits or switches to API-key billing."
)
_BILLING_HINT = (
    "Codex reported a billing or plan blocker. Resolve it in your ChatGPT "
    "account yourself; this tool never buys credits or switches billing."
)


def classify_failure(
    message: str, *, model: str, detail: str | None = None
) -> ReviewError:
    """Map a Codex failure message (event error or stderr) to a typed error."""
    text = message.lower()
    kwargs: dict[str, Any] = {"detail": detail, "model": model, "provider": PROVIDER}
    reason = message.strip().splitlines()[0][:300] if message.strip() else ""
    suffix = f" ({reason})" if reason else ""
    if _has(text, _OPTION_PHRASES) or "unexpected argument" in text:
        return ConfigError(
            "The installed Codex CLI rejected a required option; update it "
            f"(`npm install -g @openai/codex`, needs 0.167 or later){suffix}",
            **kwargs,
        )
    if "unknown feature" in text or "unknown variant" in text:
        return ConfigError(
            f"The installed Codex CLI rejected a configuration value; update it "
            f"(needs 0.167 or later){suffix}",
            **kwargs,
        )
    if _has(text, _BILLING_PHRASES):
        return ConfigError(f"{_BILLING_HINT}{suffix}", **kwargs)
    if _has(text, _QUOTA_PHRASES) or "429" in text:
        return RateLimit(f"{_QUOTA_HINT}{suffix}", **kwargs)
    if _has(text, _AUTH_PHRASES) or "401" in text or "login" in text:
        return ConfigError(f"{_AUTH_HINT}{suffix}", **kwargs)
    if _has(text, _MODEL_PHRASES) or (
        "model" in text and ("not supported" in text or "does not exist" in text)
    ):
        return ConfigError(
            f"Codex could not use model {model!r}{suffix}. `codex debug models` "
            "lists the models your sign-in offers.",
            **kwargs,
        )
    if _has(text, _CONTEXT_PHRASES) or "context_length_exceeded" in text:
        return ContextOverflow(
            f"Codex rejected the prompt as too large{suffix}. Narrow the scope "
            "(--include / --exclude / smaller --base).",
            **kwargs,
        )
    if any(p in text for p in ("stream disconnected", "server error", "503", "502")):
        return TransportError(f"Codex request failed{suffix}", **kwargs)
    return ReviewError(f"Codex reported an error{suffix}", **kwargs)


def _parse_events(stdout_text: str) -> list[CodexEvent]:
    events: list[CodexEvent] = []
    for line in stdout_text.splitlines():
        if not line.strip():
            continue
        try:
            events.append(CodexEvent.model_validate(json.loads(line)))
        except (ValueError, ValidationError):
            continue  # stray non-event lines are ignored
    return events


def _event_error_text(event: CodexEvent) -> str:
    if event.error:
        message = event.error.get("message")
        return message if isinstance(message, str) else json.dumps(event.error)
    return event.message or event.type


def interpret_outcome(
    outcome: ProcessOutcome,
    *,
    model: str,
    timeout: float,
    secret_values: Sequence[str] = (),
) -> CallResult:
    """Turn a finished ``codex exec --json`` run into a CallResult or error."""
    stderr_text = redact(outcome.stderr.decode("utf-8", "replace"), secret_values)
    stdout_text = outcome.stdout.decode("utf-8", "replace")
    events = _parse_events(stdout_text)
    tools = sorted(
        {
            e.item.type
            for e in events
            if e.item is not None and e.item.type not in _ALLOWED_ITEM_TYPES
        }
    )
    if outcome.stopped_early or tools:
        raise ReviewError(
            "Codex started a tool call ("
            + (", ".join(tools) or "non-message item")
            + ") during a review-only run; the process was stopped and its "
            "output discarded. Rerun; if it repeats, use another provider.",
            model=model,
            provider=PROVIDER,
        )
    if outcome.cancelled:
        raise TransportError(
            "Codex call cancelled; the process was stopped.",
            model=model,
            provider=PROVIDER,
        )
    if outcome.timed_out:
        raise TransportError(
            f"Codex did not finish within {timeout:.0f}s; the process was stopped "
            "and no output was used. Raise $CODE_REVIEW_CODEX_TIMEOUT or narrow "
            "the scope.",
            detail=stderr_text[:1000] or None,
            model=model,
            provider=PROVIDER,
        )
    if outcome.stdout_overflow:
        raise ContextOverflow(
            f"Codex output exceeded the {MAX_STDOUT_BYTES:,}-byte cap; the process "
            "was stopped and the output discarded.",
            model=model,
            provider=PROVIDER,
        )
    failures = [e for e in events if e.type in ("turn.failed", "error")]
    if failures:
        message = redact(_event_error_text(failures[-1]), secret_values)
        raise classify_failure(message, model=model, detail=message[:1000])
    completed = [e for e in events if e.type == "turn.completed"]
    if outcome.returncode != 0:
        fallback = redact(stdout_text.strip(), secret_values)
        raise classify_failure(
            stderr_text or fallback or f"exit status {outcome.returncode}",
            model=model,
            detail=(stderr_text or fallback)[:1000] or None,
        )
    if not completed:
        raise ProviderHiccup(
            "Codex exited without a turn.completed event (empty, malformed, or "
            "truncated output)",
            detail=redact(stdout_text, secret_values)[:1000]
            or stderr_text[:1000]
            or None,
            model=model,
            provider=PROVIDER,
        )
    messages = [
        e.item.text
        for e in events
        if e.type == "item.completed"
        and e.item is not None
        and e.item.type == "agent_message"
        and e.item.text
    ]
    if not messages or not messages[-1].strip():
        raise ProviderHiccup(
            "Codex completed the turn without a final message",
            model=model,
            provider=PROVIDER,
        )
    diagnostics: list[str] = []
    for e in events:
        if e.item is None or e.item.type != "error":
            continue
        text = redact(e.item.message or "", secret_values)
        first = text.strip().splitlines()[0] if text.strip() else ""
        if (
            first
            and first not in diagnostics
            and not first.startswith(_EXPECTED_DIAGNOSTICS)
        ):
            diagnostics.append(first)
    for line in diagnostics:
        sys.stderr.write(f"WARN: codex reported: {line[:300]}\n")
    usage = completed[-1].usage or {}
    return CallResult(
        messages[-1],
        prompt_tokens=_usage_int(usage.get("input_tokens")),
        completion_tokens=_usage_int(usage.get("output_tokens")),
    )


def call_codex(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    command: Sequence[str],
    timeout: float,
    environ: Mapping[str, str] | None = None,
    temp_parent: Path | None = None,
) -> CallResult:
    """Run one review through ``codex exec`` and return a ``CallResult``."""
    if not command:
        raise ConfigError("No Codex executable configured.", provider=PROVIDER)
    stdin_bytes = user_prompt.encode("utf-8")
    if len(stdin_bytes) > MAX_STDIN_BYTES:
        raise ContextOverflow(
            f"Review prompt is {len(stdin_bytes):,} bytes; the stdin limit for "
            f"local CLI providers is {MAX_STDIN_BYTES:,} bytes. Narrow the scope.",
            model=model,
            provider=PROVIDER,
        )
    parent_env = dict(os.environ if environ is None else environ)
    env, stripped = build_child_env(parent_env)
    if stripped:
        sys.stderr.write(
            "NOTE: codex ignores "
            + ", ".join(stripped)
            + " so the review uses Codex's ChatGPT sign-in, never API-key "
            "billing.\n"
        )
    secrets = _secret_values(parent_env)
    with tempfile.TemporaryDirectory(
        prefix="code-review-codex-", dir=temp_parent, ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp)
        workdir = root / "workspace"
        workdir.mkdir()
        instructions = root / "review-instructions.md"
        instructions.write_text(system_prompt, encoding="utf-8")
        argv = build_codex_argv(
            command, model=model, workdir=workdir, instructions_file=instructions
        )
        _check_batch_shim(argv)
        try:
            outcome = run_bounded(
                argv,
                stdin_bytes=stdin_bytes,
                cwd=workdir,
                env=env,
                timeout=timeout,
                max_stdout_bytes=MAX_STDOUT_BYTES,
                max_stderr_bytes=MAX_STDERR_BYTES,
                stop_on_stdout_line=_tool_event,
            )
        except FileNotFoundError as exc:
            raise ConfigError(
                f"Codex executable {command[0]!r} not found. Install it with "
                "`npm install -g @openai/codex` or set $CODE_REVIEW_CODEX_BIN.",
                model=model,
                provider=PROVIDER,
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"Cannot start Codex executable {command[0]!r}: {exc}",
                model=model,
                provider=PROVIDER,
            ) from exc
    return interpret_outcome(
        outcome, model=model, timeout=timeout, secret_values=secrets
    )
