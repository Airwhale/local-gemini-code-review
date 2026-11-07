"""Claude Code CLI provider: one locked-down, read-only ``claude -p`` call.

``--provider claude-code`` sends the review prompts to the locally
installed Claude Code CLI in print mode, using whatever sign-in that CLI
already has (typically a Claude subscription). The runner never logs in,
switches accounts, reads credential stores, or enables extra usage.

The call is a single model turn with every agent capability removed:

- ``--tools ""`` plus ``--disallowedTools "mcp__*"``: no built-in or MCP
  tools, so nothing can edit files or run commands.
- ``--safe-mode``: CLAUDE.md, skills, plugins, hooks, MCP servers, custom
  commands/agents and auto memory are not loaded.
- ``--strict-mcp-config`` (with no ``--mcp-config``): no MCP servers.
- ``--disable-slash-commands``: prompt text is never expanded as a skill.
- ``--permission-mode dontAsk`` / ``--permission-prompts none``: anything
  that would need approval is denied, never prompted.
- ``--no-session-persistence``: nothing is saved for resumption.
- ``--system-prompt-file``: the review system prompt replaces Claude
  Code's default agent prompt (and its git-status snapshot).
- The process runs in a fresh, empty temporary directory, so the reviewed
  repository's ``.claude/`` settings, ``.mcp.json`` and ``CLAUDE.md`` are
  not in its working tree at all.

``--bare`` is deliberately NOT used: it skips OAuth/keychain sign-in and
would force API-key billing instead of the user's existing login.

The review payload goes through stdin; arguments are an argv list with
``shell=False``. ``--output-format json`` returns one result envelope,
validated here with Pydantic v2 before its ``result`` text flows into the
existing markdown review parser like any other provider's output. A zero
exit status alone never counts as success: the envelope must say so too.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from code_review.errors import (
    ConfigError,
    ContextOverflow,
    ProviderHiccup,
    RateLimit,
    ReviewError,
    SafetyRefusal,
    TransportError,
)
from code_review.providers import CallResult, _usage_int

PROVIDER = "claude-code"

# Wall-clock ceiling for one review call. A large review on Opus with
# adaptive thinking can take several minutes; past this the process tree
# is stopped and the call fails as TRANSPORT. Override with
# $CODE_REVIEW_CLAUDE_TIMEOUT (seconds).
DEFAULT_CLAUDE_TIMEOUT = 900.0
# The JSON envelope wraps one review; anything near this size is a
# runaway, not a review. Output past the cap is discarded and the process
# stopped rather than buffered without bound.
MAX_STDOUT_BYTES = 8 * 1024 * 1024
MAX_STDERR_BYTES = 256 * 1024
# Claude Code caps piped stdin at 10 MB and exits with an error past it;
# refusing up front gives a typed CONTEXT_OVERFLOW instead.
MAX_STDIN_BYTES = 10 * 1024 * 1024
_KILL_GRACE_SECONDS = 5.0
_READ_CHUNK = 64 * 1024
# Waits are sliced so Ctrl+C (and panel cancellation) is noticed promptly:
# an unsliced Popen.wait() on Windows cannot be interrupted.
_WAIT_SLICE_SECONDS = 0.25
# Set by cancel_running_calls() when a panel is interrupted; every bounded
# run notices it within one wait slice and stops its process tree.
_CANCEL = threading.Event()


def cancel_running_calls() -> None:
    """Ask every in-flight ``run_bounded`` call to stop its process."""
    _CANCEL.set()


def reset_cancellation() -> None:
    _CANCEL.clear()


# Environment variables removed from the child process.
#
# Paid-backend selectors: with any of these set, Claude Code bills an API
# key or a cloud provider instead of the signed-in subscription. The
# intended path for this provider is the existing login, and a silent
# switch to pay-per-token billing is exactly what must not happen -- so
# they are stripped (by name; values are never printed) and the run
# either uses the login or fails with a typed auth error.
_PAID_BACKEND_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_FOUNDRY_API_KEY",
)
# This tool's own credentials. Claude Code never needs them and the call
# has no tools, but there is no reason to hand them to another process.
_RUNNER_SECRET_ENV = (
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITLAB_TOKEN",
    "GLAB_TOKEN",
)

# Claude Code CLI model aliases that float to newer models. The provider
# requires a full model ID so a review is pinned to a known model.
_FLOATING_ALIASES = frozenset(
    {"sonnet", "opus", "haiku", "fable", "default", "best", "opusplan"}
)
_MODEL_ID_RE = re.compile(r"^claude-[a-z0-9]+(?:[-.][a-z0-9]+)*(?:\[1m\])?$")

# cmd.exe metacharacters. A .cmd/.bat shim (npm-style installs on
# Windows) is executed through cmd.exe, which re-parses arguments; refuse
# to pass any argument containing these rather than trust escaping.
_BATCH_UNSAFE = frozenset('%^&|<>"!\r\n')


@dataclasses.dataclass(frozen=True)
class ProcessOutcome:
    """Raw result of one bounded subprocess run."""

    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    stdout_overflow: bool = False
    stderr_overflow: bool = False
    cancelled: bool = False


class ClaudeCodeResultEnvelope(BaseModel):
    """The ``--output-format json`` result object (outer envelope only).

    Strict on the fields that decide success, permissive (``extra=
    "ignore"``) on everything else so new CLI fields never break a
    review. ``usage`` / ``modelUsage`` stay loosely typed: they are
    optional metadata, read defensively, and must never fail a review
    that otherwise succeeded.
    """

    model_config = ConfigDict(extra="ignore", strict=True, populate_by_name=True)

    type: Literal["result"]
    subtype: str
    is_error: bool
    result: str | None = None
    stop_reason: str | None = None
    terminal_reason: str | None = None
    api_error_status: int | None = None
    num_turns: int | None = None
    errors: list[Any] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = Field(default=None, alias="modelUsage")
    permission_denials: list[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration helpers (used by config._resolve_settings)
# ---------------------------------------------------------------------------


def validate_claude_model(model: str) -> str:
    """Require a full, pinned Claude model ID (e.g. ``claude-opus-5-5``).

    Claude Code's own aliases (``sonnet``, ``opus``...) resolve to whatever
    model is newest, so a review run with them is not reproducible.
    """
    if model in _FLOATING_ALIASES or model.endswith("-latest"):
        raise ConfigError(
            f"Model {model!r} is a floating Claude Code alias. --provider "
            "claude-code needs a full model ID so the reviewer is pinned, "
            "e.g. --model claude-opus-5-5."
        )
    if not _MODEL_ID_RE.fullmatch(model):
        raise ConfigError(
            f"Model {model!r} is not a native Claude model ID. --provider "
            "claude-code takes IDs like claude-opus-5-5 (not OpenRouter "
            "slugs such as anthropic/claude-sonnet-4.5)."
        )
    return model


def _find_on_path(name: str, path_value: str | None, pathext: str | None) -> str | None:
    """PATH lookup that only ever considers ABSOLUTE PATH entries.

    ``shutil.which`` on Windows searches the current directory first
    (unless ``NoDefaultCurrentDirectoryInExePath`` is set), and the
    current directory is usually the repository under review -- a PR
    that adds ``claude.exe`` at its root would otherwise get executed.
    Relative and empty PATH entries (``.``, ``bin``, ``""``) are skipped
    for the same reason on every platform.
    """
    if sys.platform == "win32":
        suffixes = [s for s in (pathext or ".COM;.EXE;.BAT;.CMD").split(";") if s]
    else:
        suffixes = [""]
    for entry in (path_value or "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if not entry or not os.path.isabs(entry):
            continue
        for suffix in suffixes:
            candidate = Path(entry) / f"{name}{suffix}"
            if candidate.is_file() and (
                sys.platform == "win32" or os.access(candidate, os.X_OK)
            ):
                return str(candidate)
    return None


def resolve_claude_command(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Locate the Claude Code executable: ``$CODE_REVIEW_CLAUDE_BIN`` or PATH.

    Returns the argv prefix (one absolute path). Missing executable is a
    typed CONFIG error raised before any git or network work. Never
    resolves through the current directory (see ``_find_on_path``).
    """
    env = os.environ if environ is None else environ
    explicit = (env.get("CODE_REVIEW_CLAUDE_BIN") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise ConfigError(
                f"$CODE_REVIEW_CLAUDE_BIN={explicit!r} must be an absolute path "
                "(a relative path would resolve inside the reviewed repository)."
            )
        if not path.is_file():
            raise ConfigError(
                f"$CODE_REVIEW_CLAUDE_BIN={explicit!r} does not exist or is "
                "not a file. Point it at the Claude Code executable, or unset "
                "it to search PATH."
            )
        return (str(path),)
    found = _find_on_path("claude", env.get("PATH"), env.get("PATHEXT"))
    if found is None:
        raise ConfigError(
            "Claude Code executable `claude` not found on PATH (relative PATH "
            "entries and the current directory are never searched). Install "
            "Claude Code (https://code.claude.com/docs/en/setup) and sign in "
            "yourself, or set $CODE_REVIEW_CLAUDE_BIN to its absolute path."
        )
    return (found,)


# Upper bound for $CODE_REVIEW_CLAUDE_TIMEOUT: one day. Larger values (or
# ``inf``) overflow Windows' millisecond wait and would crash untyped.
MAX_CLAUDE_TIMEOUT = 86_400.0


def resolve_claude_timeout(raw: str | None, source: str) -> float:
    if raw is None:
        return DEFAULT_CLAUDE_TIMEOUT
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"claude_timeout {raw!r} (from {source}) is not a valid number (seconds)."
        ) from exc
    if not 0 < timeout <= MAX_CLAUDE_TIMEOUT:
        raise ConfigError(
            f"claude_timeout={timeout} (from {source}) must be between 0 and "
            f"{MAX_CLAUDE_TIMEOUT:.0f} seconds."
        )
    return timeout


def _user_settings_paid_auth(environ: Mapping[str, str]) -> list[str]:
    """Names of paid-auth settings in Claude Code's USER settings file.

    The child environment is scrubbed, but ``~/.claude/settings.json`` can
    still supply ``apiKeyHelper`` or an ``env`` block with an API key, which
    Claude Code would bill instead of the subscription. Only key NAMES are
    inspected and reported; values are never read into output. Best-effort:
    an unreadable or malformed file yields no warning.
    """
    base = (environ.get("CLAUDE_CONFIG_DIR") or "").strip()
    settings_path = (Path(base) if base else Path.home() / ".claude") / "settings.json"
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    found = []
    if data.get("apiKeyHelper"):
        found.append("apiKeyHelper")
    env_block = data.get("env")
    if isinstance(env_block, dict):
        paid = {name.upper() for name in _PAID_BACKEND_ENV}
        found += sorted(
            f"env.{k}" for k, v in env_block.items() if k.upper() in paid and v
        )
    return found


# ---------------------------------------------------------------------------
# Process construction
# ---------------------------------------------------------------------------


def build_claude_argv(
    command: Sequence[str], *, model: str, system_prompt_file: Path
) -> list[str]:
    """The full argument vector for one review call (no shell involved)."""
    return [
        *command,
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--system-prompt-file",
        str(system_prompt_file),
        "--tools",
        "",
        "--disallowedTools",
        "mcp__*",
        "--strict-mcp-config",
        "--safe-mode",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--permission-prompts",
        "none",
    ]


def build_child_env(parent: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """Copy ``parent`` minus paid-backend selectors and runner secrets.

    Returns ``(env, stripped_paid_backend_names)``; names only, values are
    never reported.
    """
    paid = {name.upper() for name in _PAID_BACKEND_ENV}
    secrets = {name.upper() for name in _RUNNER_SECRET_ENV}
    env: dict[str, str] = {}
    stripped: list[str] = []
    for key, value in parent.items():
        upper = key.upper()
        if upper in paid:
            if value:
                stripped.append(upper)
            continue
        if upper in secrets:
            continue
        env[key] = value
    return env, sorted(set(stripped))


def _secret_values(parent: Mapping[str, str]) -> list[str]:
    names = {n.upper() for n in (*_PAID_BACKEND_ENV, *_RUNNER_SECRET_ENV)}
    return [v for k, v in parent.items() if k.upper() in names and len(v) >= 8]


_REDACTIONS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}"),
)
_KEYED_SECRET = re.compile(
    r"(?i)((?:access|refresh|id)?_?token|api[_-]?key|authorization|secret)"
    r"(\"?'?\s*[:=]\s*\"?'?)([^\s\"',}]{6,})"
)


def redact(text: str, secret_values: Sequence[str] = ()) -> str:
    """Mask credential-shaped substrings before text reaches stderr/JSON."""
    for value in secret_values:
        text = text.replace(value, "[REDACTED]")
    for pattern in _REDACTIONS:
        text = pattern.sub("[REDACTED]", text)
    return _KEYED_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)


def _check_batch_shim(argv: Sequence[str]) -> None:
    """Refuse cmd.exe re-parsing hazards for .cmd/.bat launchers."""
    if Path(argv[0]).suffix.lower() not in (".cmd", ".bat"):
        return
    for arg in argv[1:]:
        if any(ch in _BATCH_UNSAFE for ch in arg):
            raise ConfigError(
                f"The Claude Code launcher {argv[0]!r} is a batch script, and "
                "an argument contains characters cmd.exe would reinterpret. "
                "Set $CODE_REVIEW_CLAUDE_BIN to the native claude executable "
                "(claude.exe), or use a temp directory path without "
                '%^&|<>"! characters.'
            )


def _system_taskkill() -> str | None:
    """Absolute path of the OS ``taskkill.exe``, or ``None``.

    Never launched by bare name: Windows process creation searches the
    current directory (usually the reviewed repository) before System32,
    so a planted ``taskkill.exe`` would run. The directory comes from
    ``GetSystemDirectoryW``, not from environment variables.
    """
    if sys.platform != "win32":
        return None
    import ctypes

    buffer = ctypes.create_unicode_buffer(512)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < length < len(buffer):
        return None
    candidate = Path(buffer.value) / "taskkill.exe"
    return str(candidate) if candidate.is_absolute() and candidate.is_file() else None


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Stop ``proc`` and anything it started. Never raises."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        taskkill = _system_taskkill()
        if taskkill is not None:
            try:
                subprocess.run(
                    [taskkill, "/F", "/T", "/PID", str(proc.pid)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=str(Path(taskkill).parent),
                    check=False,
                    timeout=_KILL_GRACE_SECONDS * 2,
                )
            except (OSError, subprocess.SubprocessError):
                pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=_KILL_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        proc.kill()
    except OSError:
        pass


def _kill_leftovers(proc: subprocess.Popen[bytes]) -> None:
    """After the main process ends, stop anything left in its group.

    POSIX: the child led its own session, so its process group still names
    any helper it left behind. Windows has no equivalent once the parent
    is gone; those leftovers are bounded instead by the reader-thread join
    timeout in ``run_bounded`` (the call returns; nothing hangs). Never
    raises.
    """
    if sys.platform != "win32":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass


def run_bounded(
    argv: Sequence[str],
    *,
    stdin_bytes: bytes,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    max_stdout_bytes: int = MAX_STDOUT_BYTES,
    max_stderr_bytes: int = MAX_STDERR_BYTES,
) -> ProcessOutcome:
    """Run ``argv`` with stdin piped in, bounded in time and output size.

    Reader threads drain stdout/stderr concurrently (no pipe deadlock).
    Past its cap, stdout stops the process (the result envelope can no
    longer be trusted); stderr just stops being collected. On timeout,
    overflow, or any exception (including KeyboardInterrupt) the whole
    process tree is stopped before returning or re-raising.
    """
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
        env=dict(env),
        shell=False,
        **kwargs,
    )
    assert proc.stdin is not None and proc.stdout is not None
    assert proc.stderr is not None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = {"stdout": False, "stderr": False}

    def _reader(name: str, stream: IO[bytes], cap: int) -> None:
        sink = buffers[name]
        fd = stream.fileno()
        try:
            # os.read returns whatever is available; a buffered read(n) would
            # hold a small envelope back until n bytes or EOF, and a helper
            # process keeping the pipe open would then hide the output.
            while chunk := os.read(fd, _READ_CHUNK):
                if overflow[name]:
                    continue  # keep draining so the child never blocks
                room = cap - len(sink)
                if len(chunk) > room:
                    sink.extend(chunk[: max(room, 0)])
                    overflow[name] = True
                    if name == "stdout":
                        _kill_tree(proc)
                else:
                    sink.extend(chunk)
        except (OSError, ValueError):
            pass

    def _writer(stream: IO[bytes]) -> None:
        try:
            stream.write(stdin_bytes)
        except (BrokenPipeError, OSError, ValueError):
            pass  # the child exited early; its output explains why
        finally:
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    threads = [
        threading.Thread(
            target=_reader, args=("stdout", proc.stdout, max_stdout_bytes), daemon=True
        ),
        threading.Thread(
            target=_reader, args=("stderr", proc.stderr, max_stderr_bytes), daemon=True
        ),
        threading.Thread(target=_writer, args=(proc.stdin,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    cancelled = False
    deadline = time.monotonic() + timeout
    try:
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if _CANCEL.is_set():
                cancelled = True
                break
            try:
                proc.wait(timeout=min(_WAIT_SLICE_SECONDS, remaining))
            except subprocess.TimeoutExpired:
                continue
        if timed_out or cancelled:
            _kill_tree(proc)
            try:
                proc.wait(timeout=_KILL_GRACE_SECONDS * 2)
            except subprocess.TimeoutExpired:
                pass  # reported as timed out / cancelled either way
    except BaseException:
        _kill_tree(proc)
        raise
    finally:
        _kill_leftovers(proc)
        for thread in threads:
            thread.join(timeout=_KILL_GRACE_SECONDS)
        # Never close a pipe whose reader is still blocked on it: close()
        # would wait for that read, i.e. for whatever grandchild still holds
        # the pipe open. Abandoned readers are daemon threads.
        for thread, stream in zip(threads[:2], (proc.stdout, proc.stderr), strict=True):
            if not thread.is_alive():
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
    return ProcessOutcome(
        returncode=proc.returncode,
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
        timed_out=timed_out,
        stdout_overflow=overflow["stdout"],
        stderr_overflow=overflow["stderr"],
        cancelled=cancelled,
    )


# ---------------------------------------------------------------------------
# Result interpretation
# ---------------------------------------------------------------------------

_AUTH_PHRASES = (
    "failed to authenticate",
    "authentication",
    "not logged in",
    "not signed in",
    "please run /login",
    "/login",
    "oauth",
    "invalid api key",
    "unauthorized",
    "account_on_hold",
    "oauth_org_not_allowed",
)
_BILLING_PHRASES = (
    "credit balance",
    "billing",
    "payment",
    "extra usage",
    "purchase",
)
_QUOTA_PHRASES = (
    "usage limit",
    "rate limit",
    "rate_limit",
    "quota",
    "limit reached",
    "hit your limit",
    "too many requests",
    "out of usage",
)
_CONTEXT_PHRASES = (
    "prompt is too long",
    "context window",
    "context length",
    "exceeds the maximum",
    "input is too large",
)
_MODEL_PHRASES = (
    "model_not_found",
    "model not found",
    "invalid model",
    "unknown model",
    "may not exist",
    "does not have access to the model",
    "not available on your",
)
_OPTION_PHRASES = ("unknown option", "unknown argument", "error: option")
_TRANSPORT_PHRASES = (
    "overloaded",
    "server_error",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "connection error",
    "network error",
    "econnreset",
    "econnrefused",
    "etimedout",
    "request timed out",
)

_AUTH_HINT = (
    "Claude Code is not signed in, or its sign-in expired or was rejected. "
    "Sign in yourself by running `claude` interactively (or `claude auth "
    "login`), then rerun. This tool never logs in, switches accounts, or "
    "reads credentials, and it does not fall back to API-key billing."
)
_BILLING_HINT = (
    "Claude Code reported a billing or credit blocker. Resolve it in your "
    "Claude account yourself; this tool never buys credits, enables extra "
    "usage, or switches to a paid API backend."
)
_QUOTA_HINT = (
    "Claude Code reported a usage or rate limit. Wait for the limit to "
    "reset and rerun; this tool never enables extra usage or switches to a "
    "paid API backend."
)


def _has(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def classify_failure(
    message: str,
    *,
    model: str,
    status: int | None = None,
    detail: str | None = None,
) -> ReviewError:
    """Map a Claude Code failure (envelope text, error list or stderr) to a
    typed error. Status-derived classes win over phrase matches."""
    text = message.lower()
    kwargs: dict[str, Any] = {"detail": detail, "model": model, "provider": PROVIDER}
    reason = message.strip().splitlines()[0][:300] if message.strip() else ""
    suffix = f" ({reason})" if reason else ""
    if status in (401, 403):
        return ConfigError(f"{_AUTH_HINT}{suffix}", **kwargs)
    if status == 429:
        return RateLimit(f"{_QUOTA_HINT}{suffix}", **kwargs)
    if status == 413:
        return ContextOverflow(
            f"Claude Code rejected the prompt size{suffix}", **kwargs
        )
    if status == 404:
        return ConfigError(
            f"Claude Code could not use model {model!r}{suffix}. Check the model "
            "ID and that your plan includes it.",
            **kwargs,
        )
    if status is not None and status >= 500:
        return TransportError(f"Claude API returned HTTP {status}{suffix}", **kwargs)
    if _has(text, _OPTION_PHRASES):
        return ConfigError(
            "The installed Claude Code rejected a required option; update "
            f"Claude Code (needs v2.1.259 or later){suffix}",
            **kwargs,
        )
    if _has(text, _BILLING_PHRASES):
        return ConfigError(f"{_BILLING_HINT}{suffix}", **kwargs)
    if _has(text, _QUOTA_PHRASES):
        return RateLimit(f"{_QUOTA_HINT}{suffix}", **kwargs)
    if _has(text, _AUTH_PHRASES):
        return ConfigError(f"{_AUTH_HINT}{suffix}", **kwargs)
    if _has(text, _MODEL_PHRASES):
        return ConfigError(
            f"Claude Code could not use model {model!r}{suffix}. Check the model "
            "ID and that your plan includes it.",
            **kwargs,
        )
    if _has(text, _CONTEXT_PHRASES):
        return ContextOverflow(
            f"Claude Code rejected the prompt as too large{suffix}. Narrow the "
            "scope (--include / --exclude / smaller --base).",
            **kwargs,
        )
    if _has(text, _TRANSPORT_PHRASES):
        return TransportError(f"Claude API request failed{suffix}", **kwargs)
    return ReviewError(f"Claude Code reported an error{suffix}", **kwargs)


_API_FAILURE_LINE = re.compile(
    r"(?i)^(api error\b|prompt is too long\b|failed to authenticate\b|"
    r"invalid api key\b|not logged in\b|credit balance is too low\b)"
)


def _parse_envelope(stdout_text: str) -> ClaudeCodeResultEnvelope | None:
    """Validate the outer JSON envelope; ``None`` when it is not one.

    Tolerates a stray non-JSON line before the object by retrying on the
    last non-empty line (the result object is always printed last).
    """
    candidates = [stdout_text]
    lines = [line for line in stdout_text.splitlines() if line.strip()]
    if len(lines) > 1:
        candidates.append(lines[-1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        try:
            return ClaudeCodeResultEnvelope.model_validate(data)
        except ValidationError:
            continue
    return None


def _error_text(envelope: ClaudeCodeResultEnvelope) -> str:
    parts = [envelope.result or ""]
    parts += [e if isinstance(e, str) else json.dumps(e) for e in envelope.errors]
    if envelope.terminal_reason:
        parts.append(envelope.terminal_reason)
    return "\n".join(p for p in parts if p)


def interpret_outcome(
    outcome: ProcessOutcome,
    *,
    model: str,
    timeout: float,
    secret_values: Sequence[str] = (),
) -> CallResult:
    """Turn a finished process into a ``CallResult`` or a typed error."""
    stderr_text = redact(outcome.stderr.decode("utf-8", "replace"), secret_values)
    stdout_text = outcome.stdout.decode("utf-8", "replace").strip()
    if outcome.cancelled:
        raise TransportError(
            "Claude Code call cancelled; the process was stopped.",
            model=model,
            provider=PROVIDER,
        )
    if outcome.timed_out:
        raise TransportError(
            f"Claude Code did not finish within {timeout:.0f}s; the process "
            "was stopped and no output was used. Raise "
            "$CODE_REVIEW_CLAUDE_TIMEOUT or narrow the scope.",
            detail=stderr_text[:1000] or None,
            model=model,
            provider=PROVIDER,
        )
    if outcome.stdout_overflow:
        raise ContextOverflow(
            f"Claude Code output exceeded the {MAX_STDOUT_BYTES:,}-byte cap; the "
            "process was stopped and the output discarded.",
            model=model,
            provider=PROVIDER,
        )
    envelope = _parse_envelope(stdout_text)
    if envelope is None:
        redacted_stdout = redact(stdout_text, secret_values)
        snippet = redacted_stdout[:1000] or stderr_text[:1000]
        if outcome.returncode != 0:
            raise classify_failure(
                stderr_text or redacted_stdout or f"exit status {outcome.returncode}",
                model=model,
                detail=snippet or None,
            )
        raise ProviderHiccup(
            "Claude Code returned no valid JSON result envelope (empty, "
            "malformed, or truncated output)",
            detail=snippet or None,
            model=model,
            provider=PROVIDER,
        )
    detail = redact(_error_text(envelope), secret_values)[:1000] or None
    if envelope.is_error or envelope.subtype != "success":
        message = redact(_error_text(envelope), secret_values) or stderr_text
        if not message:
            message = f"subtype={envelope.subtype} exit status {outcome.returncode}"
        raise classify_failure(
            message,
            model=model,
            status=envelope.api_error_status,
            detail=detail,
        )
    if outcome.returncode != 0:
        # The envelope claims success but the process failed. Its result
        # text is a (possibly partial) review, not an error message, so
        # classify from stderr only: a review that mentions "authentication"
        # must not turn into a CONFIG error.
        raise classify_failure(
            stderr_text or f"exit status {outcome.returncode}",
            model=model,
            status=envelope.api_error_status,
            detail=stderr_text[:1000] or None,
        )
    result = envelope.result
    if envelope.stop_reason == "refusal":
        raise SafetyRefusal(
            "Claude declined to complete the review (stop_reason=refusal)",
            detail=detail,
            model=model,
            provider=PROVIDER,
        )
    truncated = envelope.stop_reason == "max_tokens"
    if result is None or not result.strip():
        if truncated:
            raise ContextOverflow(
                "Claude hit its output limit before producing a review "
                "(stop_reason=max_tokens). Narrow the scope.",
                model=model,
                provider=PROVIDER,
            )
        raise ProviderHiccup(
            "Claude Code reported success but the envelope has no result text",
            model=model,
            provider=PROVIDER,
        )
    stripped_result = result.strip()
    if "\n" not in stripped_result and _API_FAILURE_LINE.match(stripped_result):
        # Defensive: a bare one-line API failure reported as a "successful"
        # result is not a review; treating it as one would read as a clean
        # pass with no findings.
        raise classify_failure(
            redact(stripped_result, secret_values),
            model=model,
            status=envelope.api_error_status,
        )
    if truncated:
        sys.stderr.write(
            "WARN: claude-code output was truncated at the model's output "
            "limit; the review below may be incomplete.\n"
        )
    if envelope.permission_denials:
        sys.stderr.write(
            f"WARN: claude-code denied {len(envelope.permission_denials)} tool "
            "request(s); no tools were available and nothing was executed.\n"
        )
    if envelope.model_usage and not any(
        key.startswith(model) for key in envelope.model_usage
    ):
        sys.stderr.write(
            f"WARN: claude-code reported usage for {sorted(envelope.model_usage)} "
            f"but not the requested model {model}.\n"
        )
    usage = envelope.usage or {}
    prompt_parts = [
        _usage_int(usage.get(key))
        for key in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    ]
    known = [p for p in prompt_parts if p is not None]
    return CallResult(
        result,
        prompt_tokens=sum(known) if known else None,
        completion_tokens=_usage_int(usage.get("output_tokens")),
        truncated=truncated,
    )


def call_claude_code(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    command: Sequence[str],
    timeout: float,
    environ: Mapping[str, str] | None = None,
    temp_parent: Path | None = None,
) -> CallResult:
    """Run one review through ``claude -p`` and return a ``CallResult``.

    ``command`` is the launcher argv prefix (normally the resolved
    ``claude`` path). ``temp_parent`` chooses where the private working
    directory is created (default: the system temp dir).
    """
    if not command:
        raise ConfigError("No Claude Code executable configured.", provider=PROVIDER)
    stdin_bytes = user_prompt.encode("utf-8")
    if len(stdin_bytes) > MAX_STDIN_BYTES:
        raise ContextOverflow(
            f"Review prompt is {len(stdin_bytes):,} bytes; Claude Code accepts "
            f"at most {MAX_STDIN_BYTES:,} bytes on stdin. Narrow the scope.",
            model=model,
            provider=PROVIDER,
        )
    parent_env = dict(os.environ if environ is None else environ)
    env, stripped = build_child_env(parent_env)
    if stripped:
        sys.stderr.write(
            "NOTE: claude-code ignores "
            + ", ".join(stripped)
            + " so the review uses Claude Code's signed-in account, never "
            "pay-per-token API billing.\n"
        )
    settings_auth = _user_settings_paid_auth(parent_env)
    if settings_auth:
        sys.stderr.write(
            "WARN: Claude Code user settings define "
            + ", ".join(settings_auth)
            + "; Claude Code may bill that API key instead of your "
            "subscription. Remove it from your Claude Code settings if that "
            "is not intended.\n"
        )
    secrets = _secret_values(parent_env)
    with tempfile.TemporaryDirectory(
        prefix="code-review-claude-", dir=temp_parent, ignore_cleanup_errors=True
    ) as tmp:
        workdir = Path(tmp)
        prompt_file = workdir / "review-system-prompt.md"
        prompt_file.write_text(system_prompt, encoding="utf-8")
        argv = build_claude_argv(command, model=model, system_prompt_file=prompt_file)
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
            )
        except FileNotFoundError as exc:
            raise ConfigError(
                f"Claude Code executable {command[0]!r} not found. Install Claude "
                "Code or set $CODE_REVIEW_CLAUDE_BIN.",
                model=model,
                provider=PROVIDER,
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"Cannot start Claude Code executable {command[0]!r}: {exc}",
                model=model,
                provider=PROVIDER,
            ) from exc
    return interpret_outcome(
        outcome, model=model, timeout=timeout, secret_values=secrets
    )
