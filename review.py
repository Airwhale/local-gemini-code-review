#!/usr/bin/env python3
"""Standalone code-review runner for the Gemini CLI code-review extension.

This fork keeps the upstream `skills/code-review-commons/SKILL.md` and
`commands/code-review.toml` prompts intact (Apache-2.0, unmodified) and adds a
thin Python runner that sends them to a Gemini-or-other-model via one of two
providers selectable at the command line:

  --provider openrouter (default)
      POSTs to OpenRouter's OpenAI-compatible chat-completions endpoint
      (https://openrouter.ai/api/v1/chat/completions). Requires
      `OPENROUTER_API_KEY`. Good if you want to mix models from different
      vendors (Gemini, Claude, GPT, DeepSeek) without separate API keys.

  --provider gemini
      POSTs to Google AI Studio's `generateContent` endpoint directly
      (https://generativelanguage.googleapis.com/v1beta/models/...). Requires
      `GEMINI_API_KEY`. Slightly lower latency (one less hop) and uses the
      same key the GitHub bot uses on the backend.

Both paths default to ``gemini-2.5-pro``. `--model <slug>` overrides; the
``--provider openrouter`` path also accepts named aliases (see
``MODEL_ALIASES`` below) so you can write ``--model claude`` instead of
``--model anthropic/claude-sonnet-4.5``.

Diff modes (default) review a git diff:

    uv run review.py                          # diff current branch vs origin/HEAD merge-base
    uv run review.py --base main              # diff vs an explicit base ref
    uv run review.py --pr 6                   # review a GitHub PR (uses `gh pr diff`)
    uv run review.py --staged                 # staged changes only

Whole-codebase mode reviews tracked files (filtered):

    uv run review.py --codebase
    uv run review.py --codebase --include 'backend/**/*.py'
    uv run review.py --codebase --exclude '**/test_*'

The .env loaded from this script's directory (not CWD) -- copy `.env.example`
to `.env` once at the runner location and invoke from any project folder.
Set whichever of `OPENROUTER_API_KEY` / `GEMINI_API_KEY` your chosen provider
needs.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable, TypeVar

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------
#
# The runner is designed to be called by an LLM agent in a loop, not just by a
# human at a shell. That means the exit code is a contract: an agent caller
# needs to be able to react differently to "model refused" vs "rate limit"
# vs "diff too big" vs "network down" without parsing prose. README's "Error
# model" section is the public-facing version of this table; the constants
# below are the in-code source of truth.
#
# Exit-code budget: 0 success, 1 catchall, 2 config, 10-19 reserved for typed
# review errors. Codes 3-9 are conventionally Unix-shell-reserved; we skip
# them.


T = TypeVar("T")


class ReviewError(Exception):
    """Base class for typed runner errors.

    Subclasses set ``exit_code``, ``category``, and ``suggested`` so the error
    formatter in ``main`` can emit a structured stderr block an LLM caller
    can pattern-match (``ERROR: <CATEGORY> [exit <N>]``).
    """

    exit_code: int = 1
    category: str = "UNKNOWN"
    suggested: str = "Read stderr for details; escalate if unclear."

    def __init__(
        self,
        message: str,
        *,
        detail: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.detail = detail
        self.model = model
        self.provider = provider


class ConfigError(ReviewError):
    """Missing API key, invalid CLI / env value, or other static-config bug."""

    exit_code = 2
    category = "CONFIG"
    suggested = (
        "Check the relevant env var or CLI flag (see the error message). "
        "Do not retry without fixing the config -- retry will hit the same "
        "error."
    )


class SafetyRefusal(ReviewError):
    """Model refused on content-filter grounds.

    Returned content was null with ``finish_reason`` indicating SAFETY,
    ``content_filter``, or equivalent. Re-trying the same model with the
    same prompt almost always reproduces the refusal; switch model first.
    """

    exit_code = 10
    category = "SAFETY_REFUSAL"
    suggested = (
        "Retry with a different model: ``--model claude`` is the most "
        "refusal-resistant on security / policy / adversarial-fixture code. "
        "If still refused across models, the content may need human review."
    )


class RateLimit(ReviewError):
    """Provider HTTP 429 -- request quota or RPS cap exceeded."""

    exit_code = 11
    category = "RATE_LIMIT"
    suggested = (
        "Wait 30-60 seconds and retry. If the limit is per-key per-day "
        "(common on free tiers), switch ``--provider`` or ``--model`` to "
        "one with separate quota."
    )


class ContextOverflow(ReviewError):
    """Diff exceeded the model's input or output token budget."""

    exit_code = 12
    category = "CONTEXT_OVERFLOW"
    suggested = (
        "Narrow the diff scope: ``--include`` / ``--exclude`` in codebase "
        "mode, or a smaller ``--base`` ref in diff mode. Do not retry "
        "without reducing scope -- a second call with the same payload "
        "will hit the same limit."
    )


class ProviderHiccup(ReviewError):
    """Null content with no clear safety / length signal.

    Empirically recovers ~always on a single retry. The runner auto-retries
    once before raising this; if you see this exit code, both attempts
    failed.
    """

    exit_code = 13
    category = "PROVIDER_HICCUP"
    suggested = (
        "The runner already auto-retried once. Wait a few seconds and "
        "retry; if still hicupped, switch ``--provider`` or ``--model``."
    )


class TransportError(ReviewError):
    """Network / HTTP-5xx failure reaching the provider."""

    exit_code = 14
    category = "TRANSPORT"
    suggested = (
        "Retry with exponential backoff (2s, 4s, 8s). The runner already "
        "retried once at 2s. If three additional retries fail, check the "
        "provider's status page; escalate if the provider is up."
    )


# ---------------------------------------------------------------------------
# Safety context prefix
# ---------------------------------------------------------------------------
#
# Prepended to every review prompt (overridable via --context / --no-context /
# $CODE_REVIEW_CONTEXT). Lowers the false-positive refusal rate when the
# diff under review contains words/patterns the model's safety filter
# associates with harm in other contexts (security testing, policy code,
# adversarial fixtures, AML domain language, etc.).
#
# Wording deliberately avoids "ignore safety guidelines" or similar
# jailbreak-shaped phrasing -- that itself triggers safety filters. The
# framing is "this is authorized code review of legitimate work"; the
# specific examples (sanctions, attack, prompt injection, tampering,
# redaction) are the words we've observed triggering false-positive
# refusals on our own PRs.

DEFAULT_CONTEXT = (
    "The diff below is from a legitimate software-engineering project "
    "undergoing authorized code review. The code may include defensive "
    "security measures, adversarial test fixtures, policy enforcement "
    "logic, or domain language that looks adversarial in isolation "
    "(e.g. 'sanctions', 'attack', 'prompt injection', 'tampering', "
    "'redaction'). Treat this as benign code review by the maintainers. "
    "Do not refuse on the basis of subject matter."
)

# Provider configuration. The default model slug differs by provider because
# OpenRouter prefixes vendor names (``google/...``) while the Gemini API
# accepts the bare model name. Override per-call with ``--model``.
PROVIDERS = ("openrouter", "gemini")
DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "openrouter": "google/gemini-2.5-pro",
    "gemini": "gemini-2.5-pro",
}
DEFAULT_PROVIDER = "openrouter"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
HTTP_TIMEOUT = 300.0  # Gemini 2.5 Pro on a ~5K-line diff lands ~30-60s; pad
                     # generously for very large diffs and whole-codebase
                     # bundles.

# Sampling temperature for the model. Raised from 0.2 to 0.5 after running
# several review cycles at 0.2 and observing 1-2 findings per round on
# diffs that plausibly contained more. Higher temperature broadens the
# model's exploration so more findings surface per call, at the cost of
# a moderately higher hallucination rate -- the decline-comment contract
# in the runbook handles those cleanly. Override per call with
# ``--temperature`` or per environment with ``$CODE_REVIEW_TEMPERATURE``.
DEFAULT_TEMPERATURE = 0.5

# Maximum output tokens the model may emit. Raised from the implicit
# provider default (~8K for Gemini 2.5 Pro) to 16K so a thorough review
# isn't truncated mid-finding. This is a *ceiling*, not a target -- you
# pay only for tokens actually emitted, not for unused headroom.
# Override with ``--max-tokens`` or ``$CODE_REVIEW_MAX_TOKENS``.
DEFAULT_MAX_TOKENS = 16000

# Named aliases for OpenRouter model slugs. The OpenRouter URL form
# ``vendor/model`` is awkward to type; an alias collapses it to a short
# name (``claude``, ``gpt``, ``deepseek``...) for ergonomics. Aliases are
# resolved before the call is made; the resolved slug is what shows up
# in stderr "Reviewing ... with ..." and what OpenRouter actually
# dispatches to.
#
# Aliases apply only to ``--provider openrouter``. The Gemini API direct
# path takes bare Gemini model names (no vendor prefix) so a slug like
# ``anthropic/claude-sonnet-4.5`` is a category error for that provider;
# we error out with a clear message instead of silently sending an
# invalid model name to Google's API.
#
# Keep this table small and curated. Adding every model on OpenRouter is
# not the goal -- users who want exotic models can pass the raw slug via
# ``--model``. The aliases here are the models that earned their place
# as a "second-opinion reviewer" in practice.
MODEL_ALIASES: dict[str, str] = {
    # Gemini family (OpenRouter route to the same models the
    # ``--provider gemini`` direct path serves).
    "pro": "google/gemini-2.5-pro",
    "gemini-pro": "google/gemini-2.5-pro",
    "flash": "google/gemini-2.5-flash",
    "gemini-flash": "google/gemini-2.5-flash",
    # Anthropic / Claude family.
    "claude": "anthropic/claude-sonnet-4.5",
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "claude-opus": "anthropic/claude-opus-4.5",
    # OpenAI / GPT family. These slugs are current OpenRouter catalog
    # entries; an older reviewer with a pre-2025 training cutoff may
    # flag them as nonexistent because GPT-5 / GPT-5-mini postdate that
    # cutoff. Verify against the live catalog before "fixing" them back
    # to gpt-4o.
    "gpt": "openai/gpt-5",
    "gpt-mini": "openai/gpt-5-mini",
    # DeepSeek -- cheap, surprisingly strong at code review.
    "deepseek": "deepseek/deepseek-chat-v3.1",
}

# Upstream `code-review.toml` instructs the model to *call* git itself via a
# tool. We have no tool layer; instead we extract the diff up front and
# substitute it into the prompt. This is the literal sentence from
# `commands/code-review.toml`; if upstream rewords it the substitution
# silently no-ops and the script still works (the diff is also appended
# unconditionally if the substitution missed).
TOOL_CALL_INSTRUCTION = (
    "**Code Changes**: call the `git diff -U5 --merge-base origin/HEAD` "
    "tool to retrieve the changes."
)

# Substitution sentinel for the codebase-review command template. The
# fork-added ``commands/codebase-review.toml`` puts this literal token
# where the codebase bundle goes; if it's missing (e.g. a future rewrite
# of the command template) the bundle is appended unconditionally so the
# model still has the content.
CODEBASE_PLACEHOLDER = "<CODEBASE_BUNDLE_PLACEHOLDER>"

# Whole-codebase mode constants.
#
# ``MAX_BUNDLE_CHARS`` is the hard pre-flight cap on the concatenated
# codebase bundle. 700K chars is ~175K tokens at the standard 4-chars-
# per-token estimate, which is conservative against both Gemini 2.5 Pro
# (1M-token context) and Claude Sonnet 4.5 (200K-token context). Cap
# means the runner errors out before paying for a request that would
# fail mid-flight on the smaller-context model.
MAX_BUNDLE_CHARS = 700_000

# Individual-file size cap: skip files larger than this when bundling.
# Most files this large are data fixtures, vendored blobs, or generated
# artifacts that ``git ls-files`` happens to track but that don't
# benefit from a code review. Skipped paths are logged on stderr so the
# user can ``--include`` them back if a real source file got caught.
MAX_INDIVIDUAL_FILE_BYTES = 100_000

# Defensive built-in exclusions applied after the user's ``--include``
# and ``--exclude`` filters. ``git ls-files`` already respects
# ``.gitignore``, so vendored dirs (``node_modules``, ``.venv``, etc.)
# are usually already absent. These patterns catch the residue:
# lock files, minified output, common binary / asset extensions. Match
# is case-sensitive (file extensions in practice are lowercase).
BUILTIN_CODEBASE_EXCLUDES: tuple[str, ...] = (
    # Lock files for various package managers.
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "uv.lock",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    # Minified / generated bundles.
    "*.min.js",
    "*.min.css",
    # Build outputs occasionally tracked by mistake.
    "*/dist/*",
    "*/build/*",
    # Binary / asset extensions: skip outright (model can't review).
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.ico", "*.webp",
    "*.woff", "*.woff2", "*.ttf", "*.eot",
    "*.pdf", "*.zip", "*.tar", "*.gz",
    "*.mp3", "*.mp4", "*.mov", "*.avi",
)

# Per-file delimiter for whole-codebase bundles. The shape is chosen so
# the model can pattern-match file boundaries reliably and quote the
# exact path back in its per-file findings.
FILE_DELIMITER_TEMPLATE = "======== FILE: {path} ========"


def load_skill(name: str = "code-review-commons") -> str:
    """Return the SKILL.md content for the named skill directory.

    Defaults to the upstream ``code-review-commons`` skill (the
    diff-review one). Whole-codebase mode passes ``code-review-codebase``
    (fork-added) which differs only in the Critical Constraints section:
    it permits commenting on any line in any file in the bundle, instead
    of the upstream skill's hardcoded "only lines beginning with +/-"
    rule that's correct for diff review but forbids all comments on
    whole-file input.
    """
    path = ROOT / "skills" / name / "SKILL.md"
    return path.read_text(encoding="utf-8")


def load_command_prompt(name: str) -> str:
    """Load `commands/<name>.toml` and return the `prompt` field verbatim."""
    path = ROOT / "commands" / f"{name}.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return data["prompt"]


def _apply_context(user_prompt: str, context: str | None) -> str:
    """Prepend a safety-context block to the user prompt.

    Wrapping the context in a labeled XML-style tag (``<CONTEXT_FOR_REVIEWER>``)
    rather than free-floating prose keeps the model from accidentally treating
    the framing as part of the code it should review. ``None`` / empty
    short-circuits to the bare prompt for ``--no-context`` mode.
    """
    if not context:
        return user_prompt
    return (
        "<CONTEXT_FOR_REVIEWER>\n"
        f"{context}\n"
        "</CONTEXT_FOR_REVIEWER>\n\n"
        f"{user_prompt}"
    )


def build_diff_prompts(diff: str, context: str | None) -> tuple[str, str]:
    """Construct ``(system, user)`` prompts for diff-mode review.

    Loads the upstream ``code-review-commons`` skill (system prompt) and
    the upstream ``code-review`` command (user prompt template), then
    substitutes the diff into the command's tool-call placeholder and
    prepends the optional safety-context block to the user prompt.
    """
    system_prompt = load_skill("code-review-commons")
    user_template = load_command_prompt("code-review")
    diff_block = f"**Code Changes**:\n\n```diff\n{diff}\n```"
    user_prompt = user_template.replace(TOOL_CALL_INSTRUCTION, diff_block)
    # Defensive: if upstream rewords the tool-call sentence and our literal
    # substitution missed, append the diff so the model still has it.
    if diff_block not in user_prompt:
        user_prompt = f"{user_prompt}\n\n{diff_block}"
    return system_prompt, _apply_context(user_prompt, context)


def build_codebase_prompts(bundle: str, context: str | None) -> tuple[str, str]:
    """Construct ``(system, user)`` prompts for whole-codebase review.

    Uses the fork-added ``code-review-codebase`` skill and
    ``codebase-review`` command. The skill differs from the upstream
    ``code-review-commons`` only in the Critical Constraints section:
    it permits commenting on any line in any file in the bundle (the
    upstream rule "comment only on +/- lines" forbids all comments on
    whole-file content, which is correct for diff review but wrong for
    codebase review).

    TODO: v2 -- architectural-summary mode (proposed flag ``--summary``)
    would prepend a leading "patterns / structure / smells" section to
    the per-file findings. Trade-offs (hallucination risk on
    architectural takes, less actionable output, token-budget
    contention) documented in ``docs/llm-code-review-runbook.md`` under
    "Future modes". The current codebase prompt produces per-file
    findings only.
    """
    system_prompt = load_skill("code-review-codebase")
    user_template = load_command_prompt("codebase-review")
    bundle_block = f"**Codebase**:\n\n{bundle}\n"
    if CODEBASE_PLACEHOLDER in user_template:
        user_prompt = user_template.replace(CODEBASE_PLACEHOLDER, bundle_block)
    else:
        # Defensive: same shape as ``build_diff_prompts`` defensive append.
        user_prompt = f"{user_template}\n\n{bundle_block}"
    return system_prompt, _apply_context(user_prompt, context)


def _format_size(n_bytes: int) -> str:
    """Format a byte count in decimal-prefix units the way OS file managers do.

    Returns ``"<n> B"`` below 1 KB, ``"<n> KB"`` (1000-based, integer) up to
    1 MB, and ``"<n.x> MB"`` with one decimal above that. Decimal (1000)
    rather than binary (1024) so the displayed value matches what users see
    in their file manager and matches the decimal byte constants declared
    in this module (``MAX_INDIVIDUAL_FILE_BYTES = 100_000`` reads as
    "100 KB"). One source of truth for size display means the messages
    stay sensible if the constants ever change scale.
    """
    if n_bytes >= 1_000_000:
        return f"{n_bytes / 1_000_000:.1f} MB"
    if n_bytes >= 1_000:
        return f"{n_bytes // 1_000} KB"
    return f"{n_bytes} B"


def _run_git(args: list[str]) -> str:
    """Run a git command in the current working directory and return stdout.
    Surfaces non-zero exits with the command and stderr so the user sees why.

    ``encoding="utf-8"`` is explicit: without it, ``text=True`` decodes via
    ``locale.getpreferredencoding(False)``, which on Windows is cp1252.
    Source files containing non-ASCII characters (em-dashes, arrows, the
    section sign) then come back to Python as mojibake (``â€"`` for ``--``,
    ``â†'`` for ``->``, ``Â§`` for ``§``), the model sees the mojibake in
    the diff, and the next review iteration flags "character encoding
    artifacts in documentation files" -- a finding that doesn't exist in
    the source, only in the runner's decoding step. ``errors="replace"``
    is defensive in case git ever emits bytes that aren't valid UTF-8
    (rare; usually a corrupted file).
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"ERROR: `{' '.join(args)}` failed (exit {exc.returncode})\n"
            f"{exc.stderr.strip()}\n"
        )
        sys.exit(exc.returncode)
    return result.stdout


def git_diff_local(base: str | None, staged: bool) -> str:
    """Produce a unified diff matching the upstream `git diff -U5
    --merge-base origin/HEAD` shape so the model sees what the GitHub bot
    would see.

    Working-tree changes are included by default. The iterative-review use
    case (run, fix, re-run before committing) breaks if we restrict to
    ``base...HEAD`` because uncommitted fixes are invisible and the model
    re-flags the same issues forever. ``git diff <base>`` (two-dot;
    merge-base..working-tree) covers committed + staged + unstaged in one
    pass, which is what "review everything I'm proposing to ship" means.
    """
    if staged:
        return _run_git(["git", "diff", "--cached", "-U5"])
    if base:
        return _run_git(["git", "diff", "-U5", base])
    return _run_git(["git", "diff", "-U5", "--merge-base", "origin/HEAD"])


def pr_diff(pr_number: int) -> str:
    """Pull a PR's diff via `gh`. Requires `gh auth login` to have run.

    Same Windows-locale concern as ``_run_git``: explicit
    ``encoding="utf-8"`` keeps PR diffs containing em-dashes, arrows,
    section signs, and other non-ASCII characters from being mangled
    into cp1252 mojibake before the model sees them.
    """
    try:
        result = subprocess.run(
            ["gh", "pr", "diff", str(pr_number), "--patch"],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        sys.stderr.write(
            "ERROR: `gh` not found on PATH. Install GitHub CLI to use --pr.\n"
        )
        sys.exit(2)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"ERROR: `gh pr diff {pr_number}` failed (exit {exc.returncode})\n"
            f"{exc.stderr.strip()}\n"
        )
        sys.exit(exc.returncode)
    return result.stdout


def _glob_match(path: Path, patterns: tuple[str, ...] | list[str]) -> bool:
    """Return True if ``path`` matches any of the glob ``patterns``.

    Each pattern is tested against both the full POSIX path
    (e.g. ``backend/api/views.py``) and the basename (e.g. ``views.py``)
    so a pattern like ``test_*.py`` catches test files at any depth
    rather than only at the repo root. fnmatch treats ``*`` as matching
    everything including ``/``, so ``*.py`` matches all Python files
    regardless of nesting; this is intentional and documented.

    Verified: ``fnmatch.fnmatch("foo/bar.py", "*.py") == True``. Python
    docs (fnmatch module): "Note that the filename separator (os.sep on
    Unix) is not special to this module." This is the OPPOSITE of
    ``glob`` semantics where ``*`` stops at ``/``; do not assume glob
    behavior when reading or modifying this function.

    The ``tuple[str, ...] | list[str]`` signature is intentionally
    explicit rather than ``Sequence[str]`` -- a ``str`` is itself a
    ``Sequence[str]`` (it iterates as single-character strings), so a
    looser annotation would silently accept a caller bug like
    ``_glob_match(p, "*.py")`` and iterate over individual characters
    instead of treating the string as one pattern. The verbose union
    blocks that footgun at type-check time.
    """
    posix = path.as_posix()
    name = path.name
    return any(
        fnmatch.fnmatch(posix, pat) or fnmatch.fnmatch(name, pat)
        for pat in patterns
    )


def gather_codebase_files(
    includes: list[str], excludes: list[str]
) -> list[Path]:
    """Return the list of files to bundle for whole-codebase review.

    Pipeline (in order):
      1. ``git ls-files`` -> all tracked files (so ``.gitignore`` already
         excludes ``node_modules``, ``.venv``, build artifacts, etc.).
      2. Apply ``--include`` globs if any; otherwise keep all files.
      3. Apply user-supplied ``--exclude`` globs.
      4. Apply ``BUILTIN_CODEBASE_EXCLUDES`` (lock files, asset
         extensions, etc.).
      5. Drop files larger than ``MAX_INDIVIDUAL_FILE_BYTES``; log to
         stderr so the user can ``--include`` them back if needed.

    Returns paths relative to the current working directory (which is
    expected to be the project being reviewed, since we run ``git
    ls-files`` against CWD).
    """
    output = _run_git(["git", "ls-files"])
    paths = [Path(line) for line in output.splitlines() if line]

    # Step 2: user --include filter.
    if includes:
        paths = [p for p in paths if _glob_match(p, includes)]

    # Step 3: user --exclude filter.
    if excludes:
        paths = [p for p in paths if not _glob_match(p, excludes)]

    # Step 4: built-in defensive excludes.
    paths = [p for p in paths if not _glob_match(p, BUILTIN_CODEBASE_EXCLUDES)]

    # Step 5: per-file size cap.
    kept: list[Path] = []
    for p in paths:
        try:
            size = p.stat().st_size
        except OSError:
            # File listed by ``git ls-files`` but missing on disk
            # (typo, symlink to nowhere, race). Skip silently rather
            # than crash; the user can re-run if a file was meant to
            # be present.
            continue
        if size > MAX_INDIVIDUAL_FILE_BYTES:
            sys.stderr.write(
                f"skip (>{_format_size(MAX_INDIVIDUAL_FILE_BYTES)}): "
                f"{p.as_posix()} ({_format_size(size)})\n"
            )
            continue
        kept.append(p)

    return kept


def _number_lines(content: str) -> str:
    """Prefix each line with its 1-indexed line number, ``cat -n`` style.

    LLMs cannot reliably count lines in long files. Without explicit
    line numbers in the bundle, the model estimates line positions from
    visual context and drifts by 50-150 lines on files >500 lines (and
    5-15 lines on files >100). Prefixing every line with its number
    turns "report a line number" from an arithmetic task (which
    transformers cannot do reliably) into a transcription task (which
    they do well).

    Diff mode doesn't need this -- ``git diff -U5`` already embeds
    ``@@ -L,N +L,N @@`` anchors plus context lines, so the model
    transcribes from those. The whole-codebase bundle has no such
    anchors, which is why this helper exists.

    Format: ``<width>d: <line>``, right-aligned, minimum 6-char width
    matching ``cat -n``. Trailing newline (if any) is preserved.
    """
    if not content:
        return content
    had_trailing_newline = content.endswith("\n")
    lines = content.splitlines()
    if not lines:
        return content
    width = max(6, len(str(len(lines))))
    numbered = "\n".join(
        f"{index:>{width}d}: {line}"
        for index, line in enumerate(lines, start=1)
    )
    return numbered + "\n" if had_trailing_newline else numbered


def bundle_codebase(file_paths: list[Path]) -> str:
    """Concatenate the given files into a single delimited bundle.

    Encoding errors are replaced silently (``errors="replace"``) so a
    non-UTF8 byte sequence in a tracked file doesn't crash the runner;
    the model sees a replacement character but the rest of the file is
    still reviewable. In practice this only triggers on files that
    should have been excluded by the asset-extension filter -- a true
    source file with a stray non-UTF8 byte is rare.

    Each file's content is line-numbered via ``_number_lines`` before
    bundling so the model can reference accurate line numbers in its
    findings without having to count lines itself. See that helper for
    the rationale.
    """
    parts: list[str] = []
    for path in file_paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        delimiter = FILE_DELIMITER_TEMPLATE.format(path=path.as_posix())
        parts.append(f"{delimiter}\n{_number_lines(content)}")
    return "\n\n".join(parts)


def _classify_http_error(
    status: int,
    body: str,
    *,
    model: str,
    provider: str,
) -> ReviewError:
    """Map a 4xx/5xx response to the right typed error.

    Provider-side messages are inconsistent across vendors, so we pattern-
    match on a small set of substrings the major providers use today. The
    classification is best-effort -- a future provider could land a 4xx
    that doesn't match any pattern, and that's fine: it falls through to
    a generic ``ReviewError`` which surfaces as exit 1 with the response
    body in ``detail`` so a caller can decide.
    """
    body_lower = body.lower()
    if status == 429:
        return RateLimit(
            f"{provider} returned HTTP 429",
            detail=body[:1000],
            model=model,
            provider=provider,
        )
    if status == 413 or "context_length" in body_lower or "too long" in body_lower or "exceeds the maximum" in body_lower:
        return ContextOverflow(
            f"{provider} returned HTTP {status} with context-length indication",
            detail=body[:1000],
            model=model,
            provider=provider,
        )
    if 500 <= status < 600:
        return TransportError(
            f"{provider} returned HTTP {status} (provider-side failure)",
            detail=body[:1000],
            model=model,
            provider=provider,
        )
    return ReviewError(
        f"{provider} returned HTTP {status}",
        detail=body[:1000],
        model=model,
        provider=provider,
    )


def call_openrouter(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    referer: str,
    title: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """POST to OpenRouter's chat-completions endpoint and return the review
    markdown. Caller builds the prompts so this function stays mode-agnostic
    (diff review and codebase review share the same wire path).

    Raises typed ``ReviewError`` subclasses on failure -- see the README's
    "Error model" section for the contract. ``main`` catches and formats
    them with the correct exit code.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Temperature broadens exploration so more findings surface per
        # call; the upstream prompt's "Critical Constraints" section
        # still gates *quality*. ``max_tokens`` is a ceiling so a
        # thorough review isn't truncated mid-finding -- the user pays
        # only for tokens actually emitted, not the unused headroom.
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter surfaces these in its dashboard for attribution; they
        # also help the platform-side routing decide which provider tier to
        # use. Harmless if absent but recommended.
        "HTTP-Referer": referer,
        "X-Title": title,
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.post(OPENROUTER_URL, headers=headers, json=payload)
    except httpx.RequestError as exc:
        # Network-level failure: DNS, TCP, timeout, connection reset.
        # Distinct from a provider 5xx (which is also a transport-class
        # error but at least returned a response).
        raise TransportError(
            f"OpenRouter request failed before response: {exc}",
            model=model,
            provider="openrouter",
        ) from exc

    if response.status_code >= 400:
        raise _classify_http_error(
            response.status_code, response.text, model=model, provider="openrouter"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderHiccup(
            f"OpenRouter returned non-JSON response: {exc}",
            detail=response.text[:1000],
            model=model,
            provider="openrouter",
        ) from exc

    choices = data.get("choices") or []
    if not choices:
        raise ProviderHiccup(
            "OpenRouter response had no choices",
            detail=str(data)[:1000],
            model=model,
            provider="openrouter",
        )
    choice = choices[0]
    finish_reason = (choice.get("finish_reason") or choice.get("native_finish_reason") or "").lower()
    message = choice.get("message") or {}
    content = message.get("content")

    if content:
        return content

    # Null / empty content -- classify by finish_reason.
    if finish_reason in ("safety", "content_filter", "content-filter", "blocked"):
        raise SafetyRefusal(
            f"Model refused with finish_reason={finish_reason!r}",
            detail=str(data)[:1000],
            model=model,
            provider="openrouter",
        )
    if finish_reason in ("length", "max_tokens"):
        raise ContextOverflow(
            f"Hit max_tokens ({max_tokens}) before producing content "
            f"(finish_reason={finish_reason!r})",
            detail=str(data)[:1000],
            model=model,
            provider="openrouter",
        )
    raise ProviderHiccup(
        f"Null content with finish_reason={finish_reason!r}",
        detail=str(data)[:1000],
        model=model,
        provider="openrouter",
    )


def call_gemini(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """POST to Google AI Studio's ``generateContent`` endpoint directly.
    Caller builds the prompts (same as ``call_openrouter``) so the wire
    path is mode-agnostic. Raises typed ``ReviewError`` subclasses; see
    README "Error model".
    """
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            # camelCase keys per the v1beta generateContent spec.
            # ``maxOutputTokens`` is the ceiling on generated tokens;
            # ``temperature`` matches the OpenRouter side so review
            # behavior is consistent across providers.
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    headers = {
        "Content-Type": "application/json",
        # ``x-goog-api-key`` is the documented auth header for the v1beta
        # generative-language API. Passing the key as a query string also
        # works but leaks it into shell history / proxy logs more readily.
        "x-goog-api-key": api_key,
    }
    url = GEMINI_URL_TEMPLATE.format(model=model)

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        raise TransportError(
            f"Gemini request failed before response: {exc}",
            model=model,
            provider="gemini",
        ) from exc

    if response.status_code >= 400:
        raise _classify_http_error(
            response.status_code, response.text, model=model, provider="gemini"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderHiccup(
            f"Gemini API returned non-JSON response: {exc}",
            detail=response.text[:1000],
            model=model,
            provider="gemini",
        ) from exc

    # Gemini can refuse at the prompt level (before generation) by
    # returning ``promptFeedback.blockReason`` with no candidates.
    prompt_feedback = data.get("promptFeedback") or {}
    block_reason = prompt_feedback.get("blockReason")
    if block_reason:
        raise SafetyRefusal(
            f"Gemini blocked prompt: blockReason={block_reason!r}",
            detail=str(data)[:1000],
            model=model,
            provider="gemini",
        )

    candidates = data.get("candidates") or []
    if not candidates:
        raise ProviderHiccup(
            "Gemini response had no candidates",
            detail=str(data)[:1000],
            model=model,
            provider="gemini",
        )
    candidate = candidates[0]
    finish_reason = (candidate.get("finishReason") or "").upper()
    content_block = candidate.get("content") or {}
    parts = content_block.get("parts") or []
    text = "".join(part.get("text", "") for part in parts)

    if text:
        return text

    # Null / empty content -- classify by finishReason.
    if finish_reason == "SAFETY":
        raise SafetyRefusal(
            "Gemini refused output with finishReason=SAFETY",
            detail=str(data)[:1000],
            model=model,
            provider="gemini",
        )
    if finish_reason == "MAX_TOKENS":
        raise ContextOverflow(
            f"Hit maxOutputTokens ({max_tokens}) before producing content "
            "(finishReason=MAX_TOKENS)",
            detail=str(data)[:1000],
            model=model,
            provider="gemini",
        )
    raise ProviderHiccup(
        f"Empty content with finishReason={finish_reason!r}",
        detail=str(data)[:1000],
        model=model,
        provider="gemini",
    )


def _retry_on_recoverable(call: Callable[[], T], *, label: str) -> T:
    """Run ``call``; on ``ProviderHiccup`` or ``TransportError`` retry once.

    Single retry only -- we deliberately do NOT compound on safety refusals
    or rate limits (a second call to the same model with the same prompt
    almost always reproduces those, and burns tokens) or context overflows
    (the scope is wrong, not the call). Caller can chain its own retries
    on the surviving exception if it wants exponential backoff.

    ``label`` shows up in the retry-notice stderr line so a viewer scrolling
    the log can tell which call retried.
    """
    try:
        return call()
    except (ProviderHiccup, TransportError) as exc:
        sys.stderr.write(
            f"[retry] {exc.category} on first attempt ({label}); "
            "retrying once in 2s...\n"
        )
        time.sleep(2)
        return call()


def _resolve_model(args: argparse.Namespace) -> str:
    """Resolve the final model slug from CLI flag, env var, alias table,
    or provider default. Raises ``ConfigError`` if an alias is used with the
    wrong provider (the Gemini API direct path takes bare Gemini model names
    only -- a non-Gemini alias like ``claude`` is a category error there).
    """
    if args.model is not None:
        model = args.model
    elif args.provider == "openrouter":
        model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL_BY_PROVIDER["openrouter"])
    else:  # gemini
        model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL_BY_PROVIDER["gemini"])

    if model in MODEL_ALIASES:
        if args.provider != "openrouter":
            raise ConfigError(
                f"Model alias `{model}` is only valid with --provider openrouter. "
                f"The Gemini API direct path takes bare Gemini model names only "
                f"(e.g. gemini-2.5-pro, gemini-2.5-flash). Either pass "
                f"--provider openrouter to use this alias, or pass "
                f"--model gemini-2.5-pro / --model gemini-2.5-flash for the "
                f"direct path."
            )
        model = MODEL_ALIASES[model]

    return model


def _print_error(err: ReviewError) -> None:
    """Emit a structured stderr block for ``err``.

    First line is the stable machine-parseable prefix:
    ``ERROR: <CATEGORY> [exit <N>]``. Subsequent lines are human-readable
    detail. LLM callers can grep for the first line to classify; humans can
    read the body.
    """
    sys.stderr.write(f"ERROR: {err.category} [exit {err.exit_code}]\n")
    sys.stderr.write(f"Reason: {err}\n")
    if err.model:
        sys.stderr.write(f"Model: {err.model}\n")
    if err.provider:
        sys.stderr.write(f"Provider: {err.provider}\n")
    sys.stderr.write(f"Suggested: {err.suggested}\n")
    if err.detail:
        # Truncate to keep the stderr block scannable; full body is on
        # ``err.detail`` if a caller wants programmatic access.
        snippet = err.detail if len(err.detail) <= 400 else err.detail[:400] + "..."
        sys.stderr.write(f"Detail: {snippet}\n")


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 if they aren't already.

    The model's output regularly contains Unicode characters (``->`` rendered
    as ``\\u2192``, em-dashes, smart quotes, mermaid arrows) that cp1252 -- the
    default stdout encoding on Windows -- cannot encode. Without this, the
    very last line of main(), ``print(output)``, crashes with
    ``UnicodeEncodeError`` after the model call has already succeeded and the
    user has already paid for the tokens. Forcing UTF-8 with
    ``errors="replace"`` keeps the runner robust on Windows without changing
    anything on macOS/Linux (which are already UTF-8 by default).
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                if (stream.encoding or "").lower() != "utf-8":
                    stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                # Best-effort. Some shells / CI pipes wrap stdout in a way
                # that doesn't expose ``reconfigure``; in those cases we
                # fall through and accept a possible UnicodeEncodeError
                # rather than hide a real configuration problem.
                pass


def main() -> None:
    _ensure_utf8_stdout()
    # Load env from this script's directory so `.env` is configured once at
    # the runner location rather than in every project we review from.
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "Standalone code-review runner using the Gemini CLI "
            "code-review extension prompts. Sends them to a Gemini-or-"
            "other model via OpenRouter or the Gemini API directly."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--base",
        help="Base ref to diff against (e.g. main, origin/main).",
    )
    source.add_argument(
        "--pr",
        type=int,
        help="GitHub PR number to review (uses `gh pr diff`).",
    )
    source.add_argument(
        "--staged",
        action="store_true",
        help="Review staged changes only.",
    )
    source.add_argument(
        "--codebase",
        action="store_true",
        help=(
            "Review the whole tracked codebase via ``git ls-files`` "
            "instead of a diff. Narrow with --include / --exclude. "
            "Output shape is per-file findings (severity-tagged) -- "
            "the architectural-summary shape is a v2 TODO documented "
            "in the runbook's 'Future modes' section."
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Glob to include in --codebase mode (e.g. "
            "``backend/**/*.py``). Can be passed multiple times. "
            "Ignored outside --codebase."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Glob to exclude in --codebase mode (e.g. "
            "``**/test_*.py``). Can be passed multiple times. "
            "Ignored outside --codebase."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=os.getenv("CODE_REVIEW_PROVIDER", DEFAULT_PROVIDER),
        help=(
            "Which API to call. ``openrouter`` (default) goes through "
            "OpenRouter's chat-completions endpoint and needs "
            "``OPENROUTER_API_KEY``. ``gemini`` calls Google AI Studio's "
            "generateContent endpoint directly and needs ``GEMINI_API_KEY``. "
            "Override with $CODE_REVIEW_PROVIDER."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model slug or alias. Defaults to the provider-appropriate "
            "``gemini-2.5-pro`` variant. Override with $OPENROUTER_MODEL "
            "or $GEMINI_MODEL respectively. Aliases (--provider "
            "openrouter only): pro, flash, claude, claude-opus, gpt, "
            "gpt-mini, deepseek. ``gemini-2.5-flash`` / ``flash`` is "
            "~3x faster with some quality loss."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            f"Sampling temperature. Default {DEFAULT_TEMPERATURE} -- "
            "raised from the previous 0.2 default to surface more "
            "findings per round at the cost of a higher hallucination "
            "rate (the decline-comment contract handles those). Range "
            "typically 0.0-1.0. Override with $CODE_REVIEW_TEMPERATURE."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        dest="max_tokens",
        help=(
            f"Maximum output tokens the model may emit. Default "
            f"{DEFAULT_MAX_TOKENS} -- raised from the implicit ~8K "
            "provider default so a thorough review isn't truncated "
            "mid-finding. This is a ceiling, not a target: you pay only "
            "for tokens actually emitted. Override with "
            "$CODE_REVIEW_MAX_TOKENS."
        ),
    )
    parser.add_argument(
        "--context",
        default=None,
        metavar="TEXT",
        help=(
            "Safety-context prefix prepended to every review prompt. "
            "Reduces false-positive content-filter refusals on security "
            "/ policy / adversarial-fixture code (the kind that contains "
            "words like 'attack', 'sanctions', 'prompt injection' out of "
            "context). Defaults to a generic 'authorized code review' "
            "framing; override with this flag or $CODE_REVIEW_CONTEXT to "
            "match your project's subject matter."
        ),
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help=(
            "Disable the safety-context prefix entirely. Useful only if "
            "the default phrasing itself is what triggers a refusal "
            "(rare). Mutually exclusive with --context."
        ),
    )
    args = parser.parse_args()

    if args.no_context and args.context is not None:
        raise ConfigError(
            "--no-context and --context are mutually exclusive. Pick one."
        )

    model = _resolve_model(args)

    # The two blocks below deliberately duplicate the (CLI -> env ->
    # default + validate) resolution pattern instead of extracting a
    # generic helper. Two call sites is too few to justify a 6-param
    # abstraction (cli_val, env_var_name, default, converter, type_name,
    # flag_name), and keeping the env-var names inline makes them
    # grep-discoverable ("where does CODE_REVIEW_TEMPERATURE get
    # resolved?" returns a single hit). Revisit if a third tunable
    # arrives.
    # Resolve temperature: explicit CLI flag wins, then env, then default.
    if args.temperature is not None:
        temperature = args.temperature
    else:
        env_temp = os.getenv("CODE_REVIEW_TEMPERATURE")
        try:
            temperature = float(env_temp) if env_temp is not None else DEFAULT_TEMPERATURE
        except ValueError as exc:
            raise ConfigError(
                f"$CODE_REVIEW_TEMPERATURE={env_temp!r} is not a valid float. "
                "Unset it or pass --temperature explicitly."
            ) from exc

    # Resolve max_tokens with the same precedence.
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    else:
        env_max = os.getenv("CODE_REVIEW_MAX_TOKENS")
        try:
            max_tokens = int(env_max) if env_max is not None else DEFAULT_MAX_TOKENS
        except ValueError as exc:
            raise ConfigError(
                f"$CODE_REVIEW_MAX_TOKENS={env_max!r} is not a valid integer. "
                "Unset it or pass --max-tokens explicitly."
            ) from exc

    # Resolve safety context: --no-context wins, then explicit --context,
    # then env, then default. Empty string from env is treated as "use
    # default" rather than "disabled" -- pass --no-context explicitly to
    # disable, since an env value of "" is more likely a misconfig than
    # intent.
    if args.no_context:
        context: str | None = None
    elif args.context is not None:
        context = args.context
    else:
        context = os.getenv("CODE_REVIEW_CONTEXT") or DEFAULT_CONTEXT

    # Resolve and validate the API key for the chosen provider before
    # touching git / GitHub so the user fails fast on configuration errors.
    if args.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigError(
                "OPENROUTER_API_KEY not set. Copy .env.example to "
                f".env at {ROOT} and fill in your key, or rerun with "
                "--provider gemini if you only have a Gemini API key."
            )
    else:  # gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigError(
                "GEMINI_API_KEY not set. Copy .env.example to .env "
                f"at {ROOT} and fill in your Google AI Studio key, or "
                "rerun with --provider openrouter if you only have an "
                "OpenRouter key."
            )

    # Build the payload for the chosen mode. ``--include`` / ``--exclude``
    # are codebase-mode-only -- warn if they show up alongside a diff
    # mode rather than silently dropping them.
    if args.codebase:
        files = gather_codebase_files(args.include, args.exclude)
        if not files:
            sys.stderr.write(
                "No files matched after --include / --exclude / built-in "
                "filters. Nothing to review.\n"
            )
            sys.exit(0)
        bundle = bundle_codebase(files)
        if len(bundle) > MAX_BUNDLE_CHARS:
            # Show the 10 largest files so the user can target
            # ``--exclude`` flags effectively rather than guessing.
            # We re-stat in this branch rather than threading the
            # sizes through ``gather_codebase_files``'s return type:
            # this is a cold error path (only fires when the bundle
            # exceeds the cap), so the redundant syscalls don't matter,
            # and the alternative -- returning ``list[tuple[Path, int]]``
            # from a function that 99% of callers only need ``list[Path]``
            # from -- is a worse signature for a non-hot-path saving.
            sized = sorted(
                ((p, p.stat().st_size) for p in files),
                key=lambda x: x[1],
                reverse=True,
            )
            largest = "\n".join(
                f"  {_format_size(size):>10}  {path.as_posix()}"
                for path, size in sized[:10]
            )
            raise ContextOverflow(
                f"Codebase bundle is {len(bundle):,} chars "
                f"(limit {MAX_BUNDLE_CHARS:,}). Narrow with --include "
                "or --exclude.",
                detail="Largest files in current selection:\n" + largest,
                model=model,
                provider=args.provider,
            )
        sys.stderr.write(
            f"Reviewing {len(files)} file(s) ({len(bundle):,} chars) "
            f"with `{model}` via {args.provider} "
            f"(T={temperature}, max_tokens={max_tokens})...\n"
        )
        system_prompt, user_prompt = build_codebase_prompts(bundle, context)
    else:
        if args.include or args.exclude:
            sys.stderr.write(
                "WARN: --include / --exclude are ignored outside "
                "--codebase mode.\n"
            )
        if args.pr:
            diff = pr_diff(args.pr)
        else:
            diff = git_diff_local(args.base, args.staged)
        if not diff.strip():
            sys.stderr.write("No diff found. Nothing to review.\n")
            sys.exit(0)
        sys.stderr.write(
            f"Reviewing {len(diff):,}-char diff with `{model}` via "
            f"{args.provider} (T={temperature}, max_tokens={max_tokens})...\n"
        )
        system_prompt, user_prompt = build_diff_prompts(diff, context)

    # Wire-format dispatch. Both providers take the same (system, user)
    # prompt pair; only the request shape differs. ``_retry_on_recoverable``
    # wraps the call so a single provider hiccup or 5xx is absorbed
    # automatically; other typed errors (safety, rate limit, context
    # overflow, config) surface immediately for the caller to handle.
    if args.provider == "openrouter":
        referer = os.getenv(
            "OPENROUTER_HTTP_REFERER",
            "https://github.com/Airwhale/local-gemini-code-review",
        )
        title = os.getenv("OPENROUTER_X_TITLE", "OpenRouter Code Review")
        output = _retry_on_recoverable(
            lambda: call_openrouter(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,
                referer=referer,
                title=title,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            label="openrouter",
        )
    else:  # gemini
        output = _retry_on_recoverable(
            lambda: call_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            label="gemini",
        )
    print(output)


def _entrypoint() -> None:
    """Top-level entry that maps typed errors to exit codes.

    Keeping the try/except out of ``main`` itself means ``main`` can be
    imported and unit-tested without the process-exit side effect.
    """
    try:
        main()
    except ReviewError as err:
        _print_error(err)
        sys.exit(err.exit_code)


if __name__ == "__main__":
    _entrypoint()
