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
import tomllib
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).parent.resolve()

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
    # OpenAI / GPT family.
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


def build_diff_prompts(diff: str) -> tuple[str, str]:
    """Construct ``(system, user)`` prompts for diff-mode review.

    Loads the upstream ``code-review-commons`` skill (system prompt) and
    the upstream ``code-review`` command (user prompt template), then
    substitutes the diff into the command's tool-call placeholder.
    """
    system_prompt = load_skill("code-review-commons")
    user_template = load_command_prompt("code-review")
    diff_block = f"**Code Changes**:\n\n```diff\n{diff}\n```"
    user_prompt = user_template.replace(TOOL_CALL_INSTRUCTION, diff_block)
    # Defensive: if upstream rewords the tool-call sentence and our literal
    # substitution missed, append the diff so the model still has it.
    if diff_block not in user_prompt:
        user_prompt = f"{user_prompt}\n\n{diff_block}"
    return system_prompt, user_prompt


def build_codebase_prompts(bundle: str) -> tuple[str, str]:
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
    return system_prompt, user_prompt


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


def bundle_codebase(file_paths: list[Path]) -> str:
    """Concatenate the given files into a single delimited bundle.

    Encoding errors are replaced silently (``errors="replace"``) so a
    non-UTF8 byte sequence in a tracked file doesn't crash the runner;
    the model sees a replacement character but the rest of the file is
    still reviewable. In practice this only triggers on files that
    should have been excluded by the asset-extension filter -- a true
    source file with a stray non-UTF8 byte is rare.
    """
    parts: list[str] = []
    for path in file_paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        delimiter = FILE_DELIMITER_TEMPLATE.format(path=path.as_posix())
        parts.append(f"{delimiter}\n{content}")
    return "\n\n".join(parts)


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

    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            sys.stderr.write(
                f"ERROR: OpenRouter returned HTTP {response.status_code}\n"
                f"{response.text}\n"
            )
            sys.exit(1)
        data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        sys.stderr.write(
            f"ERROR: unexpected OpenRouter response shape ({exc}):\n{data}\n"
        )
        sys.exit(1)


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
    path is mode-agnostic.
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

    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            sys.stderr.write(
                f"ERROR: Gemini API returned HTTP {response.status_code}\n"
                f"{response.text}\n"
            )
            sys.exit(1)
        data = response.json()

    try:
        # The generateContent response wraps the model output in
        # candidates[0].content.parts[]; concatenate parts in case the
        # model returned multiple text parts (rare for code review but
        # documented as possible).
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError) as exc:
        sys.stderr.write(
            f"ERROR: unexpected Gemini response shape ({exc}):\n{data}\n"
        )
        sys.exit(1)


def _resolve_model(args: argparse.Namespace) -> str:
    """Resolve the final model slug from CLI flag, env var, alias table,
    or provider default. Errors out if an alias is used with the wrong
    provider (the Gemini API direct path takes bare Gemini model names
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
            sys.stderr.write(
                f"ERROR: model alias `{model}` is only valid with "
                f"--provider openrouter. The Gemini API direct path takes "
                f"bare Gemini model names only (e.g. gemini-2.5-pro, "
                f"gemini-2.5-flash). Either pass --provider openrouter to "
                f"use this alias, or pass --model gemini-2.5-pro / "
                f"--model gemini-2.5-flash for the direct path.\n"
            )
            sys.exit(2)
        model = MODEL_ALIASES[model]

    return model


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
        default=None,
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
        default=None,
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
    args = parser.parse_args()

    model = _resolve_model(args)

    # Resolve temperature: explicit CLI flag wins, then env, then default.
    if args.temperature is not None:
        temperature = args.temperature
    else:
        env_temp = os.getenv("CODE_REVIEW_TEMPERATURE")
        try:
            temperature = float(env_temp) if env_temp is not None else DEFAULT_TEMPERATURE
        except ValueError:
            sys.stderr.write(
                f"ERROR: $CODE_REVIEW_TEMPERATURE={env_temp!r} is not a "
                "valid float. Unset it or pass --temperature explicitly.\n"
            )
            sys.exit(2)

    # Resolve max_tokens with the same precedence.
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    else:
        env_max = os.getenv("CODE_REVIEW_MAX_TOKENS")
        try:
            max_tokens = int(env_max) if env_max is not None else DEFAULT_MAX_TOKENS
        except ValueError:
            sys.stderr.write(
                f"ERROR: $CODE_REVIEW_MAX_TOKENS={env_max!r} is not a "
                "valid integer. Unset it or pass --max-tokens explicitly.\n"
            )
            sys.exit(2)

    # Resolve and validate the API key for the chosen provider before
    # touching git / GitHub so the user fails fast on configuration errors.
    if args.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            sys.stderr.write(
                "ERROR: OPENROUTER_API_KEY not set. Copy .env.example to "
                f".env at {ROOT} and fill in your key, or rerun with "
                "--provider gemini if you only have a Gemini API key.\n"
            )
            sys.exit(2)
    else:  # gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            sys.stderr.write(
                "ERROR: GEMINI_API_KEY not set. Copy .env.example to .env "
                f"at {ROOT} and fill in your Google AI Studio key, or "
                "rerun with --provider openrouter if you only have an "
                "OpenRouter key.\n"
            )
            sys.exit(2)

    # Build the payload for the chosen mode. ``--include`` / ``--exclude``
    # are codebase-mode-only -- warn if they show up alongside a diff
    # mode rather than silently dropping them.
    if args.codebase:
        files = gather_codebase_files(args.include or [], args.exclude or [])
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
            sized = sorted(
                ((p, p.stat().st_size) for p in files),
                key=lambda x: x[1],
                reverse=True,
            )
            sys.stderr.write(
                f"ERROR: codebase bundle is {len(bundle):,} chars "
                f"(limit {MAX_BUNDLE_CHARS:,}). Narrow with --include "
                "or --exclude. Largest files in current selection:\n"
            )
            for path, size in sized[:10]:
                # Right-justify the formatted size for visual alignment;
                # 10 chars fits "999 KB", "99.9 MB", etc.
                sys.stderr.write(
                    f"  {_format_size(size):>10}  {path.as_posix()}\n"
                )
            sys.exit(1)
        sys.stderr.write(
            f"Reviewing {len(files)} file(s) ({len(bundle):,} chars) "
            f"with `{model}` via {args.provider} "
            f"(T={temperature}, max_tokens={max_tokens})...\n"
        )
        system_prompt, user_prompt = build_codebase_prompts(bundle)
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
        system_prompt, user_prompt = build_diff_prompts(diff)

    # Wire-format dispatch. Both providers take the same (system, user)
    # prompt pair; only the request shape differs.
    if args.provider == "openrouter":
        referer = os.getenv(
            "OPENROUTER_HTTP_REFERER",
            "https://github.com/Airwhale/local-gemini-code-review",
        )
        title = os.getenv("OPENROUTER_X_TITLE", "OpenRouter Code Review")
        output = call_openrouter(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            api_key=api_key,
            referer=referer,
            title=title,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:  # gemini
        output = call_gemini(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    print(output)


if __name__ == "__main__":
    main()
