#!/usr/bin/env python3
"""Standalone code-review runner for the Gemini CLI code-review extension.

This fork keeps the upstream `skills/code-review-commons/SKILL.md` and
`commands/code-review.toml` prompts intact (Apache-2.0, unmodified) and adds a
thin Python runner that sends them to a Gemini model via one of two providers
selectable at the command line:

  --provider openrouter (default)
      POSTs to OpenRouter's OpenAI-compatible chat-completions endpoint
      (https://openrouter.ai/api/v1/chat/completions). Requires
      `OPENROUTER_API_KEY`. Good if you already have an OpenRouter account or
      want one bill across multiple providers.

  --provider gemini
      POSTs to Google AI Studio's `generateContent` endpoint directly
      (https://generativelanguage.googleapis.com/v1beta/models/...). Requires
      `GEMINI_API_KEY`. Slightly lower latency (one less hop) and uses the
      same key the GitHub bot uses on the backend.

Both paths send the same system + user prompts (loaded verbatim from the
upstream skill / command files) and the same `gemini-2.5-pro` model by
default, so the review output is materially equivalent. The behavior matches
the GitHub `/gemini review` bot, just without the GitHub webhook -> Google
job-queue round-trip that adds 5-15 minutes of wall time.

Usage:
    uv run review.py                          # diff current branch vs origin/HEAD merge-base
    uv run review.py --base main              # diff vs an explicit base ref
    uv run review.py --pr 6                   # review a GitHub PR (uses `gh pr diff`)
    uv run review.py --staged                 # staged changes only
    uv run review.py --provider gemini        # use Gemini API directly
    uv run review.py --model gemini-2.5-flash # faster, somewhat lower quality

The .env loaded from this script's directory (not CWD) -- copy `.env.example`
to `.env` once at the runner location and invoke from any project folder.
Set whichever of `OPENROUTER_API_KEY` / `GEMINI_API_KEY` your chosen provider
needs.
"""

from __future__ import annotations

import argparse
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
                     # generously for very large diffs.

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


def load_skill() -> str:
    """Return the full `code-review-commons` SKILL.md content (with YAML
    frontmatter). The model treats the markdown as a system prompt; including
    the frontmatter is harmless and preserves the exact upstream behavior.
    """
    path = ROOT / "skills" / "code-review-commons" / "SKILL.md"
    return path.read_text(encoding="utf-8")


def load_command_prompt(name: str) -> str:
    """Load `commands/<name>.toml` and return the `prompt` field verbatim."""
    path = ROOT / "commands" / f"{name}.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return data["prompt"]


def build_prompts(diff: str) -> tuple[str, str]:
    """Construct ``(system_prompt, user_prompt)`` from the upstream skill /
    command files, with the diff substituted into the tool-call placeholder.
    Same pair is used regardless of provider so review behavior matches.
    """
    system_prompt = load_skill()
    user_template = load_command_prompt("code-review")
    diff_block = f"**Code Changes**:\n\n```diff\n{diff}\n```"
    user_prompt = user_template.replace(TOOL_CALL_INSTRUCTION, diff_block)
    # Defensive: if upstream rewords the tool-call sentence and our literal
    # substitution missed, append the diff so the model still has it.
    if diff_block not in user_prompt:
        user_prompt = f"{user_prompt}\n\n{diff_block}"
    return system_prompt, user_prompt


def _run_git(args: list[str]) -> str:
    """Run a git command in the current working directory and return stdout.
    Surfaces non-zero exits with the command and stderr so the user sees why.
    """
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, check=True
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
    """Pull a PR's diff via `gh`. Requires `gh auth login` to have run."""
    try:
        result = subprocess.run(
            ["gh", "pr", "diff", str(pr_number), "--patch"],
            capture_output=True, text=True, check=True,
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


def call_openrouter(
    *,
    diff: str,
    model: str,
    api_key: str,
    referer: str,
    title: str,
) -> str:
    """POST to OpenRouter's chat-completions endpoint and return the review
    markdown. Raises on transport or HTTP errors so the caller surfaces them.
    """
    system_prompt, user_prompt = build_prompts(diff)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # The upstream prompt is highly structured; we want minimal sampling
        # noise so suggested-change blocks come back as exact diffs.
        "temperature": 0.2,
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
    diff: str,
    model: str,
    api_key: str,
) -> str:
    """POST to Google AI Studio's ``generateContent`` endpoint directly.
    Returns the review markdown. The request shape is camelCase and uses a
    ``systemInstruction`` field rather than a system-role message.
    """
    system_prompt, user_prompt = build_prompts(diff)
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": 0.2,
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


def main() -> None:
    # Load env from this script's directory so `.env` is configured once at
    # the runner location rather than in every project we review from.
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "Standalone code-review runner using the Gemini CLI "
            "code-review extension prompts. Sends them to a Gemini model "
            "via OpenRouter or the Gemini API directly."
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
            "Model slug. Defaults to the provider-appropriate "
            "``gemini-2.5-pro`` variant -- ``google/gemini-2.5-pro`` for "
            "OpenRouter, ``gemini-2.5-pro`` for the Gemini API. Override "
            "with $OPENROUTER_MODEL or $GEMINI_MODEL respectively. "
            "``gemini-2.5-flash`` is ~3x faster with some quality loss."
        ),
    )
    args = parser.parse_args()

    # Resolve the model: explicit ``--model`` wins; otherwise a provider-
    # specific env var; otherwise the provider default.
    if args.model is not None:
        model = args.model
    elif args.provider == "openrouter":
        model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL_BY_PROVIDER["openrouter"])
    else:  # gemini
        model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL_BY_PROVIDER["gemini"])

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

    if args.pr:
        diff = pr_diff(args.pr)
    else:
        diff = git_diff_local(args.base, args.staged)

    if not diff.strip():
        sys.stderr.write("No diff found. Nothing to review.\n")
        sys.exit(0)

    sys.stderr.write(
        f"Reviewing {len(diff):,}-char diff with `{model}` via "
        f"{args.provider}...\n"
    )

    if args.provider == "openrouter":
        referer = os.getenv(
            "OPENROUTER_HTTP_REFERER",
            "https://github.com/Airwhale/code-review",
        )
        title = os.getenv("OPENROUTER_X_TITLE", "OpenRouter Code Review")
        output = call_openrouter(
            diff=diff,
            model=model,
            api_key=api_key,
            referer=referer,
            title=title,
        )
    else:  # gemini
        output = call_gemini(diff=diff, model=model, api_key=api_key)
    print(output)


if __name__ == "__main__":
    main()
