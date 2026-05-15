#!/usr/bin/env python3
"""OpenRouter-backed code review using the Gemini CLI code-review extension prompts.

This fork keeps the upstream `skills/code-review-commons/SKILL.md` and
`commands/code-review.toml` prompts intact (Apache-2.0, unmodified) and adds a
thin Python runner that POSTs them to OpenRouter's chat-completions endpoint
with a Gemini model. The behavior matches the GitHub `/gemini review` bot
because we send the same system prompt, the same user prompt, and the same
underlying model -- just without the GitHub webhook -> Google job queue
round-trip that adds 5-15 minutes of wall time.

Usage:
    uv run review.py                    # diff current branch vs origin/HEAD merge-base
    uv run review.py --base main        # diff vs an explicit base
    uv run review.py --pr 6             # review a GitHub PR (uses `gh pr diff`)
    uv run review.py --staged           # review staged changes only
    uv run review.py --model google/gemini-2.5-flash

The .env loaded from this script's directory (not CWD) -- copy `.env.example`
to `.env` once and invoke from any project folder. Set `OPENROUTER_API_KEY`.
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
DEFAULT_MODEL = "google/gemini-2.5-pro"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_TIMEOUT = 300.0  # Gemini 2.5 Pro on a ~5K-line diff lands ~30-60s; pad
                     # generously for very large diffs.

# Upstream `code-review.toml` instructs the model to *call* git itself via a
# tool. We have no tool layer; instead we extract the diff up front and
# substitute it into the prompt. This is the literal sentence from
# `commands/code-review.toml`; if upstream rewords it the substitution silently
# no-ops and the script still works (the diff still ends up appended below).
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
    """
    if staged:
        return _run_git(["git", "diff", "--cached", "-U5"])
    if base:
        return _run_git(["git", "diff", "-U5", f"{base}...HEAD"])
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
    system_prompt = load_skill()
    user_template = load_command_prompt("code-review")
    diff_block = f"**Code Changes**:\n\n```diff\n{diff}\n```"
    user_prompt = user_template.replace(TOOL_CALL_INSTRUCTION, diff_block)
    # Defensive: if upstream rewords the tool-call sentence and our literal
    # substitution missed, append the diff anyway so the model still has it.
    if diff_block not in user_prompt:
        user_prompt = f"{user_prompt}\n\n{diff_block}"

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


def main() -> None:
    # Load env from this script's directory so `.env` is configured once at
    # the runner location rather than in every project we review from.
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "OpenRouter-backed code review using the Gemini CLI code-review "
            "extension prompts."
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
        "--model",
        default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        help=(
            "OpenRouter model slug. Defaults to $OPENROUTER_MODEL or "
            f"`{DEFAULT_MODEL}`. `google/gemini-2.5-flash` is ~3x faster."
        ),
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.stderr.write(
            "ERROR: OPENROUTER_API_KEY not set. Copy .env.example to .env "
            f"at {ROOT} and fill in your key.\n"
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
        f"Reviewing {len(diff):,}-char diff with `{args.model}`...\n"
    )
    referer = os.getenv(
        "OPENROUTER_HTTP_REFERER", "https://github.com/Airwhale/code-review"
    )
    title = os.getenv("OPENROUTER_X_TITLE", "OpenRouter Code Review")
    output = call_openrouter(
        diff=diff,
        model=args.model,
        api_key=api_key,
        referer=referer,
        title=title,
    )
    print(output)


if __name__ == "__main__":
    main()
