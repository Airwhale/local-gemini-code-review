"""Review sources: git/gh subprocess plumbing and codebase file gathering.

Everything that decides WHAT gets reviewed -- local diffs, PR diffs
(repo-pinned gh calls plus the resolved-URL announce), --diff-file /
stdin, and the filtered file set for --codebase / --full-files.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
from pathlib import Path

from code_review.errors import ConfigError
from code_review.prompts import MAX_INDIVIDUAL_FILE_BYTES, _format_size

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
    # Build outputs occasionally tracked by mistake. Both spellings are
    # needed: fnmatch's ``*`` can match the empty string but the literal
    # ``/`` in ``*/dist/*`` cannot, so ``dist/bundle.js`` at the repo
    # root only matches the un-prefixed variant.
    "dist/*",
    "build/*",
    "*/dist/*",
    "*/build/*",
    # Binary / asset extensions: skip outright (model can't review).
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.webp",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.mp3",
    "*.mp4",
    "*.mov",
    "*.avi",
)


# Per-file delimiter for whole-codebase bundles. The shape is chosen so
# the model can pattern-match file boundaries reliably and quote the
# exact path back in its per-file findings.
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
    except FileNotFoundError as exc:
        # Same typed-error contract as ``pr_diff`` gives missing ``gh``:
        # a raw FileNotFoundError traceback would break the documented
        # ``ERROR: <CATEGORY> [exit <N>]`` stderr contract.
        raise ConfigError(
            "`git` not found on PATH. Install git (or fix PATH) to use "
            "the diff / codebase modes.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        # Raise a typed error rather than ``sys.exit(exc.returncode)``
        # so an LLM caller sees the documented ``ERROR: CONFIG [exit 2]``
        # contract instead of a raw subprocess exit code (which collides
        # with the UNKNOWN/1 bucket). Git failures are almost always a
        # misconfigured ref / non-git directory / similar setup issue
        # the caller has to fix before retry makes sense.
        raise ConfigError(
            f"`{' '.join(args)}` failed (exit {exc.returncode})",
            detail=exc.stderr.strip(),
        ) from exc
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
        _warn_if_base_ahead(base)
        return _run_git(["git", "diff", "-U5", base])
    return _run_git(["git", "diff", "-U5", "--merge-base", "origin/HEAD"])


def _warn_if_base_ahead(base: str) -> None:
    """Warn when ``base`` has commits HEAD lacks (two-dot drift).

    ``git diff <base>`` is two-dot: it compares base's TIP to the working
    tree. That is the right default (see ``git_diff_local``), but it has a
    sharp edge once base moves ahead: every commit on base that HEAD lacks
    shows up in the diff as a REMOVAL, as though this branch deleted it.
    Models read that literally and report confident, wrong findings --
    "this reverts the fix in X", "you dropped the guard on Y" -- about code
    the author never touched.

    Observed in the wild: a branch cut before a one-commit base fix was
    reviewed against the moved base; three of four models independently
    flagged the (untouched) file as "behavior changed". The findings were
    artifacts of drift, not of the change under review.

    Warn rather than switch to three-dot: three-dot would HIDE the drift,
    and drift genuinely matters at merge time. The caller deserves to know
    the diff is mixed, and can rebase or pass an explicit merge-base ref.
    Best-effort -- an unresolvable ref just skips the check (the diff call
    right after will produce the real error).
    """
    try:
        behind = _run_git(["git", "rev-list", "--count", f"HEAD..{base}"]).strip()
    except ConfigError:
        return
    if not behind.isdigit() or behind == "0":
        return
    sys.stderr.write(
        f"WARN: {base} has {behind} commit(s) not in HEAD. `git diff {base}` "
        "is two-dot, so those commits appear in the diff as removals -- the "
        "model may report them as intentional deletions/reverts. Rebase onto "
        f"{base}, or diff against the merge-base "
        f"(--diff-file with `git diff $(git merge-base HEAD {base})`) to "
        "review only this branch's changes.\n"
    )


def _run_gh(args: list[str], repo: str | None) -> str:
    """Run a ``gh pr …`` command and return stdout.

    ``repo`` pins the target repository (``--repo owner/name``). This
    matters: bare ``gh pr N`` resolves through gh's own default-repo
    logic (``gh repo set-default``), which on forks routinely points at
    the UPSTREAM repo -- so ``--pr 3`` would silently review someone
    else's PR #3. The runner also announces the resolved PR URL on
    stderr (see ``_read_diff_source``) so the wrong target is visible
    even without ``--repo``.

    Same Windows-locale concern as ``_run_git``: explicit
    ``encoding="utf-8"`` keeps PR diffs containing em-dashes, arrows,
    section signs, and other non-ASCII characters from being mangled
    into cp1252 mojibake before the model sees them.
    """
    if repo:
        args = [*args, "--repo", repo]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        # Same typed-error contract as ``_run_git``: callers see
        # ``ERROR: CONFIG [exit 2]`` and know to install ``gh`` /
        # adjust PATH before retrying.
        raise ConfigError(
            "`gh` not found on PATH. Install GitHub CLI to use --pr.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ConfigError(
            f"`{' '.join(args)}` failed (exit {exc.returncode})",
            detail=exc.stderr.strip(),
        ) from exc
    return result.stdout


def pr_diff(pr_number: int, repo: str | None = None) -> str:
    """Pull a PR's diff via `gh`. Requires `gh auth login` to have run."""
    return _run_gh(["gh", "pr", "diff", str(pr_number), "--patch"], repo)


def pr_changed_files(pr_number: int, repo: str | None = None) -> list[Path]:
    """List a PR's changed file paths via ``gh pr diff --name-only``."""
    output = _run_gh(["gh", "pr", "diff", str(pr_number), "--name-only"], repo)
    return [Path(line) for line in output.splitlines() if line]


def _gh_pr_view_field(pr_number: int, field: str, repo: str | None) -> str:
    """One field from ``gh pr view --json`` (e.g. headRefOid, url)."""
    output = _run_gh(
        ["gh", "pr", "view", str(pr_number), "--json", field, "--jq", f".{field}"],
        repo,
    )
    return output.strip()


def pr_head_sha(pr_number: int, repo: str | None = None) -> str:
    """The PR's current head commit SHA via ``gh pr view``."""
    return _gh_pr_view_field(pr_number, "headRefOid", repo)


def pr_url(pr_number: int, repo: str | None = None) -> str:
    """The PR's canonical URL -- names the owner/repo it resolved to."""
    return _gh_pr_view_field(pr_number, "url", repo)


def _guard_pr_full_files(pr_number: int, repo: str | None = None) -> None:
    """Refuse ``--full-files`` with ``--pr`` unless HEAD is the PR head.

    The PR diff comes from GitHub, but ``--full-files`` reads reference
    file bodies from the LOCAL checkout. When the checkout isn't at the
    PR head, the model silently receives file content unrelated to the
    diff it is reviewing -- worse than no reference at all, because the
    mismatch is invisible in the output. A matching-but-dirty tree only
    gets a WARN: uncommitted edits are the user's own visible state, and
    hard-failing on them would break the fix-and-re-review loop.
    """
    head = pr_head_sha(pr_number, repo)
    local = _run_git(["git", "rev-parse", "HEAD"]).strip()
    if local != head:
        # Carry the --repo pin into the suggested command: bare
        # `gh pr checkout N` resolves through gh's default repo, which
        # is the exact wrong-repo footgun --repo exists to avoid.
        checkout = f"gh pr checkout {pr_number}"
        if repo:
            checkout += f" --repo {repo}"
        raise ConfigError(
            "--full-files with --pr reads file content from the local "
            f"checkout, but HEAD ({local[:12]}) is not the head of PR "
            f"#{pr_number} ({head[:12]}) -- the reference files would not "
            f"match the diff. Run `{checkout}` first, or drop --full-files."
        )
    if _run_git(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        sys.stderr.write(
            "WARN: working tree has uncommitted changes; --full-files "
            "reference content may differ from the PR head.\n"
        )


def _rebase_repo_relative(paths: list[Path]) -> list[Path]:
    """Re-base repo-root-relative paths onto the current directory.

    ``git diff --name-only`` and ``gh pr diff --name-only`` emit paths
    relative to the REPO ROOT, but the runner may be invoked from a
    subdirectory -- read as-is, every reference file would stat-fail
    and be silently dropped from the --full-files set. From the root
    (the common case) this is a no-op; from a subdirectory the paths
    gain ``../`` prefixes, which read correctly and stay recognizable
    in the reference delimiters.
    """
    top = _run_git(["git", "rev-parse", "--show-toplevel"]).strip()
    if not top:
        return paths
    root = Path(top)
    if root.resolve() == Path.cwd().resolve():
        return paths
    return [Path(os.path.relpath(root / p)) for p in paths]


def changed_file_paths(args: argparse.Namespace) -> list[Path]:
    """The files touched by the active diff mode (for ``--full-files``).

    Mirrors ``git_diff_local`` / ``pr_diff`` argument-for-argument so the
    reference set always matches the diff the model reviews.
    """
    if args.pr:
        return _rebase_repo_relative(pr_changed_files(args.pr, args.repo))
    if args.staged:
        output = _run_git(["git", "diff", "--cached", "--name-only"])
    elif args.base:
        output = _run_git(["git", "diff", "--name-only", args.base])
    else:
        output = _run_git(["git", "diff", "--name-only", "--merge-base", "origin/HEAD"])
    return _rebase_repo_relative([Path(line) for line in output.splitlines() if line])


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
        fnmatch.fnmatch(posix, pat) or fnmatch.fnmatch(name, pat) for pat in patterns
    )


def gather_codebase_files(includes: list[str], excludes: list[str]) -> list[Path]:
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

    # Steps 4-5: built-in excludes + size cap (shared with --full-files).
    return _filter_reviewable(paths)


def _filter_reviewable(paths: list[Path]) -> list[Path]:
    """Apply the built-in defensive excludes and the per-file size cap.

    Shared by codebase mode and ``--full-files`` reference gathering.
    Files missing on disk are skipped silently -- in codebase mode
    that's a stat race; in --full-files it's the normal case for files
    the diff *deletes*.
    """
    paths = [p for p in paths if not _glob_match(p, BUILTIN_CODEBASE_EXCLUDES)]

    kept: list[Path] = []
    for p in paths:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > MAX_INDIVIDUAL_FILE_BYTES:
            sys.stderr.write(
                f"skip (>{_format_size(MAX_INDIVIDUAL_FILE_BYTES)}): "
                f"{p.as_posix()} ({_format_size(size)})\n"
            )
            continue
        kept.append(p)

    return kept


def _read_diff_source(args: argparse.Namespace) -> str:
    """Fetch the diff for the active diff mode (git, gh, file, or stdin)."""
    if args.diff_file is not None:
        if args.diff_file == "-":
            if sys.stdin.isatty():
                # Reading a TTY blocks forever waiting for input the
                # user doesn't know to type -- fail fast with the fix.
                raise ConfigError(
                    "--diff-file - reads the diff from stdin, but stdin "
                    "is a terminal. Pipe a diff in (e.g. `git diff | "
                    "code-review --diff-file -`) or pass a file path."
                )
            return sys.stdin.read()
        try:
            # utf-8-sig: tolerate Windows-editor BOMs like the other
            # user-supplied files.
            return Path(args.diff_file).read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ConfigError(
                f"Cannot read --diff-file {args.diff_file!r}: {exc}"
            ) from exc
    if args.pr:
        # Announce the resolved PR URL: without --repo, gh's default-repo
        # logic decides which repository "--pr N" means, and on forks
        # that default often points at UPSTREAM -- the URL makes a wrong
        # resolution visible before tokens are spent (--dry-run included).
        sys.stderr.write(
            f"[gh] reviewing PR #{args.pr}: {pr_url(args.pr, args.repo)}\n"
        )
        return pr_diff(args.pr, args.repo)
    return git_diff_local(args.base, args.staged)
