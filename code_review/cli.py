#!/usr/bin/env python3
"""Standalone code-review runner for the Gemini CLI code-review extension.

This fork keeps the upstream `skills/code-review-commons/SKILL.md` and
`commands/code-review.toml` prompts intact (Apache-2.0, unmodified) and adds a
thin Python runner that sends them to a Gemini-or-other-model via one of three
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

  --provider ollama
      POSTs to a local Ollama server's native chat endpoint
      (http://localhost:11434/api/chat by default) -- native rather than
      OpenAI-compat because it accepts per-request ``options.num_ctx``
      and reports ``prompt_eval_count`` for truncation detection. No API
      key required -- the server runs on your machine (or in WSL).
      Override the URL with `--ollama-host` or `$OLLAMA_HOST` if Ollama
      listens elsewhere (different port, different machine, WSL with
      non-default networking). Best for offline / private / cost-free
      review; trade-off is quality and speed depending on local model
      size and CPU/GPU.

Provider defaults: openrouter -> ``google/gemini-2.5-pro``, gemini ->
``gemini-2.5-pro``, ollama -> ``qwen3-coder:30b`` (the MoE coder model
with ~3.3B active params, the quality/speed sweet spot on CPU). The
``--model <slug>`` flag overrides per call; ``--provider openrouter``
and ``--provider ollama`` also accept named aliases (see
``MODEL_ALIASES_BY_PROVIDER`` below) so you can write ``--model claude``
or ``--model local`` instead of the full slug.

Install as a global command (the primary interface), or run from a checkout:

    uv tool install git+https://github.com/Airwhale/local-gemini-code-review
    code-review --base origin/main            # then, from any repo
    uv run review.py --base origin/main       # equivalent, from a checkout

Diff modes (default) review a git diff:

    code-review                               # diff current branch vs origin/HEAD merge-base
    code-review --base main                   # diff vs an explicit base ref
    code-review --pr 6                        # review a GitHub PR (uses `gh pr diff`)
    code-review --staged                      # staged changes only
    code-review --diff-file changes.patch     # review a diff from a file (- = stdin)

Whole-codebase mode reviews tracked files (filtered):

    code-review --codebase
    code-review --codebase --include 'backend/**/*.py'
    code-review --codebase --exclude '**/test_*'

See --help for the full flag set (panels, chunking, JSON output, ...).

Env files load in layers, and real environment variables always win:
$CODE_REVIEW_ENV file (if set), then the per-user config dir
(%APPDATA%\\code-review\\.env on Windows, ~/.config/code-review/.env
elsewhere -- the home for an installed tool), then a checkout's repo-root
.env (see .env.example). Set whichever of `OPENROUTER_API_KEY` /
`GEMINI_API_KEY` your chosen provider needs (Ollama needs no API key --
just set `OLLAMA_HOST` if your server isn't at the default
`http://localhost:11434`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from code_review import __version__
from code_review.chunking import (
    _DIFF_FILE_ANCHOR,
    _chunk_budget,
    _pack_contiguous,
    partition_codebase,
    partition_diffs,
    split_diff_by_file,
)
from code_review.config import (
    _PROJECT_CONFIG_KEYS,
    _PROJECT_CONFIG_NAME,
    ReviewRequest,
    Settings,
    _apply_config_file_lists,
    _layered,
    _load_env_files,
    _load_project_config,
    _resolve_model,
    _resolve_model_name,
    _resolve_settings,
    _user_config_dir,
)

# The split modules; cli re-exports the full surface so
# `from code_review.cli import X` stays the stable import path
# (tests, the review.py shim, and any downstream scripts).
from code_review.errors import (
    ConfigError,
    ContextOverflow,
    ProviderHiccup,
    RateLimit,
    ReviewError,
    SafetyRefusal,
    TransportError,
    _print_error,
)
from code_review.panel import (
    _CATEGORY_PRECEDENCE,
    _SEVERITY_RANK,
    MergedFinding,
    _panel_exit_error,
    _panel_max_workers,
    build_panel_envelope,
    merge_panel_findings,
    panel_findings_match,
    render_panel_markdown,
)
from code_review.parser import (
    _BARE_LINE_RE,
    _CLEAN_RE,
    _FENCE_RE,
    _FILE_RE,
    _FINDING_RE,
    _LINE_TOKEN_RE,
    _SEVERITY_RE,
    _SUGGESTION_LEADIN_RE,
    _SUMMARY_FALLBACK_RE,
    _SUMMARY_RE,
    Finding,
    ParsedReview,
    _extract_suggestion,
    _fingerprints_match,
    _join_body,
    _location_match,
    _parse_finding_heading,
    _severity_at_or_above,
    annotate_in_hunk,
    build_chunked_envelope,
    build_json_envelope,
    diff_against_baseline,
    enforce_min_severity,
    filter_baseline_findings,
    finding_fingerprint,
    findings_match,
    hunk_ranges,
    load_baseline,
    normalize_title,
    parse_review_markdown,
    parse_review_markdown_safe,
)
from code_review.prompts import (
    _PACKAGE_DIR,
    _REPO_ROOT,
    CODEBASE_PLACEHOLDER,
    DEFAULT_CONTEXT,
    FILE_DELIMITER_TEMPLATE,
    INJECTION_GUARD,
    MAX_BUNDLE_CHARS,
    MAX_INDIVIDUAL_FILE_BYTES,
    SEVERITY_LEVELS,
    TOOL_CALL_INSTRUCTION,
    _apply_context,
    _format_size,
    _min_severity_instruction,
    _number_lines,
    _prompt_root,
    build_codebase_prompts,
    build_diff_prompts,
    build_reference_section,
    bundle_codebase,
    load_command_prompt,
    load_skill,
)
from code_review.providers import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_BY_PROVIDER,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    GEMINI_URL_TEMPLATE,
    HTTP_TIMEOUT,
    MAX_RETRY_SLEEP,
    MODEL_ALIASES,
    MODEL_ALIASES_BY_PROVIDER,
    OLLAMA_CHARS_PER_TOKEN,
    OLLAMA_CHAT_PATH,
    OLLAMA_POST_VERIFY_MARGIN,
    OLLAMA_PS_TIMEOUT,
    OLLAMA_WINDOW_FILL,
    OPENROUTER_URL,
    PROVIDERS,
    CallResult,
    T,
    _call_with_retries,
    _classify_http_error,
    _detect_ollama_num_ctx,
    _format_usage_line,
    _make_client,
    _match_loaded_context,
    _normalize_ollama_host,
    _ollama_post_verify,
    _ollama_prompt_guard,
    _parse_retry_after,
    _resolve_ollama_window,
    _usage_int,
    _warn_if_truncated,
    call_gemini,
    call_ollama,
    call_openrouter,
    estimate_cost_usd,
    format_cost,
    model_context_limit,
)
from code_review.sources import (
    BUILTIN_CODEBASE_EXCLUDES,
    _filter_reviewable,
    _gh_pr_view_field,
    _glob_match,
    _guard_pr_full_files,
    _read_diff_source,
    _rebase_repo_relative,
    _run_gh,
    _run_git,
    changed_file_paths,
    gather_codebase_files,
    git_diff_local,
    pr_changed_files,
    pr_diff,
    pr_head_sha,
    pr_url,
)

__all__ = [
    "_apply_config_file_lists",
    "_apply_context",
    "_BARE_LINE_RE",
    "_call_with_retries",
    "_CATEGORY_PRECEDENCE",
    "_chunk_budget",
    "_classify_http_error",
    "_CLEAN_RE",
    "_detect_ollama_num_ctx",
    "_DIFF_FILE_ANCHOR",
    "_extract_suggestion",
    "_FENCE_RE",
    "_FILE_RE",
    "_filter_reviewable",
    "_FINDING_RE",
    "_fingerprints_match",
    "_format_size",
    "_format_usage_line",
    "_gh_pr_view_field",
    "_glob_match",
    "_guard_pr_full_files",
    "_join_body",
    "_layered",
    "_LINE_TOKEN_RE",
    "_load_env_files",
    "_load_project_config",
    "_location_match",
    "_make_client",
    "_match_loaded_context",
    "_min_severity_instruction",
    "_normalize_ollama_host",
    "_number_lines",
    "_ollama_post_verify",
    "_ollama_prompt_guard",
    "_pack_contiguous",
    "_PACKAGE_DIR",
    "_panel_exit_error",
    "_panel_max_workers",
    "_parse_finding_heading",
    "_parse_retry_after",
    "_print_error",
    "_PROJECT_CONFIG_KEYS",
    "_PROJECT_CONFIG_NAME",
    "_prompt_root",
    "_read_diff_source",
    "_rebase_repo_relative",
    "_REPO_ROOT",
    "_resolve_model",
    "_resolve_model_name",
    "_resolve_ollama_window",
    "_resolve_settings",
    "_run_gh",
    "_run_git",
    "_severity_at_or_above",
    "_SEVERITY_RANK",
    "_SEVERITY_RE",
    "_SUGGESTION_LEADIN_RE",
    "_SUMMARY_FALLBACK_RE",
    "_SUMMARY_RE",
    "_usage_int",
    "_user_config_dir",
    "_warn_if_truncated",
    "build_chunked_envelope",
    "build_codebase_prompts",
    "build_diff_prompts",
    "build_json_envelope",
    "build_panel_envelope",
    "build_reference_section",
    "BUILTIN_CODEBASE_EXCLUDES",
    "bundle_codebase",
    "call_gemini",
    "call_ollama",
    "call_openrouter",
    "CallResult",
    "changed_file_paths",
    "CODEBASE_PLACEHOLDER",
    "ConfigError",
    "ContextOverflow",
    "DEFAULT_CONTEXT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_BY_PROVIDER",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_NUM_CTX",
    "DEFAULT_OLLAMA_TIMEOUT",
    "DEFAULT_PROVIDER",
    "DEFAULT_TEMPERATURE",
    "diff_against_baseline",
    "enforce_min_severity",
    "FILE_DELIMITER_TEMPLATE",
    "filter_baseline_findings",
    "Finding",
    "finding_fingerprint",
    "findings_match",
    "gather_codebase_files",
    "GEMINI_URL_TEMPLATE",
    "git_diff_local",
    "HTTP_TIMEOUT",
    "INJECTION_GUARD",
    "load_baseline",
    "load_command_prompt",
    "load_skill",
    "MAX_BUNDLE_CHARS",
    "MAX_INDIVIDUAL_FILE_BYTES",
    "MAX_RETRY_SLEEP",
    "merge_panel_findings",
    "MergedFinding",
    "MODEL_ALIASES",
    "MODEL_ALIASES_BY_PROVIDER",
    "normalize_title",
    "OLLAMA_CHARS_PER_TOKEN",
    "OLLAMA_CHAT_PATH",
    "OLLAMA_POST_VERIFY_MARGIN",
    "OLLAMA_PS_TIMEOUT",
    "OLLAMA_WINDOW_FILL",
    "OPENROUTER_URL",
    "panel_findings_match",
    "parse_review_markdown",
    "parse_review_markdown_safe",
    "ParsedReview",
    "partition_codebase",
    "partition_diffs",
    "pr_changed_files",
    "pr_diff",
    "pr_head_sha",
    "pr_url",
    "ProviderHiccup",
    "PROVIDERS",
    "RateLimit",
    "render_panel_markdown",
    "ReviewError",
    "ReviewRequest",
    "SafetyRefusal",
    "Settings",
    "SEVERITY_LEVELS",
    "split_diff_by_file",
    "T",
    "TOOL_CALL_INSTRUCTION",
    "TransportError",
]


def _write_output_file(text: str, path: str) -> None:
    """Write stdout content to ``--output`` as UTF-8 with ``\\n`` newlines.

    ``newline="\\n"`` keeps the bytes identical across platforms so a
    saved review can be fingerprinted / baseline-diffed on Windows and
    Linux interchangeably. Failures are the user's config (bad path,
    permission), hence ConfigError.
    """
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    except OSError as exc:
        raise ConfigError(f"Cannot write --output file {path!r}: {exc}") from exc


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


def _build_request(args: argparse.Namespace, settings: Settings) -> ReviewRequest:
    """Gather the diff / codebase bundle and build the final prompts.

    Exits 0 (not an error) when there is nothing to review. The
    ``--min-severity`` appendix is applied here, at the very end of the
    user prompt, where trailing instructions bind strongest.
    """
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
            # Skip any file that disappeared between ``bundle_codebase``
            # and now (narrow race window but possible on a busy CI box)
            # so the error path doesn't itself crash with an
            # ``OSError`` and bury the original ContextOverflow message.
            def _safe_stat(p: Path) -> tuple[Path, int] | None:
                try:
                    return (p, p.stat().st_size)
                except OSError:
                    return None

            sized_pairs = [pair for p in files if (pair := _safe_stat(p))]
            sized = sorted(sized_pairs, key=lambda x: x[1], reverse=True)
            largest = "\n".join(
                f"  {_format_size(size):>10}  {path.as_posix()}"
                for path, size in sized[:10]
            )
            raise ContextOverflow(
                f"Codebase bundle is {len(bundle):,} chars "
                f"(limit {MAX_BUNDLE_CHARS:,}). Narrow with --include "
                "or --exclude.",
                detail="Largest files in current selection:\n" + largest,
                model=settings.model,
                provider=settings.provider,
            )
        system_prompt, user_prompt = build_codebase_prompts(bundle, settings.context)
        request = ReviewRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode="codebase",
            payload_chars=len(bundle),
            files=files,
        )
    else:
        if args.include or args.exclude:
            sys.stderr.write(
                "WARN: --include / --exclude are ignored outside --codebase mode.\n"
            )
        diff = _read_diff_source(args)
        if not diff.strip():
            sys.stderr.write("No diff found. Nothing to review.\n")
            sys.exit(0)
        reference = ""
        # Tri-state: True = strict (explicit --full-files), None = auto
        # (best-effort, the default), False = off (--no-full-files). See the
        # --full-files parser entry for why auto is the default.
        #
        # The two modes differ only in how failures are handled: an explicit
        # ask gets a loud error (you asked for it, you deserve to know it
        # didn't happen); auto silently degrades to hunks-only, because the
        # user never opted in and a failed review is worse than a thinner one.
        explicit_full = args.full_files is True
        # --diff-file is excluded from auto entirely: a handed-in diff has no
        # verifiable relationship to the local tree, so attaching local file
        # bodies would pair the reviewed diff with unrelated content -- the
        # same failure the --pr HEAD guard exists to prevent, but with no
        # cheap way to detect it. (Explicit --full-files --diff-file is
        # already a typed CONFIG error; this covers the auto path.)
        if args.full_files is not False and args.diff_file is None:
            attach = True
            if args.pr:
                # --full-files reads bodies from the LOCAL checkout, so it is
                # only safe when HEAD is the PR head; otherwise the model gets
                # file content unrelated to the diff. Explicit asks get the
                # guard's ConfigError (with the `gh pr checkout` hint); auto
                # just declines and reviews the hunks.
                try:
                    _guard_pr_full_files(args.pr, args.repo)
                except ConfigError:
                    if explicit_full:
                        raise
                    attach = False
            elif args.staged and _run_git(["git", "diff", "--name-only"]).strip():
                # --staged reviews the INDEX; the reference bodies come
                # from the working tree. Unstaged edits make those two
                # diverge -- warn rather than fail, since the divergence
                # is the user's own visible state (same policy as the
                # --pr dirty-tree case). Reading staged blobs via
                # `git show :path` would close the gap entirely but
                # needs the bundler to accept non-filesystem content.
                sys.stderr.write(
                    "WARN: --staged reviews the index, but --full-files "
                    "reads the working tree, which has unstaged edits -- "
                    "reference content may not match the staged diff.\n"
                )
            ref_paths: list[Path] = []
            if attach:
                try:
                    ref_paths = _filter_reviewable(changed_file_paths(args))
                    reference = build_reference_section(ref_paths)
                except (ConfigError, OSError):
                    # Auto is best-effort. Resolving changed files shells out
                    # to git (`--merge-base origin/HEAD`), which legitimately
                    # fails in shallow clones, detached CI checkouts, or repos
                    # with no origin/HEAD. Degrade to hunks-only rather than
                    # failing a review the user never asked to enrich; an
                    # explicit --full-files still surfaces the error.
                    if explicit_full:
                        raise
                    ref_paths, reference = [], ""
            if reference and len(diff) + len(reference) > MAX_BUNDLE_CHARS:

                def _safe_size(p: Path) -> tuple[Path, int] | None:
                    try:
                        return (p, p.stat().st_size)
                    except OSError:
                        return None

                sized = sorted(
                    (pair for p in ref_paths if (pair := _safe_size(p))),
                    key=lambda x: x[1],
                    reverse=True,
                )
                largest = "\n".join(
                    f"  {_format_size(size):>10}  {path.as_posix()}"
                    for path, size in sized[:10]
                )
                if explicit_full:
                    raise ContextOverflow(
                        f"Diff ({len(diff):,} chars) plus --full-files "
                        f"reference content ({len(reference):,} chars) "
                        f"exceeds the {MAX_BUNDLE_CHARS:,}-char cap. Drop "
                        "--full-files or narrow the change.",
                        detail="Largest reference files:\n" + largest,
                        model=settings.model,
                        provider=settings.provider,
                    )
                # Auto: too big to attach. Fall back to hunks-only and say
                # so, since it changes how much the model can verify.
                sys.stderr.write(
                    f"NOTE: full-file context ({len(reference):,} chars) "
                    f"would exceed the {MAX_BUNDLE_CHARS:,}-char cap -- "
                    "reviewing hunks only. Narrow the diff, or pass "
                    "--no-full-files to silence this.\n"
                )
                reference = ""
        if reference and not explicit_full:
            # The 700K-char cap is a GLOBAL constant sized for Gemini (1M
            # tokens) and Claude (200K). Smaller models exist: deepseek-chat
            # -v3.1 is 163,840, so a payload can clear the cap and still blow
            # the model's window -- a hard HTTP 400 on a review that would
            # have succeeded hunks-only. That is auto breaking something the
            # user never asked for, so check the model's published window
            # (from the same OpenRouter feed the cost estimate uses) and
            # decline rather than fail. Unknown window -> no guard available;
            # attach and let the provider decide. Explicit --full-files is
            # left alone: you asked, so you get the error instead of silence.
            window = model_context_limit(settings.provider, settings.model)
            if window is not None:
                # ~4 chars/token, plus the output ceiling: the provider
                # counts requested completion tokens against the window too
                # (its 400 reads "169737 tokens (153737 of text input,
                # 16000 in the output)").
                est_tokens = (len(diff) + len(reference)) // 4 + settings.max_tokens
                if est_tokens > window:
                    sys.stderr.write(
                        f"NOTE: full-file context would need ~{est_tokens:,} tokens "
                        f"but {settings.model} has a {window:,}-token window -- "
                        "reviewing hunks only. Use a wider model, or "
                        "--no-full-files to silence this.\n"
                    )
                    reference = ""
        if reference:
            sys.stderr.write("Full-file context: on (changed files attached).\n")
        system_prompt, user_prompt = build_diff_prompts(
            diff, settings.context, full_files=bool(reference)
        )
        request = ReviewRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt + reference,
            mode="diff",
            payload_chars=len(diff) + len(reference),
            diff=diff,
        )

    request.user_prompt += _min_severity_instruction(settings.min_severity)
    return request


def _build_requests(
    args: argparse.Namespace, settings: Settings
) -> list[ReviewRequest]:
    """Build one request normally, or several when ``--chunk`` splits an
    oversized payload.

    Chunk boundaries never cross file boundaries (codebase chunks pack
    whole files in ``git ls-files`` order; diff chunks pack whole
    per-file diffs in diff order) -- the documented tradeoff is that the
    model cannot see importer/importee relationships across chunks.
    ``--chunk`` on a payload that already fits is a no-op single chunk.
    """
    if not args.chunk:
        return [_build_request(args, settings)]

    budget, note = _chunk_budget(settings, is_codebase=args.codebase)
    if note is not None:
        sys.stderr.write(f"WARN: {note}\n")
    severity_appendix = _min_severity_instruction(settings.min_severity)

    if args.codebase:
        files = gather_codebase_files(args.include, args.exclude)
        if not files:
            sys.stderr.write(
                "No files matched after --include / --exclude / built-in "
                "filters. Nothing to review.\n"
            )
            sys.exit(0)
        partitions = partition_codebase(files, budget)
        requests = []
        for part in partitions:
            bundle = bundle_codebase(part)
            system_prompt, user_prompt = build_codebase_prompts(
                bundle, settings.context
            )
            requests.append(
                ReviewRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt + severity_appendix,
                    mode="codebase",
                    payload_chars=len(bundle),
                    files=part,
                    chunk_label=f"{len(part)} file(s), {len(bundle):,} chars",
                )
            )
        return requests

    diff = _read_diff_source(args)
    if not diff.strip():
        sys.stderr.write("No diff found. Nothing to review.\n")
        sys.exit(0)
    chunks = partition_diffs(split_diff_by_file(diff), budget)
    requests = []
    for chunk in chunks:
        system_prompt, user_prompt = build_diff_prompts(chunk, settings.context)
        requests.append(
            ReviewRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt + severity_appendix,
                mode="diff",
                payload_chars=len(chunk),
                chunk_label=f"{len(chunk):,} chars",
                diff=chunk,
            )
        )
    return requests


def _estimate_cost_line(settings: Settings, prompt_chars: int) -> str | None:
    """Render the cost estimate, or None when it can't be sourced.

    Deliberately a CEILING, and labelled as one: the prompt side is
    estimated (chars/4, no local tokenizer) and the completion side is
    bounded by --max-tokens rather than predicted. Panels sum every model.

    Returns None -- rather than a guess -- whenever pricing is unavailable
    (Gemini publishes no unauthenticated feed; OpenRouter unreachable; an
    unknown slug). Same rule as `_format_usage_line`: never invent numbers
    about money or usage.
    """
    prompt_tokens = prompt_chars // 4
    models = list(settings.models) if settings.models is not None else [settings.model]
    total = 0.0
    for model in models:
        usd = estimate_cost_usd(
            settings.provider, model, prompt_tokens, settings.max_tokens
        )
        if usd is None:
            return None
        total += usd
    suffix = f" (x{len(models)} models)" if len(models) > 1 else ""
    return (
        f"{format_cost(total)}{suffix} -- ceiling: prompt ~{prompt_tokens:,} tok "
        f"+ completion <= {settings.max_tokens:,} tok"
    )


def _list_models_report() -> str:
    """Render the ``--list-models`` report: aliases + defaults, per provider.

    Deliberately OFFLINE. Fetching each provider's live catalogue would mean
    three network paths (OpenRouter's 300+ model list, a keyed Gemini call,
    Ollama's /api/tags) in a tool whose contract is typed, deterministic,
    no-surprise behavior -- and it would answer a question nobody asks. The
    friction is not "which of OpenRouter's 300 models exist", it is "what do
    I type here": the aliases, and which provider they belong to. That is
    local data, so this is instant and works offline.
    """
    lines = ["Model aliases (pass to --model / --models; any real slug also works):"]
    for provider in PROVIDERS:
        default = DEFAULT_MODEL_BY_PROVIDER.get(provider, "?")
        lines.append("")
        lines.append(f"  --provider {provider}   (default: {default})")
        aliases = MODEL_ALIASES_BY_PROVIDER.get(provider, {})
        if not aliases:
            # gemini takes bare model names; there is nothing to alias.
            lines.append("    (no aliases -- pass the model name directly)")
            continue
        width = max(len(a) for a in aliases)
        for alias, slug in aliases.items():
            lines.append(f"    {alias:<{width}}  ->  {slug}")
    lines += [
        "",
        "Aliases are provider-scoped: using one from another provider's table",
        "is a typed CONFIG error naming the right --provider.",
    ]
    return "\n".join(lines)


@contextlib.contextmanager
def _heartbeat(interval: float = 15.0) -> Iterator[None]:
    """Emit an elapsed-time line on stderr while a long call is in flight.

    A big review on a slow model runs for minutes with nothing on screen,
    which is indistinguishable from a hang -- the observed response is to
    kill it or background the whole runner rather than trust it.

    Gated on ``stderr.isatty()``: the stderr format is a documented contract
    for agent callers, so a piped/redirected run stays byte-for-byte what it
    was. Humans get the reassurance; parsers get the contract.
    """
    if not sys.stderr.isatty():
        yield
        return
    stop = threading.Event()
    started = time.monotonic()

    def _tick() -> None:
        while not stop.wait(interval):
            elapsed = int(time.monotonic() - started)
            sys.stderr.write(f"  ... still waiting ({elapsed}s elapsed)\n")
            sys.stderr.flush()

    thread = threading.Thread(target=_tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def _annotate_hunks(parsed: ParsedReview, request: ReviewRequest) -> None:
    """Tag each finding with whether it sits inside a changed hunk.

    Diff mode only -- ``request.diff`` is None for --codebase, where every
    line is fair game and the question is meaningless. Callers automating
    GitHub review comments need this: a ``suggestion`` block on a line the
    PR diff doesn't contain is rejected (422), so findings outside the
    hunks have to go in the review body instead.
    """
    if request.diff:
        annotate_in_hunk(parsed.findings, hunk_ranges(request.diff))


def _validate_flag_combos(args: argparse.Namespace) -> None:
    """Reject unsupported flag pairs in one greppable place.

    Each pair is a deliberate scope decision, not an oversight; see the
    messages (and the README) for the workarounds.
    """
    if args.chunk and args.models:
        raise ConfigError(
            "--chunk and --models are not supported together (a panel of "
            "chunked runs multiplies calls and loses cross-chunk merge "
            "semantics). Chunk with a single model, or panel an unchunked "
            "payload."
        )
    if args.chunk and args.full_files:
        raise ConfigError(
            "--chunk and --full-files are not supported together yet: "
            "reference content would have to be re-partitioned per chunk. "
            "Pick one."
        )
    if args.chunk and args.baseline:
        raise ConfigError(
            "--baseline is not supported with --chunk yet; baseline an "
            "unchunked run, or diff the chunked JSON rounds externally."
        )
    if args.full_files and args.codebase:
        raise ConfigError(
            "--full-files applies to diff modes only; --codebase already "
            "sends full file content."
        )
    # `found_by` only exists once several models are merged. Silently
    # ignoring this on a single-model run would let a caller believe a
    # consensus floor was applied when nothing filtered.
    if args.min_found_by is not None and args.min_found_by > 1 and not args.models:
        raise ConfigError(
            f"--min-found-by {args.min_found_by} needs --models: consensus "
            "counts only exist when several models review the same diff. "
            "Add e.g. --models pro,gpt,deepseek, or drop --min-found-by."
        )
    if args.full_files and args.diff_file is not None:
        raise ConfigError(
            "--full-files needs the diff to come from this working tree "
            "(git/gh); a --diff-file diff has no local files to reference."
        )
    if args.repo is not None and not args.pr:
        raise ConfigError(
            "--repo only applies to --pr (it pins which repository the "
            "PR number refers to). Drop it, or add --pr N."
        )


def _dry_run_report(
    settings: Settings, request: ReviewRequest, ollama_window: str | None = None
) -> str:
    """Render the ``--dry-run`` stdout report.

    Everything a live run would resolve, minus the model call: resolved
    config, prompt sizes, the estimated token count, the Ollama window
    (and its source) when applicable, and the surviving file list in
    codebase mode -- the practical way to debug --include/--exclude
    globs without paying for a review.
    """
    prompt_chars = len(request.system_prompt) + len(request.user_prompt)
    if settings.models is not None:
        model_line = f"models:            {', '.join(settings.models)} (panel)"
    else:
        model_line = f"model:             {settings.model}"
    lines = [
        "DRY RUN -- no model call made, no tokens spent.",
        f"provider:          {settings.provider}",
        model_line,
        f"mode:              {request.mode}",
        f"temperature:       {settings.temperature}",
        f"max_tokens:        {settings.max_tokens}",
        f"retries:           {settings.retries}",
        f"min_severity:      {settings.min_severity}",
        f"payload:           {request.payload_chars:,} chars",
        f"system_prompt:     {len(request.system_prompt):,} chars",
        f"user_prompt:       {len(request.user_prompt):,} chars",
        f"est_prompt_tokens: ~{prompt_chars // 4:,}",
    ]
    cost_line = _estimate_cost_line(settings, prompt_chars)
    if cost_line is not None:
        lines.append(f"est_cost:          {cost_line}")
    if ollama_window is not None:
        lines.append(f"ollama_window:     {ollama_window}")
    if request.files is not None:
        lines.append(f"files:             {len(request.files)}")
        for p in request.files:
            try:
                size = _format_size(p.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"  {size:>10}  {p.as_posix()}")
    return "\n".join(lines)


def _execute_call(
    settings: Settings, system_prompt: str, user_prompt: str, model: str
) -> CallResult:
    """Dispatch one review request to the configured provider.

    All three providers take the same (system, user) prompt pair; only
    the request shape differs. ``_call_with_retries`` wraps each call so
    transient failures are absorbed per the retry policy; other typed
    errors (safety, context overflow, config) surface immediately.

    For Ollama this is also where the context window resolves (env >
    /api/ps probe > advisory default) and the pre-flight truncation
    guard runs -- per model, per call, so multi-model panels (M3) get a
    fresh probe for each sequentially loaded model.
    """
    # Bind narrowed locals before the lambdas: assert-narrowing on
    # ``settings.x`` doesn't survive into a closure for mypy, and the
    # non-None guarantees come from _resolve_settings' config checks.
    if settings.provider == "openrouter":
        api_key = settings.api_key
        referer = settings.referer
        title = settings.title
        assert api_key is not None
        assert referer is not None
        assert title is not None
        return _call_with_retries(
            lambda: call_openrouter(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,
                referer=referer,
                title=title,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            ),
            label="openrouter",
            retries=settings.retries,
        )
    if settings.provider == "gemini":
        # Separate local (not ``api_key``) so each branch's variable has
        # a single assignment -- mypy refuses to narrow a captured
        # variable that is reassigned anywhere in the function.
        gemini_key = settings.api_key
        assert gemini_key is not None
        return _call_with_retries(
            lambda: call_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=gemini_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            ),
            label="gemini",
            retries=settings.retries,
        )
    # ollama
    host = settings.ollama_host
    timeout = settings.ollama_timeout
    assert host is not None
    assert timeout is not None
    num_ctx, enforced, _source = _resolve_ollama_window(
        host, model, settings.ollama_num_ctx_env
    )
    # Pre-flight context-window guard: refuse (typed CONTEXT_OVERFLOW)
    # rather than let Ollama silently truncate the prompt and review a
    # fragment; warn-only when the window couldn't be determined (the
    # post-call prompt_eval_count check backstops that case). Cloud
    # providers don't need this -- they 4xx on oversized prompts instead
    # of truncating.
    _ollama_prompt_guard(
        len(system_prompt) + len(user_prompt),
        num_ctx,
        model=model,
        enforced=enforced,
    )
    # Request the window only when it's authoritative (env or detected).
    # When unknown, omit num_ctx entirely: sending the advisory 4096
    # would actively shrink a VRAM-tier 32K/256K window.
    num_ctx_to_send = num_ctx if enforced else None
    return _call_with_retries(
        lambda: call_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            host=host,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=timeout,
            num_ctx=num_ctx_to_send,
        ),
        label="ollama",
        retries=settings.retries,
    )


def _run_panel(
    settings: Settings, request: ReviewRequest
) -> tuple[dict[str, CallResult], list[tuple[str, ReviewError]]]:
    """Run the panel: one ``_execute_call`` per model, concurrently for
    cloud providers, sequentially for ollama (see ``_panel_max_workers``).

    Returns ``(results_by_model, failures)`` -- results keyed in CLI
    order; each failure is that model's typed error, collected rather
    than raised so one bad model doesn't kill the panel. Threading is
    safe: every ``call_*`` builds its own ``httpx.Client``, and the
    Ollama window probe runs inside ``_execute_call`` per model, so each
    sequentially loaded model gets a fresh /api/ps read. Stderr lines
    are single ``write()`` calls to limit interleaving.
    """
    assert settings.models is not None
    workers = _panel_max_workers(settings.provider, len(settings.models))
    if settings.provider == "ollama" and len(settings.models) > 1:
        sys.stderr.write(
            "[panel] ollama models run sequentially (model-swap "
            "thrashing / RAM pressure)\n"
        )

    def _one(model: str) -> tuple[str, CallResult | ReviewError]:
        sys.stderr.write(f"[panel {model}] starting...\n")
        try:
            result = _execute_call(
                settings, request.system_prompt, request.user_prompt, model
            )
        except ReviewError as err:
            return model, err
        usage_line = _format_usage_line(result, settings.provider, model)
        if usage_line is not None:
            sys.stderr.write(f"[panel {model}] done. {usage_line}\n")
        else:
            sys.stderr.write(f"[panel {model}] done.\n")
        return model, result

    outcomes: dict[str, CallResult | ReviewError] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for model, outcome in pool.map(_one, settings.models):
            outcomes[model] = outcome

    results_by_model: dict[str, CallResult] = {}
    failures: list[tuple[str, ReviewError]] = []
    for model in settings.models:
        outcome = outcomes[model]
        if isinstance(outcome, ReviewError):
            failures.append((model, outcome))
        else:
            results_by_model[model] = outcome
    return results_by_model, failures


def _run_chunked(
    args: argparse.Namespace, settings: Settings, requests: list[ReviewRequest]
) -> None:
    """Execute a multi-chunk run sequentially, fail-fast.

    Chunks are disjoint content: a failed chunk means unreviewed files
    (no redundancy, unlike a panel member), so the first surviving typed
    error aborts the run with that error's exit code -- exit 0 iff every
    chunk succeeded. Markdown streams per chunk (long local runs show
    progress); JSON buffers into one envelope.
    """
    n = len(requests)
    sys.stderr.write(f"[chunk] payload split into {n} chunks\n")

    if args.dry_run:
        lines = [
            "DRY RUN -- no model call made, no tokens spent.",
            f"provider:          {settings.provider}",
            f"model:             {settings.model}",
            f"mode:              {requests[0].mode} (chunked)",
            f"chunks:            {n}",
        ]
        for idx, request in enumerate(requests, start=1):
            lines.append(f"  chunk {idx}/{n}: {request.chunk_label}")
        print("\n".join(lines))
        return

    chunk_data: list[tuple[str, ParsedReview, CallResult, str]] = []
    streamed_parts: list[str] = []
    for idx, request in enumerate(requests, start=1):
        sys.stderr.write(
            f"[chunk {idx}/{n}] reviewing {request.chunk_label} with "
            f"`{settings.model}` via {settings.provider}...\n"
        )
        try:
            result = _execute_call(
                settings, request.system_prompt, request.user_prompt, settings.model
            )
        except ReviewError:
            done = f"chunks 1-{idx - 1} completed" if idx > 1 else "no chunks completed"
            sys.stderr.write(
                f"WARN: [chunk] {done}; chunk {idx} failed -- review is incomplete.\n"
            )
            raise
        usage_line = _format_usage_line(result, settings.provider, settings.model)
        if usage_line is not None:
            sys.stderr.write(f"[chunk {idx}/{n}] {usage_line}\n")
        label = request.chunk_label or f"chunk {idx}"
        parsed = parse_review_markdown_safe(result.content)
        _annotate_hunks(parsed, request)
        if settings.format == "json":
            # Markdown chunk output streams the model's text verbatim;
            # only the JSON envelope enforces the severity floor.
            parsed = enforce_min_severity(parsed, settings.min_severity)
        chunk_data.append((label, parsed, result, result.content))
        if settings.format == "markdown":
            part = (
                f"\n---\n# Review chunk {idx}/{n} ({label})\n\n"
                f"{result.content.rstrip()}\n"
                if idx > 1
                else f"# Review chunk {idx}/{n} ({label})\n\n{result.content.rstrip()}\n"
            )
            print(part, flush=True)
            streamed_parts.append(part)

    if settings.format == "json":
        stdout_text = json.dumps(
            build_chunked_envelope(
                mode=requests[0].mode,
                provider=settings.provider,
                model=settings.model,
                temperature=settings.temperature,
                chunk_data=chunk_data,
            ),
            indent=2,
            ensure_ascii=False,
        )
        print(stdout_text)
    else:
        stdout_text = "".join(streamed_parts)
    if settings.output is not None:
        _write_output_file(stdout_text, settings.output)


def main() -> None:
    _ensure_utf8_stdout()
    # Env layering: $CODE_REVIEW_ENV > user config dir > repo-root .env
    # (see _load_env_files); process env always wins over file values.
    _load_env_files()

    parser = argparse.ArgumentParser(
        description=(
            "Standalone code-review runner using the Gemini CLI "
            "code-review extension prompts. Sends them to a Gemini-or-"
            "other model via OpenRouter, the Gemini API directly, or "
            "a local Ollama server (offline / no API key)."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        dest="list_models",
        help=(
            "Print the model aliases and per-provider defaults, then exit. "
            "Offline and instant -- shows what you can type for --model / "
            "--models, not the provider's full remote catalogue."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--base",
        help="Base ref to diff against (e.g. main, origin/main).",
    )
    source.add_argument(
        "--pr",
        type=int,
        help=(
            "GitHub PR number to review (uses `gh pr diff`). Without "
            "--repo, gh's own default-repo resolution decides which "
            "repository the number refers to -- the runner announces "
            "the resolved PR URL on stderr so a wrong default is "
            "visible."
        ),
    )
    parser.add_argument(
        "--repo",
        default=None,
        metavar="OWNER/NAME",
        help=(
            "Pin the GitHub repository for --pr (passed to every gh "
            "call). Recommended on forks, where gh's default "
            "(`gh repo set-default`) often points at the upstream repo "
            "and a bare --pr N would review the wrong project's PR."
        ),
    )
    source.add_argument(
        "--staged",
        action="store_true",
        help="Review staged changes only.",
    )
    source.add_argument(
        "--diff-file",
        default=None,
        metavar="PATH",
        dest="diff_file",
        help=(
            "Review a unified diff read from this file instead of "
            "invoking git ('-' reads stdin). Powers the eval harness "
            "and lets other tools hand the runner a diff directly."
        ),
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
        default=None,  # resolved via CLI > env > project config > default
        help=(
            "Which API to call. ``openrouter`` (default) goes through "
            "OpenRouter's chat-completions endpoint and needs "
            "``OPENROUTER_API_KEY``. ``gemini`` calls Google AI Studio's "
            "generateContent endpoint directly and needs ``GEMINI_API_KEY``. "
            "``ollama`` posts to a local Ollama server's native chat "
            "endpoint (no API key; configure with ``--ollama-host`` / "
            "$OLLAMA_HOST / $OLLAMA_MODEL / $OLLAMA_TIMEOUT / "
            "$OLLAMA_NUM_CTX). Override with $CODE_REVIEW_PROVIDER."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model slug or alias. Defaults to the provider-appropriate "
            "value (``google/gemini-2.5-pro`` for openrouter, "
            "``gemini-2.5-pro`` for gemini, ``qwen3-coder:30b`` for "
            "ollama). Override with $OPENROUTER_MODEL / $GEMINI_MODEL / "
            "$OLLAMA_MODEL respectively. Aliases: pro/gemini-pro, "
            "flash/gemini-flash, claude/claude-sonnet, claude-opus, "
            "gpt, gpt-mini, deepseek (openrouter); local, local-pro "
            "(ollama) -- full table in the README. ``flash`` is ~3x "
            "faster than ``pro`` with some quality loss."
        ),
    )
    # Tri-state (True / False / None=auto). Auto is the default because
    # hunk-only context is the single largest source of false findings:
    # the model cannot see the rest of the file, so it invents it (claims a
    # symbol is undefined when it is defined 100 lines up, "adds" a
    # docstring that already exists, suggests a replacement identical to
    # the current code). Attaching the changed files when they fit is the
    # structural fix; the prompt-level rule in prompts.py is the backstop
    # for when they don't.
    #
    # Explicit --full-files keeps the strict contract (ContextOverflow if
    # the bundle exceeds the cap -- you asked, so you get told). Auto is
    # best-effort: if it does not fit, fall back to diff-only with a note
    # rather than failing a review the user never opted into.
    parser.add_argument(
        "--full-files",
        action="store_true",
        default=None,
        dest="full_files",
        help=(
            "Diff modes only: also send the full current content of "
            "every changed file as reference context, so the model can "
            "judge changes against code outside the +/-5-line hunk "
            "windows. The review target stays the diff. Budgeted "
            "against the same 700K-char cap as --codebase. Default: "
            "AUTO -- attached automatically when it fits under the cap; "
            "passing --full-files makes it strict (error if it does not "
            "fit). Use --no-full-files to force hunks-only."
        ),
    )
    parser.add_argument(
        "--no-full-files",
        action="store_false",
        dest="full_files",
        help=(
            "Diff modes only: never attach changed-file reference "
            "content; review the hunks alone. Cheaper in tokens, but "
            "expect more false 'X is missing' findings."
        ),
    )
    parser.add_argument(
        "--min-found-by",
        type=int,
        default=None,
        dest="min_found_by",
        metavar="N",
        help=(
            "Panel mode (--models) only: drop merged findings that fewer "
            "than N models reported. Cross-model agreement is the "
            "strongest cheap filter for plausible-but-wrong findings, so "
            "--min-found-by 2 keeps only what at least two models found "
            "independently. Default: 1 (keep everything). Env: "
            "CODE_REVIEW_MIN_FOUND_BY."
        ),
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help=(
            "Opt-in: when the payload exceeds the budget (700K chars, "
            "or the Ollama context window), split it at file boundaries "
            "into sequential chunk reviews instead of erroring. "
            "Tradeoff: the model cannot see cross-file relationships "
            "across chunk boundaries. Exit 0 only if every chunk "
            "succeeds."
        ),
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="CSV",
        help=(
            "Comma-separated model slugs/aliases for a multi-model "
            "panel (e.g. ``pro,claude,deepseek``). Each model reviews "
            "the same payload; findings are merged with consensus "
            "annotations (Found by: ...). Exit 0 if at least one model "
            "succeeds. Mutually exclusive with --model. Panels shine "
            "with --provider openrouter (one key, many vendors); "
            "ollama panels run sequentially."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        metavar="URL",
        help=(
            "Ollama server URL when --provider ollama. Defaults to "
            f"{DEFAULT_OLLAMA_HOST} or $OLLAMA_HOST. Useful if Ollama "
            "is on a non-default port, on another machine, or running "
            "inside WSL with non-default networking. Ignored for other "
            "providers."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            f"Sampling temperature. Default {DEFAULT_TEMPERATURE} -- "
            "tuned between the original 0.2 (too conservative, "
            "missed real findings) and a brief 0.5 default (caught "
            "more but produced hallucinated findings on cross-model "
            "review). Range typically 0.0-1.0; higher widens "
            "exploration at higher hallucination risk. Override with "
            "$CODE_REVIEW_TEMPERATURE."
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
            "(rare). Mutually exclusive with --context. Note this also "
            "disables the embedded-instruction (prompt-injection) guard "
            "that normally rides inside the context wrapper."
        ),
    )
    parser.add_argument(
        "--min-severity",
        type=str.upper,
        choices=list(SEVERITY_LEVELS),
        default=None,  # resolved via CLI > env > project config > LOW
        metavar="LEVEL",
        help=(
            "Only report findings at or above this severity "
            "(LOW/MEDIUM/HIGH/CRITICAL; case-insensitive). Default LOW "
            "= no filter. Override with $CODE_REVIEW_MIN_SEVERITY. "
            "Asked of the model via a fork-owned prompt appendix "
            "(upstream prompt files untouched) and ENFORCED after "
            "parsing wherever the runner synthesizes findings (--format "
            "json envelopes, panel reports); verbatim markdown output "
            "remains best-effort."
        ),
    )
    parser.add_argument(
        "--no-project-config",
        action="store_true",
        help=(
            "Ignore any .code-review.toml found for the reviewed repo. "
            "Recommended when auditing untrusted checkouts: the file "
            "can shape the review (model, temperature, include/exclude "
            "-- exclude can hide files from --codebase). Env and CLI "
            "settings still apply."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Extra retry attempts beyond the built-in single 2s retry "
            "on transient failures. N > 0 also enables rate-limit "
            "retries (sleeping the provider's Retry-After, clamped to "
            f"{MAX_RETRY_SLEEP:.0f}s). Default 0. Override with "
            "$CODE_REVIEW_RETRIES. CONFIG / SAFETY_REFUSAL / "
            "CONTEXT_OVERFLOW are never retried."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Also write the review (exact stdout content) to this file, "
            "UTF-8 with LF newlines. Useful on Windows where `tee` "
            "isn't at hand, and for saving reviews across rounds."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default=None,
        help=(
            "Output format. ``markdown`` (default) prints the model's "
            "review verbatim. ``json`` parses the review into a "
            "structured findings envelope (schema_version 1) -- the "
            "prompts are unchanged; parsing is local and deterministic. "
            "On parse failure the envelope carries parse_ok=false plus "
            "the raw markdown, still exit 0. Override with "
            "$CODE_REVIEW_FORMAT."
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="PATH",
        help=(
            "A prior run's --format json output. Current findings are "
            "marked new/persisting against it and disappeared findings "
            "are reported as resolved -- the round-over-round workflow: "
            "--format json --output r.json, fix things, then re-run "
            "with --baseline r.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Resolve config, gather the diff / bundle, build the "
            "prompts, print a report, and exit without calling the "
            "model. No tokens are spent, but read-only subprocesses "
            "(git, gh) and the read-only Ollama /api/ps window probe "
            "still run -- the probe keeps exit-12 behavior identical "
            "to a live run. The best way to debug --include/--exclude."
        ),
    )
    args = parser.parse_args()

    # Informational, like --version: print and exit before any config
    # resolution, so it works with no API key and in any directory.
    if args.list_models:
        print(_list_models_report())
        return

    if args.no_context and args.context is not None:
        raise ConfigError(
            "--no-context and --context are mutually exclusive. Pick one."
        )
    _validate_flag_combos(args)

    # Per-project config from the REVIEWED repo (upward walk from CWD);
    # loading one is always announced on stderr. API keys, context, and
    # ollama_* never come from it; --no-project-config skips it whole.
    project_config = {} if args.no_project_config else _load_project_config()
    _apply_config_file_lists(args, project_config)

    settings = _resolve_settings(args, project_config)
    # Validate the baseline BEFORE the model call so a bad file never
    # burns tokens.
    baseline_doc = (
        load_baseline(settings.baseline) if settings.baseline is not None else None
    )
    requests = _build_requests(args, settings)

    if len(requests) > 1:
        _run_chunked(args, settings, requests)
        return
    request = requests[0]

    if args.dry_run:
        ollama_window: str | None = None
        if settings.provider == "ollama":
            assert settings.ollama_host is not None
            prompt_chars = len(request.system_prompt) + len(request.user_prompt)
            if settings.models is not None:
                # Panel parity: live panels treat a guard trip as a
                # per-model failure (WARN + skip), never exit 12, so the
                # dry-run annotates each model's window instead of
                # raising. Every model gets its own probe.
                notes = []
                for panel_model in settings.models:
                    num_ctx, enforced, window_source = _resolve_ollama_window(
                        settings.ollama_host,
                        panel_model,
                        settings.ollama_num_ctx_env,
                    )
                    would_fail = (
                        enforced and prompt_chars // OLLAMA_CHARS_PER_TOKEN >= num_ctx
                    )
                    suffix = " -- WOULD FAIL pre-flight" if would_fail else ""
                    notes.append(
                        f"{panel_model}: {num_ctx:,} tokens ({window_source}){suffix}"
                    )
                ollama_window = "; ".join(notes)
            else:
                num_ctx, enforced, window_source = _resolve_ollama_window(
                    settings.ollama_host,
                    settings.model,
                    settings.ollama_num_ctx_env,
                )
                # Run the same guard a live run would, so --dry-run exits
                # 12 exactly when a live run would (warn-only when the
                # window is unknown).
                _ollama_prompt_guard(
                    prompt_chars,
                    num_ctx,
                    model=settings.model,
                    enforced=enforced,
                )
                ollama_window = f"{num_ctx:,} tokens ({window_source})"
        print(_dry_run_report(settings, request, ollama_window=ollama_window))
        return

    reviewer = (
        ", ".join(settings.models) if settings.models is not None else settings.model
    )
    if request.mode == "codebase":
        assert request.files is not None
        sys.stderr.write(
            f"Reviewing {len(request.files)} file(s) "
            f"({request.payload_chars:,} chars) with `{reviewer}` "
            f"via {settings.provider} (T={settings.temperature}, "
            f"max_tokens={settings.max_tokens})...\n"
        )
    else:
        sys.stderr.write(
            f"Reviewing {request.payload_chars:,}-char diff with "
            f"`{reviewer}` via {settings.provider} "
            f"(T={settings.temperature}, max_tokens={settings.max_tokens})...\n"
        )

    # Money before the spend, not after. Auto full-file context makes payloads
    # several times larger than the bare diff, so "what is this about to cost"
    # stopped being guessable from the char count. Silent when pricing can't
    # be sourced -- never a fabricated number.
    _cost = _estimate_cost_line(
        settings, len(request.system_prompt) + len(request.user_prompt)
    )
    if _cost is not None:
        sys.stderr.write(f"[cost] est {_cost}\n")

    # Single-model runs: point at panel mode once, on the way past.
    # Cross-model agreement is the cheapest high-precision filter this tool
    # has (see parser.merge/panel_findings_match) -- a finding two models
    # raise independently is almost always real, and the long tail of
    # single-model noise mostly doesn't survive a second opinion. It is
    # under-discovered: it lives behind --models while --model is the
    # obvious flag, so the default path is also the noisiest one.
    # Suppress with CODE_REVIEW_NO_TIPS=1 for scripted/agent callers.
    if settings.models is None and not os.environ.get("CODE_REVIEW_NO_TIPS"):
        sys.stderr.write(
            "TIP: --models pro,gpt,deepseek cross-checks several models and "
            "flags agreement; findings only one model raises are usually "
            "noise. (CODE_REVIEW_NO_TIPS=1 to hide this.)\n"
        )

    if settings.models is not None:
        with _heartbeat():
            results_by_model, failures = _run_panel(settings, request)
        for model, err in failures:
            # Never starts with "ERROR:" -- that prefix is reserved for
            # the single terminal error block.
            sys.stderr.write(
                f"WARN: [panel] {model} failed: {err.category} "
                f"[exit {err.exit_code}] -- {err}\n"
            )
        if not results_by_model:
            # All models failed: exit with one typed error, chosen by
            # the documented category precedence.
            raise _panel_exit_error(failures)
        raw_by_model = {m: r.content for m, r in results_by_model.items()}
        # Panel reports are runner-synthesized in BOTH formats (the
        # markdown is generated from parsed findings, not verbatim), so
        # the severity floor is enforced before merging; the per-model
        # raw appendix still shows everything.
        parsed_by_model = {
            m: enforce_min_severity(
                parse_review_markdown_safe(raw), settings.min_severity
            )
            for m, raw in raw_by_model.items()
        }
        for _parsed in parsed_by_model.values():
            _annotate_hunks(_parsed, request)
        merged = merge_panel_findings(parsed_by_model)
        if settings.min_found_by > 1:
            # Applied AFTER merging (found_by only exists post-merge) and to
            # BOTH formats, like the severity floor: panel reports are
            # runner-synthesized in markdown too, so the filter is a real
            # contract rather than a JSON-only convenience. The per-model raw
            # appendix still shows everything that was dropped.
            kept = [m for m in merged if len(m.found_by) >= settings.min_found_by]
            dropped = len(merged) - len(kept)
            if dropped:
                sys.stderr.write(
                    f"[panel] --min-found-by {settings.min_found_by}: dropped "
                    f"{dropped} finding(s) below the consensus floor "
                    f"({len(kept)} kept).\n"
                )
            merged = kept
        if settings.format == "json":
            stdout_text = json.dumps(
                build_panel_envelope(
                    mode=request.mode,
                    provider=settings.provider,
                    temperature=settings.temperature,
                    models=settings.models,
                    merged=merged,
                    parsed_by_model=parsed_by_model,
                    results_by_model=results_by_model,
                    raw_by_model=raw_by_model,
                    failures=failures,
                ),
                indent=2,
                ensure_ascii=False,
            )
        else:
            stdout_text = render_panel_markdown(
                merged,
                parsed_by_model,
                raw_by_model,
                failures,
                len(settings.models),
            )
        print(stdout_text)
        if settings.output is not None:
            _write_output_file(stdout_text, settings.output)
        return

    with _heartbeat():
        result = _execute_call(
            settings, request.system_prompt, request.user_prompt, settings.model
        )
    usage_line = _format_usage_line(result, settings.provider, settings.model)
    if usage_line is not None:
        sys.stderr.write(usage_line + "\n")

    # Structured-output tail: parse only when something consumes the
    # parse (--format json or --baseline); markdown stdout stays the
    # model's verbatim output either way. In JSON mode the severity
    # floor is enforced post-parse (on both current findings and the
    # baseline, so `resolved` can't fill up with merely-filtered
    # entries); markdown mode leaves it prompt-level best-effort, since
    # the verbatim output shows everything anyway.
    parsed: ParsedReview | None = None
    statuses: list[str] | None = None
    resolved: list[dict] | None = None
    if settings.format == "json" or baseline_doc is not None:
        parsed = parse_review_markdown_safe(result.content)
        _annotate_hunks(parsed, request)
        if settings.format == "json":
            parsed = enforce_min_severity(parsed, settings.min_severity)
            if baseline_doc is not None:
                baseline_doc = filter_baseline_findings(
                    baseline_doc, settings.min_severity
                )
    if baseline_doc is not None:
        assert parsed is not None
        if parsed.parse_ok:
            statuses, resolved = diff_against_baseline(parsed.findings, baseline_doc)
            new = statuses.count("new")
            persisting = statuses.count("persisting")
            sys.stderr.write(
                f"[baseline] {len(parsed.findings)} finding(s): {new} new, "
                f"{persisting} persisting, {len(resolved)} resolved\n"
            )
        else:
            sys.stderr.write(
                "WARN: --baseline skipped; the review output could not be "
                "parsed into findings (see parse problems).\n"
            )

    if settings.format == "json":
        assert parsed is not None
        envelope = build_json_envelope(
            mode=request.mode,
            provider=settings.provider,
            model=settings.model,
            temperature=settings.temperature,
            parsed=parsed,
            result=result,
            raw_markdown=result.content,
            statuses=statuses,
            resolved=resolved,
        )
        stdout_text = json.dumps(envelope, indent=2, ensure_ascii=False)
    else:
        stdout_text = result.content

    print(stdout_text)
    if settings.output is not None:
        _write_output_file(stdout_text, settings.output)


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
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted.\n")
        sys.exit(130)
    except Exception as exc:
        # Honor the README's stderr contract (``ERROR: UNKNOWN [exit 1]``)
        # even for unexpected bugs, so an LLM caller can classify the
        # failure without parsing a raw traceback. The traceback still
        # ships in the Detail line for humans debugging the runner.
        wrapped = ReviewError(
            f"unhandled {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )
        _print_error(wrapped)
        sys.exit(wrapped.exit_code)


if __name__ == "__main__":
    _entrypoint()
