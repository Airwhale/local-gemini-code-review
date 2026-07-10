"""Big-input chunking (--chunk): lossless splitting and budget math.

Splits payloads at file boundaries (whole per-file diffs, whole files)
and packs them against a per-chunk budget -- the bundle cap for cloud,
the enforced Ollama window (with the 0.85 fill margin) for local.
"""

from __future__ import annotations

import re
from pathlib import Path

from code_review.config import Settings
from code_review.errors import ContextOverflow
from code_review.prompts import (
    MAX_BUNDLE_CHARS,
    _min_severity_instruction,
    build_codebase_prompts,
    build_diff_prompts,
    bundle_codebase,
)
from code_review.providers import (
    DEFAULT_OLLAMA_NUM_CTX,
    OLLAMA_CHARS_PER_TOKEN,
    OLLAMA_WINDOW_FILL,
    _resolve_ollama_window,
)

_DIFF_FILE_ANCHOR = re.compile(r"^diff --git ", re.M)


def split_diff_by_file(diff: str) -> list[str]:
    """Split a unified diff into per-file parts, losslessly.

    ``"".join(parts) == diff`` always holds: any preamble before the
    first ``diff --git`` line stays attached to the first part.
    """
    starts = [m.start() for m in _DIFF_FILE_ANCHOR.finditer(diff)]
    if not starts:
        return [diff] if diff else []
    parts: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(diff)
        parts.append(diff[start:end])
    if starts[0] > 0:
        parts[0] = diff[: starts[0]] + parts[0]
    return parts


def _pack_contiguous(sizes: list[int], budget: int) -> list[list[int]]:
    """Order-preserving next-fit packing of item indices into chunks.

    Contiguous (never reorders) so codebase chunks follow ``git
    ls-files`` order and diff chunks follow diff order -- neighboring
    files stay together, which is the best cheap approximation of
    keeping related code in one chunk. Callers must pre-validate that
    no single item exceeds the budget.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    current_size = 0
    for idx, size in enumerate(sizes):
        if current and current_size + size > budget:
            chunks.append(current)
            current, current_size = [], 0
        current.append(idx)
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def partition_codebase(files: list[Path], budget: int) -> list[list[Path]]:
    """Partition codebase files into chunks whose bundled size fits ``budget``."""
    sized = [(p, len(bundle_codebase([p]))) for p in files]
    for p, size in sized:
        if size > budget:
            raise ContextOverflow(
                f"Single file {p.as_posix()} bundles to {size:,} chars, "
                f"over the {budget:,}-char chunk budget -- chunking cannot "
                "help. Exclude it (--exclude) or raise the budget "
                "($OLLAMA_NUM_CTX for ollama)."
            )
    index_chunks = _pack_contiguous([s for _, s in sized], budget)
    return [[sized[i][0] for i in chunk] for chunk in index_chunks]


def partition_diffs(parts: list[str], budget: int) -> list[str]:
    """Pack per-file diff parts into chunk-sized diff strings."""
    for part in parts:
        if len(part) > budget:
            first_line = part.splitlines()[0] if part else "(empty)"
            raise ContextOverflow(
                f"A single file's diff ({first_line!r}) is {len(part):,} "
                f"chars, over the {budget:,}-char chunk budget -- chunking "
                "cannot help. Use a smaller --base or exclude the file "
                "from the change."
            )
    index_chunks = _pack_contiguous([len(p) for p in parts], budget)
    return ["".join(parts[i] for i in chunk) for chunk in index_chunks]


def _chunk_budget(
    settings: Settings, *, is_codebase: bool = False
) -> tuple[int, str | None]:
    """Per-chunk payload budget in chars, plus an optional WARN note.

    Cloud providers budget against the standard bundle cap. Ollama
    budgets against the ENFORCED window (env-set or /api/ps-detected,
    probed once here before partitioning) minus the fixed prompt
    overhead, so chunks don't trip the pre-flight guard. When the window
    is unknown even after the probe, sizing assumes the smallest stock
    tier -- safety over efficiency: a 32K machine wastes calls, but
    never silently truncates -- with a WARN recommending $OLLAMA_NUM_CTX.
    """
    if settings.provider != "ollama":
        return MAX_BUNDLE_CHARS, None
    assert settings.ollama_host is not None
    num_ctx, enforced, _source = _resolve_ollama_window(
        settings.ollama_host, settings.model, settings.ollama_num_ctx_env
    )
    note = None
    if not enforced:
        note = (
            "chunk sizing assumes the smallest stock Ollama window "
            f"({DEFAULT_OLLAMA_NUM_CTX:,} tokens) because the actual window "
            "is unknown; set $OLLAMA_NUM_CTX to size chunks to your real "
            "window and cut the call count"
        )
    # Overhead: the prompts minus the payload (skill + command template +
    # context wrapper + severity appendix). Measured, not estimated --
    # against the prompt set the chunks will actually use: the codebase
    # skill/template is larger than the diff one, so measuring the diff
    # overhead for --codebase chunks would oversize them and trip the
    # post-call truncation verify.
    build_prompts = build_codebase_prompts if is_codebase else build_diff_prompts
    empty_system, empty_user = build_prompts("", settings.context)
    overhead = (
        len(empty_system)
        + len(empty_user)
        + len(_min_severity_instruction(settings.min_severity))
    )
    budget = min(
        MAX_BUNDLE_CHARS,
        int(num_ctx * OLLAMA_CHARS_PER_TOKEN * OLLAMA_WINDOW_FILL) - overhead,
    )
    if budget <= 0:
        raise ContextOverflow(
            f"The review prompt overhead alone (~{overhead:,} chars) fills "
            f"the {num_ctx:,}-token Ollama window; chunking cannot help. "
            "Raise $OLLAMA_NUM_CTX (RAM permitting).",
            provider="ollama",
            model=settings.model,
        )
    return budget, note
