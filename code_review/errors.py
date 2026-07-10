"""The typed error model: exception hierarchy, exit codes, stderr block.

Exit codes (0/1/2/10-14/130) and the ``ERROR: <CATEGORY> [exit <N>]``
stderr first line are public API pinned by tests -- see the README's
"Error model" section for the caller-facing contract.
"""

from __future__ import annotations

import sys


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
    """Provider HTTP 429 -- request quota or RPS cap exceeded.

    ``retry_after_seconds`` is the machine-readable parse of the
    Retry-After header: delta-seconds directly, HTTP-dates converted to
    a delta, ``None`` when the header was absent or unparseable. The
    ``--retries`` sleep and the human-readable hint in the message are
    derived from the same header value so they can never disagree.
    """

    exit_code = 11
    category = "RATE_LIMIT"
    suggested = (
        "Wait 30-60 seconds and retry. If the limit is per-key per-day "
        "(common on free tiers), switch ``--provider`` or ``--model`` to "
        "one with separate quota."
    )
    retry_after_seconds: float | None = None


class ContextOverflow(ReviewError):
    """Diff exceeded the model's input or output token budget."""

    exit_code = 12
    category = "CONTEXT_OVERFLOW"
    suggested = (
        "Narrow the diff scope: ``--include`` / ``--exclude`` in codebase "
        "mode, or a smaller ``--base`` ref in diff mode. Do not retry "
        "without reducing scope -- a second call with the same payload "
        "will hit the same limit. Exception: if the message says "
        "max_tokens was hit before any content appeared (reasoning "
        "models can spend the whole budget thinking), raise "
        "``--max-tokens`` instead of narrowing scope."
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
