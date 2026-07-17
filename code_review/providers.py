"""Provider wire layer: OpenRouter, Gemini API, and native Ollama.

One request per call through ``_make_client`` (the single HTTP
construction point wire tests mock), typed HTTP-error classification,
the retry policy, model aliases/defaults, and the Ollama
context-window machinery (probe, pre-flight guard, post-call
truncation verify).
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

import httpx

from code_review.errors import (
    ConfigError,
    ContextOverflow,
    ProviderHiccup,
    RateLimit,
    ReviewError,
    SafetyRefusal,
    TransportError,
)

T = TypeVar("T")


@dataclasses.dataclass
class CallResult:
    """What a provider call returns beyond the review text itself.

    ``prompt_tokens`` / ``completion_tokens`` come from the provider's
    usage block and are ``None`` when the response carried none (the
    ``[usage]`` stderr line is skipped in that case, never estimated).
    ``truncated`` is True when the model returned content but stopped at
    the max_tokens ceiling -- callers must treat the findings list as
    potentially incomplete (a ``WARN:`` line accompanies it on stderr).
    """

    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False


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

MAX_RETRY_SLEEP = 300.0

# Severity ladder used by --min-severity (and, in M2+, finding sorting).
# Order matters: index position defines rank.
PROVIDERS = ("openrouter", "gemini", "ollama")
DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "openrouter": "google/gemini-2.5-pro",
    "gemini": "gemini-2.5-pro",
    # Ollama default. ``qwen3-coder:30b`` is the MoE coder model with
    # ~3.3B active params -- best quality/speed balance on CPU. If the
    # user hasn't pulled it, ``call_ollama`` raises a typed ConfigError
    # pointing them to ``ollama pull qwen3-coder:30b``. Override per env
    # with $OLLAMA_MODEL or per call with --model.
    "ollama": "qwen3-coder:30b",
}
DEFAULT_PROVIDER = "openrouter"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
# Ollama exposes both its native API (``/api/chat``) and an OpenAI-compatible
# endpoint (``/v1/chat/completions``). We use the NATIVE endpoint: it accepts
# per-request ``options.num_ctx`` (so the runner can request the context
# window instead of guessing what the server loaded) and reports
# ``prompt_eval_count``, which lets ``_ollama_post_verify`` detect silent
# prompt truncation after the fact. The OpenAI-compat endpoint offers
# neither.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_CHAT_PATH = "/api/chat"
HTTP_TIMEOUT = 300.0  # Gemini 2.5 Pro on a ~5K-line diff lands ~30-60s; pad
# generously for very large diffs and whole-codebase
# bundles.
# Local CPU inference on a 30B MoE coder model takes ~10-25 tok/sec on
# modern hardware, so a thorough review (1500-3000 output tokens) can run
# 1-5 minutes; cold-start model load adds another 10-60s on the first
# call after server start. ``HTTP_TIMEOUT`` (300s = 5 min) is too tight
# for that worst case, so Ollama gets its own ceiling. Override with
# $OLLAMA_TIMEOUT if you're on slower hardware or running larger models.
DEFAULT_OLLAMA_TIMEOUT = 1800.0  # 30 minutes

# Ollama context-window guard. Unlike the cloud providers -- which return
# a 4xx when the prompt exceeds the model's context -- Ollama silently
# TRUNCATES a prompt that doesn't fit the loaded context window
# (``num_ctx``) and generates from whatever survived. For a code review
# that's the worst failure mode: the model reviews a fragment of the
# diff, returns exit 0, and the output looks like a legitimate "few
# issues found" review. The native endpoint accepts a per-request
# ``options.num_ctx``, so the runner requests the window when it knows
# it, refuses pre-flight (typed CONTEXT_OVERFLOW) when the prompt
# exceeds a known window, and verifies ``prompt_eval_count`` post-call
# as the backstop for the unknown case.
#
# Stock Ollama picks the window from available VRAM (per
# docs.ollama.com/context-length: <24 GiB -> 4K, 24-48 GiB -> 32K,
# >=48 GiB -> 256K), and ``OLLAMA_CONTEXT_LENGTH`` / the app settings /
# a Modelfile ``num_ctx`` can override it. The runner resolves the
# window per model per call and REQUESTS it via the native endpoint's
# ``options.num_ctx``:
#
#   1. ``$OLLAMA_NUM_CTX`` set -> sent as num_ctx; guard is a hard
#      pre-flight error. (RAM warning: the KV cache scales with the
#      window; a huge value can OOM/swap the server.)
#   2. Model currently loaded -> its actual ``context_length`` from
#      ``GET /api/ps`` (see ``_detect_ollama_num_ctx``) is sent back as
#      num_ctx (same value = no model reload); hard pre-flight error.
#      Covers every round after the first in an iterative review loop.
#   3. Unknown -> ``num_ctx`` is OMITTED from the request so the server
#      keeps its own VRAM-tier default (sending the 4096 advisory value
#      would actively SHRINK a 32K/256K window and cause the very
#      truncation the guard exists to prevent). The pre-flight guard
#      only WARNs against the smallest-tier estimate, and
#      ``_ollama_post_verify`` backstops with a hard error after the
#      call if ``prompt_eval_count`` shows the prompt filled the window.
#
# Token estimate is the standard ~4 chars/token.
DEFAULT_OLLAMA_NUM_CTX = 4096
OLLAMA_CHARS_PER_TOKEN = 4
OLLAMA_PS_TIMEOUT = 5.0  # /api/ps probe is local and fast; never let a
# hung probe delay the actual review call long.
# Post-call truncation check: prompt_eval_count at or above this
# fraction of the effective window means the prompt almost certainly
# filled (and overflowed) it.
OLLAMA_POST_VERIFY_MARGIN = 0.98

# Fraction of the Ollama window that --chunk budgets for prompt content.
# The ~4-chars/token estimate runs DENSER on real tokenizers for
# code-heavy text: budgeting chunks to 100% of the window sized a chunk
# to an estimated 19.9K tokens that actually tokenized to >=20,000 on a
# 20K window -- the prompt filled it and post-verify (correctly)
# discarded the output. The margin absorbs tokenizer variance and
# leaves generation room (Ollama's output shares num_ctx).
OLLAMA_WINDOW_FILL = 0.85

# Sampling temperature for the model. History of this constant:
#
#   0.2  (original): too conservative -- 1-2 findings per round on diffs
#        that plausibly contained more; 5-7 rounds to converge.
#   0.5  (raised after observing the above): more findings per round
#        (3-5 typical), but on a later cross-model comparison
#        ``google/gemini-2.5-pro`` produced a HIGH-severity finding
#        that referenced a CLI flag (``--timeout``) and quoted "help
#        text" that don't exist in the codebase -- a clean
#        hallucination that would have crashed the runner if its
#        suggested fix had been applied verbatim.
#   0.3  (current): split the difference. Tighter than 0.5 to cut the
#        hallucination rate; looser than 0.2 to keep the "surface more
#        findings" benefit. Empirical re-tuning is encouraged if you
#        observe drift either way; override per call with
#        ``--temperature`` or per environment with
#        ``$CODE_REVIEW_TEMPERATURE``.
DEFAULT_TEMPERATURE = 0.3

# Maximum output tokens the model may emit. Raised from the implicit
# provider default (~8K for Gemini 2.5 Pro) to 16K so a thorough review
# isn't truncated mid-finding. This is a *ceiling*, not a target -- you
# pay only for tokens actually emitted, not for unused headroom.
# Override with ``--max-tokens`` or ``$CODE_REVIEW_MAX_TOKENS``.
DEFAULT_MAX_TOKENS = 16000

# Named aliases for model slugs, scoped per provider. Each provider has
# its own slug format (OpenRouter: ``vendor/model``, Gemini API: bare
# ``gemini-...`` names, Ollama: ``family:tag``), and an alias is only
# valid for its declared provider. Aliases are resolved before the call
# is made; the resolved slug is what shows up in stderr "Reviewing ...
# with ..." and what the provider actually dispatches to.
#
# When a user passes ``--model <alias>`` with the wrong provider, the
# runner raises a typed ConfigError pointing them at the correct
# ``--provider`` instead of silently sending an invalid model name to
# the upstream API.
#
# Keep these tables small and curated. The aliases here are the models
# that earned their place as a "second-opinion reviewer" in practice;
# users who want exotic models can pass the raw slug via ``--model``.
MODEL_ALIASES_BY_PROVIDER: dict[str, dict[str, str]] = {
    "openrouter": {
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
    },
    "ollama": {
        # Local model tiers, both qwen3-coder. ``local`` is the
        # recommended default (30B MoE with ~3.3B active params, the
        # quality/speed sweet spot on CPU because active param count
        # drives inference speed, not total params). ``local-pro`` is
        # the larger MoE (80B/3B active) for users who want higher
        # quality and can spare the disk (~52 GB) for marginal speed
        # cost. Users who haven't pulled a given tag get a typed
        # ConfigError from ``call_ollama`` pointing them at the right
        # ``ollama pull`` command.
        #
        # No ``local-fast`` alias: qwen3-coder doesn't ship a small
        # dense variant on Ollama, and the qwen2.5-coder family is a
        # generation behind on code review quality. Users who need a
        # smaller model should pass the explicit ``--model`` slug
        # (e.g. ``--model qwen2.5-coder:7b``) rather than rely on a
        # "fast" alias that papers over the generation gap.
        "local": "qwen3-coder:30b",
        "local-pro": "qwen3-coder-next",
    },
    # ``gemini`` (direct-API) has no aliases -- it takes bare Gemini
    # model names only. Passing an alias with --provider gemini raises
    # a ConfigError so the user gets a clear message rather than a
    # cryptic 404 from Google's API.
}

# Backwards-compatible flat alias map. Some external tooling or tests
# may have imported the old top-level ``MODEL_ALIASES`` dict that
# contained only the OpenRouter aliases. The new authoritative table
# is ``MODEL_ALIASES_BY_PROVIDER``; this name is preserved as a
# read-only view over the OpenRouter slice for compatibility.
MODEL_ALIASES: dict[str, str] = MODEL_ALIASES_BY_PROVIDER["openrouter"]


# Upstream `code-review.toml` instructs the model to *call* git itself via a
# tool. We have no tool layer; instead we extract the diff up front and
# substitute it into the prompt. This is the literal sentence from
# `commands/code-review.toml`; if upstream rewords it the substitution
# silently no-ops and the script still works (the diff is also appended
# unconditionally if the substitution missed).
def _parse_retry_after(value: str) -> float | None:
    """Parse a Retry-After header to seconds-from-now.

    Delta-seconds (``"30"``) parse directly -- gated on the same
    ``strip()/isdigit()`` check the message hint uses, so the sleep and
    the hint can never disagree. HTTP-dates (``"Fri, 31 Dec 1999
    23:59:59 GMT"``) convert via the stdlib email parser to a
    non-negative delta. Anything unparseable returns ``None`` (callers
    fall back to a fixed wait).
    """
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError, IndexError):
        # The email parser raises more than ValueError on pathological
        # inputs (OverflowError on absurd years, IndexError on some
        # malformed structures); any parse failure must stay a
        # RATE_LIMIT with no hint, never escape as UNKNOWN.
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())


def _classify_http_error(
    status: int,
    body: str,
    *,
    model: str,
    provider: str,
    retry_after: str | None = None,
) -> ReviewError:
    """Map a 4xx/5xx response to the right typed error.

    Provider-side messages are inconsistent across vendors, so we pattern-
    match on a small set of substrings the major providers use today. The
    classification is best-effort -- a future provider could land a 4xx
    that doesn't match any pattern, and that's fine: it falls through to
    a generic ``ReviewError`` which surfaces as exit 1 with the response
    body in ``detail`` so a caller can decide.

    Order matters: the 5xx and 401/403 checks run BEFORE the substring
    match so a provider-side failure page or auth-rejection body that
    happens to contain a phrase like "too long" or "token limit" keeps
    its status-derived classification (retryable TRANSPORT / fix-the-key
    CONFIG) instead of being misclassified as a do-not-retry
    CONTEXT_OVERFLOW.
    """
    body_lower = body.lower()
    if status == 429:
        # Retry-After is either delta-seconds ("30") or an HTTP-date
        # ("Fri, 31 Dec 1999 23:59:59 GMT") -- only append the seconds
        # unit when the value is numeric. ``retry_after_seconds`` is the
        # machine-readable twin for the --retries sleep.
        wait_hint = ""
        if retry_after:
            retry_after = retry_after.strip()
            suffix = "s" if retry_after.isdigit() else ""
            wait_hint = f" (Retry-After: {retry_after}{suffix})"
        err = RateLimit(
            f"{provider} returned HTTP 429{wait_hint}",
            detail=body[:1000],
            model=model,
            provider=provider,
        )
        if retry_after:
            err.retry_after_seconds = _parse_retry_after(retry_after)
        return err
    if 500 <= status < 600:
        return TransportError(
            f"{provider} returned HTTP {status} (provider-side failure)",
            detail=body[:1000],
            model=model,
            provider=provider,
        )
    if status in (401, 403):
        # Auth failures are a fix-your-key problem, so they belong in the
        # CONFIG lane (exit 2, never retried) rather than falling through
        # to the generic UNKNOWN bucket. One carve-out: OpenRouter uses
        # 403 when its moderation layer flags the INPUT -- a content
        # decision, not a credentials problem -- which stays in the
        # SAFETY_REFUSAL lane so the exit code steers the caller to the
        # safety-context docs instead of key rotation.
        if status == 403 and ("moderation" in body_lower or "flagged" in body_lower):
            return SafetyRefusal(
                f"{provider} returned HTTP 403 (input flagged by moderation)",
                detail=body[:1000],
                model=model,
                provider=provider,
            )
        key_hint = {
            "openrouter": " Check OPENROUTER_API_KEY.",
            "gemini": " Check GEMINI_API_KEY.",
        }.get(provider, "")
        return ConfigError(
            f"{provider} returned HTTP {status} (authentication/authorization "
            f"failed).{key_hint}",
            detail=body[:1000],
            model=model,
            provider=provider,
        )
    # 4xx only from here down. Match the family of provider phrases that
    # signal context-length overflow, kept as a tuple so adding a new
    # variant is a one-line edit. Each phrase was added based on an
    # actual provider response observed in the wild or in published
    # vendor docs; do not add speculative phrases without verifying a
    # provider uses them, or you risk false-positive ContextOverflow
    # classifications on unrelated 4xx errors.
    CONTEXT_OVERFLOW_PHRASES = (
        "context_length",
        # OpenRouter's 400 reads "This endpoint's maximum context length is
        # 163840 tokens. However, you requested about 169737 tokens" -- no
        # underscore, and not "exceeds the maximum", so it fell through to
        # UNKNOWN (exit 1, "escalate if unclear") when the truth is a typed
        # CONTEXT_OVERFLOW (exit 12, "the scope is wrong, not the call").
        "maximum context length",
        "too long",
        "exceeds the maximum",
        "token limit",
        "input too large",
        "payload size",
    )
    if status == 413 or any(
        phrase in body_lower for phrase in CONTEXT_OVERFLOW_PHRASES
    ):
        return ContextOverflow(
            f"{provider} returned HTTP {status} with context-length indication",
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


def _warn_if_truncated(hit_token_ceiling: bool, max_tokens: int, provider: str) -> None:
    """Warn on stderr when the model returned content but stopped at the
    ``max_tokens`` ceiling.

    Without this, a review cut off mid-finding is indistinguishable from
    a complete one: the runner exits 0 and an LLM caller parses the
    partial findings list as if it were the whole review. Truncation with
    non-empty content is a warning rather than an error because the
    partial output is still useful; empty content at the ceiling stays a
    hard ContextOverflow in the provider callers.
    """
    if hit_token_ceiling:
        sys.stderr.write(
            f"WARN: {provider} output was truncated at max_tokens="
            f"{max_tokens}; the review below may be incomplete. "
            "Re-run with a higher --max-tokens for full output.\n"
        )


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Pricing moves (models get repriced, new ones appear), so a hardcoded table
# would quietly go stale -- and a WRONG money number is worse than none.
# OpenRouter publishes live per-token prices unauthenticated, so fetch and
# cache. One day is short enough to track repricing, long enough that the
# fetch is invisible in normal use.
PRICING_CACHE_TTL_SECONDS = 24 * 3600
PRICING_FETCH_TIMEOUT = 10.0


def _pricing_cache_path() -> Path:
    from code_review.config import _user_config_dir

    # Filename bumped from openrouter-pricing.json when context_length was
    # added: a v1 cache has no windows, and a stale hit would silently
    # disable the context guard for a day.
    return _user_config_dir() / "openrouter-models-v2.json"


def _pricing_float(value: object) -> float | None:
    """Coerce a pricing/cache numeric field; anything odd is invalid."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_cached_pricing() -> dict[str, dict[str, float]] | None:
    """Return cached OpenRouter pricing, or None if absent/stale/corrupt."""
    path = _pricing_cache_path()
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(blob["fetched_at"]) > PRICING_CACHE_TTL_SECONDS:
            return None
        models = blob["models"]
    except (OSError, ValueError, KeyError, TypeError):
        # Corrupt/absent cache is not an error -- just a miss.
        return None
    if not isinstance(models, dict):
        return None
    validated: dict[str, dict[str, float]] = {}
    for slug, entry in models.items():
        if not isinstance(slug, str) or not isinstance(entry, dict):
            continue
        prompt = _pricing_float(entry.get("prompt"))
        completion = _pricing_float(entry.get("completion"))
        if prompt is None or completion is None:
            continue
        validated_entry = {"prompt": prompt, "completion": completion}
        context_length = _pricing_float(entry.get("context_length"))
        if context_length is not None:
            validated_entry["context_length"] = context_length
        validated[slug] = validated_entry
    return validated or None


def _store_cached_pricing(models: dict[str, dict[str, float]]) -> None:
    path = _pricing_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "models": models}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        # A read-only/absent config dir must not break a review.
        pass


def openrouter_pricing() -> dict[str, dict[str, float]] | None:
    """Per-token USD prices keyed by model slug, or None if unavailable.

    Cached to the user config dir for a day. Every failure path returns
    None rather than raising: a cost *estimate* is a courtesy, and must
    never be the reason a review fails or a --dry-run errors.
    """
    cached = _load_cached_pricing()
    if cached is not None:
        return cached
    try:
        with _make_client(PRICING_FETCH_TIMEOUT) as client:
            resp = client.get(OPENROUTER_MODELS_URL)
            resp.raise_for_status()
            data = resp.json()
        models: dict[str, dict[str, float]] = {}
        for entry in data.get("data") or []:
            slug = entry.get("id")
            pricing = entry.get("pricing") or {}
            try:
                # OpenRouter sends prices as per-token decimal STRINGS
                # ("0.00000125"); floats here, formatting at the edge.
                entry_info: dict[str, float] = {
                    "prompt": float(pricing["prompt"]),
                    "completion": float(pricing["completion"]),
                }
            except (TypeError, ValueError, KeyError):
                continue  # skip unpriced/odd entries, keep the rest
            # Context window, when published. Drives the auto full-files
            # guard: the global 700K-char cap is sized for Gemini (1M) and
            # Claude (200K), but deepseek-chat-v3.1 is 163,840 -- smaller
            # than the cap allows, so a payload can clear the cap and still
            # blow the model's window.
            try:
                entry_info["context_length"] = float(entry["context_length"])
            except (TypeError, ValueError, KeyError):
                pass
            models[slug] = entry_info
        if not models:
            return None
        _store_cached_pricing(models)
        return models
    except Exception:
        # Offline, rate-limited, shape change -- all just "no estimate".
        return None


def model_context_limit(provider: str, model: str) -> int | None:
    """Published context window (tokens) for ``model``, or None if unknown.

    Only OpenRouter publishes this in a feed we already fetch. None means
    "don't know" -- callers must treat that as "no guard available" rather
    than assuming any particular size.
    """
    if provider != "openrouter":
        return None
    info = openrouter_pricing()
    if not info or model not in info:
        return None
    entry = info.get(model)
    if not isinstance(entry, dict):
        return None
    window = _pricing_float(entry.get("context_length"))
    return int(window) if window else None


def estimate_cost_usd(
    provider: str, model: str, prompt_tokens: int, max_completion_tokens: int
) -> float | None:
    """Upper-bound USD estimate for one call, or None when unknowable.

    Prompt side is an estimate (chars/4); completion is bounded by
    ``max_tokens``, so the total is a ceiling, not a prediction. Ollama is
    local and free. Gemini publishes no unauthenticated price feed, so it
    returns None rather than a guess -- consistent with the rule that this
    tool never invents usage numbers.
    """
    if provider == "ollama":
        return 0.0
    if provider != "openrouter":
        return None
    prices = openrouter_pricing()
    if not prices or model not in prices:
        return None
    p = prices.get(model)
    if not isinstance(p, dict):
        return None
    prompt_price = _pricing_float(p.get("prompt"))
    completion_price = _pricing_float(p.get("completion"))
    if prompt_price is None or completion_price is None:
        return None
    return prompt_tokens * prompt_price + max_completion_tokens * completion_price


def format_cost(usd: float) -> str:
    """Human-readable USD, with enough precision for sub-cent estimates."""
    if usd == 0:
        return "$0.00 (local)"
    if usd < 0.01:
        return f"~${usd:.4f}"
    return f"~${usd:.2f}"


def _make_client(timeout: httpx.Timeout | float) -> httpx.Client:
    """Single construction point for HTTP clients.

    Every wire call (all three providers plus the /api/ps probe) builds
    its client here, so wire tests can monkeypatch one function with an
    ``httpx.MockTransport``-backed factory and cover the full HTTP
    surface offline.
    """
    return httpx.Client(timeout=timeout)


def _usage_int(value: object) -> int | None:
    """Coerce a provider usage field to int; anything else -> None.

    ``isinstance(value, bool)`` is excluded explicitly because bool is a
    subclass of int and a buggy provider sending ``true`` should read as
    "no usage data", not "1 token".
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


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
) -> CallResult:
    """POST to OpenRouter's chat-completions endpoint and return the review
    as a ``CallResult`` (markdown + usage + truncation flag). Caller builds
    the prompts so this function stays mode-agnostic (diff review and
    codebase review share the same wire path).

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
        with _make_client(HTTP_TIMEOUT) as client:
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
            response.status_code,
            response.text,
            model=model,
            provider="openrouter",
            retry_after=response.headers.get("retry-after"),
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
    if not isinstance(data, dict):
        # A misbehaving proxy can return a JSON array/string; without
        # this, .get() raises AttributeError and exits UNKNOWN instead
        # of the retryable PROVIDER_HICCUP it really is.
        raise ProviderHiccup(
            "OpenRouter returned non-object JSON",
            detail=response.text[:1000],
            model=model,
            provider="openrouter",
        )

    choices = data.get("choices") or []
    # The isinstance guard covers a dict/string where the list should
    # be: {"choices": {...}} is truthy, so the emptiness check alone
    # would pass and choices[0] would raise KeyError -> UNKNOWN.
    if not isinstance(choices, list) or not choices:
        raise ProviderHiccup(
            "OpenRouter response had no choices list",
            detail=str(data)[:1000],
            model=model,
            provider="openrouter",
        )
    choice = choices[0]
    # Same malformed-shape contract as the top-level check: nested
    # non-object values (a string in choices[], a list for message)
    # must surface as retryable PROVIDER_HICCUP, not as an
    # AttributeError escaping to UNKNOWN.
    if not isinstance(choice, dict):
        raise ProviderHiccup(
            "OpenRouter choices[0] is not an object",
            detail=str(data)[:1000],
            model=model,
            provider="openrouter",
        )
    # Inspect both ``finish_reason`` (OpenAI-standard normalized value)
    # and ``native_finish_reason`` (OpenRouter's pass-through of the
    # underlying provider's raw reason). They can disagree: a provider
    # that safety-blocked a completion may surface as
    # ``native_finish_reason="safety"`` while the normalized
    # ``finish_reason="stop"`` -- preferring one over the other would
    # miss that signal. We classify by the UNION of both fields, so a
    # safety signal from either source wins over a generic "stop".
    # str() coercion: a numeric/oddly-typed reason must not crash
    # ``.lower()``.
    finish_reason = str(choice.get("finish_reason") or "").lower()
    native_finish_reason = str(choice.get("native_finish_reason") or "").lower()
    finish_reasons = {finish_reason, native_finish_reason} - {""}
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise ProviderHiccup(
            "OpenRouter message is not an object",
            detail=str(data)[:1000],
            model=model,
            provider="openrouter",
        )
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        # Non-string content falls through to the empty-content
        # classification below, which always ends in a typed raise.
        content = None
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}  # optional metadata -- degrade, don't fail the review

    if content:
        truncated = bool(finish_reasons & {"length", "max_tokens"})
        _warn_if_truncated(truncated, max_tokens, "openrouter")
        return CallResult(
            content,
            prompt_tokens=_usage_int(usage.get("prompt_tokens")),
            completion_tokens=_usage_int(usage.get("completion_tokens")),
            truncated=truncated,
        )

    # Null / empty content -- classify by the union of finish_reasons.
    if finish_reasons & {"safety", "content_filter", "content-filter", "blocked"}:
        raise SafetyRefusal(
            f"Model refused with finish_reasons={sorted(finish_reasons)!r}",
            detail=str(data)[:1000],
            model=model,
            provider="openrouter",
        )
    if finish_reasons & {"length", "max_tokens"}:
        raise ContextOverflow(
            f"Hit max_tokens ({max_tokens}) before producing content "
            f"(finish_reasons={sorted(finish_reasons)!r})",
            detail=str(data)[:1000],
            model=model,
            provider="openrouter",
        )
    raise ProviderHiccup(
        f"Null content with finish_reasons={sorted(finish_reasons)!r}",
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
) -> CallResult:
    """POST to Google AI Studio's ``generateContent`` endpoint directly and
    return a ``CallResult``. Caller builds the prompts (same as
    ``call_openrouter``) so the wire path is mode-agnostic. Raises typed
    ``ReviewError`` subclasses; see README "Error model".
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
        with _make_client(HTTP_TIMEOUT) as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        raise TransportError(
            f"Gemini request failed before response: {exc}",
            model=model,
            provider="gemini",
        ) from exc

    if response.status_code >= 400:
        raise _classify_http_error(
            response.status_code,
            response.text,
            model=model,
            provider="gemini",
            retry_after=response.headers.get("retry-after"),
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
    if not isinstance(data, dict):
        raise ProviderHiccup(
            "Gemini API returned non-object JSON",
            detail=response.text[:1000],
            model=model,
            provider="gemini",
        )

    # Gemini can refuse at the prompt level (before generation) by
    # returning ``promptFeedback.blockReason`` with no candidates.
    prompt_feedback = data.get("promptFeedback") or {}
    if not isinstance(prompt_feedback, dict):
        prompt_feedback = {}  # malformed feedback -> fall through to candidates
    block_reason = prompt_feedback.get("blockReason")
    if block_reason:
        raise SafetyRefusal(
            f"Gemini blocked prompt: blockReason={block_reason!r}",
            detail=str(data)[:1000],
            model=model,
            provider="gemini",
        )

    candidates = data.get("candidates") or []
    # isinstance: a dict/string here is truthy, so the emptiness check
    # alone would pass and candidates[0] would raise KeyError -> UNKNOWN.
    if not isinstance(candidates, list) or not candidates:
        raise ProviderHiccup(
            "Gemini response had no candidates list",
            detail=str(data)[:1000],
            model=model,
            provider="gemini",
        )
    candidate = candidates[0]
    # Same malformed-shape contract as the top-level check: nested
    # non-object values must surface as retryable PROVIDER_HICCUP, not
    # as an AttributeError escaping to UNKNOWN. Non-dict parts entries
    # and non-string texts are skipped rather than fatal -- an empty
    # result then flows into the empty-content classification below,
    # which always ends in a typed raise.
    if not isinstance(candidate, dict):
        raise ProviderHiccup(
            "Gemini candidates[0] is not an object",
            detail=str(data)[:1000],
            model=model,
            provider="gemini",
        )
    finish_reason = str(candidate.get("finishReason") or "").upper()
    content_block = candidate.get("content") or {}
    if not isinstance(content_block, dict):
        raise ProviderHiccup(
            "Gemini candidate content is not an object",
            detail=str(data)[:1000],
            model=model,
            provider="gemini",
        )
    parts = content_block.get("parts") or []
    text = "".join(
        part_text
        for part in parts
        if isinstance(part, dict) and isinstance(part_text := part.get("text"), str)
    )
    usage = data.get("usageMetadata") or {}
    if not isinstance(usage, dict):
        usage = {}  # optional metadata -- degrade, don't fail the review

    if text:
        truncated = finish_reason == "MAX_TOKENS"
        _warn_if_truncated(truncated, max_tokens, "gemini")
        # Thinking models (gemini-2.5-pro et al.) bill thought tokens
        # separately in ``thoughtsTokenCount``; candidatesTokenCount is
        # only the visible output, so summing keeps the [usage] line an
        # honest cost signal instead of understating by the (often
        # large) reasoning budget.
        completion = _usage_int(usage.get("candidatesTokenCount"))
        thoughts = _usage_int(usage.get("thoughtsTokenCount"))
        if thoughts:
            completion = (completion or 0) + thoughts
        return CallResult(
            text,
            prompt_tokens=_usage_int(usage.get("promptTokenCount")),
            completion_tokens=completion,
            truncated=truncated,
        )

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


def _normalize_ollama_host(host: str) -> str:
    """Return ``host`` as a scheme-qualified base URL.

    Ollama's own convention for ``$OLLAMA_HOST`` is scheme-less
    ``host:port`` (e.g. ``0.0.0.0:11434`` -- the exact form WSL / remote
    users already have exported for the ``ollama`` CLI). Passed to httpx
    verbatim, that raises ``UnsupportedProtocol``, which is a
    ``RequestError`` subclass -- so it would be misclassified as a
    retryable TRANSPORT error when the real problem is config. Prepending
    ``http://`` when no scheme is present accepts both conventions.
    Trailing slashes are stripped so path-joining with
    ``OLLAMA_CHAT_PATH`` can't produce ``//``. An empty / whitespace-only
    value (e.g. ``--ollama-host ""`` or ``OLLAMA_HOST="  "``) raises a
    typed ConfigError rather than degrading to the invalid URL ``http:``.
    """
    host = host.strip()
    if not host:
        raise ConfigError(
            "Ollama host is empty. Set --ollama-host or $OLLAMA_HOST to a "
            "URL like http://localhost:11434, or unset them to use the "
            f"default ({DEFAULT_OLLAMA_HOST})."
        )
    if "://" not in host:
        host = f"http://{host}"
    # A scheme without a hostname (``http://``, ``http://:11434``) would
    # otherwise surface later as an untyped URL error from httpx instead
    # of a CONFIG error the caller can act on.
    if urlparse(host).hostname is None:
        raise ConfigError(
            f"Ollama host {host!r} has no hostname. Set --ollama-host or "
            "$OLLAMA_HOST to a URL like http://localhost:11434 (or "
            "host:port -- the scheme is optional)."
        )
    return host.rstrip("/")


def _match_loaded_context(ps_data: object, model: str) -> int | None:
    """Extract the loaded ``model``'s ``context_length`` from ``/api/ps`` data.

    Pure so it's unit-testable without a server. Ollama normalizes
    untagged names to ``:latest`` when loading, so ``qwen3-coder-next``
    must match a loaded ``qwen3-coder-next:latest``. Returns ``None``
    when the model isn't in the loaded list or the server predates the
    ``context_length`` field.
    """
    if not isinstance(ps_data, dict):
        # A top-level list/string/None from a misbehaving proxy would
        # otherwise AttributeError; the caller's blanket except would
        # mask it, but don't rely on that.
        return None
    wanted = {model} if ":" in model else {model, f"{model}:latest"}
    models_list = ps_data.get("models") or []
    if not isinstance(models_list, list):
        # A non-list `models` (int/bool/string) would TypeError below;
        # the caller's blanket except would mask it, but be explicit.
        return None
    for entry in models_list:
        if not isinstance(entry, dict):
            continue
        if {entry.get("name"), entry.get("model")} & wanted:
            ctx = entry.get("context_length")
            if isinstance(ctx, int) and ctx > 0:
                return ctx
    return None


def _detect_ollama_num_ctx(host: str, model: str) -> int | None:
    """Best-effort read of the model's actual loaded context window.

    ``GET /api/ps`` lists models currently in memory, including the
    ``context_length`` each was loaded with -- the operative window for
    our request, since Ollama reuses the loaded instance. Returns
    ``None`` (never raises) when the model isn't loaded yet, the server
    is unreachable, or the field is absent (older Ollama): the caller
    falls back to the advisory default. In the primary iterative-review
    workflow the model is loaded from round 1's call onward, so every
    subsequent round gets an exact window instead of an estimate.
    """
    try:
        with _make_client(OLLAMA_PS_TIMEOUT) as client:
            response = client.get(host.rstrip("/") + "/api/ps")
        if response.status_code != 200:
            return None
        return _match_loaded_context(response.json(), model)
    except Exception:
        # Probe must never break the run; the real call surfaces any
        # genuine connectivity problem as a typed error.
        return None


def _ollama_prompt_guard(
    prompt_chars: int, num_ctx: int, *, model: str, enforced: bool = True
) -> None:
    """Pre-flight guard against Ollama's silent prompt truncation.

    Ollama silently truncates prompts that don't fit ``num_ctx`` and
    generates from the fragment that survived -- no error, exit 0, and a
    plausible-looking review of a fraction of the code. The runner
    requests the window per call via the native endpoint (see
    ``DEFAULT_OLLAMA_NUM_CTX`` for the tier policy); this check catches
    known-too-small windows before tokens are spent, and
    ``_ollama_post_verify`` backstops after the call.

    ``enforced=True`` (window known: $OLLAMA_NUM_CTX set, or read from
    ``/api/ps``; that value is what gets sent as num_ctx) -> likely
    overflow raises a typed ``ContextOverflow``. ``enforced=False``
    (window unknown; ``num_ctx`` here is only the smallest stock VRAM
    tier and is NOT sent -- the server keeps its own default) -> likely
    overflow only WARNs, because the actual window on a >=24 GiB machine
    is 32K/256K and a hard error would reject a valid run.
    """
    approx_tokens = prompt_chars // OLLAMA_CHARS_PER_TOKEN
    if approx_tokens < num_ctx:
        return
    if enforced:
        raise ContextOverflow(
            f"Prompt is ~{approx_tokens:,} tokens ({prompt_chars:,} chars) "
            f"but the Ollama context window is {num_ctx:,} tokens. "
            "Ollama silently truncates oversized prompts, which would "
            "produce a review of only a fragment of the code. Either "
            "narrow the scope (--include / --exclude / smaller --base), "
            "or raise $OLLAMA_NUM_CTX -- the runner requests the window "
            "per call, no server restart needed, but the KV cache scales "
            "with it, so stay within your RAM/VRAM.",
            model=model,
            provider="ollama",
        )
    sys.stderr.write(
        f"WARN: prompt is ~{approx_tokens:,} tokens but the Ollama context "
        f"window is unknown (model `{model}` not loaded yet; "
        "$OLLAMA_NUM_CTX unset), so no num_ctx is requested and the "
        "server's VRAM-tier default (4K/32K/256K) applies. If the prompt "
        "exceeds it, Ollama truncates silently -- the runner verifies "
        "prompt_eval_count after the call and hard-fails if truncation "
        "is likely. Set $OLLAMA_NUM_CTX to your actual window for a hard "
        "pre-flight check instead.\n"
    )


def _ollama_post_verify(
    prompt_eval_count: int | None,
    num_ctx_sent: int | None,
    *,
    host: str,
    model: str,
) -> None:
    """Post-call truncation backstop using the native API's usage counts.

    ``prompt_eval_count`` at or above ``OLLAMA_POST_VERIFY_MARGIN`` of
    the effective window means the prompt filled it -- with our review
    prompts that means server-side truncation, and the output must be
    discarded (hard ContextOverflow, never a warn: an exit-0 partial
    review poisons agent loops). The effective window is the num_ctx we
    sent; when none was sent (unknown tier), the model is loaded by now,
    so ``/api/ps`` is re-probed for its actual window. If that still
    fails, verification is skipped with a WARN rather than guessed.

    Known limitation (do not "fix" by lowering the margin): KV-cache
    reuse makes ``prompt_eval_count`` an UNDERCOUNT on repeated
    identical prefixes, so this check can false-negative but not
    false-positive.
    """
    if prompt_eval_count is None:
        return  # pre-usage-fields Ollama; nothing to verify against
    window = num_ctx_sent
    source = "requested"
    if window is None:
        window = _detect_ollama_num_ctx(host, model)
        source = "detected post-call"
        if window is None:
            sys.stderr.write(
                "WARN: could not verify the prompt fit the Ollama context "
                "window (window unknown even after the call); if findings "
                "look sparse, set $OLLAMA_NUM_CTX and re-run.\n"
            )
            return
    if prompt_eval_count >= int(window * OLLAMA_POST_VERIFY_MARGIN):
        raise ContextOverflow(
            f"prompt_eval_count={prompt_eval_count:,} is at the context "
            f"window ({window:,} tokens, {source}) -- the prompt was "
            "almost certainly truncated server-side and the review covers "
            "only a fragment, so the output was discarded. Narrow the "
            "scope (--include / --exclude / smaller --base) or raise "
            "$OLLAMA_NUM_CTX (RAM permitting).",
            model=model,
            provider="ollama",
        )


def call_ollama(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    host: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    num_ctx: int | None = None,
) -> CallResult:
    """POST to a local Ollama server via its NATIVE ``/api/chat`` endpoint
    and return a ``CallResult``. No API key required (local). Caller builds
    the prompts (same as ``call_openrouter`` and ``call_gemini``) so the
    wire path is mode-agnostic. Raises typed ``ReviewError`` subclasses;
    see README "Error model".

    ``num_ctx`` is the context window to request: the env-set or
    /api/ps-detected value, or ``None`` to omit it so the server keeps
    its own VRAM-tier default (see ``DEFAULT_OLLAMA_NUM_CTX`` -- never
    pass the advisory constant here; it would shrink a larger window).

    Differences from cloud providers:

    - **No auth header.** Local server, no token-bearer logic.
    - **First request after server start is slow** (~10-60s) because Ollama
      lazy-loads the model into RAM. Subsequent requests within the
      keep-alive window are fast. ``timeout`` is generously larger than
      ``HTTP_TIMEOUT`` for cloud providers to absorb this cold start.
    - **Connection refused** is a distinct failure mode (server not
      running) versus a 4xx/5xx (server up, problem with the request).
      We surface it as a ``ConfigError`` so the user gets actionable
      guidance instead of a generic transport error.
    - **404 means the model isn't pulled**, not "endpoint not found."
      The body usually contains the model name; we surface it as a
      ConfigError pointing at the right ``ollama pull`` command.
    - **No content filter / safety mode.** Ollama doesn't refuse output
      on safety grounds, so the empty-content classifier checks only
      length / generic hiccup -- no SafetyRefusal branch.
    - **Silent truncation is checked post-call**: the native response's
      ``prompt_eval_count`` feeds ``_ollama_post_verify``, which raises
      a hard ContextOverflow when the prompt filled the window.
    """
    options: dict = {
        "temperature": temperature,
        # Native name for the output-token ceiling (max_tokens
        # elsewhere). Always sent explicitly -- the server default
        # differs from the cloud providers'.
        "num_predict": max_tokens,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": options,
        # Force non-streaming so we get the whole response in one JSON
        # blob -- simpler to parse and matches how we handle the cloud
        # providers.
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
    }
    url = host.rstrip("/") + OLLAMA_CHAT_PATH

    # Split the timeout into a short connect window and a long read window.
    # A single ``timeout=1800`` applies to all phases of the request,
    # including the TCP connect -- which means an unreachable server would
    # have us wait 30 minutes for the connection to time out before we
    # raise ``ConnectError``. The fix is to keep the long timeout for
    # the *read* phase (where CPU-local inference legitimately takes
    # minutes) but cap the *connect* phase at 10 seconds so "server
    # not running" surfaces as a fast ConfigError instead of a 30-minute
    # hang. Write / pool fall back to the ``timeout`` default (the long
    # one) since those phases don't have the same "unreachable server"
    # failure mode.
    timeout_config = httpx.Timeout(timeout, connect=10.0)

    try:
        with _make_client(timeout_config) as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.ConnectError as exc:
        # Server unreachable. Most likely Ollama isn't running, or
        # OLLAMA_HOST points at the wrong place. ConfigError so the
        # caller (human or LLM agent) doesn't retry pointlessly -- they
        # need to fix config first.
        raise ConfigError(
            f"Cannot reach Ollama at {host}. Is the server running? "
            f"Start it with `ollama serve` (or `wsl -d Ubuntu -- ollama serve` "
            f"if Ollama lives in WSL). Override the URL with --ollama-host "
            f"or $OLLAMA_HOST if it's on a non-default port/machine.",
            detail=str(exc),
            model=model,
            provider="ollama",
        ) from exc
    except httpx.RequestError as exc:
        # Other network-layer failure (timeout reading response, DNS for
        # a non-localhost host, etc.). TransportError so the auto-retry
        # in ``_call_with_retries`` gives it one more try before
        # raising.
        raise TransportError(
            f"Ollama request to {host} failed: {exc}",
            model=model,
            provider="ollama",
        ) from exc

    # 404 from Ollama means the model isn't pulled. The response body
    # is JSON like {"error": "model 'x' not found, try pulling it
    # first"} -- the "try pulling" substring doubles as a fallback in
    # case a future Ollama version moves the status code. Surface as a
    # ConfigError with the exact pull command so the user can fix it in
    # one step.
    if response.status_code == 404 or (
        response.status_code >= 400 and "try pulling" in response.text.lower()
    ):
        raise ConfigError(
            f"Model `{model}` not available on Ollama server at {host}. "
            f"Run `ollama pull {model}` (or `wsl -d Ubuntu -- ollama pull "
            f"{model}` if Ollama is in WSL), then retry.",
            detail=response.text[:500],
            model=model,
            provider="ollama",
        )

    if response.status_code >= 400:
        raise _classify_http_error(
            response.status_code,
            response.text,
            model=model,
            provider="ollama",
            retry_after=response.headers.get("retry-after"),
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderHiccup(
            f"Ollama returned non-JSON response: {exc}",
            detail=response.text[:1000],
            model=model,
            provider="ollama",
        ) from exc
    if not isinstance(data, dict):
        raise ProviderHiccup(
            "Ollama returned non-object JSON",
            detail=response.text[:1000],
            model=model,
            provider="ollama",
        )

    # Native /api/chat shape: top-level message/done_reason plus
    # prompt_eval_count / eval_count usage fields.
    message = data.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if content is not None and not isinstance(content, str):
        content = None  # non-string content -> typed empty-content path
    done_reason = str(data.get("done_reason") or "").lower()
    prompt_eval = _usage_int(data.get("prompt_eval_count"))
    eval_count = _usage_int(data.get("eval_count"))

    # Truncation backstop runs even when content came back: a review of
    # a silently truncated prompt is worse than no review.
    _ollama_post_verify(prompt_eval, num_ctx, host=host, model=model)

    if content:
        truncated = done_reason == "length"
        _warn_if_truncated(truncated, max_tokens, "ollama")
        return CallResult(
            content,
            prompt_tokens=prompt_eval,
            completion_tokens=eval_count,
            truncated=truncated,
        )

    # Empty content. Ollama doesn't have content-filter / safety
    # refusals, so the only diagnosable empty-content cause is hitting
    # the output-token ceiling. Anything else falls through to a generic
    # ProviderHiccup which the caller can retry once.
    if done_reason == "length":
        raise ContextOverflow(
            f"Hit num_predict ({max_tokens}) before producing content "
            f"(done_reason={done_reason!r})",
            detail=str(data)[:1000],
            model=model,
            provider="ollama",
        )
    raise ProviderHiccup(
        f"Ollama returned empty content with done_reason={done_reason!r}",
        detail=str(data)[:1000],
        model=model,
        provider="ollama",
    )


def _call_with_retries(call: Callable[[], T], *, label: str, retries: int = 0) -> T:
    """Run ``call`` with the runner's retry policy.

    ``retries=0`` (the default) reproduces the historical behavior
    exactly: one automatic 2s retry on ``ProviderHiccup`` /
    ``TransportError``, nothing else. ``--retries N`` grants N
    *additional* attempts on top of that: hiccup/transport back off at
    ``min(2**attempt, 60)`` seconds (2, 4, 8, ...); ``RateLimit`` is
    retried only when N > 0, sleeping the provider's parsed
    ``retry_after_seconds`` (or 60s when the header was absent or
    unparseable), clamped to ``MAX_RETRY_SLEEP`` with a WARN so a
    hostile/buggy header can't stall an agent caller.

    Never retried, by design: ConfigError (fix config first),
    SafetyRefusal (same prompt reproduces it; switch model),
    ContextOverflow (the scope is wrong, not the call).

    ``label`` shows up in the retry-notice stderr lines so a viewer
    scrolling the log can tell which call retried.
    """
    transient_budget = 1 + retries
    ratelimit_budget = retries
    transient_used = 0
    ratelimit_used = 0
    while True:
        try:
            return call()
        except (ProviderHiccup, TransportError) as exc:
            transient_used += 1
            if transient_used > transient_budget:
                raise
            delay = min(2.0**transient_used, 60.0)
            sys.stderr.write(
                f"[retry] {exc.category} on attempt {transient_used} "
                f"({label}); retrying in {delay:.0f}s...\n"
            )
            time.sleep(delay)
        except RateLimit as exc:
            ratelimit_used += 1
            if ratelimit_used > ratelimit_budget:
                raise
            delay = (
                exc.retry_after_seconds if exc.retry_after_seconds is not None else 60.0
            )
            if delay > MAX_RETRY_SLEEP:
                sys.stderr.write(
                    f"WARN: Retry-After of {delay:.0f}s clamped to "
                    f"{MAX_RETRY_SLEEP:.0f}s; bail out and retry later if "
                    "the provider really needs that long.\n"
                )
                delay = MAX_RETRY_SLEEP
            sys.stderr.write(
                f"[retry] RATE_LIMIT on attempt {ratelimit_used} ({label}); "
                f"retrying in {delay:.0f}s...\n"
            )
            time.sleep(delay)


def _format_usage_line(result: CallResult, provider: str, model: str) -> str | None:
    """Render the ``[usage]`` stderr line, or ``None`` when the provider
    sent no usage data at all (never estimate)."""
    if result.prompt_tokens is None and result.completion_tokens is None:
        return None
    prompt = "?" if result.prompt_tokens is None else f"{result.prompt_tokens:,}"
    completion = (
        "?" if result.completion_tokens is None else f"{result.completion_tokens:,}"
    )
    if result.prompt_tokens is not None and result.completion_tokens is not None:
        total = f" total={result.prompt_tokens + result.completion_tokens:,}"
    else:
        total = ""
    return (
        f"[usage] prompt={prompt} completion={completion}{total} tokens "
        f"({provider}/{model})"
    )


def _resolve_ollama_window(
    host: str, model: str, env_num_ctx: int | None
) -> tuple[int, bool, str]:
    """Resolve the Ollama context window for one model, per call.

    Returns ``(num_ctx, enforced, source)`` where source is one of
    ``"env"`` / ``"detected"`` / ``"advisory-default"``. Runs per model
    per call (not once at settings time) because /api/ps only knows
    about *loaded* models -- in a multi-model panel, model k+1 isn't
    loaded until its own call starts. Emits the ``[ollama]`` stderr line
    on detection.
    """
    if env_num_ctx is not None:
        return env_num_ctx, True, "env"
    detected = _detect_ollama_num_ctx(host, model)
    if detected is not None:
        sys.stderr.write(
            f"[ollama] detected context window {detected:,} tokens "
            "for loaded model via /api/ps\n"
        )
        return detected, True, "detected"
    return DEFAULT_OLLAMA_NUM_CTX, False, "advisory-default"


# Per-project configuration file, discovered by upward walk from CWD.
# Lives in the REVIEWED repo (unlike .env, which configures the runner
# installation), so different projects can pin different models,
# excludes, or context strings.
