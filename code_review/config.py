"""Configuration resolution: env layering, project config, Settings.

Precedence: CLI > process env (with .env files merged in) > the
reviewed repo's .code-review.toml > built-in default. The project
config is UNTRUSTED (it ships with the code under review): loading is
announced, unknown keys are dropped with a WARN, and credentials /
prompt context / ollama_* endpoints are never read from it.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from code_review.errors import ConfigError
from code_review.prompts import _REPO_ROOT, DEFAULT_CONTEXT, SEVERITY_LEVELS
from code_review.providers import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_BY_PROVIDER,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    MODEL_ALIASES_BY_PROVIDER,
    PROVIDERS,
    _normalize_ollama_host,
)


def _user_config_dir() -> Path:
    """Per-user config directory for an installed ``code-review``.

    Windows: ``%APPDATA%\\code-review``. Elsewhere: ``$XDG_CONFIG_HOME/
    code-review`` falling back to ``~/.config/code-review``. Hand-rolled
    (three lines) rather than a platformdirs dependency.
    """
    if os.name == "nt":
        base = os.getenv("APPDATA")
        return (
            Path(base) if base else Path.home() / "AppData" / "Roaming"
        ) / "code-review"
    xdg = os.getenv("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "code-review"


def _load_env_files() -> None:
    """Load .env files without overriding real environment variables.

    Order (earlier wins among files; the process environment always
    wins over all of them because ``override=False``):
      1. ``$CODE_REVIEW_ENV`` -- explicit file; being set but missing is
         a ConfigError, because explicit config must not fail silently.
      2. The user config dir -- the natural home for an installed
         ``code-review`` (there is no repo checkout to put a .env in).
      3. The repo root next to this package -- preserves the documented
         clone workflow (configure once at the runner location, invoke
         from any project directory).
    """
    explicit = os.getenv("CODE_REVIEW_ENV")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ConfigError(
                f"$CODE_REVIEW_ENV={explicit!r} does not exist or is not a file."
            )
        load_dotenv(path, override=False)
    load_dotenv(_user_config_dir() / ".env", override=False)
    load_dotenv(_REPO_ROOT / ".env", override=False)


def _resolve_model_name(name: str, provider: str) -> str:
    """Resolve one model name or alias for ``provider``.

    Aliases are scoped per provider (see ``MODEL_ALIASES_BY_PROVIDER``):
    OpenRouter aliases like ``claude`` resolve only with --provider
    openrouter; Ollama aliases like ``local`` resolve only with
    --provider ollama. Using an alias from the wrong table raises a
    typed ``ConfigError`` pointing the caller at the correct
    ``--provider`` instead of silently sending an invalid slug to the
    upstream API. Panel mode maps this over every ``--models`` entry
    pre-flight, so alias mistakes exit 2 before any network call.
    """
    provider_aliases = MODEL_ALIASES_BY_PROVIDER.get(provider, {})
    if name in provider_aliases:
        return provider_aliases[name]

    # If the model name matches an alias for a DIFFERENT provider, the
    # user almost certainly meant to switch providers. Surface that with
    # a typed ConfigError naming the right --provider, so an LLM caller
    # parsing stderr can self-correct instead of hammering the wrong
    # endpoint.
    for other_provider, other_aliases in MODEL_ALIASES_BY_PROVIDER.items():
        if other_provider != provider and name in other_aliases:
            raise ConfigError(
                f"Model alias `{name}` is only valid with "
                f"--provider {other_provider} (currently --provider "
                f"{provider}). Either switch with "
                f"--provider {other_provider}, or pass an actual model "
                f"name supported by --provider {provider}."
            )

    return name


def _resolve_model(args: argparse.Namespace, project_config: dict | None = None) -> str:
    """Resolve the single-model slug: CLI flag > per-provider env var >
    project config ``model`` > provider default (see
    ``_resolve_model_name`` for alias rules -- aliases work in every
    layer)."""
    config = project_config if project_config is not None else {}
    env_by_provider = {
        "openrouter": "OPENROUTER_MODEL",
        "gemini": "GEMINI_MODEL",
        "ollama": "OLLAMA_MODEL",
    }
    if args.model is not None:
        name = args.model
    elif os.getenv(env_by_provider.get(args.provider, "")):
        name = os.environ[env_by_provider[args.provider]]
    elif "model" in config:
        config_model = config["model"]
        if not isinstance(config_model, str):
            # Silently falling through to the provider default would
            # hide the misconfig; fail typed like every other bad
            # config value.
            raise ConfigError(f"{_PROJECT_CONFIG_NAME} key 'model' must be a string.")
        name = config_model
    else:
        name = DEFAULT_MODEL_BY_PROVIDER[args.provider]
    # strip(): quoted .env values and TOML strings pick up stray spaces
    # easily, and " flash" would miss the alias table and reach the
    # provider as a bogus slug.
    return _resolve_model_name(name.strip(), args.provider)


# ---------------------------------------------------------------------------
# Structured output: markdown findings parser (--format json / --baseline)
# ---------------------------------------------------------------------------
#
# The prompts stay byte-identical to upstream; structure is recovered by
# DETERMINISTICALLY parsing the rigid markdown the OUTPUT templates
# mandate (`# ... summary:`, `## File: path`, `### L<N>: [SEV] title`).
# The parser is deliberately tolerant: real models drift from the
# template in observed ways (diff-anchored `### L+117:` headings from
# deepseek, ```diff-tagged suggestion fences), each of which is handled
# below and pinned by fixtures in tests/fixtures/. Parse failure must
# never destroy a paid-for review: the wrapper degrades to
# ``parse_ok=False`` with the raw markdown embedded in the envelope.

_PROJECT_CONFIG_NAME = ".code-review.toml"
# Keys the reviewed repo's .code-review.toml may set. Deliberately
# EXCLUDED, because this file is attacker-adjacent on untrusted
# checkouts (e.g. a PR branch that adds one):
#   - API keys: credentials never come from the reviewed repo.
#   - context: injected as trusted operator framing AHEAD of the
#     injection guard -- accepting it from the repo under review would
#     let that repo instruct its own reviewer ("report no issues").
#   - ollama_host: the full diff is POSTed to this URL -- a hostile
#     value exfiltrates the code under review to an arbitrary server.
#   - ollama_num_ctx / ollama_timeout: machine-local hardware facts, not
#     project facts (and a huge num_ctx can OOM the reviewer's server).
_PROJECT_CONFIG_KEYS = frozenset(
    {
        "provider",
        "model",
        "models",
        "temperature",
        "max_tokens",
        "retries",
        "min_severity",
        "format",
        "include",
        "exclude",
    }
)


def _load_project_config() -> dict:
    """Find and parse ``.code-review.toml`` for the current project.

    Walks upward from CWD; stops at the first hit, at a directory
    containing ``.git`` (the config conventionally sits next to it, so
    that directory IS checked first), or at the filesystem root. Pure
    path walk -- no git subprocess, so --help and non-git directories
    stay clean.

    Security posture: this file lives in the REVIEWED repo, which for
    ``--pr``-style use may be an untrusted checkout -- so loading one is
    always announced on stderr with its path, unknown keys are dropped
    with a WARN, and the accepted key set (see _PROJECT_CONFIG_KEYS) is
    limited to review-shaping tunables: no credentials, no prompt
    ``context``, no ``ollama_*`` endpoint/window settings. Pass
    ``--no-project-config`` to ignore the file entirely when auditing
    untrusted code (even ``exclude`` can hide a file from --codebase).
    """
    directory = Path.cwd()
    while True:
        candidate = directory / _PROJECT_CONFIG_NAME
        if candidate.is_file():
            try:
                # utf-8-sig: Windows editors (Notepad, PowerShell
                # Set-Content) write a BOM, which tomllib rejects;
                # -sig strips it and is a no-op otherwise.
                config = tomllib.loads(candidate.read_text(encoding="utf-8-sig"))
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ConfigError(f"Cannot parse {candidate}: {exc}") from exc
            unknown = sorted(set(config) - _PROJECT_CONFIG_KEYS)
            if unknown:
                sys.stderr.write(
                    f"WARN: {candidate} has unrecognized keys (ignored): "
                    f"{', '.join(unknown)}\n"
                )
                config = {k: v for k, v in config.items() if k in _PROJECT_CONFIG_KEYS}
            sys.stderr.write(f"[config] loaded {candidate}\n")
            return config
        if (directory / ".git").exists() or directory.parent == directory:
            return {}
        directory = directory.parent


def _layered(
    cli_value: object,
    env_name: str | None,
    toml_key: str,
    project_config: dict,
) -> tuple[Any, str]:
    """One lookup through the precedence layers: CLI > env > project
    config. Returns ``(value, source)``; ``(None, "default")`` when no
    layer provided a value (the caller applies its built-in default).
    ``source`` feeds error messages so a bad value says where it came
    from. Empty-string env values read as unset (matching the
    long-standing $CODE_REVIEW_CONTEXT semantics).
    """
    if cli_value is not None:
        return cli_value, "cli"
    if env_name:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value, f"${env_name}"
    if toml_key in project_config:
        return project_config[toml_key], _PROJECT_CONFIG_NAME
    return None, "default"


def _apply_config_file_lists(args: argparse.Namespace, config: dict) -> None:
    """Adopt ``include``/``exclude`` from project config when the CLI
    passed none (CLI globs always win outright -- list merging would be
    surprising)."""
    for key in ("include", "exclude"):
        if getattr(args, key):
            continue
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ConfigError(
                f"{_PROJECT_CONFIG_NAME} key {key!r} must be a list of strings."
            )
        setattr(args, key, list(value))


@dataclasses.dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration (CLI > env > default).

    Frozen: everything here is decided once, before any git or network
    activity. ``ollama_num_ctx_env`` carries only the explicit
    ``$OLLAMA_NUM_CTX`` value -- the /api/ps-detected window is
    deliberately NOT a setting; it resolves per model per call in
    ``_execute_call`` (see ``_resolve_ollama_window``).
    """

    provider: str
    model: str
    temperature: float
    max_tokens: int
    retries: int
    min_severity: str
    context: str | None
    output: str | None
    format: str = "markdown"
    baseline: str | None = None
    models: tuple[str, ...] | None = None  # panel mode (--models)
    api_key: str | None = None
    referer: str | None = None
    title: str | None = None
    ollama_host: str | None = None
    ollama_timeout: float | None = None
    ollama_num_ctx_env: int | None = None


@dataclasses.dataclass
class ReviewRequest:
    """A fully built review payload plus the metadata ``--dry-run`` prints."""

    system_prompt: str
    user_prompt: str
    mode: str  # "diff" | "codebase"
    payload_chars: int  # diff length or bundle length (pre-prompt-wrapping)
    files: list[Path] | None = None  # codebase mode only
    chunk_label: str | None = None  # --chunk mode: e.g. "3 file(s), 41,209 chars"
    # Raw diff (diff mode only), retained so findings can be annotated with
    # `in_hunk` after parsing -- the parser sees only the model's markdown,
    # and hunk membership needs the diff. None in codebase mode.
    diff: str | None = None


def _resolve_settings(
    args: argparse.Namespace, project_config: dict | None = None
) -> Settings:
    """Resolve and validate all runtime configuration before any git or
    network activity, so misconfig fails fast as a typed CONFIG error.

    Precedence: **CLI > process env (with .env files already merged in)
    > project ``.code-review.toml`` > built-in default**, implemented
    per-tunable via ``_layered`` so error messages can name the layer a
    bad value came from. API keys are deliberately NOT layered -- they
    come from the environment only, never from project config.
    """
    config = project_config if project_config is not None else {}

    # Provider first: argparse ``choices`` validates only user-typed
    # flags, so env/config-sourced values are checked here. The resolved
    # value is written back onto ``args`` because everything downstream
    # (model resolution, provider dispatch) reads ``args.provider``.
    provider_raw, provider_source = _layered(
        args.provider, "CODE_REVIEW_PROVIDER", "provider", config
    )
    # strip(): quoted .env values pick up stray whitespace easily, and
    # " ollama" failing validation over an invisible space is hostile.
    # Same treatment for the other validated string tunables below.
    if isinstance(provider_raw, str):
        provider_raw = provider_raw.strip()
    provider = provider_raw if provider_raw is not None else DEFAULT_PROVIDER
    if provider not in PROVIDERS:
        raise ConfigError(
            f"provider {provider!r} (from {provider_source}) is not valid "
            "(check $CODE_REVIEW_PROVIDER / .code-review.toml). Use one "
            "of: " + ", ".join(PROVIDERS) + "."
        )
    args.provider = provider

    # Panel model list: resolved pre-flight so alias errors exit 2
    # before any network call. --models and --model on the CLI together
    # are genuinely ambiguous (manual check, same pattern as --context /
    # --no-context) -- but a CLI --model OVERRIDES a project-config
    # panel outright, per the documented CLI > project-config
    # precedence: the config list only activates when the CLI didn't
    # pick a single model.
    models: tuple[str, ...] | None = None
    names: list[str] | None = None
    if args.models is not None:
        if args.model is not None:
            raise ConfigError(
                "--models and --model are mutually exclusive. Use "
                "--models for a panel, --model for a single reviewer."
            )
        names = [n.strip() for n in args.models.split(",") if n.strip()]
    elif "models" in config and args.model is None:
        config_models = config["models"]
        if not isinstance(config_models, list) or not all(
            isinstance(m, str) for m in config_models
        ):
            raise ConfigError(
                f"{_PROJECT_CONFIG_NAME} key 'models' must be a list of strings."
            )
        names = [n.strip() for n in config_models if n.strip()]
    if names is not None:
        if args.chunk:
            # _validate_flag_combos already rejects CLI --chunk+--models,
            # but it runs before project config loads -- re-check here so
            # a config-sourced panel can't slip into a chunked run (which
            # would silently review with only the first model).
            raise ConfigError(
                "--chunk and a models panel are not supported together "
                "(the panel here comes from .code-review.toml). Chunk "
                "with a single --model, or panel an unchunked payload."
            )
        if len(names) < 2:
            raise ConfigError(
                "a panel needs at least two model entries "
                "(use --model for a single reviewer)."
            )
        resolved = [_resolve_model_name(n, provider) for n in names]
        dupes = {m for m in resolved if resolved.count(m) > 1}
        if dupes:
            raise ConfigError(
                f"panel models resolve to duplicate entries: {sorted(dupes)}. "
                "Aliases and slugs for the same model count as one."
            )
        models = tuple(resolved)
        if args.baseline is not None:
            raise ConfigError(
                "--baseline is not supported with --models yet; run the "
                "panel with --format json and diff rounds externally, or "
                "baseline a single-model run."
            )

    model = models[0] if models else _resolve_model(args, config)

    # Temperature.
    temp_raw, temp_source = _layered(
        args.temperature, "CODE_REVIEW_TEMPERATURE", "temperature", config
    )
    try:
        temperature = float(temp_raw) if temp_raw is not None else DEFAULT_TEMPERATURE
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"temperature {temp_raw!r} (from {temp_source}) is not a valid float."
        ) from exc
    # Validate range here rather than letting the provider 4xx -- catches
    # the misconfig as a typed CONFIG error (exit 2) the LLM caller can
    # react to, instead of an opaque provider UNKNOWN. ``2.0`` is the
    # common ceiling across OpenAI / Anthropic / Gemini; providers that
    # accept higher will simply not see it, which is fine.
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError(
            f"temperature={temperature} (from {temp_source}) is out of "
            "range [0.0, 2.0]."
        )

    # Max output tokens.
    max_raw, max_source = _layered(
        args.max_tokens, "CODE_REVIEW_MAX_TOKENS", "max_tokens", config
    )
    try:
        max_tokens = int(max_raw) if max_raw is not None else DEFAULT_MAX_TOKENS
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"max_tokens {max_raw!r} (from {max_source}) is not a valid integer."
        ) from exc
    if max_tokens <= 0:
        raise ConfigError(
            f"max_tokens={max_tokens} (from {max_source}) must be positive."
        )

    # Retry budget.
    retries_raw, retries_source = _layered(
        args.retries, "CODE_REVIEW_RETRIES", "retries", config
    )
    try:
        retries = int(retries_raw) if retries_raw is not None else 0
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"retries {retries_raw!r} (from {retries_source}) is not a valid integer."
        ) from exc
    if retries < 0:
        raise ConfigError(f"retries={retries} (from {retries_source}) must be >= 0.")

    # Severity floor.
    severity_raw, severity_source = _layered(
        args.min_severity, "CODE_REVIEW_MIN_SEVERITY", "min_severity", config
    )
    min_severity = (
        str(severity_raw).strip().upper() if severity_raw is not None else "LOW"
    )
    if min_severity not in SEVERITY_LEVELS:
        raise ConfigError(
            f"min_severity {severity_raw!r} (from {severity_source}) is "
            "not valid. Use one of: " + ", ".join(SEVERITY_LEVELS) + "."
        )

    # Output format.
    format_raw, format_source = _layered(
        args.format, "CODE_REVIEW_FORMAT", "format", config
    )
    if isinstance(format_raw, str):
        format_raw = format_raw.strip()
    output_format = format_raw if format_raw is not None else "markdown"
    if output_format not in ("markdown", "json"):
        raise ConfigError(
            f"format {output_format!r} (from {format_source}) is not "
            "valid. Use 'markdown' or 'json'."
        )

    # Safety context: --no-context wins, then explicit --context, then
    # env, then default. Empty string from env is treated as "use
    # default" rather than "disabled" -- pass --no-context explicitly to
    # disable, since an env value of "" is more likely a misconfig than
    # intent. Project config deliberately CANNOT set context: the block
    # is injected as trusted operator framing ahead of the injection
    # guard, and the config file lives in the (possibly untrusted)
    # reviewed repo -- accepting it would let that repo instruct its
    # own reviewer.
    if args.no_context:
        context: str | None = None
    elif args.context is not None:
        context = args.context
    else:
        context = os.getenv("CODE_REVIEW_CONTEXT") or DEFAULT_CONTEXT

    api_key: str | None = None
    referer: str | None = None
    title: str | None = None
    ollama_host: str | None = None
    ollama_timeout: float | None = None
    ollama_num_ctx_env: int | None = None

    if args.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigError(
                "OPENROUTER_API_KEY not set. Put it in a .env at "
                f"{_user_config_dir()} (installed) or {_REPO_ROOT} "
                "(checkout; see .env.example), or rerun with "
                "--provider gemini (Google AI Studio key) or "
                "--provider ollama (local server, no key needed)."
            )
        referer = os.getenv(
            "OPENROUTER_HTTP_REFERER",
            "https://github.com/Airwhale/local-gemini-code-review",
        )
        title = os.getenv("OPENROUTER_X_TITLE", "OpenRouter Code Review")
    elif args.provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigError(
                "GEMINI_API_KEY not set. Put your Google AI Studio key "
                f"in a .env at {_user_config_dir()} (installed) or "
                f"{_REPO_ROOT} (checkout; see .env.example), or rerun "
                "with --provider openrouter (OpenRouter key) or "
                "--provider ollama (local server, no key needed)."
            )
    else:  # ollama
        # No API key for local provider. A window pinned by CLI or env
        # is user-specified and therefore enforced, exactly like
        # $OLLAMA_NUM_CTX always was; the /api/ps-detected window is NOT
        # a config layer -- it resolves per model per call in
        # _execute_call. All ollama_* settings are machine-local (where
        # YOUR server is, what fits YOUR RAM) and security-sensitive (a
        # hostile ollama_host would receive the full diff), so they are
        # never read from the reviewed repo's project config -- the
        # empty dict below keeps that layer out of the lookup.
        host_raw, host_source = _layered(
            args.ollama_host, "OLLAMA_HOST", "ollama_host", {}
        )
        if host_raw is not None and not isinstance(host_raw, str):
            raise ConfigError(
                f"ollama_host {host_raw!r} (from {host_source}) must be a string."
            )
        ollama_host = _normalize_ollama_host(host_raw or DEFAULT_OLLAMA_HOST)
        num_ctx_raw, num_ctx_source = _layered(
            None, "OLLAMA_NUM_CTX", "ollama_num_ctx", {}
        )
        if num_ctx_raw is not None:
            try:
                ollama_num_ctx_env = int(num_ctx_raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"ollama_num_ctx {num_ctx_raw!r} (from {num_ctx_source}) "
                    "is not a valid integer (tokens). Fix it, or remove it "
                    "to let the runner detect the window from a loaded "
                    "model."
                ) from exc
            if ollama_num_ctx_env <= 0:
                raise ConfigError(
                    f"ollama_num_ctx={ollama_num_ctx_env} (from "
                    f"{num_ctx_source}) must be positive (tokens)."
                )
        timeout_raw, timeout_source = _layered(
            None, "OLLAMA_TIMEOUT", "ollama_timeout", {}
        )
        try:
            ollama_timeout = (
                float(timeout_raw)
                if timeout_raw is not None
                else DEFAULT_OLLAMA_TIMEOUT
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"ollama_timeout {timeout_raw!r} (from {timeout_source}) is "
                "not a valid float (seconds)."
            ) from exc
        if ollama_timeout <= 0:
            raise ConfigError(
                f"ollama_timeout={ollama_timeout} (from {timeout_source}) "
                "must be positive (seconds)."
            )

    return Settings(
        provider=args.provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        min_severity=min_severity,
        context=context,
        output=args.output,
        format=output_format,
        baseline=args.baseline,
        models=models,
        api_key=api_key,
        referer=referer,
        title=title,
        ollama_host=ollama_host,
        ollama_timeout=ollama_timeout,
        ollama_num_ctx_env=ollama_num_ctx_env,
    )
