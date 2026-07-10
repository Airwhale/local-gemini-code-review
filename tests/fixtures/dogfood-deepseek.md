# Change summary: This change adds a context-window guard for Ollama to prevent silent prompt truncation, introduces a CI workflow, updates documentation, and adds unit tests.

## File: review.py
### L+117: [MEDIUM] The `_classify_http_error` function's docstring incorrectly states that 5xx errors are checked before substring matching, but the code does the opposite.

The docstring says "Order matters: the 5xx check runs BEFORE the substring match so a provider-side failure page that happens to contain a phrase like 'too long' or 'token limit' stays a retryable TRANSPORT error instead of being misclassified as a do-not-retry CONTEXT_OVERFLOW." However, the actual code checks for 429, then 5xx, then the substring matches for context overflow. This means a 5xx error with a context overflow phrase would be correctly classified as TRANSPORT, but the docstring is misleading about the order relative to substring matching.

Suggested change:
```diff
     """
     body_lower = body.lower()
     if status == 429:
         # Retry-After is either delta-seconds ("30") or an HTTP-date
         # ("Fri, 31 Dec 1999 23:59:59 GMT") -- only append the seconds
         # unit when the value is numeric.
         wait_hint = ""
         if retry_after:
             retry_after = retry_after.strip()
             suffix = "s" if retry_after.isdigit() else ""
             wait_hint = f" (Retry-After: {retry_after}{suffix})"
         return RateLimit(
             f"{provider} returned HTTP 429{wait_hint}",
             detail=body[:1000],
             model=model,
             provider=provider,
         )
-    if 500 <= status < 600:
-        return TransportError(
-            f"{provider} returned HTTP {status} (provider-side failure)",
-            detail=body[:1000],
-            model=model,
-            provider=provider,
-        )
     # 4xx only from here down. Match the family of provider phrases that
     # signal context-length overflow, kept as a tuple so adding a new
     # variant is a one-line edit. Each phrase was added based on an
     # actual provider response observed in the wild or in published
     # vendor docs; do not add speculative phrases without verifying a
     # provider uses them, or you risk false-positive ContextOverflow
     # classifications on unrelated 4xx errors.
     CONTEXT_OVERFLOW_PHRASES = (
         "context_length",
         "too long",
         "exceeds the maximum",
         "token limit",
     )
     if any(phrase in body_lower for phrase in CONTEXT_OVERFLOW_PHRASES):
         return ContextOverflow(
             f"{provider} returned HTTP {status} with context-length indication",
             detail=body[:1000],
             model=model,
             provider=provider,
         )
+    if 500 <= status < 600:
+        return TransportError(
+            f"{provider} returned HTTP {status} (provider-side failure)",
+            detail=body[:1000],
+            model=model,
+            provider=provider,
+        )
     return ReviewError(
         f"{provider} returned HTTP {status}",
         detail=body[:1000],
         model=model,
         provider=provider,
     )
```

### L+229: [MEDIUM] The `_warn_if_truncated` function's warning message could be clearer about the impact.

The warning message says "the review below may be incomplete." This is accurate but could be more specific about the risk of missing findings. Also, the warning suggests re-running with a higher `--max-tokens`, but doesn't mention that the user should also check if the model is hitting the ceiling due to verbose reasoning.

Suggested change:
```diff
     if hit_token_ceiling:
         sys.stderr.write(
             f"WARN: {provider} output was truncated at max_tokens="
             f"{max_tokens}; the review below may be incomplete. "
-            "Re-run with a higher --max-tokens for full output.\n"
+            "Findings may be cut off mid-sentence. Re-run with a higher "
+            "--max-tokens for full output, or consider using a model with "
+            "a larger context window if this happens frequently.\n"
         )
```

### L+268: [MEDIUM] The `_normalize_ollama_host` function does not validate the URL scheme beyond prepending 'http://'.

The function prepends 'http://' if no scheme is present, but does not validate that the resulting URL is well-formed (e.g., contains a hostname). While an empty/whitespace host is caught, a malformed host like 'http://' or 'http://:11434' would likely cause a connection error later, which is less helpful than an early config error.

Suggested change:
```diff
     if "://" not in host:
         host = f"http://{host}"
-    return host.rstrip("/")
+    # Basic validation: ensure the URL has a netloc (hostname).
+    # This catches malformed inputs like 'http://' or 'http://:11434'.
+    from urllib.parse import urlparse
+    parsed = urlparse(host)
+    if not parsed.netloc:
+        raise ConfigError(
+            f"Invalid Ollama host URL: {host!r}. Expected format: "
+            "http://host:port or host:port."
+        )
+    return host.rstrip("/")
```

### L+303: [MEDIUM] The `_match_loaded_context` function's docstring is missing from the diff, but the function is referenced; ensure it's properly documented.

The function `_match_loaded_context` is defined but its docstring is not included in the diff. Since this is a new function, it should have a clear docstring explaining its purpose and return values. Check that the actual implementation includes a docstring.

### L+346: [MEDIUM] The `_ollama_prompt_guard` function uses integer division for token estimation, which may underestimate for very short prompts.

The calculation `approx_tokens = prompt_chars // OLLAMA_CHARS_PER_TOKEN` uses floor division. For a prompt of 1-3 characters, this yields 0 tokens, which could incorrectly pass the guard even if the window is tiny. While this edge case is unlikely in practice (prompts are large), it's a potential inaccuracy.

Suggested change:
```diff
-    approx_tokens = prompt_chars // OLLAMA_CHARS_PER_TOKEN
+    approx_tokens = (prompt_chars + OLLAMA_CHARS_PER_TOKEN - 1) // OLLAMA_CHARS_PER_TOKEN  # ceil division
```

### L+414: [MEDIUM] The `call_ollama` function's error handling for `httpx.RequestError` could provide more specific guidance for scheme-less host errors.

The `except httpx.RequestError as exc:` block catches all request errors, including invalid URLs due to missing scheme. However, the error message is generic. Since `_normalize_ollama_host` now validates the URL, this may be less critical, but the error could still benefit from a hint about the `OLLAMA_HOST` format.

Suggested change:
```diff
     except httpx.RequestError as exc:
         # Invalid host URL, unreachable server, or a lower-level network
         # fault. Raise a typed ConfigError (exit 2) rather than a
         # retryable TransportError, because a bad host / unreachable
         # server won't heal by retrying.
+        # If the error is about an unsupported protocol, it might be due
+        # to a scheme-less host (e.g., 'localhost:11434' without 'http://').
+        hint = ""
+        if "unsupported protocol" in str(exc).lower():
+            hint = " (Did you forget 'http://' prefix? The runner adds it automatically.)"
         raise ConfigError(
-            f"Ollama server unreachable at {host!r}: {exc}",
+            f"Ollama server unreachable at {host!r}: {exc}{hint}",
             detail="Check that `ollama serve` is running and that "
             "--ollama-host / $OLLAMA_HOST points to the correct URL.",
             model=model,
             provider="ollama",
         )
```

### L+480: [MEDIUM] The `main` function's argument parser description for `--provider` includes outdated information about `OLLAMA_NUM_CTX`.

The help text mentions `$OLLAMA_NUM_CTX` as part of the Ollama configuration, but the variable is actually `OLLAMA_NUM_CTX` (without the `$` in the help text). The `$` is used to denote environment variables in prose, but it's inconsistent with other env var references in the same description.

Suggested change:
```diff
             "``OPENROUTER_API_KEY``. ``gemini`` calls Google AI Studio's "
             "generateContent endpoint directly and needs ``GEMINI_API_KEY``. "
             "``ollama`` posts to a local Ollama server's OpenAI-compatible "
             "endpoint (no API key; configure with ``--ollama-host`` / "
-            "$OLLAMA_HOST / $OLLAMA_MODEL / $OLLAMA_TIMEOUT / "
-            "$OLLAMA_NUM_CTX). Override with $CODE_REVIEW_PROVIDER."
+            "$OLLAMA_HOST / $OLLAMA_MODEL / $OLLAMA_TIMEOUT / "
+            "$OLLAMA_NUM_CTX). Override with $CODE_REVIEW_PROVIDER."
```

### L+555: [MEDIUM] The context-window resolution logic in `main` does not handle the case where `OLLAMA_NUM_CTX` is set to a string that cannot be converted to int.

The code uses `int(env_num_ctx)` inside a try/except block, which catches `ValueError`. However, if the conversion fails, the error message could be more user-friendly by showing the invalid value.

Suggested change:
```diff
             try:
                 ollama_num_ctx = int(env_num_ctx)
             except ValueError as exc:
                 raise ConfigError(
                     f"$OLLAMA_NUM_CTX={env_num_ctx!r} is not a valid integer "
                     "(tokens). Set it to the context window your Ollama "
                     "server loads models with, or unset it to let the "
                     "runner detect the window from a loaded model."
                 ) from exc
```

### L+572: [MEDIUM] The warning message when `ollama_num_ctx_enforced` is False is printed to stderr but may be interleaved with other output.

The warning is written via `sys.stderr.write` without a newline flush, which is fine, but in a concurrent environment (e.g., if the runner is integrated into a pipeline), the warning could appear out of order. Consider using `print(..., file=sys.stderr)` for atomic line writing.

Suggested change:
```diff
             else:
                 ollama_num_ctx = DEFAULT_OLLAMA_NUM_CTX
                 ollama_num_ctx_enforced = False
+                print(
+                    f"[ollama] context window unknown, assuming {ollama_num_ctx:,} tokens "
+                    "(set $OLLAMA_NUM_CTX to your actual window for a hard check)",
+                    file=sys.stderr,
+                )
```

### L+614: [MEDIUM] The `_ollama_prompt_guard` call in `main` uses `len(system_prompt) + len(user_prompt)` for character count, which may slightly overestimate due to JSON encoding overhead.

The prompt sent to Ollama is JSON-encoded, which adds extra characters (quotes, escapes). The guard uses a rough estimate of 4 chars per token, so the overestimation is likely within margin of error, but it's worth noting in a comment.

Suggested change:
```diff
         # Pre-flight context-window guard: refuse (typed CONTEXT_OVERFLOW)
         # rather than let Ollama silently truncate the prompt and review
         # a fragment; warn-only when the window couldn't be determined.
         # Cloud providers don't need this -- they 4xx on oversized
         # prompts instead of truncating.
+        # Note: prompt_chars is an underestimate (JSON encoding adds overhead),
+        # but the 4-char-per-token heuristic is conservative.
         _ollama_prompt_guard(
             len(system_prompt) + len(user_prompt),
             ollama_num_ctx,
             model=model,
             enforced=ollama_num_ctx_enforced,
         )
```

### L+640: [MEDIUM] The `_entrypoint` function's catch-all exception handler prints a traceback in the detail, which could be large and may contain sensitive paths.

While the traceback is useful for debugging, it may include local file paths. Since the error is printed to stderr, this is acceptable for a local tool, but consider truncating the traceback or sanitizing paths if privacy is a concern.

Suggested change:
```diff
         # Honor the README's stderr contract (``ERROR: UNKNOWN [exit 1]``)
         # even for unexpected bugs, so an LLM caller can classify the
         # failure without parsing a raw traceback. The traceback still
         # ships in the Detail line for humans debugging the runner.
+        # The traceback may contain local file paths; users should review
+        # before sharing logs.
         err = ReviewError(
             f"unhandled {type(exc).__name__}: {exc}",
             detail=traceback.format_exc(),
         )
```

## File: tests/test_review.py
### L+44: [MEDIUM] The test `test_5xx_with_overflow_phrase_stays_transport` may not be testing the intended order of checks.

The test passes a 500 status with a body containing "token limit". According to the current code (5xx check before substring match), this should be classified as `TransportError`. However, the test comment says "A provider-side failure page mentioning 'token limit' must NOT be misclassified as a do-not-retry CONTEXT_OVERFLOW." This is correct, but the test should also verify that the 5xx check indeed runs before substring matching. The test suite should include a test that a 400 status with "token limit" is `ContextOverflow` and a 500 status with "token limit" is `TransportError`.

### L+70: [MEDIUM] The test `test_429_retry_after_http_date_gets_no_seconds_suffix` uses a hardcoded date that may be confusing.

The date "Wed, 21 Oct 2015 07:28:00 GMT" is arbitrary. While it's fine for testing, a comment explaining why this date format is important would be helpful.

Suggested change:
```diff
     def test_429_retry_after_http_date_gets_no_seconds_suffix(self):
         # Retry-After may be an HTTP-date; appending "s" would mangle it.
+        # Example from RFC 9110.
         err = self._classify(
             429, "slow down", retry_after="Wed, 21 Oct 2015 07:28:00 GMT"
         )
```

### L+130: [MEDIUM] The test `test_provider_defaults` uses `monkeypatch.delenv` for multiple env vars in a loop, which may have side effects.

The loop `for var in ("OPENROUTER_MODEL", "GEMINI_MODEL", "OLLAMA_MODEL"): monkeypatch.delenv(var, raising=False)` runs before each assertion, but because the assertions are in the same test, the env vars are deleted for all providers. This is fine because each assertion calls `_resolve_model` with a different provider, but it's slightly inefficient. Consider using a fixture to clear env vars once.

### L+192: [MEDIUM] The test `test_unenforced_oversize_warns_instead_of_raising` does not verify the warning content beyond a prefix.

The test checks that the warning starts with "WARN:" and contains "OLLAMA_NUM_CTX". It would be stronger to also check that the model name appears in the warning.

Suggested change:
```diff
         err = capsys.readouterr().err
         assert err.startswith("WARN:")
         assert "OLLAMA_NUM_CTX" in err
+        assert "m" in err  # model name
```

### L+231: [MEDIUM] The test `TestGlobMatch.test_top_level_dist_excluded` relies on `BUILTIN_CODEBASE_EXCLUDES` containing "dist/*". Ensure the constant includes this pattern.

The test passes, but it's important that the constant `BUILTIN_CODEBASE_EXCLUDES` indeed contains "dist/*" (it does, as seen in the diff). However, the test could be more explicit by checking the constant directly.

### L+309: [MEDIUM] The test `TestErrorModelContract.test_exit_codes_match_readme_table` hardcodes exit codes that could drift from the actual class attributes.

The test asserts specific exit code values (e.g., `assert ConfigError.exit_code == 2`). This is good for ensuring the documented contract, but if the exit codes change, the test will fail. This is by design, but consider adding a comment linking to the README section.

## File: .env.example
### L+38: [LOW] The comment about `OLLAMA_HOST` mentions scheme-less values are accepted, but the example uses `http://localhost:11434`.

The comment says "Scheme-less `host:port` values (Ollama's own convention, e.g. `0.0.0.0:11434`) are accepted -- the runner prepends `http://`." However, the example line is `# OLLAMA_HOST=http://localhost:11434`. Consider adding a second example line showing the scheme-less version.

Suggested change:
```diff
 # from `wsl hostname -I` in that case). Scheme-less `host:port` values
 # (Ollama's own convention, e.g. `0.0.0.0:11434`) are accepted -- the
 # runner prepends `http://`.
-# OLLAMA_HOST=http://localhost:11434
+# OLLAMA_HOST=http://localhost:11434   # or just `localhost:11434`
```

### L+62: [LOW] The `OLLAMA_NUM_CTX` description says "Set this explicitly to make the guard a hard check everywhere", but it's already a hard check when the window is known.

The comment could be clearer: the guard is a hard check when `OLLAMA_NUM_CTX` is set OR when the model is loaded and the window is detected. The phrase "make the guard a hard check everywhere" might imply that without it, the guard is always soft, which is not true (detected windows also cause hard checks).

Suggested change:
```diff
 # make the guard a hard check everywhere -- match it to the server's
-# window (`OLLAMA_CONTEXT_LENGTH=32768 ollama serve`, or the app settings).
+# window (`OLLAMA_CONTEXT_LENGTH=32768 ollama serve`, or the app settings). If
+# unset, the runner will try to detect the window from a loaded model; if
+# detection fails, it assumes a small window and only warns on overflow.
 # OLLAMA_NUM_CTX=32768
```

## File: README.md
### L+146: [LOW] The download size for `qwen3-coder-next` is updated from ~40 GB to ~52 GB, but the "Local vs cloud" section still says ~40 GB.

In the "Local vs cloud" section, the text says "Higher quality: `qwen3-coder-next` (80B/3B active MoE, ~40 GB download)." This should be updated to ~52 GB for consistency.

Suggested change:
```diff
 | `local-pro` | `qwen3-coder-next` | 80B/3B MoE. Higher quality at the cost of ~52 GB download + slightly slower active path. |
```

### L+202: [LOW] The `--max-tokens` description mentions a warning on stderr for truncation, but the warning message format is not specified.

The README says "If the model does hit the ceiling mid-review, the runner still prints the partial output but emits a `WARN: ... truncated at max_tokens` line on stderr." This is accurate, but the exact warning message is defined in the code. Consider adding an example.

### L+296: [LOW] The "Error model" table's CONTEXT_OVERFLOW row includes a long exception note that may be hard to parse.

The note "Exception: if the message says max_tokens was hit before any content appeared (reasoning models can spend the whole budget thinking), raise `--max-tokens` instead." is correct but could be simplified. The table is meant for quick reference; consider moving the nuance to the prose above.

## File: docs/llm-code-review-runbook.md
### L+25: [LOW] The runbook's environment variable list includes `OLLAMA_NUM_CTX` but does not explain its interaction with detection.

The description says "Usually unset: the runner reads the real window from a loaded model via `/api/ps` and hard-enforces it." This is good, but could mention that setting it overrides detection.

Suggested change:
```diff
 - `OLLAMA_NUM_CTX` — the context window (tokens) the Ollama server loads models with. Usually unset: the runner reads the real window from a loaded model via `/api/ps` and hard-enforces it (`CONTEXT_OVERFLOW`, exit 12, because Ollama silently truncates oversized prompts instead of erroring). When the window can't be determined it assumes the smallest stock VRAM tier (4096) and only warns. Set explicitly to make the guard hard everywhere.
+  Set explicitly to override detection and make the guard hard everywhere.
```

### L+124: [LOW] The runbook's "Ollama silently truncates oversized prompts" note is numbered 10, but the previous note is 9, causing a gap in numbering.

The numbering should be sequential. Check that the preceding notes are numbered 1-9.

### L+265: [LOW] The runbook's "Future modes (TODO)" section is unchanged, but consider if any of the new features affect the TODO items.

The new context-window guard might relate to "Model-specific tuning (temperature, max_tokens per model)" or "Batch review (run N diffs in parallel, aggregate findings)". No changes needed, but keep in mind.

## File: pyproject.toml
### L+1: [LOW] The project name changed from "code-review-openrouter" to "local-gemini-code-review", but the description still mentions "OpenRouter-backed code review".

The description says "OpenRouter-backed code review using the Gemini CLI code-review extension prompts." This is now inaccurate because the runner supports multiple providers. Update to reflect the multi-provider nature.

Suggested change:
```diff
 name = "local-gemini-code-review"
 version = "0.1.0"
-description = "OpenRouter-backed code review using the Gemini CLI code-review extension prompts."
+description = "Multi-provider code review runner (OpenRouter, Gemini API, local Ollama) using the Gemini CLI code-review extension prompts."
 requires-python = ">=3.11"
```

## File: .github/workflows/ci.yml
### L+0: [LOW] The CI workflow runs tests on Ubuntu and Windows, but does not specify a Python version.

The workflow uses `ubuntu-latest` and `windows-latest`, which come with default Python versions. It's better to explicitly set the Python version to ensure consistency, especially since the project requires `>=3.11`.

Suggested change:
```diff
       - name: Install uv
         uses: astral-sh/setup-uv@v5
+      - name: Set up Python
+        run: uv python install 3.11
       - name: Run tests
         run: uv run --group dev pytest tests/ -v
```

## File: uv.lock
### L+0: [LOW] The lock file includes new dependencies (pytest, colorama, etc.) but the diff does not show the full addition; ensure the lock file is up-to-date.

The lock file diff shows additions for `colorama`, `iniconfig`, `packaging`, `pluggy`, `pygments`, and `pytest`. This is expected due to the new dev dependency group. No action needed, but ensure the lock file is committed.
