# Change summary: This change introduces support for a local Ollama provider, including robust configuration, documentation, and specific guards against silent prompt truncation, while also adding a CI workflow and unit tests for the runner.

## File: .github/workflows/ci.yml
### L21: [HIGH] The `uv run` command is incorrect and will cause the CI to fail.
The `uv run` command does not have a `--group` flag, so this step will fail with an invalid argument error. Furthermore, `uv run` executes a command within the virtual environment but does not install dependencies. The dev dependencies (like `pytest`) must be installed in a prior step for the test run to succeed. This requires a corresponding change in `pyproject.toml` to use the standard `[project.optional-dependencies]` section.

Suggested change:
```diff
-      - name: Run tests
-        run: uv run --group dev pytest tests/ -v
+      - name: Install dependencies
+        run: uv pip install -e .[dev]
+      - name: Run tests
+        run: uv run pytest tests/ -v
```

## File: pyproject.toml
### L11: [MEDIUM] The `[dependency-groups]` table is non-standard and not supported by `uv`.
The `[dependency-groups]` table is a PDM-specific extension and is not part of the PEP 621 standard. To ensure portability and compatibility with standard tooling like `uv`, optional dependencies should be defined under the `[project.optional-dependencies]` table. This change is necessary for the suggested fix in `ci.yml` (`uv pip install -e .[dev]`) to work correctly.

Suggested change:
```diff
- [dependency-groups]
- dev = [
-     "pytest>=8",
- ]
+ [project.optional-dependencies]
+ dev = [
+     "pytest>=8",
+ ]
```

## File: review.py
### L422: [LOW] The comment explaining `fnmatch` behavior could be more precise.
The current comment is slightly confusing. The core reason both patterns are needed is that `*` in `fnmatch` (as used by `pathlib.Path.match`) does not match the path separator `/`. A more direct explanation would improve clarity for future maintainers.

Suggested change:
```diff
-    # Build outputs occasionally tracked by mistake. Both spellings are
-    # needed: fnmatch's ``*`` can match the empty string but the literal
-    # ``/`` in ``*/dist/*`` cannot, so ``dist/bundle.js`` at the repo
-    # root only matches the un-prefixed variant.
+    # Build outputs occasionally tracked by mistake. Both patterns are
+    # needed because ``*`` does not match path separators (`/`).
+    # ``dist/*`` matches at the repo root, while ``*/dist/*`` matches
+    # in subdirectories.
```
