## What

## Why

## Checklist

- [ ] `uv run --group dev pytest`, `ruff check .`, `ruff format --check .`, and `mypy` all pass locally (CI gates on these, ubuntu + windows)
- [ ] Upstream prompt files untouched (`skills/code-review-commons/`, `commands/code-review.toml`, `commands/pr-code-review.toml`) — fork prompt content is runtime-appended in Python, never edited into those files
- [ ] No new stderr line starts with `ERROR:` (reserved for the single terminal error block); any new informational prefix is added to the README's stderr-prefix table
- [ ] Changes to exit codes, error classification, or the JSON envelope are reflected in the README contract tables **and** the pinned tests, and noted in `CHANGELOG.md`
- [ ] Docs updated where behavior changed (README / runbook / `.env.example` / `--help` text)
- [ ] Behavior-level changes (prompts, temperature/model defaults): eval harness run (`uv run evals/run.py --model flash`) and the recall/noise table pasted in this description — it spends real tokens, so it's not in CI
