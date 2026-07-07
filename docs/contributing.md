# How to Contribute

> **Fork note:** this repository is a fork of
> [gemini-cli-extensions/code-review](https://github.com/gemini-cli-extensions/code-review)
> maintained by @Airwhale. The Google CLA process below applies to the
> UPSTREAM project only — contributions to this fork just need a PR
> against `main` here. The upstream text is kept intact for reference.
>
> **Developing on this fork:** CI gates every PR on the following (all
> offline — the tests mock the HTTP transport, no API keys needed):
>
> ```bash
> uv run --group dev pytest              # test suite (runs on ubuntu + windows in CI;
>                                        #  Windows matters -- encoding special-cases)
> uv run --group dev ruff check .        # lint
> uv run --group dev ruff format --check .
> uv run --group dev mypy                # type check (config in pyproject.toml)
> ```
>
> Two invariants PRs must respect: the upstream prompt files
> (`skills/code-review-commons/`, `commands/code-review.toml`,
> `commands/pr-code-review.toml`) stay byte-identical, and the error
> model (exit codes, `ERROR:` stderr prefix) is public API pinned by
> tests. For behavior-level changes (prompts, temperature, models),
> also run the eval harness — `uv run evals/run.py --model flash` — it
> spends real tokens and asks for confirmation unless `--yes`; the
> `Evals` GitHub workflow is manual-dispatch for the same reason.

We would love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code Reviews

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.
