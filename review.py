#!/usr/bin/env python3
"""Back-compat shim: the runner moved to ``code_review/cli.py``.

Keeps the documented ``uv run review.py ...`` invocation working from a
checkout. Not an API surface -- import :mod:`code_review.cli` instead.
"""

from code_review.cli import _entrypoint

if __name__ == "__main__":
    _entrypoint()
