"""local-gemini-code-review: multi-provider LLM code-review runner.

The implementation lives in :mod:`code_review.cli` (one module by
design -- see the README's Non-goals). This package exists so the
project can ship as a wheel: ``uv tool install`` gives a global
``code-review`` command, with the prompt assets (``skills/``,
``commands/``) force-included next to the module.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-gemini-code-review")
except PackageNotFoundError:  # running from a raw checkout, not installed
    __version__ = "0+unknown"
