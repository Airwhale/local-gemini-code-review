"""Git's C-quoted path encoding shared by diff consumers."""

import ast

from code_review.errors import ConfigError


def decode_patch_path(value: str) -> str:
    """Decode octal-escaped UTF-8 bytes without interpreting unquoted paths."""
    if not value.startswith('"'):
        return value
    try:
        return ast.literal_eval("b" + value).decode("utf-8")
    except (ValueError, SyntaxError, UnicodeError) as exc:
        raise ConfigError("Invalid quoted path in patch.") from exc
