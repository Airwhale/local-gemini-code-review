"""Fake OpenAI Codex CLI for tests. Synthetic data only; no network.

Invoked as ``[sys.executable, this_file, "exec", ...]``. Behavior is chosen
by ``$FAKE_CODEX_SCENARIO``; when ``$FAKE_CODEX_RECORD`` names a file, the
received argv, stdin, working directory, environment variable names and the
``model_instructions_file`` contents are written there as JSON.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REVIEW = """# Change summary: Adds a synthetic input parser.

## File: app.py

### L3: [HIGH] Empty input crashes parser.

The parser indexes the first item without checking whether the input is empty.
"""


def emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def message(text: str, item_id: str = "item_1") -> dict[str, Any]:
    return {
        "type": "item.completed",
        "item": {"id": item_id, "type": "agent_message", "text": text},
    }


def completed() -> dict[str, Any]:
    return {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 1200,
            "cached_input_tokens": 300,
            "output_tokens": 77,
            "reasoning_output_tokens": 20,
        },
    }


def _instructions(args: list[str]) -> str | None:
    for i, arg in enumerate(args):
        if arg == "-c" and args[i + 1].startswith("model_instructions_file="):
            value = args[i + 1].split("=", 1)[1]
            return Path(value.strip("'")).read_text(encoding="utf-8")
    return None


def main() -> int:
    stdin = sys.stdin.buffer.read()
    args = sys.argv[1:]
    record = os.environ.get("FAKE_CODEX_RECORD")
    if record:
        Path(record).write_text(
            json.dumps(
                {
                    "argv": args,
                    "stdin": stdin.decode("utf-8"),
                    "cwd": os.getcwd(),
                    "cwd_entries": sorted(os.listdir(".")),
                    "env_keys": sorted(k.upper() for k in os.environ),
                    "instructions": _instructions(args),
                }
            ),
            encoding="utf-8",
        )

    scenario = os.environ.get("FAKE_CODEX_SCENARIO", "success")
    emit({"type": "thread.started", "thread_id": "00000000-0000-4000-8000-0000000000"})
    emit({"type": "turn.started"})
    if scenario == "success":
        emit({"type": "item.completed", "item": {"id": "r", "type": "reasoning", "text": "x"}})
        emit(message(REVIEW))
        emit(completed())
    elif scenario == "diagnostics":
        # Shapes observed from Codex CLI 0.167: diagnostics arrive as items
        # of type "error" before the turn and are not tool calls.
        for text in (
            "Codex is ignoring 1 unrecognized configuration settings.\n"
            "  session-flags: `x` is ignored.",
            "Codex is ignoring 1 unrecognized configuration settings.\n"
            "  session-flags: `x` is ignored.",
            "Code Mode is unavailable because code-mode host is disabled. "
            "Code mode will fail closed.",
        ):
            emit({"type": "item.completed", "item": {"type": "error", "message": text}})
        emit(message(REVIEW))
        emit(completed())
    elif scenario == "extra_lines":
        sys.stdout.write("Note: an informational line that is not JSON\n")
        emit(message(REVIEW))
        emit(completed())
        sys.stdout.write("trailing banner\n")
    elif scenario == "tool_then_sleep":
        emit(
            {
                "type": "item.started",
                "item": {"id": "c", "type": "command_execution", "command": "cat x"},
            }
        )
        time.sleep(60)
        emit(message(REVIEW))
        emit(completed())
    elif scenario == "tool_completed":
        emit({"type": "item.completed", "item": {"id": "f", "type": "file_change"}})
        emit(message(REVIEW))
        emit(completed())
    elif scenario == "turn_failed_auth":
        emit(
            {
                "type": "turn.failed",
                "error": {"message": "401 Unauthorized: run `codex login` to sign in"},
            }
        )
        return 1
    elif scenario == "error_quota":
        emit({"type": "error", "message": "You've hit your usage limit. Try later."})
        return 1
    elif scenario == "turn_failed_context":
        emit(
            {
                "type": "turn.failed",
                "error": {"message": "context_length_exceeded: input too long"},
            }
        )
        return 1
    elif scenario == "secret_in_error":
        emit(
            {
                "type": "turn.failed",
                "error": {"message": "Bad key sk-proj-SYNTHETICSECRET0123456789abcd"},
            }
        )
        return 1
    elif scenario == "no_turn_completed":
        emit(message(REVIEW))
    elif scenario == "empty_message":
        emit(completed())
    elif scenario == "unexpected_argument":
        sys.stderr.write("error: unexpected argument '--ignore-rules' found\n")
        return 2
    elif scenario == "malformed":
        sys.stdout.write("{not json\n")
    elif scenario == "sleep":
        time.sleep(60)
        emit(message(REVIEW))
        emit(completed())
    else:
        sys.stderr.write(f"unknown scenario {scenario}\n")
        return 99
    return 0


if __name__ == "__main__":
    sys.exit(main())
