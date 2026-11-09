"""Fake Claude Code executable for tests. Synthetic data only; no network.

Invoked as ``[sys.executable, this_file, *claude_args]``. Behavior is
chosen by ``$FAKE_CLAUDE_SCENARIO``; when ``$FAKE_CLAUDE_RECORD`` names a
file, the received argv, stdin, working directory, environment variable
names and system-prompt file contents are written there as JSON.
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


def envelope(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": REVIEW,
        "stop_reason": "end_turn",
        "terminal_reason": "completed",
        "num_turns": 1,
        "session_id": "00000000-0000-4000-8000-000000000000",
        "total_cost_usd": 0.0,
        "usage": {
            "input_tokens": 120,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
            "output_tokens": 42,
        },
        "modelUsage": {"claude-opus-5-5": {"inputTokens": 120}},
        "permission_denials": [],
        "some_future_field": {"ignored": True},
    }
    base.update(overrides)
    return base


def emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")


def main() -> int:
    stdin = sys.stdin.buffer.read()
    args = sys.argv[1:]
    record = os.environ.get("FAKE_CLAUDE_RECORD")
    if record:
        system_prompt = None
        if "--system-prompt-file" in args:
            spf = Path(args[args.index("--system-prompt-file") + 1])
            system_prompt = spf.read_text(encoding="utf-8")
        Path(record).write_text(
            json.dumps(
                {
                    "argv": args,
                    "stdin": stdin.decode("utf-8"),
                    "cwd": os.getcwd(),
                    "env_keys": sorted(k.upper() for k in os.environ),
                    "system_prompt": system_prompt,
                }
            ),
            encoding="utf-8",
        )

    scenario = os.environ.get("FAKE_CLAUDE_SCENARIO", "success")
    if scenario == "success":
        emit(envelope())
    elif scenario == "mutate":
        # A misbehaving agent writing to its working directory.
        Path("written-by-fake.txt").write_text("x", encoding="utf-8")
        emit(envelope())
    elif scenario == "warning_line":
        sys.stdout.write("Note: something informational\n")
        emit(envelope())
    elif scenario == "trailing_line":
        emit(envelope())
        sys.stdout.write("A newer version is available\n")
    elif scenario == "truncated_review":
        emit(envelope(stop_reason="max_tokens", result=REVIEW[:80]))
    elif scenario == "empty_max_tokens":
        emit(envelope(stop_reason="max_tokens", result=""))
    elif scenario == "refusal":
        emit(envelope(stop_reason="refusal", result="I can't help with that."))
    elif scenario == "missing_result":
        data = envelope()
        del data["result"]
        emit(data)
    elif scenario == "bad_types":
        emit(envelope(is_error="false"))
    elif scenario == "auth_error":
        # Shape observed from a real signed-out CLI: subtype "success" with
        # is_error true and a nonzero exit.
        emit(
            envelope(
                is_error=True,
                stop_reason="stop_sequence",
                terminal_reason="api_error",
                result="Failed to authenticate: OAuth session expired",
                usage={"input_tokens": 0, "output_tokens": 0},
                modelUsage={},
            )
        )
        return 1
    elif scenario == "auth_error_exit0":
        emit(envelope(is_error=True, result="Not logged in. Please run /login"))
    elif scenario == "secret_in_error":
        emit(
            envelope(
                is_error=True,
                result="Invalid API key sk-ant-api03-SYNTHETICSECRET0123456789",
            )
        )
        return 1
    elif scenario == "quota":
        emit(
            envelope(
                is_error=True,
                api_error_status=429,
                result="Claude usage limit reached. Your limit will reset at 5pm.",
            )
        )
        return 1
    elif scenario == "billing":
        emit(envelope(is_error=True, result="Credit balance is too low"))
        return 1
    elif scenario == "error_subtype":
        data = envelope(
            subtype="error_during_execution",
            is_error=True,
            errors=["API Error: 529 overloaded_error"],
        )
        del data["result"]
        emit(data)
        return 1
    elif scenario == "plain_secret_stdout":
        sys.stdout.write("Invalid API key sk-ant-api03-SYNTHETICSECRET0123456789\n")
        return 1
    elif scenario == "api_error_as_success":
        emit(envelope(result="API Error: 500 internal server error"))
    elif scenario == "grandchild":
        # Leave a helper holding stdout/stderr open after this process exits.
        import subprocess

        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        emit(envelope())
        sys.stdout.flush()
    elif scenario == "unknown_option":
        sys.stderr.write("error: unknown option '--permission-prompts'\n")
        return 1
    elif scenario == "crash":
        sys.stderr.write("something went badly wrong\n")
        return 3
    elif scenario == "malformed":
        sys.stdout.write("{not json at all\n")
    elif scenario == "truncated_json":
        sys.stdout.write(json.dumps(envelope())[:60])
    elif scenario == "empty":
        pass
    elif scenario == "sleep":
        time.sleep(60)
        emit(envelope())
    elif scenario == "flood":
        sys.stdout.write("x" * (2 * 1024 * 1024))
        sys.stdout.flush()
        time.sleep(60)
    else:
        sys.stderr.write(f"unknown scenario {scenario}\n")
        return 99
    return 0


if __name__ == "__main__":
    sys.exit(main())
