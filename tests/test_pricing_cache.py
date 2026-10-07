"""Malformed cached models must not prevent reviews of valid models."""

import json
import time
from pathlib import Path

import pytest

from code_review import providers


@pytest.mark.parametrize("field", ["prompt", "completion", "context_length"])
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(10**400, id="oversized-integer"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(True, id="boolean"),
        pytest.param("bad", id="text"),
    ],
)
def test_bad_cached_field_preserves_valid_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: object,
) -> None:
    malformed: dict[str, object] = {
        "prompt": 1e-6,
        "completion": 2e-6,
        "context_length": 64000,
    }
    malformed[field] = invalid
    cache = tmp_path / "prices.json"
    cache.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "models": {
                    "selected": {
                        "prompt": 1e-6,
                        "completion": 2e-6,
                        "context_length": 64000,
                    },
                    "unrelated": malformed,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(providers, "_pricing_cache_path", lambda: cache)
    assert providers.estimate_cost_usd(
        "openrouter", "selected", 1000, 500
    ) == pytest.approx(0.002)
    assert providers.model_context_limit("openrouter", "selected") == 64000
    assert providers.model_context_limit("openrouter", "unrelated") is None
    if field == "context_length":
        # Bad context metadata must not discard otherwise usable prices.
        assert providers.estimate_cost_usd(
            "openrouter", "unrelated", 1000, 500
        ) == pytest.approx(0.002)
    else:
        assert providers.estimate_cost_usd("openrouter", "unrelated", 1000, 500) is None
