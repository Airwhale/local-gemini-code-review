"""Wire-layer tests: canned HTTP responses through the ``_make_client``
seam via ``httpx.MockTransport`` -- the full provider surface, offline.

Every ``call_*`` function and the ``/api/ps`` probe build their client
through ``_make_client``, so monkeypatching that one factory covers all
four HTTP call sites.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import code_review.cli as review
from code_review.cli import (
    CallResult,
    ConfigError,
    ContextOverflow,
    ProviderHiccup,
    RateLimit,
    SafetyRefusal,
    TransportError,
    call_gemini,
    call_ollama,
    call_openrouter,
)


def _mock(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        review,
        "_make_client",
        lambda timeout: httpx.Client(transport=transport, timeout=timeout),
    )


def _json_response(
    payload: dict | list, status: int = 200, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers or {})


def _openrouter(**kwargs) -> CallResult:
    defaults: dict[str, Any] = dict(
        system_prompt="s",
        user_prompt="u",
        model="m",
        api_key="k",
        referer="r",
        title="t",
        temperature=0.3,
        max_tokens=100,
    )
    defaults.update(kwargs)
    return call_openrouter(**defaults)


def _gemini(**kwargs) -> CallResult:
    defaults: dict[str, Any] = dict(
        system_prompt="s",
        user_prompt="u",
        model="m",
        api_key="k",
        temperature=0.3,
        max_tokens=100,
    )
    defaults.update(kwargs)
    return call_gemini(**defaults)


def _ollama(**kwargs) -> CallResult:
    defaults: dict[str, Any] = dict(
        system_prompt="s",
        user_prompt="u",
        model="m",
        host="http://localhost:11434",
        temperature=0.3,
        max_tokens=100,
        timeout=5.0,
        num_ctx=None,
    )
    defaults.update(kwargs)
    return call_ollama(**defaults)


class TestOpenRouterWire:
    def test_success_with_usage(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "review text"}}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ),
        )
        result = _openrouter()
        assert result.content == "review text"
        assert (result.prompt_tokens, result.completion_tokens) == (10, 5)
        assert not result.truncated

    def test_native_safety_signal_wins_over_normalized_stop(self, monkeypatch):
        # OpenRouter can normalize finish_reason to "stop" while the
        # underlying provider blocked on safety.
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "native_finish_reason": "SAFETY",
                            "message": {"content": None},
                        }
                    ],
                }
            ),
        )
        with pytest.raises(SafetyRefusal):
            _openrouter()

    def test_empty_content_at_length_is_overflow(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "choices": [
                        {"finish_reason": "length", "message": {"content": ""}}
                    ],
                }
            ),
        )
        with pytest.raises(ContextOverflow):
            _openrouter()

    def test_truncated_content_flagged(self, monkeypatch, capsys):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "choices": [
                        {"finish_reason": "length", "message": {"content": "partial"}}
                    ],
                }
            ),
        )
        result = _openrouter()
        assert result.truncated
        assert "WARN:" in capsys.readouterr().err

    def test_429_carries_retry_after_seconds(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {"error": "slow down"},
                status=429,
                headers={"retry-after": "30"},
            ),
        )
        with pytest.raises(RateLimit) as exc_info:
            _openrouter()
        assert exc_info.value.retry_after_seconds == 30.0

    @pytest.mark.parametrize(
        "payload",
        [
            # dict where the LIST should be: truthy, so an emptiness
            # check alone passes and [0] raises KeyError -> UNKNOWN
            {"choices": {"message": {"content": "ok"}}},
            {"choices": ["not-an-object"]},  # non-dict choice
            {"choices": [{"message": ["not-an-object"]}]},  # non-dict message
            {"choices": [{"message": {"content": {"nested": "x"}}}]},  # non-str content
        ],
    )
    def test_malformed_nested_shapes_are_hiccup(self, monkeypatch, payload):
        # The malformed-response contract (exit 13, retryable) must hold
        # for nested shapes too, not just top-level non-object JSON.
        _mock(monkeypatch, lambda req: _json_response(payload))
        with pytest.raises(ProviderHiccup):
            _openrouter()

    def test_401_is_config_error_naming_the_key(self, monkeypatch):
        # Reproduced live: an invalid OpenRouter key returns 401, which
        # used to fall through to UNKNOWN [exit 1]. It's a fix-your-key
        # problem -> CONFIG [exit 2] with the env var named.
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {"error": {"message": "User not found.", "code": 401}},
                status=401,
            ),
        )
        with pytest.raises(ConfigError) as exc_info:
            _openrouter()
        assert "OPENROUTER_API_KEY" in str(exc_info.value)

    def test_5xx_with_overflow_phrase_stays_transport(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: httpx.Response(
                502,
                text="upstream token limit tracker exploded",
            ),
        )
        with pytest.raises(TransportError):
            _openrouter()

    def test_non_json_is_hiccup(self, monkeypatch):
        _mock(monkeypatch, lambda req: httpx.Response(200, text="<html>lol</html>"))
        with pytest.raises(ProviderHiccup):
            _openrouter()

    def test_non_dict_json_is_hiccup(self, monkeypatch):
        _mock(monkeypatch, lambda req: _json_response(["not", "an", "object"]))
        with pytest.raises(ProviderHiccup):
            _openrouter()

    def test_no_choices_is_hiccup(self, monkeypatch):
        _mock(monkeypatch, lambda req: _json_response({"choices": []}))
        with pytest.raises(ProviderHiccup):
            _openrouter()

    def test_connect_error_is_transport(self, monkeypatch):
        def handler(req):
            raise httpx.ConnectError("boom", request=req)

        _mock(monkeypatch, handler)
        with pytest.raises(TransportError):
            _openrouter()


class TestGeminiWire:
    @pytest.mark.parametrize(
        "payload",
        [
            # dict where the LIST should be (truthy -> [0] KeyError)
            {"candidates": {"content": {"parts": []}}},
            {"candidates": ["not-an-object"]},  # non-dict candidate
            {"candidates": [{"content": "not-an-object"}]},  # non-dict content
            # Junk parts entries are skipped -> empty text -> typed
            # empty-content classification (finishReason STOP -> hiccup).
            {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [123, {"text": 5}]}}
                ]
            },
        ],
    )
    def test_malformed_nested_shapes_are_hiccup(self, monkeypatch, payload):
        _mock(monkeypatch, lambda req: _json_response(payload))
        with pytest.raises(ProviderHiccup):
            _gemini()

    def test_success_with_usage_metadata(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": "review"}]},
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 20,
                        "candidatesTokenCount": 7,
                    },
                }
            ),
        )
        result = _gemini()
        assert result.content == "review"
        assert (result.prompt_tokens, result.completion_tokens) == (20, 7)

    def test_prompt_level_block(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "promptFeedback": {"blockReason": "SAFETY"},
                }
            ),
        )
        with pytest.raises(SafetyRefusal):
            _gemini()

    def test_empty_max_tokens_is_overflow(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "candidates": [
                        {"finishReason": "MAX_TOKENS", "content": {"parts": []}}
                    ],
                }
            ),
        )
        with pytest.raises(ContextOverflow):
            _gemini()

    def test_truncated_content_flagged(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "candidates": [
                        {
                            "finishReason": "MAX_TOKENS",
                            "content": {"parts": [{"text": "partial"}]},
                        }
                    ],
                }
            ),
        )
        assert _gemini().truncated

    def test_non_dict_json_is_hiccup(self, monkeypatch):
        _mock(monkeypatch, lambda req: _json_response([1, 2, 3]))
        with pytest.raises(ProviderHiccup):
            _gemini()


class TestOllamaWire:
    def test_non_string_content_is_typed_not_crash(self, monkeypatch):
        # Malformed nested shape -> typed empty-content classification,
        # never a raw AttributeError/TypeError escaping to UNKNOWN.
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {"message": {"content": 42}, "done_reason": "stop"}
            ),
        )
        with pytest.raises(ProviderHiccup):
            _ollama()

    def test_native_success_shape(self, monkeypatch):
        def handler(req):
            body = json.loads(req.content)
            # The native payload shape: options carry the tuning knobs.
            assert body["options"]["num_predict"] == 100
            assert "num_ctx" not in body["options"]  # None -> omitted
            assert body["stream"] is False
            return _json_response(
                {
                    "message": {"role": "assistant", "content": "local review"},
                    "done_reason": "stop",
                    "prompt_eval_count": 50,
                    "eval_count": 20,
                }
            )

        _mock(monkeypatch, handler)
        result = _ollama()
        assert result.content == "local review"
        assert (result.prompt_tokens, result.completion_tokens) == (50, 20)

    def test_num_ctx_sent_when_pinned(self, monkeypatch):
        seen = {}

        def handler(req):
            seen.update(json.loads(req.content)["options"])
            return _json_response(
                {
                    "message": {"content": "ok"},
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                }
            )

        _mock(monkeypatch, handler)
        _ollama(num_ctx=32768)
        assert seen["num_ctx"] == 32768

    def test_done_reason_length_empty_is_overflow(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "message": {"content": ""},
                    "done_reason": "length",
                    "prompt_eval_count": 10,
                }
            ),
        )
        with pytest.raises(ContextOverflow):
            _ollama()

    def test_404_not_pulled_is_config_error(self, monkeypatch):
        # Body shape verified against a live Ollama 0.24.0 server.
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {"error": "model 'qwen3-coder:30b' not found"},
                status=404,
            ),
        )
        with pytest.raises(ConfigError) as exc_info:
            _ollama()
        assert "ollama pull" in str(exc_info.value)

    def test_try_pulling_body_fallback_on_400(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {"error": "model 'x' not found, try pulling it first"},
                status=400,
            ),
        )
        with pytest.raises(ConfigError) as exc_info:
            _ollama()
        assert "ollama pull" in str(exc_info.value)

    def test_connect_error_is_config_error(self, monkeypatch):
        def handler(req):
            raise httpx.ConnectError("refused", request=req)

        _mock(monkeypatch, handler)
        with pytest.raises(ConfigError) as exc_info:
            _ollama()
        assert "ollama serve" in str(exc_info.value)

    def test_post_verify_trips_on_filled_window(self, monkeypatch):
        # prompt_eval_count at the requested window: truncation likely,
        # output discarded even though content came back.
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "message": {"content": "bogus fragment review"},
                    "done_reason": "stop",
                    "prompt_eval_count": 4090,
                    "eval_count": 5,
                }
            ),
        )
        with pytest.raises(ContextOverflow) as exc_info:
            _ollama(num_ctx=4096)
        assert "truncated server-side" in str(exc_info.value)

    def test_non_dict_json_is_hiccup(self, monkeypatch):
        _mock(monkeypatch, lambda req: _json_response(["nope"]))
        with pytest.raises(ProviderHiccup):
            _ollama()


class TestApiPsProbeWire:
    def test_detected_window(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "models": [
                        {
                            "name": "m:latest",
                            "model": "m:latest",
                            "context_length": 32768,
                        }
                    ],
                }
            ),
        )
        assert review._detect_ollama_num_ctx("http://localhost:11434", "m") == 32768

    def test_missing_field_returns_none(self, monkeypatch):
        _mock(
            monkeypatch,
            lambda req: _json_response(
                {
                    "models": [{"name": "m:latest", "model": "m:latest"}],
                }
            ),
        )
        assert review._detect_ollama_num_ctx("http://localhost:11434", "m") is None

    def test_non_200_returns_none(self, monkeypatch):
        _mock(monkeypatch, lambda req: httpx.Response(500, text="down"))
        assert review._detect_ollama_num_ctx("http://x", "m") is None

    def test_top_level_non_dict_returns_none(self, monkeypatch):
        _mock(monkeypatch, lambda req: _json_response([]))
        assert review._detect_ollama_num_ctx("http://x", "m") is None

    def test_transport_error_returns_none(self, monkeypatch):
        def handler(req):
            raise httpx.ConnectError("refused", request=req)

        _mock(monkeypatch, handler)
        assert review._detect_ollama_num_ctx("http://x", "m") is None

    def test_probe_and_chat_share_the_seam(self, monkeypatch):
        """One path-dispatching transport serves /api/ps AND /api/chat --
        proving both call sites construct clients through _make_client."""
        paths = []

        def handler(req):
            paths.append(req.url.path)
            if req.url.path == "/api/ps":
                return _json_response(
                    {
                        "models": [
                            {
                                "name": "m:latest",
                                "model": "m:latest",
                                "context_length": 8192,
                            }
                        ],
                    }
                )
            return _json_response(
                {
                    "message": {"content": "ok"},
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                }
            )

        _mock(monkeypatch, handler)
        window = review._detect_ollama_num_ctx("http://localhost:11434", "m")
        result = _ollama(num_ctx=window)
        assert window == 8192
        assert result.content == "ok"
        assert paths == ["/api/ps", "/api/chat"]
