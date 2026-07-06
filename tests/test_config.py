"""Tests for M6: prompt-asset resolution, env-file layering, the
per-project .code-review.toml, and the 4-layer settings precedence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import code_review.cli as review
from code_review.cli import (
    ConfigError,
    _apply_config_file_lists,
    _layered,
    _load_env_files,
    _load_project_config,
    _prompt_root,
    _resolve_settings,
    _user_config_dir,
    load_command_prompt,
    load_skill,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "CODE_REVIEW_PROMPT_DIR",
        "CODE_REVIEW_ENV",
        "CODE_REVIEW_PROVIDER",
        "CODE_REVIEW_TEMPERATURE",
        "CODE_REVIEW_MAX_TOKENS",
        "CODE_REVIEW_RETRIES",
        "CODE_REVIEW_MIN_SEVERITY",
        "CODE_REVIEW_FORMAT",
        "CODE_REVIEW_CONTEXT",
        "OPENROUTER_MODEL",
        "OLLAMA_NUM_CTX",
        "OLLAMA_TIMEOUT",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


class TestPromptRoot:
    def test_checkout_fallback_finds_repo_root(self):
        root = _prompt_root()
        assert (root / "skills").is_dir()
        assert (root / "commands").is_dir()

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        (tmp_path / "skills").mkdir()
        monkeypatch.setenv("CODE_REVIEW_PROMPT_DIR", str(tmp_path))
        assert str(_prompt_root()) == str(tmp_path)

    def test_env_override_without_skills_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CODE_REVIEW_PROMPT_DIR", str(tmp_path))
        with pytest.raises(ConfigError):
            _prompt_root()

    def test_loaders_read_real_assets(self):
        assert "Code Review" in load_skill("code-review-commons")
        assert "<OUTPUT>" in load_command_prompt("code-review")


class TestUserConfigDir:
    def test_windows_uses_appdata(self, monkeypatch: pytest.MonkeyPatch):
        if review.os.name != "nt":
            pytest.skip("windows-only expectation")
        monkeypatch.setenv("APPDATA", r"C:\Users\t\AppData\Roaming")
        assert _user_config_dir() == Path(r"C:\Users\t\AppData\Roaming\code-review")


class TestLoadEnvFiles:
    def test_explicit_env_file_missing_is_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CODE_REVIEW_ENV", str(tmp_path / "nope.env"))
        with pytest.raises(ConfigError):
            _load_env_files()

    def test_explicit_env_file_loads_without_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        env_file = tmp_path / "custom.env"
        env_file.write_text(
            "M6_TEST_MARKER=from-file\nOPENROUTER_API_KEY=file-key\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CODE_REVIEW_ENV", str(env_file))
        monkeypatch.delenv("M6_TEST_MARKER", raising=False)
        _load_env_files()
        assert review.os.environ["M6_TEST_MARKER"] == "from-file"
        # override=False: the process env (set by the fixture) wins.
        assert review.os.environ["OPENROUTER_API_KEY"] == "test-key"
        monkeypatch.delenv("M6_TEST_MARKER", raising=False)


class TestLoadProjectConfig:
    def test_found_in_parent_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        (tmp_path / ".code-review.toml").write_text(
            'model = "flash"\n', encoding="utf-8"
        )
        child = tmp_path / "sub" / "deeper"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        config = _load_project_config()
        assert config == {"model": "flash"}
        err = capsys.readouterr().err
        assert "[config] loaded" in err  # always announced

    def test_stops_at_git_boundary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Config ABOVE a repo root must not leak into the repo.
        (tmp_path / ".code-review.toml").write_text(
            'model = "flash"\n', encoding="utf-8"
        )
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        monkeypatch.chdir(repo)
        assert _load_project_config() == {}

    def test_none_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        assert _load_project_config() == {}

    def test_unknown_keys_dropped_with_warn(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".code-review.toml").write_text(
            'model = "flash"\nopenrouter_api_key = "sneaky"\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        config = _load_project_config()
        assert config == {"model": "flash"}  # credentials never accepted
        assert "openrouter_api_key" in capsys.readouterr().err

    def test_invalid_toml_is_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".code-review.toml").write_text("not = [toml", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigError):
            _load_project_config()

    def test_bom_tolerated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Windows editors (Notepad, PowerShell Set-Content) write a BOM;
        # tomllib rejects it unless we read with utf-8-sig. Found live:
        # the installed-tool smoke test failed on exactly this.
        (tmp_path / ".git").mkdir()
        (tmp_path / ".code-review.toml").write_bytes(b'\xef\xbb\xbfmodel = "flash"\n')
        monkeypatch.chdir(tmp_path)
        assert _load_project_config() == {"model": "flash"}


class TestLayered:
    def test_cli_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("X_TEST", "env-val")
        value, source = _layered("cli-val", "X_TEST", "x", {"x": "cfg-val"})
        assert (value, source) == ("cli-val", "cli")

    def test_env_beats_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("X_TEST", "env-val")
        value, source = _layered(None, "X_TEST", "x", {"x": "cfg-val"})
        assert (value, source) == ("env-val", "$X_TEST")

    def test_config_beats_default(self):
        value, source = _layered(None, "X_TEST", "x", {"x": "cfg-val"})
        assert (value, source) == ("cfg-val", ".code-review.toml")

    def test_empty_env_reads_as_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("X_TEST", "")
        value, source = _layered(None, "X_TEST", "x", {})
        assert (value, source) == (None, "default")


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        base=None,
        pr=None,
        staged=False,
        codebase=False,
        include=[],
        exclude=[],
        provider=None,
        model=None,
        models=None,
        ollama_host=None,
        temperature=None,
        max_tokens=None,
        retries=None,
        min_severity=None,
        context=None,
        no_context=False,
        output=None,
        format=None,
        baseline=None,
        dry_run=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestSettingsPrecedence:
    def test_config_supplies_tunables(self):
        config = {
            "temperature": 0.7,
            "max_tokens": 4000,
            "retries": 2,
            "min_severity": "high",
            "format": "json",
            "model": "flash",
        }
        settings = _resolve_settings(_args(), config)
        assert settings.temperature == 0.7
        assert settings.max_tokens == 4000
        assert settings.retries == 2
        assert settings.min_severity == "HIGH"
        assert settings.format == "json"
        assert settings.model == "google/gemini-2.5-flash"  # alias resolved

    def test_cli_beats_config(self):
        settings = _resolve_settings(_args(temperature=0.1), {"temperature": 0.9})
        assert settings.temperature == 0.1

    def test_env_beats_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CODE_REVIEW_TEMPERATURE", "0.5")
        settings = _resolve_settings(_args(), {"temperature": 0.9})
        assert settings.temperature == 0.5

    def test_bad_config_value_names_the_source(self):
        with pytest.raises(ConfigError) as exc_info:
            _resolve_settings(_args(), {"temperature": "toasty"})
        assert ".code-review.toml" in str(exc_info.value)

    def test_config_provider_and_models(self, monkeypatch: pytest.MonkeyPatch):
        settings = _resolve_settings(
            _args(), {"provider": "openrouter", "models": ["pro", "claude"]}
        )
        assert settings.models == (
            "google/gemini-2.5-pro",
            "anthropic/claude-sonnet-4.5",
        )

    def test_config_ollama_num_ctx_is_pinned(self):
        settings = _resolve_settings(
            _args(provider="ollama"), {"ollama_num_ctx": 32768}
        )
        assert settings.ollama_num_ctx_env == 32768  # enforced tier

    def test_defaults_without_config(self):
        settings = _resolve_settings(_args())
        assert settings.provider == "openrouter"
        assert settings.temperature == review.DEFAULT_TEMPERATURE
        assert settings.min_severity == "LOW"
        assert settings.format == "markdown"

    def test_config_context_used(self):
        settings = _resolve_settings(_args(), {"context": "project framing"})
        assert settings.context == "project framing"


class TestApplyConfigFileLists:
    def test_adopted_when_cli_empty(self):
        args = _args()
        _apply_config_file_lists(args, {"include": ["src/**"], "exclude": ["gen/*"]})
        assert args.include == ["src/**"]
        assert args.exclude == ["gen/*"]

    def test_cli_wins_outright(self):
        args = _args(include=["cli/**"])
        _apply_config_file_lists(args, {"include": ["cfg/**"]})
        assert args.include == ["cli/**"]

    def test_bad_type_rejected(self):
        with pytest.raises(ConfigError):
            _apply_config_file_lists(_args(), {"include": "not-a-list"})
