"""Offline GitLab source and CLI contracts."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from code_review import cli, sources
from code_review.config import Settings
from code_review.errors import ConfigError
from code_review.parser import hunk_ranges
from code_review.providers import CallResult

REPO = "https://gitlab.example.com/group/nested/project"
URL = f"{REPO}/-/merge_requests/7"
SHA = "a" * 40
PATCH = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"


def metadata(sha: str = SHA, url: str = URL) -> str:
    return json.dumps(
        {"iid": 7, "web_url": url, "sha": sha, "diff_refs": {"head_sha": sha}}
    )


def args(**overrides: object) -> argparse.Namespace:
    value = argparse.Namespace(
        mr=7,
        repo=REPO,
        pr=None,
        staged=False,
        base=None,
        diff_file=None,
        chunk=False,
        models=None,
        min_found_by=None,
        full_files=False,
        baseline=None,
        codebase=False,
        include=[],
        exclude=[],
    )
    for key, item in overrides.items():
        setattr(value, key, item)
    return value


@pytest.fixture
def glab(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert kwargs["text"] is False
        assert kwargs["check"] is True
        assert "shell" not in kwargs
        return subprocess.CompletedProcess(
            command, 0, (PATCH if command[2] == "diff" else metadata()).encode("utf-8")
        )

    monkeypatch.setattr(sources.subprocess, "run", run)
    return calls


def test_pinning_resolution_and_announce(
    glab: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    value = args()
    assert sources._read_diff_source(value) == PATCH
    assert value._gitlab_mr.sha == SHA
    assert URL in capsys.readouterr().err
    assert glab == [
        [
            "glab",
            "mr",
            "view",
            "7",
            "--output",
            "json",
            "--repo",
            f"{REPO}",
        ],
        [
            "glab",
            "mr",
            "diff",
            "7",
            "--raw",
            "--repo",
            f"{REPO}",
        ],
        [
            "glab",
            "mr",
            "view",
            "7",
            "--output",
            "json",
            "--repo",
            f"{REPO}",
        ],
    ]


@pytest.mark.parametrize(
    "repo",
    [
        None,
        "project",
        "../project",
        "group/../project",
        "http://other/group/project",
        "group/project --help",
        "group/project/",
        "group/project",
        "https://user:password@gitlab.example.com/group/project",
        "https://gitlab.example.com/group/project?token=placeholder",
        "https://gitlab.example.com/group/project#fragment",
        "https://gitlab.example.com:bad/group/project",
        "https://gitlab.example.com:0/group/project",
        "https://gitlab.example.com/group/../project",
        "https://gitlab.example.com/group/project/",
        "\nhttps://gitlab.example.com/group/project",
    ],
)
def test_repository_required_and_validated(repo: str | None) -> None:
    with pytest.raises(ConfigError):
        cli._validate_flag_combos(args(repo=repo))


@pytest.mark.parametrize("iid", [0, -7])
def test_positive_iid(iid: int) -> None:
    with pytest.raises(ConfigError, match="positive"):
        cli._validate_flag_combos(args(mr=iid))


def test_valid_flags() -> None:
    cli._validate_flag_combos(args())
    cli._validate_flag_combos(args(full_files=True))
    cli._validate_flag_combos(args(mr=None, pr=7, repo="owner/repo"))
    with pytest.raises(ConfigError, match="only applies"):
        cli._validate_flag_combos(args(mr=None))


@pytest.mark.parametrize(
    "repo",
    [
        "https://gitlab.com/group/project",
        "https://gitlab.example.com:8443/group/nested/project",
    ],
)
def test_explicit_hosts_and_ports(repo: str) -> None:
    assert sources.gitlab_repo(repo) == repo


@pytest.mark.parametrize(
    "flag", ["--pr", "--base", "--staged", "--diff-file", "--codebase"]
)
def test_source_exclusivity(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setattr(cli, "_load_env_files", lambda: None)
    extra = [] if flag in {"--staged", "--codebase"} else ["7"]
    monkeypatch.setattr(
        sys, "argv", ["code-review", "--mr", "7", "--repo", REPO, flag, *extra]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "output",
    [
        "not JSON",
        "{}",
        metadata(url="https://other/group/project/-/merge_requests/7"),
        metadata(url=URL.replace("/7", "/8")),
        metadata(sha="short"),
    ],
)
def test_bad_metadata(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    monkeypatch.setattr(sources, "_run_glab", lambda *a: output)
    with pytest.raises(ConfigError):
        sources.mr_metadata(7, REPO)


@pytest.mark.parametrize("head_sha", [None, ""])
def test_empty_diff_refs_use_sha(
    monkeypatch: pytest.MonkeyPatch, head_sha: str | None
) -> None:
    output = json.loads(metadata())
    output["diff_refs"]["head_sha"] = head_sha
    monkeypatch.setattr(sources, "_run_glab", lambda *a: json.dumps(output))
    assert sources.mr_metadata(7, REPO).sha == SHA


def test_mismatched_diff_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    output = json.loads(metadata())
    output["diff_refs"]["head_sha"] = "b" * 40
    monkeypatch.setattr(sources, "_run_glab", lambda *a: json.dumps(output))
    with pytest.raises(ConfigError, match="diff head"):
        sources.mr_metadata(7, REPO)


def test_push_during_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter([metadata(), PATCH, metadata("b" * 40)])
    monkeypatch.setattr(sources, "_run_glab", lambda *a: next(outputs))
    with pytest.raises(ConfigError, match="head changed"):
        sources._read_diff_source(args())


def test_missing_glab(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(*a: object, **kw: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(sources.subprocess, "run", run)
    with pytest.raises(ConfigError, match="Install GitLab CLI"):
        sources._run_glab(["view", "7"], REPO)


def test_failure_does_not_expose_transport_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*a: object, **kw: object) -> None:
        raise subprocess.CalledProcessError(
            1, "glab", stderr="https://secret-token@host: authentication failed"
        )

    monkeypatch.setattr(sources.subprocess, "run", run)
    with pytest.raises(
        ConfigError, match="authentication for the selected host"
    ) as error:
        sources._run_glab(["view", "7"], REPO)
    assert "secret-token" not in str(error.value)
    assert error.value.exit_code == 2
    assert error.value.detail is None


def test_patch_paths() -> None:
    patch = PATCH + (
        'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n'
        '--- "a/caf\\303\\251.py"\n+++ "b/caf\\303\\251.py"\n@@ -1 +1 @@\n-x\n+y\n'
        "diff --git a/old name.py b/new name.py\nsimilarity index 100%\nrename from old name.py\nrename to new name.py\n"
        "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n"
        "diff --git a/日本語.py b/日本語.py\n--- a/日本語.py\n+++ b/日本語.py\n"
        "diff --git a/image.png b/image.png\nBinary files a/image.png and b/image.png differ\n"
    )
    assert sources.mr_patch_paths(patch) == [
        Path(p) for p in ["x.py", "café.py", "new name.py", "日本語.py", "image.png"]
    ]
    assert sources.mr_patch_paths(PATCH + "+++ b/fake.py\n") == [Path("x.py")]


@pytest.mark.parametrize(
    "name", ["foo b/.env", "foo b/bar.txt", "nested/with spaces.bin"]
)
def test_real_binary_patch_does_not_select_unrelated_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\0synthetic binary content\n")
    (tmp_path / ".env").write_text("LOCAL_SENTINEL_NOT_FOR_REVIEW", encoding="utf-8")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), *arguments],
            check=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout

    git("init")
    git("add", "--", name)
    patch = git("diff", "--cached", "--no-ext-diff")
    assert "Binary files" in patch
    assert sources.mr_patch_paths(patch) == [Path(name)]

    def read(options: argparse.Namespace) -> str:
        options._gitlab_mr = sources.GitLabMergeRequest.model_validate_json(metadata())
        options._gitlab_patch = patch
        return patch

    def local_git(command: list[str]) -> str:
        if command == ["git", "rev-parse", "HEAD"]:
            return SHA
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return str(tmp_path)
        return ""

    monkeypatch.setattr(cli, "_read_diff_source", read)
    monkeypatch.setattr(sources, "_run_git", local_git)
    settings = Settings(
        provider="openrouter",
        model="m",
        temperature=0.3,
        max_tokens=16000,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    request = cli._build_request(args(full_files=True), settings)
    assert "LOCAL_SENTINEL_NOT_FOR_REVIEW" not in request.user_prompt


@pytest.mark.parametrize("operation", ["rename", "copy"])
def test_binary_destination_metadata_resolves_ambiguous_headers(operation: str) -> None:
    patch = f"diff --git a/old b/name b/new b/name\n{operation} from old b/name\n{operation} to new b/name\nBinary files differ\n"
    assert sources.mr_patch_paths(patch) == [Path("new b/name")]


def test_ambiguous_binary_header_without_metadata_fails_closed() -> None:
    with pytest.raises(ConfigError, match="Ambiguous"):
        sources.mr_patch_paths(
            "diff --git a/old name b/new name\nBinary files differ\n"
        )
    assert sources.mr_patch_paths("diff --git a/old b/new\nBinary files differ\n") == [
        Path("new")
    ]


@pytest.mark.parametrize(
    "separator", ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029", "\r"]
)
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_patch_content_cannot_add_unrelated_local_references(
    separator: str,
    line_ending: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.py").write_text("changed_file_marker = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SENTINEL_SHOULD_STAY_LOCAL", encoding="utf-8")
    patch = PATCH + f"+content{separator}diff --git a/.env b/.env\n"
    patch = patch.replace("\n", line_ending)

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert kwargs["text"] is False
        output = patch if command[2] == "diff" else metadata()
        return subprocess.CompletedProcess(command, 0, output.encode("utf-8"))

    def git(command: list[str]) -> str:
        if command == ["git", "rev-parse", "HEAD"]:
            return SHA
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return str(tmp_path)
        return ""

    monkeypatch.setattr(sources.subprocess, "run", run)
    monkeypatch.setattr(sources, "_run_git", git)
    settings = Settings(
        provider="openrouter",
        model="m",
        temperature=0.3,
        max_tokens=16000,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    request = cli._build_request(args(full_files=True), settings)
    assert request.diff == patch
    assert "<REFERENCE_FILES>" in request.user_prompt
    assert "changed_file_marker" in request.user_prompt
    assert "SENTINEL_SHOULD_STAY_LOCAL" not in request.user_prompt


@pytest.mark.parametrize("full", [None, True])
def test_unsafe_reference_auto_fallback(
    monkeypatch: pytest.MonkeyPatch, full: bool | None
) -> None:
    monkeypatch.setattr(cli, "_read_diff_source", lambda a: PATCH)
    monkeypatch.setattr(cli, "_guard_mr_full_files", lambda mr: None)
    value = args(full_files=full)
    value._gitlab_mr = sources.GitLabMergeRequest.model_validate_json(metadata())

    def paths(unused: argparse.Namespace) -> list[Path]:
        raise ConfigError("unsafe reference")

    monkeypatch.setattr(cli, "changed_file_paths", paths)
    settings = Settings(
        provider="openrouter",
        model="m",
        temperature=0.3,
        max_tokens=16000,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    if full is True:
        with pytest.raises(ConfigError, match="unsafe reference"):
            cli._build_request(value, settings)
    else:
        assert (
            "<REFERENCE_FILES>" not in cli._build_request(value, settings).user_prompt
        )


def test_quoted_unicode_hunk_evidence() -> None:
    patch = 'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n+++ "b/caf\\303\\251.py"\n@@ -1 +1,2 @@\n+x\n+y\n'
    assert hunk_ranges(patch) == {"café.py": [(1, 2)]}
    assert hunk_ranges('+++ "invalid\\xzz"\n@@ -1 +1 @@\n') == {}
    assert hunk_ranges("+++ b/b/file with spaces.py\n@@ -1 +1 @@\n") == {
        "b/file with spaces.py": [(1, 1)]
    }


@pytest.mark.parametrize(
    "header,path",
    [
        ("b/x.py", "x.py"),
        ('"b/caf\\303\\251.py"', "café.py"),
        ("b/trailing space.py ", "trailing space.py "),
    ],
)
def test_crlf_hunk_paths_preserve_filename_spaces(header: str, path: str) -> None:
    assert hunk_ranges(f"+++ {header}\r\n@@ -1 +1 @@\r\n+x\r\n") == {path: [(1, 1)]}


@pytest.mark.parametrize(
    "header", ['"a/old.py" "b/new.py"', 'a/old.py "b/new.py"', '"a/old.py" b/new.py']
)
def test_quoted_binary_headers(header: str) -> None:
    assert sources.mr_patch_paths(f"diff --git {header}\nBinary files differ\n") == [
        Path("new.py")
    ]


@pytest.mark.parametrize(
    "patch", ['diff --git "unterminated\n', 'diff --git a/x b/x\n+++ "bad\\xzz"\n']
)
def test_invalid_quoted_patch(patch: str) -> None:
    with pytest.raises(ConfigError):
        sources.mr_patch_paths(patch)


@pytest.mark.parametrize("path", ["../outside.py", "/absolute.py"])
def test_unsafe_patch_path(path: str) -> None:
    with pytest.raises(ConfigError, match="Unsafe"):
        sources.mr_patch_paths(f"diff --git a/x b/x\n+++ b/{path}\n")


@pytest.mark.parametrize("path", ["folder\\name.py", "colon:name.py", "C:/outside.py"])
def test_platform_path_rules(path: str) -> None:
    patch = f"diff --git a/x b/x\n+++ {json.dumps('b/' + path)}\n"
    if os.name == "nt":
        with pytest.raises(ConfigError, match="Unsafe"):
            sources.mr_patch_paths(patch)
    else:
        assert sources.mr_patch_paths(patch) == [Path(path)]


def test_wrong_head_and_dirty_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mr = sources.GitLabMergeRequest.model_validate_json(metadata())
    monkeypatch.setattr(sources, "_run_git", lambda command: "b" * 40)
    with pytest.raises(ConfigError, match="requires local HEAD"):
        sources._guard_mr_full_files(mr)
    monkeypatch.setattr(
        sources,
        "_run_git",
        lambda command: SHA if command[1] == "rev-parse" else " M x.py",
    )
    sources._guard_mr_full_files(mr)
    assert "uncommitted changes" in capsys.readouterr().err


@pytest.mark.parametrize("full", [None, True, False])
def test_context_guard_flow(monkeypatch: pytest.MonkeyPatch, full: bool | None) -> None:
    value = args(full_files=full)
    value._gitlab_mr = sources.GitLabMergeRequest.model_validate_json(metadata())
    monkeypatch.setattr(cli, "_read_diff_source", lambda a: PATCH)

    def guard(mr: sources.GitLabMergeRequest) -> None:
        raise ConfigError("wrong head")

    monkeypatch.setattr(cli, "_guard_mr_full_files", guard)
    settings = Settings(
        provider="openrouter",
        model="m",
        temperature=0.3,
        max_tokens=16000,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    if full is True:
        with pytest.raises(ConfigError, match="wrong head"):
            cli._build_request(value, settings)
    else:
        assert (
            "<REFERENCE_FILES>" not in cli._build_request(value, settings).user_prompt
        )


def test_reference_paths_rebased(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.setattr(sources, "_run_git", lambda a: str(tmp_path))
    value = args()
    value._gitlab_patch = PATCH
    assert sources.changed_file_paths(value) == [Path("../x.py")]


def test_cli_dry_run(
    glab: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_load_env_files", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setenv("CODE_REVIEW_PROVIDER", "openrouter")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "code-review",
            "--mr",
            "7",
            "--repo",
            REPO,
            "--no-full-files",
            "--no-project-config",
            "--dry-run",
        ],
    )
    cli.main()
    captured = capsys.readouterr()
    assert URL in captured.err
    assert "DRY RUN" in captured.out
    assert len(glab) == 3


def test_matching_head_attaches_changed_file_context(
    glab: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.py").write_text("new\n", encoding="utf-8")

    def git(command: list[str]) -> str:
        if command == ["git", "rev-parse", "HEAD"]:
            return SHA
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return str(tmp_path)
        return ""

    monkeypatch.setattr(sources, "_run_git", git)
    settings = Settings(
        provider="openrouter",
        model="m",
        temperature=0.3,
        max_tokens=16000,
        retries=0,
        min_severity="LOW",
        context=None,
        output=None,
    )
    request = cli._build_request(args(full_files=True), settings)
    assert "<REFERENCE_FILES>" in request.user_prompt
    assert "1: new" in request.user_prompt
    assert len(glab) == 3


def test_url_precedes_model_call_and_json_contract(
    glab: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_load_env_files", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setenv("CODE_REVIEW_PROVIDER", "openrouter")
    monkeypatch.setenv("CODE_REVIEW_NO_TIPS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "code-review",
            "--mr",
            "7",
            "--repo",
            REPO,
            "--no-full-files",
            "--no-project-config",
            "--format",
            "json",
        ],
    )

    def call(*unused: object) -> CallResult:
        assert URL in capsys.readouterr().err
        return CallResult(content="# Change summary: No issues found.\n")

    monkeypatch.setattr(cli, "_execute_call", call)
    monkeypatch.setattr(cli, "_estimate_cost_line", lambda *a: None)
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["mode"] == "diff"
    assert report["provider"] == "openrouter"
    assert len(glab) == 3


def test_escaped_reference_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sources, "_run_git", lambda a: str(tmp_path / "other-root"))
    monkeypatch.setattr(
        sources, "_rebase_repo_relative", lambda paths: [tmp_path / "outside.py"]
    )
    value = args()
    value._gitlab_patch = PATCH
    with pytest.raises(ConfigError, match="escapes"):
        sources.changed_file_paths(value)
