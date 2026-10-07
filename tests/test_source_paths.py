"""Real Git regressions for paths that line-oriented listings quote or split."""

import argparse
import os
import subprocess
from pathlib import Path

import pytest

from code_review.errors import ConfigError
from code_review.prompts import build_reference_section, bundle_codebase
from code_review.sources import (
    _filter_reviewable,
    changed_file_paths,
    gather_codebase_files,
)

SOURCE_PATHS = ("café.py", "日本語.py", "file with spaces.py", "nested/café.py")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init")
    hooks = tmp_path / ".git" / "test-hooks"
    hooks.mkdir()
    _git(tmp_path, "config", "core.hooksPath", str(hooks))
    _git(tmp_path, "config", "core.quotePath", "true")
    _git(
        tmp_path,
        "-c",
        "user.name=Source path tests",
        "-c",
        "user.email=tests@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-m",
        "Baseline",
    )
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(
        tmp_path, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"
    )
    for name in (*SOURCE_PATHS, "ignored.txt", "image.png"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# content of {name}\n", encoding="utf-8")
    _git(tmp_path, "add", "--all")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_codebase_bundle_keeps_unicode_and_space_paths(repo: Path) -> None:
    paths = gather_codebase_files(["*.py"], ["nested/*"])
    assert set(paths) == {Path(name) for name in SOURCE_PATHS if "/" not in name}
    bundle = bundle_codebase(paths)
    for path in paths:
        assert f"# content of {path.as_posix()}" in bundle
    assert Path("image.png") not in gather_codebase_files([], [])


def test_codebase_paths_are_relative_to_subdirectory(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo / "nested")
    paths = gather_codebase_files([], [])
    assert paths == [Path("café.py")]
    assert "# content of nested/café.py" in bundle_codebase(paths)


@pytest.mark.parametrize("mode", ["staged", "base", "default"])
@pytest.mark.parametrize("subdirectory", [False, True])
def test_changed_file_references_keep_exact_paths(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    subdirectory: bool,
) -> None:
    if subdirectory:
        monkeypatch.chdir(repo / "nested")
    args = argparse.Namespace(
        pr=None, staged=mode == "staged", base="HEAD" if mode == "base" else None
    )
    paths = _filter_reviewable(changed_file_paths(args))
    assert {path.resolve() for path in paths} == {
        repo / name for name in (*SOURCE_PATHS, "ignored.txt")
    }
    references = build_reference_section(paths)
    for name in SOURCE_PATHS:
        assert f"# content of {name}" in references


@pytest.mark.skipif(
    os.name == "nt", reason="Windows forbids control characters and quotes in filenames"
)
@pytest.mark.parametrize(
    "name", ["line\nbreak.py", "carriage\rreturn.py", "tab\tfile.py", 'quoted"file.py']
)
def test_git_delimiters_preserve_control_characters(repo: Path, name: str) -> None:
    (repo / name).write_text("special_path_marker = True\n", encoding="utf-8")
    _git(repo, "add", "--", name)
    codebase = gather_codebase_files([], [])
    changed = changed_file_paths(argparse.Namespace(pr=None, staged=True, base=None))
    for paths in (codebase, changed):
        assert Path(name) in paths
        assert "special_path_marker = True" in bundle_codebase(paths)


def test_nul_delimited_git_error_keeps_typed_diagnostic(repo: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        changed_file_paths(
            argparse.Namespace(pr=None, staged=False, base="missing-ref")
        )
    assert "missing-ref" in str(exc.value)
    assert exc.value.detail is not None
    assert "missing-ref" in exc.value.detail
