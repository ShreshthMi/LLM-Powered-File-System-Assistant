"""Shared fixtures: every test runs inside its own throwaway sandbox folder."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty sandbox root: FS_ASSISTANT_ROOT points at it and it is the working directory.

    A sibling ``outside`` folder holds a secret file, so tests can prove it never leaks.
    """
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET-MARKER", encoding="utf-8")
    monkeypatch.setenv("FS_ASSISTANT_ROOT", str(root))
    monkeypatch.chdir(root)
    return root


@pytest.fixture(autouse=True)
def portable_symlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip symlink probes only when the host cannot create symlinks."""
    import errno
    original = Path.symlink_to

    def create(path: Path, target: str | Path, target_is_directory: bool = False) -> None:
        try:
            original(path, target, target_is_directory=target_is_directory)
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.ENOSYS, errno.EACCES):
                pytest.skip("This host does not permit creating symlinks")
            raise

    monkeypatch.setattr(Path, "symlink_to", create)
