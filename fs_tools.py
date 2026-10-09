"""File system tools for the LLM-powered resume assistant.

Four tools the language model can call (wired up in ``llm_file_assistant.py``):

* :func:`read_file` - extract text and metadata from PDF, DOCX and plain-text files.
* :func:`list_files` - list the files in a folder, optionally filtered by extension.
* :func:`write_file` - write a UTF-8 text file, creating folders as needed.
* :func:`search_in_file` - case-insensitive keyword search with surrounding context.

Rules shared by every tool:

* **Never raise.** Failures come back as data (``"success": False`` plus a readable
  ``"error"``), so a mistyped path or a corrupt file can't crash the agent loop.
* **Sandboxed.** Paths resolve against a root folder: ``$FS_ASSISTANT_ROOT`` if set,
  otherwise the current working directory. Anything that resolves outside it (``..``,
  absolute paths, ``~``, symlinks) is refused, and so are hidden files and folders
  (``.env``, ``.git``, ``.venv``, ...) - the model can never read the API key or plant
  code. Files are then opened one folder at a time without following symlinks, so the
  check can't be dodged by swapping a symlink in between checking and opening.
* **Careful writes.** Only plain-text formats, atomic replacement, and an existing file
  is replaced only if these tools created it - a prompt-injected model can't clobber your
  resumes, README or requirements.txt.
* **Model-friendly errors.** Messages say what went wrong and how to fix it, including
  "Did you mean ...?" suggestions for mistyped names.

Text extraction for each format lives in :mod:`extractors`.
"""

from __future__ import annotations

import contextlib
import difflib
import errno
import json
import os
import re
import secrets
import stat
import unicodedata
from datetime import datetime, timezone
from typing import Any, NamedTuple

import extractors
from extractors import SUPPORTED_EXTENSIONS, TEXT_EXTENSIONS, ExtractionError

__all__ = ["list_files", "read_file", "sandbox_root", "search_in_file", "write_file"]

ROOT_ENV_VAR = "FS_ASSISTANT_ROOT"
WRITABLE_EXTENSIONS = TEXT_EXTENSIONS
WRITE_LOG_NAME = ".fs_tools_written.json"  # hidden, so the tools themselves can't read or forge it

MAX_READ_BYTES = 20 * 1024 * 1024
MAX_WRITE_BYTES = 5 * 1024 * 1024
MAX_WRITE_LOG_BYTES = 1024 * 1024
MAX_KEYWORD_CHARS = 200
MAX_MATCHES = 50
CONTEXT_CHARS = 80
MAX_DISPLAY_CHARS = 200
MAX_SUGGESTION_DEPTH = 8
MAX_SUGGESTION_SCAN = 2000  # directory entries examined when hunting for a "Did you mean" match
SUGGESTION_CUTOFF = 0.6  # minimum similarity (0-1) for a suggestion

# Flags missing on some platforms (Windows has none of the first four).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)  # opening a FIFO must not block
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)
# Race-free opening needs the *at() system calls; without them (Windows) fall back to paths.
_WALK_BY_FD = bool(_O_NOFOLLOW and _O_DIRECTORY) and {os.open, os.stat, os.mkdir, os.rename, os.unlink} <= os.supports_dir_fd

Folder = int | str  # an open directory descriptor, or a path where descriptors aren't supported


class _ToolError(Exception):
    """An expected failure whose message is shown to the user or the model."""


class _Resolved(NamedTuple):
    path: str  # absolute, symlinks resolved
    display: str  # relative to the sandbox root, "/"-separated
    root: str


# --------------------------------------------------------------------------- paths


def sandbox_root() -> str:
    """Return the real path of the folder every tool is confined to."""
    configured = (os.environ.get(ROOT_ENV_VAR) or "").strip()
    return os.path.realpath(os.path.expanduser(configured) if configured else os.getcwd())


def _is_within(path: str, root: str) -> bool:
    """Check whether a resolved path belongs to the sandbox root."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # e.g. different drives on Windows
        return False


def _relative(path: str, root: str) -> str:
    """Represent a path relative to the root with portable separators."""
    rel = os.path.relpath(path, root)
    return "." if rel == "." else rel.replace(os.sep, "/")


def _is_hidden_name(name: str) -> bool:
    """Dotfiles and Office lock files (``~$resume.docx``)."""
    return name.startswith((".", "~$"))


def _has_hidden_part(display: str) -> bool:
    """Detect forbidden dotfile or dot-directory components."""
    return any(part.startswith(".") for part in display.split("/") if part not in ("", "."))


def _safe_display(raw: Any) -> str:
    """Best-effort printable form of a path argument, for error messages."""
    if isinstance(raw, (str, os.PathLike)):
        text = os.fspath(raw)
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return text.strip()[:MAX_DISPLAY_CHARS]
    return repr(raw)[:MAX_DISPLAY_CHARS]


def _coerce_path(raw: Any, what: str) -> str:
    """Validate a path argument and strip surrounding whitespace."""
    if isinstance(raw, os.PathLike):
        raw = os.fspath(raw)
    if not isinstance(raw, str):
        raise _ToolError(f"{what} must be a string, got {type(raw).__name__}.")
    if "\x00" in raw:
        raise _ToolError(f"{what} contains a null byte.")
    return raw.strip()


def _resolve(raw: Any, *, what: str = "filepath", allow_root: bool = False) -> _Resolved:
    """Resolve a path from the user or the model to a location inside the sandbox."""
    text = _coerce_path(raw, what)
    if not text:
        if not allow_root:
            raise _ToolError(f"{what} is empty.")
        text = "."
    root = sandbox_root()
    expanded = os.path.expanduser(text)
    candidate = expanded if os.path.isabs(expanded) else os.path.join(root, expanded)
    resolved = os.path.realpath(candidate)
    if not _is_within(resolved, root):
        raise _ToolError(
            f"Access denied: '{text}' is outside the allowed folder. "
            "Use a path inside the project folder, such as 'resumes/...'."
        )
    display = _relative(resolved, root)
    if _has_hidden_part(display):
        raise _ToolError(
            f"Access denied: '{text}' is a hidden file or folder (names starting with '.' are off-limits)."
        )
    return _Resolved(resolved, display, root)


def _parts(target: _Resolved) -> list[str]:
    """Split a validated relative path into directory components."""
    return [] if target.display == "." else target.display.split("/")


# ------------------------------------------------------------------ suggestions


def _split_at_first_missing(path: str) -> tuple[str, str, str] | None:
    """Split ``path`` into (deepest existing folder, first missing name, remainder)."""
    current = path
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            return None
        if os.path.isdir(parent):
            rest = os.path.relpath(path, current)
            return parent, os.path.basename(current), "" if rest == "." else rest
        current = parent


def _closest_entry(folder: str, wanted: str) -> str | None:
    """The visible entry of ``folder`` whose name best matches ``wanted`` (case-insensitive)."""
    try:
        names = {name.lower(): name for name in os.listdir(folder) if not _is_hidden_name(name)}
    except OSError:
        return None
    close = difflib.get_close_matches(wanted.lower(), list(names), n=1, cutoff=SUGGESTION_CUTOFF)
    return os.path.join(folder, names[close[0]]) if close else None


def _find_by_name(start: str, wanted: str) -> str | None:
    """Look in ``start`` and up to two folders below it for an entry named like ``wanted``."""
    best, best_score, seen = None, SUGGESTION_CUTOFF, 0
    for folder, folders, files in os.walk(start):
        level = 0 if folder == start else os.path.relpath(folder, start).count(os.sep) + 1
        visible_folders = sorted(name for name in folders if not _is_hidden_name(name))
        folders[:] = visible_folders if level < 2 else []  # don't descend any deeper
        for name in (*visible_folders, *sorted(name for name in files if not _is_hidden_name(name))):
            seen += 1
            score = difflib.SequenceMatcher(None, wanted.lower(), name.lower()).ratio()
            if score > best_score:
                best, best_score = os.path.join(folder, name), score
        if seen > MAX_SUGGESTION_SCAN:
            break
    return best


def _did_you_mean(missing: str, root: str, depth: int = 0) -> str | None:
    """Suggest an existing path whose names closely match a mistyped one."""
    split = _split_at_first_missing(missing)
    if split is None or depth > MAX_SUGGESTION_DEPTH or not _is_within(split[0], root):
        return None
    base, wanted, rest = split
    corrected = _closest_entry(base, wanted)
    if corrected is None:  # e.g. an invented 'data/' prefix: look for the last name nearby instead
        found = _find_by_name(base, os.path.basename(missing))
        return _relative(found, root) if found else None
    if not rest:
        return _relative(corrected, root)
    full = os.path.join(corrected, rest)
    if os.path.exists(full):
        return _relative(full, root)
    return _did_you_mean(full, root, depth + 1) or _relative(corrected, root)


def _not_found(target: _Resolved, kind: str) -> _ToolError:
    """Build a missing-path error with a nearby-name suggestion."""
    hint = _did_you_mean(target.path, target.root)
    advice = f" Did you mean '{hint}'?" if hint else " Use list_files to see what exists."
    return _ToolError(f"{kind} not found: '{target.display}'.{advice}")


def _os_error(exc: OSError, target: _Resolved, kind: str) -> _ToolError:
    """Translate an OSError from opening ``target`` into a message for the model."""
    if isinstance(exc, FileNotFoundError):
        return _not_found(target, kind)
    if isinstance(exc, PermissionError):
        return _ToolError(f"Permission denied: cannot access '{target.display}'.")
    if exc.errno == errno.ELOOP:
        return _ToolError(f"Access denied: '{target.display}' changed while it was being opened (a symlink appeared).")
    if isinstance(exc, NotADirectoryError):
        if kind == "Folder":
            return _ToolError(f"'{target.display}' is a file, not a folder. Use read_file to read it.")
        return _not_found(target, kind)  # part of the path is a file
    return _ToolError(f"Cannot access '{target.display}': {exc.strerror or exc}.")


# ------------------------------------------------- race-free file system access


def _in(folder: Folder, name: str) -> str:
    """Resolve a child name relative to a folder path or descriptor."""
    return name if isinstance(folder, int) else os.path.join(folder, name)


def _at(folder: Folder) -> dict[str, int]:
    """Supply a directory descriptor only when the platform supports it."""
    return {"dir_fd": folder} if isinstance(folder, int) else {}


def _close(folder: Folder) -> None:
    """Close a directory descriptor while leaving path handles alone."""
    if isinstance(folder, int):
        os.close(folder)


def _open_folder(root: str, parts: list[str], *, create: bool = False) -> Folder:
    """Open ``root/parts...`` one folder at a time, refusing symlinks on the way.

    Creates missing folders when ``create`` is set, but never the root itself.
    """
    if not os.path.isdir(root):
        raise _ToolError(f"The assistant's folder doesn't exist: {root}. Check {ROOT_ENV_VAR}.")
    if not _WALK_BY_FD:
        path = os.path.join(root, *parts)
        if create:
            os.makedirs(path, exist_ok=True)
        return path
    folder = os.open(root, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC)
    try:
        for part in parts:
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, dir_fd=folder)
            child = os.open(part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC, dir_fd=folder)
            os.close(folder)
            folder = child
    except BaseException:
        os.close(folder)
        raise
    return folder


def _open_regular_file(target: _Resolved) -> tuple[int, os.stat_result]:
    """Open a regular file inside the sandbox for reading; returns (descriptor, stat)."""
    parts = _parts(target)
    if not parts:
        raise _ToolError(f"'{target.display}' is a folder, not a file. Use list_files to see what it contains.")
    try:
        folder = _open_folder(target.root, parts[:-1])
        try:
            flags = os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC | _O_BINARY
            descriptor = os.open(_in(folder, parts[-1]), flags, **_at(folder))
        finally:
            _close(folder)
    except OSError as exc:
        raise _os_error(exc, target, "File") from None
    info = os.fstat(descriptor)
    if stat.S_ISREG(info.st_mode):
        return descriptor, info
    os.close(descriptor)
    if stat.S_ISDIR(info.st_mode):
        raise _ToolError(f"'{target.display}' is a folder, not a file. Use list_files to see what it contains.")
    raise _ToolError(f"'{target.display}' is not a regular file.")


def _new_file_mode() -> int:
    """Compute normal file permissions from the current process umask."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def _replace_in(folder: Folder, name: str, data: bytes, mode: int) -> None:
    """Write ``data`` to a temp file, then rename it over ``name``: never a half-written file."""
    temp = f".tmp-{secrets.token_hex(8)}.part"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC | _O_BINARY
    descriptor = os.open(_in(folder, temp), flags, 0o600, **_at(folder))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), mode)
        renames = {"src_dir_fd": folder, "dst_dir_fd": folder} if isinstance(folder, int) else {}
        os.replace(_in(folder, temp), _in(folder, name), **renames)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(_in(folder, temp), **_at(folder))
        raise


def _existing(folder: Folder, name: str) -> os.stat_result | None:
    """Inspect a target without following symlinks; return None if absent."""
    try:
        return os.stat(_in(folder, name), follow_symlinks=False, **_at(folder))
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- write record


def _written_files(root: str) -> frozenset[str]:
    """Paths these tools wrote earlier. Fail-safe: an unreadable record protects every file."""
    try:
        folder = _open_folder(root, [])
        try:
            flags = os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC | _O_BINARY
            descriptor = os.open(_in(folder, WRITE_LOG_NAME), flags, **_at(folder))
        finally:
            _close(folder)
        with os.fdopen(descriptor, "rb") as handle:
            files = json.loads(handle.read(MAX_WRITE_LOG_BYTES))["files"]
        return frozenset(entry for entry in files if isinstance(entry, str))
    except (OSError, ValueError, KeyError, TypeError, _ToolError):
        return frozenset()


def _record_write(target: _Resolved) -> str | None:
    """Remember that ``target`` was written by these tools; returns a warning if that fails."""
    files = sorted(_written_files(target.root) | {target.display})
    data = json.dumps({"files": files}, indent=1).encode("utf-8")
    try:
        folder = _open_folder(target.root, [])
        try:
            _replace_in(folder, WRITE_LOG_NAME, data, _new_file_mode())
        finally:
            _close(folder)
    except (OSError, _ToolError) as exc:
        return f"Saved, but {WRITE_LOG_NAME} couldn't be updated ({exc}), so this file can't be replaced later."
    return None


# ------------------------------------------------------------------------- helpers


def _extension(path: str) -> str:
    """Return the lowercase suffix of a filename."""
    return os.path.splitext(path)[1].lower()


def _iso(timestamp: float) -> str | None:
    """Format a filesystem timestamp, tolerating out-of-range values."""
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def _megabytes(size: int) -> str:
    """Format a byte count as megabytes for user-facing messages."""
    return f"{size / (1024 * 1024):.1f} MB"


def _unsupported_message(extension: str, display: str) -> str:
    """Explain an unsupported format and suggest supported alternatives."""
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    if extension == ".doc":
        return f"'{display}' is a legacy Word .doc file, which isn't supported. Save it as .docx or PDF first."
    kind = f"'{extension}' files" if extension else "files without an extension"
    return f"Can't read '{display}': {kind} aren't supported. Supported types: {supported}."


# --------------------------------------------------------------------------- tools


def _read_result(success: bool, filepath: str, **fields: Any) -> dict[str, Any]:
    """Build the consistent read_file result shape."""
    return {
        "success": success,
        "filepath": filepath,
        "content": fields.get("content"),
        "metadata": fields.get("metadata"),
        "warning": fields.get("warning"),
        "error": fields.get("error"),
    }


def _read_bytes(target: _Resolved) -> tuple[bytes, os.stat_result, str]:
    """Read a supported file inside the sandbox: (data, stat, lower-case extension)."""
    descriptor, info = _open_regular_file(target)
    with os.fdopen(descriptor, "rb") as handle:
        extension = _extension(target.path)
        if extension not in SUPPORTED_EXTENSIONS:
            raise _ToolError(_unsupported_message(extension, target.display))
        if info.st_size > MAX_READ_BYTES:
            raise _ToolError(f"'{target.display}' is too large to read ({_megabytes(info.st_size)}; the limit is {_megabytes(MAX_READ_BYTES)}).")
        data = handle.read(MAX_READ_BYTES + 1)
    if len(data) > MAX_READ_BYTES:  # the file grew while being read
        raise _ToolError(f"'{target.display}' is too large to read (the limit is {_megabytes(MAX_READ_BYTES)}).")
    return data, info, extension


def read_file(filepath: str) -> dict[str, Any]:
    """Read a resume or text file and return its text plus metadata.

    Supports PDF (text layer via pypdf), DOCX (body, tables, headers, footers, text
    boxes) and plain text (.txt, .md, .csv, .json). The file's real format is detected
    from its contents, so a mislabelled file is still read correctly (with a warning).

    Args:
        filepath: Path to the file, relative to the sandbox root (or absolute inside it).

    Returns:
        ``{"success", "filepath", "content", "metadata", "warning", "error"}`` where
        ``metadata`` holds ``name``, ``extension``, ``size_bytes``, ``modified`` (ISO 8601),
        ``char_count``, ``word_count`` and, for PDFs, ``page_count`` (plus ``pages_read``
        when not every page could be read). ``warning`` flags partial results (empty file,
        scanned PDF, non-UTF-8 text, ...). On failure ``success`` is False,
        ``content``/``metadata`` are None and ``error`` says why.
    """
    display = _safe_display(filepath)
    try:
        target = _resolve(filepath)
        display = target.display
        data, info, extension = _read_bytes(target)
        extraction = extractors.extract(data, extension)
        content = unicodedata.normalize("NFC", extraction.text)
        metadata = {
            "name": os.path.basename(target.path),
            "extension": extension,
            "size_bytes": info.st_size,
            "modified": _iso(info.st_mtime),
            "char_count": len(content),
            "word_count": len(content.split()),
            **extraction.metadata,
        }
        warning = " ".join(extraction.warnings) or None
        return _read_result(True, display, content=content, metadata=metadata, warning=warning)
    except (_ToolError, ExtractionError) as exc:
        return _read_result(False, display, error=str(exc))
    except PermissionError:
        return _read_result(False, display, error=f"Permission denied: cannot read '{display}'.")
    except OSError as exc:
        return _read_result(False, display, error=f"Could not read '{display}': {exc.strerror or exc}.")
    except Exception as exc:  # last line of defence: tools must never raise
        return _read_result(False, display, error=f"Unexpected error reading '{display}': {type(exc).__name__}: {exc}")


def _normalize_extensions(extension: Any) -> tuple[str, ...] | None:
    """Turn '.pdf', 'pdf', '*.PDF' or '.pdf, .docx' into a tuple of lowercase suffixes."""
    if extension is None:
        return None
    if not isinstance(extension, str):
        raise _ToolError(f"extension must be a string such as '.pdf', got {type(extension).__name__}.")
    if extension.strip().lower() in ("none", "null", "all", "any"):
        return None
    suffixes = []
    for part in re.split(r"[\s,;|]+", extension):
        cleaned = part.strip().lstrip("*").lower()
        if cleaned in ("", ".", ".*"):
            continue
        if "/" in cleaned or "\\" in cleaned:
            raise _ToolError(f"Invalid extension filter '{part}'. Use something like '.pdf'.")
        suffixes.append(cleaned if cleaned.startswith(".") else f".{cleaned}")
    return tuple(suffixes) or None


def _file_entry(entry: os.DirEntry, folder: _Resolved, suffixes: tuple[str, ...] | None) -> dict[str, Any] | None:
    """Describe one directory entry, or return None if it should not be listed."""
    name = entry.name
    if _is_hidden_name(name) or (suffixes and not name.lower().endswith(suffixes)):
        return None
    try:
        if not entry.is_file():
            return None
        if entry.is_symlink():
            real = os.path.realpath(os.path.join(folder.path, name))
            if not _is_within(real, folder.root) or _has_hidden_part(_relative(real, folder.root)):
                return None  # never advertise files the other tools would refuse
        info = entry.stat()
    except OSError:  # broken symlink, permission problem, file vanished
        return None
    return {
        "name": name,
        "path": name if folder.display == "." else f"{folder.display}/{name}",
        "extension": _extension(name),
        "size_bytes": info.st_size,
        "modified": _iso(info.st_mtime),
    }


def _open_listing(folder: _Resolved) -> Folder:
    """Open a folder and translate failures into tool-friendly errors."""
    try:
        return _open_folder(folder.root, _parts(folder))
    except OSError as exc:
        raise _os_error(exc, folder, "Folder") from None


def list_files(directory: str, extension: str | None = None) -> list[dict[str, Any]]:
    """List the files in a folder (not recursive), optionally filtered by extension.

    Hidden files, Office lock files (``~$...``) and sub-folders are skipped. The filter
    accepts ``".pdf"``, ``"pdf"``, ``"*.PDF"`` or several at once (``".pdf,.docx"``).

    Args:
        directory: Folder to list, relative to the sandbox root; ``""`` or ``"."`` is the root.
        extension: Optional extension filter, case-insensitive.

    Returns:
        A list of ``{"name", "path", "extension", "size_bytes", "modified"}`` dicts sorted
        by name (an empty list if nothing matches). Because the return type is a list,
        an error - missing folder, a file instead of a folder, access denied - is
        reported as a one-element list ``[{"error": "..."}]``.
    """
    try:
        folder = _resolve(directory, what="directory", allow_root=True)
        suffixes = _normalize_extensions(extension)
        handle = _open_listing(folder)
        try:
            with os.scandir(handle) as entries:
                files = [item for entry in entries if (item := _file_entry(entry, folder, suffixes)) is not None]
        finally:
            _close(handle)
        return sorted(files, key=lambda item: (item["name"].casefold(), item["name"]))
    except _ToolError as exc:
        return [{"error": str(exc)}]
    except PermissionError:
        return [{"error": f"Permission denied: cannot list '{_safe_display(directory)}'."}]
    except Exception as exc:  # last line of defence: tools must never raise
        return [{"error": f"Unexpected error listing '{_safe_display(directory)}': {type(exc).__name__}: {exc}"}]


def _alternative_name(display: str) -> str:
    """Suggest a destination that avoids replacing the existing file."""
    stem, extension = os.path.splitext(display)
    return f"{stem}_new{extension}"


def _check_replaceable(info: os.stat_result, target: _Resolved) -> None:
    """An existing file may be replaced only if these tools wrote it and it is writable."""
    if stat.S_ISDIR(info.st_mode):
        raise _ToolError(f"'{target.display}' is a folder. Add a file name, e.g. '{target.display}/notes.md'.")
    if not stat.S_ISREG(info.st_mode):
        raise _ToolError(f"'{target.display}' exists and is not a regular file (or it changed while being written).")
    if target.display not in _written_files(target.root):
        raise _ToolError(
            f"'{target.display}' already exists and wasn't created by this assistant, so it won't be overwritten. "
            f"Write to a new file name instead (for example '{_alternative_name(target.display)}'), "
            "or ask the user to delete or rename the existing file."
        )
    if not os.access(target.path, os.W_OK):
        raise _ToolError(f"'{target.display}' is read-only; refusing to overwrite it.")


def _save(target: _Resolved, data: bytes) -> bool:
    """Create or replace ``target`` race-free; returns True if a file was replaced."""
    parts = _parts(target)
    folder = _open_folder(target.root, parts[:-1], create=True)
    try:
        existing = _existing(folder, parts[-1])
        if existing is not None:
            _check_replaceable(existing, target)
        mode = stat.S_IMODE(existing.st_mode) if existing is not None else _new_file_mode()
        _replace_in(folder, parts[-1], data, mode)
        return existing is not None
    finally:
        _close(folder)


def _write_result(success: bool, filepath: str, **fields: Any) -> dict[str, Any]:
    """Build the consistent write_file result shape."""
    return {
        "success": success,
        "filepath": filepath,
        "bytes_written": fields.get("bytes_written", 0),
        "overwritten": fields.get("overwritten", False),
        "warning": fields.get("warning"),
        "error": fields.get("error"),
    }


def write_file(filepath: str, content: str) -> dict[str, Any]:
    """Write text to a file (UTF-8), creating any missing parent folders.

    Only plain-text formats can be written (.txt, .md, .csv, .json), so the model can't
    produce scripts, config or corrupt PDF/DOCX files. Hidden paths and anything outside
    the sandbox root are refused. The write is atomic, and an existing file is replaced
    only if these tools created it (they keep a hidden record of what they wrote).

    Args:
        filepath: Destination path, relative to the sandbox root.
        content: Text to write.

    Returns:
        ``{"success", "filepath", "bytes_written", "overwritten", "warning", "error"}``.
    """
    display = _safe_display(filepath)
    try:
        if not isinstance(content, str):
            raise _ToolError(f"content must be a string, got {type(content).__name__}.")
        raw_path = _coerce_path(filepath, "filepath")
        target = _resolve(filepath)
        display = target.display
        if raw_path.endswith(("/", os.sep)) or display == "." or os.path.isdir(target.path):
            folder = raw_path.rstrip("/" + os.sep) or "."
            raise _ToolError(f"'{folder}' is a folder. Add a file name, e.g. '{folder}/notes.md'.")
        extension = _extension(target.path)
        if extension not in WRITABLE_EXTENSIONS:
            allowed = ", ".join(sorted(WRITABLE_EXTENSIONS))
            kind = f"'{extension}' files" if extension else "files without an extension"
            raise _ToolError(f"Refusing to write {kind}: write_file only creates plain-text files ({allowed}).")
        data = content.encode("utf-8", errors="replace")
        if len(data) > MAX_WRITE_BYTES:
            raise _ToolError(f"Content is too large ({_megabytes(len(data))}; the limit is {_megabytes(MAX_WRITE_BYTES)}).")
        overwritten = _save(target, data)
        warning = _record_write(target)
        return _write_result(True, display, bytes_written=len(data), overwritten=overwritten, warning=warning)
    except _ToolError as exc:
        return _write_result(False, display, error=str(exc))
    except OSError as exc:
        if exc.errno == errno.ELOOP or isinstance(exc, NotADirectoryError):
            return _write_result(False, display, error=f"Could not write '{display}': part of that path is a file or a symlink, not a folder.")
        return _write_result(False, display, error=f"Could not write '{display}': {exc.strerror or exc}.")
    except Exception as exc:  # last line of defence: tools must never raise
        return _write_result(False, display, error=f"Unexpected error writing '{display}': {type(exc).__name__}: {exc}")


def _validate_keyword(keyword: Any) -> str:
    """Normalize and validate a bounded, nonempty search keyword."""
    if not isinstance(keyword, str):
        raise _ToolError(f"keyword must be a string, got {type(keyword).__name__}.")
    needle = unicodedata.normalize("NFKC", keyword).strip()
    if not needle:
        raise _ToolError("keyword is empty. Give a word or phrase to search for.")
    if len(needle) > MAX_KEYWORD_CHARS:
        raise _ToolError(f"keyword is too long ({len(needle)} characters; the limit is {MAX_KEYWORD_CHARS}).")
    return needle


def _keyword_pattern(needle: str, whole_word: bool) -> re.Pattern[str]:
    """Literal, case-insensitive pattern; spaces in the keyword match any whitespace run."""
    body = r"\s+".join(re.escape(word) for word in needle.split())
    if whole_word:
        body = rf"(?<!\w){body}(?!\w)"
    return re.compile(body, re.IGNORECASE)


def _context(text: str, start: int, end: int) -> str:
    """Return whitespace-collapsed surrounding text with truncation markers."""
    low, high = max(0, start - CONTEXT_CHARS), min(len(text), end + CONTEXT_CHARS)
    snippet = " ".join(text[low:high].split())
    return f"{'...' if low > 0 else ''}{snippet}{'...' if high < len(text) else ''}"


def _find_matches(text: str, pattern: re.Pattern[str]) -> tuple[list[dict[str, Any]], int]:
    """Count all matches and retain a bounded list with line numbers."""
    matches: list[dict[str, Any]] = []
    total, line, previous = 0, 1, 0
    for found in pattern.finditer(text):
        total += 1
        if len(matches) >= MAX_MATCHES:
            continue
        line += text.count("\n", previous, found.start())
        previous = found.start()
        matches.append({
            "position": found.start(),
            "line": line,
            "match": " ".join(found.group(0).split()),
            "context": _context(text, found.start(), found.end()),
        })
    return matches, total


def _search_result(success: bool, filepath: str, keyword: Any, **fields: Any) -> dict[str, Any]:
    """Build the consistent search_in_file result shape."""
    return {
        "success": success,
        "filepath": filepath,
        "keyword": keyword,
        "match_count": fields.get("match_count", 0),
        "matches": fields.get("matches", []),
        "truncated": fields.get("truncated", False),
        "warning": fields.get("warning"),
        "error": fields.get("error"),
    }


def search_in_file(filepath: str, keyword: str, whole_word: bool = False) -> dict[str, Any]:
    """Search a file for a keyword (case-insensitive) and return each match with context.

    Works on every format :func:`read_file` supports. The keyword is matched literally
    (``C++`` and ``.NET`` work as typed) and a space in it matches any run of whitespace,
    so ``"machine learning"`` also finds the phrase split across a line break. Text is
    NFKC-normalised first, so PDF ligatures such as "ﬁ" match "fi"; positions are offsets
    into that normalised text.

    Args:
        filepath: File to search, relative to the sandbox root.
        keyword: Word or phrase to find.
        whole_word: If True, only match whole words ("Java" won't match "JavaScript").

    Returns:
        ``{"success", "filepath", "keyword", "match_count", "matches", "truncated",
        "warning", "error"}``. Each match is ``{"position", "line", "match", "context"}``
        with about 80 characters of context either side. ``match_count`` counts every
        match; at most 50 are listed (``truncated`` is True when more exist).
    """
    display = _safe_display(filepath)
    echoed = keyword.strip() if isinstance(keyword, str) else None
    try:
        needle = _validate_keyword(keyword)
        if not isinstance(whole_word, bool):
            raise _ToolError(f"whole_word must be true or false, got {type(whole_word).__name__}.")
        document = read_file(filepath)
        display = document["filepath"]
        if not document["success"]:
            raise _ToolError(document["error"])
        text = unicodedata.normalize("NFKC", document["content"])
        matches, total = _find_matches(text, _keyword_pattern(needle, whole_word))
        return _search_result(
            True, display, echoed,
            match_count=total, matches=matches, truncated=total > len(matches), warning=document["warning"],
        )
    except _ToolError as exc:
        return _search_result(False, display, echoed, error=str(exc))
    except Exception as exc:  # last line of defence: tools must never raise
        return _search_result(False, display, echoed, error=f"Unexpected error searching '{display}': {type(exc).__name__}: {exc}")
