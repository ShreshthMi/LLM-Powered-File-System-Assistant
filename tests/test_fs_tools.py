"""Tests for fs_tools: normal use, malformed input, and attempts to escape the sandbox."""

from __future__ import annotations

import json
import os
import stat
import time
import zipfile
from datetime import datetime
from pathlib import Path

import docx
import pytest
from docx.shared import Inches

import fs_tools
from builders import content_control_xml, make_docx, make_pdf, text_box_xml

SECRET = "TOP-SECRET-MARKER"  # planted outside the sandbox by the `sandbox` fixture


def leaks_secret(result: object) -> bool:
    return SECRET in json.dumps(result, default=str)


def assert_failed(result: dict, fragment: str) -> None:
    assert result["success"] is False
    assert fragment.lower() in result["error"].lower()


# ---------------------------------------------------------------------- read_file


class TestReadFile:
    def test_reads_text_file_with_metadata(self, sandbox: Path) -> None:
        (sandbox / "notes.txt").write_text("Hello résumé world\n", encoding="utf-8")

        result = fs_tools.read_file("notes.txt")

        assert result["success"] is True and result["error"] is None
        assert result["content"] == "Hello résumé world\n"
        metadata = result["metadata"]
        assert (metadata["name"], metadata["extension"]) == ("notes.txt", ".txt")
        assert metadata["size_bytes"] == len("Hello résumé world\n".encode())
        assert (metadata["char_count"], metadata["word_count"]) == (19, 3)
        assert datetime.fromisoformat(metadata["modified"]).tzinfo is not None
        assert "page_count" not in metadata

    def test_reads_pdf_text_from_every_page(self, sandbox: Path) -> None:
        make_pdf(sandbox / "cv.pdf", [["Jane Roe", "Python developer"], ["Page two: Java 17"]])

        result = fs_tools.read_file("cv.pdf")

        assert result["success"] is True
        assert "Python developer" in result["content"] and "Page two: Java 17" in result["content"]
        assert result["metadata"]["page_count"] == 2

    def test_reads_docx_header_paragraphs_and_tables(self, sandbox: Path) -> None:
        make_docx(sandbox / "cv.docx", ["Summary: Python engineer"], header="Jane Roe | jane@example.com",
                  table=[["Skill", "Years"], ["Python", "5"]])

        content = fs_tools.read_file("cv.docx")["content"]

        assert content.splitlines()[0] == "Jane Roe | jane@example.com"
        assert "Summary: Python engineer" in content
        assert "Python | 5" in content

    def test_reads_docx_content_controls_and_text_boxes_once(self, sandbox: Path) -> None:
        make_docx(sandbox / "template.docx", ["Body"],
                  raw_body_xml=(content_control_xml("Inside a content control"), text_box_xml("Sidebar text box")))

        content = fs_tools.read_file("template.docx")["content"]

        assert "Inside a content control" in content
        assert content.count("Sidebar text box") == 1  # the VML fallback copy is skipped

    def test_docx_tab_stops_are_not_tab_characters(self, sandbox: Path) -> None:
        document = docx.Document()
        document.add_paragraph("Skills").paragraph_format.tab_stops.add_tab_stop(Inches(1))
        document.save(str(sandbox / "tabs.docx"))

        assert fs_tools.read_file("tabs.docx")["content"] == "Skills"

    @pytest.mark.parametrize("name", ["CV.PDF", "Cv.Pdf"])
    def test_extension_case_does_not_matter(self, sandbox: Path, name: str) -> None:
        make_pdf(sandbox / name, [["Upper case extension"]])

        result = fs_tools.read_file(name)

        assert result["success"] is True and "Upper case extension" in result["content"]

    @pytest.mark.parametrize(("data", "warning"), [(b"", "empty"), (b"  \n\t\n", "whitespace")])
    def test_empty_text_file_succeeds_with_warning(self, sandbox: Path, data: bytes, warning: str) -> None:
        (sandbox / "blank.txt").write_bytes(data)

        result = fs_tools.read_file("blank.txt")

        assert result["success"] is True
        assert warning in result["warning"]

    @pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
    def test_decodes_byte_order_marks(self, sandbox: Path, encoding: str) -> None:
        (sandbox / "bom.txt").write_bytes("Café Python".encode(encoding))

        result = fs_tools.read_file("bom.txt")

        assert result["content"] == "Café Python" and result["warning"] is None

    def test_falls_back_to_windows_1252_with_warning(self, sandbox: Path) -> None:
        (sandbox / "legacy.txt").write_bytes("Café – Python".encode("cp1252"))

        result = fs_tools.read_file("legacy.txt")

        assert result["content"] == "Café – Python"
        assert "Windows-1252" in result["warning"]

    def test_normalizes_line_endings(self, sandbox: Path) -> None:
        (sandbox / "crlf.txt").write_bytes(b"one\r\ntwo\rthree")

        assert fs_tools.read_file("crlf.txt")["content"] == "one\ntwo\nthree"

    def test_binary_data_in_text_file_is_rejected(self, sandbox: Path) -> None:
        (sandbox / "image.txt").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00junk")

        assert_failed(fs_tools.read_file("image.txt"), "binary")

    def test_corrupt_pdf_returns_error(self, sandbox: Path) -> None:
        (sandbox / "broken.pdf").write_bytes(b"%PDF-1.7\nthis is not really a pdf")

        assert_failed(fs_tools.read_file("broken.pdf"), "pdf")

    def test_zero_byte_pdf_returns_error(self, sandbox: Path) -> None:
        (sandbox / "zero.pdf").write_bytes(b"")

        assert_failed(fs_tools.read_file("zero.pdf"), "empty")

    def test_password_protected_pdf_returns_error(self, sandbox: Path) -> None:
        make_pdf(sandbox / "locked.pdf", [["Secret resume"]], user_password="hunter2")

        assert_failed(fs_tools.read_file("locked.pdf"), "password")

    def test_permissions_only_encrypted_pdf_is_read(self, sandbox: Path) -> None:
        make_pdf(sandbox / "owner_only.pdf", [["Readable despite encryption"]], user_password="")

        result = fs_tools.read_file("owner_only.pdf")

        assert result["success"] is True and "Readable despite encryption" in result["content"]

    def test_pdf_without_text_layer_warns_about_scans(self, sandbox: Path) -> None:
        make_pdf(sandbox / "scan.pdf", [[]])

        result = fs_tools.read_file("scan.pdf")

        assert result["success"] is True and result["content"] == ""
        assert "scanned" in result["warning"]

    def test_corrupt_docx_returns_error(self, sandbox: Path) -> None:
        (sandbox / "broken.docx").write_text("definitely not a zip", encoding="utf-8")

        assert_failed(fs_tools.read_file("broken.docx"), "docx")

    def test_zip_that_is_not_a_word_document_returns_error(self, sandbox: Path) -> None:
        with zipfile.ZipFile(sandbox / "fake.docx", "w") as archive:
            archive.writestr("hello.txt", "hi")

        assert_failed(fs_tools.read_file("fake.docx"), "docx")

    def test_pdf_with_wrong_extension_is_read_by_content(self, sandbox: Path) -> None:
        make_pdf(sandbox / "actually_pdf.txt", [["Mislabelled resume"]])

        result = fs_tools.read_file("actually_pdf.txt")

        assert result["success"] is True and "Mislabelled resume" in result["content"]
        assert "PDF" in result["warning"]

    def test_encrypted_docx_is_not_misidentified_as_legacy_doc(self, sandbox: Path) -> None:
        # Office encryption uses a CFB container with an EncryptedPackage stream.
        data = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
        data += "EncryptedPackage".encode("utf-16-le")
        (sandbox / "protected.docx").write_bytes(data)

        result = fs_tools.read_file("protected.docx")

        assert_failed(result, "password-protected")
        assert "legacy" not in result["error"]

    @pytest.mark.parametrize("name", ["old.doc", "old.docx"])
    def test_legacy_word_files_are_rejected_with_a_hint(self, sandbox: Path, name: str) -> None:
        (sandbox / name).write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)

        assert_failed(fs_tools.read_file(name), ".docx")

    @pytest.mark.parametrize("name", ["photo.png", "README", "script.py"])
    def test_unsupported_types_list_supported_formats(self, sandbox: Path, name: str) -> None:
        (sandbox / name).write_text("data", encoding="utf-8")

        assert_failed(fs_tools.read_file(name), "Supported types")

    def test_missing_file_suggests_closest_name(self, sandbox: Path) -> None:
        (sandbox / "resumes").mkdir()
        (sandbox / "resumes" / "resume_john_doe.pdf").write_bytes(b"%PDF-1.4")

        result = fs_tools.read_file("resumes/resume_jon_doe.pdf")

        assert_failed(result, "not found")
        assert "Did you mean 'resumes/resume_john_doe.pdf'?" in result["error"]

    def test_mistyped_folder_suggests_the_full_corrected_path(self, sandbox: Path) -> None:
        (sandbox / "resumes").mkdir()
        (sandbox / "resumes" / "cv.txt").write_text("x", encoding="utf-8")

        result = fs_tools.read_file("data/../resumess/cv.txt")

        assert "Did you mean 'resumes/cv.txt'?" in result["error"]

    def test_folder_is_not_a_file(self, sandbox: Path) -> None:
        (sandbox / "resumes").mkdir()

        assert_failed(fs_tools.read_file("resumes"), "folder")

    def test_file_over_size_limit_is_rejected(self, sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fs_tools, "MAX_READ_BYTES", 10)
        (sandbox / "big.txt").write_text("x" * 11, encoding="utf-8")

        assert_failed(fs_tools.read_file("big.txt"), "too large")

    @pytest.mark.skipif(os.name == "nt" or getattr(os, "geteuid", lambda: -1)() == 0, reason="permission bits are not enforced on Windows or for root")
    def test_unreadable_file_returns_permission_error(self, sandbox: Path) -> None:
        target = sandbox / "private.txt"
        target.write_text("x", encoding="utf-8")
        target.chmod(0)
        try:
            assert_failed(fs_tools.read_file("private.txt"), "permission denied")
        finally:
            target.chmod(stat.S_IRUSR | stat.S_IWUSR)

    @pytest.mark.parametrize("bad", [None, 42, ["a.txt"], "", "   "])
    def test_invalid_paths_return_errors(self, sandbox: Path, bad: object) -> None:
        result = fs_tools.read_file(bad)  # type: ignore[arg-type]

        assert result["success"] is False and result["error"]

    def test_null_byte_in_path_returns_error(self, sandbox: Path) -> None:
        assert_failed(fs_tools.read_file("notes\x00.txt"), "null byte")

    def test_accepts_path_objects_and_absolute_paths_inside_the_root(self, sandbox: Path) -> None:
        (sandbox / "notes.txt").write_text("inside", encoding="utf-8")

        assert fs_tools.read_file(sandbox / "notes.txt")["content"] == "inside"
        assert fs_tools.read_file(str(sandbox / "notes.txt"))["filepath"] == "notes.txt"

    def test_unexpected_errors_are_reported_not_raised(self, sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (sandbox / "notes.txt").write_text("x", encoding="utf-8")
        monkeypatch.setattr(fs_tools.extractors, "extract", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))

        assert_failed(fs_tools.read_file("notes.txt"), "unexpected error")

    def test_text_that_merely_starts_like_a_pdf_is_still_text(self, sandbox: Path) -> None:
        (sandbox / "note.txt").write_text("%PDF- is the PDF signature. 5 years Python.", encoding="utf-8")
        (sandbox / "fake.txt").write_text("%PDF-1.7 looks like a header but is just text", encoding="utf-8")

        assert "5 years Python" in fs_tools.read_file("note.txt")["content"]
        assert fs_tools.read_file("fake.txt")["content"].startswith("%PDF-1.7 looks like")

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
    def test_named_pipe_is_refused_without_hanging(self, sandbox: Path) -> None:
        os.mkfifo(sandbox / "pipe.txt")

        assert_failed(fs_tools.read_file("pipe.txt"), "not a regular file")

    def test_pages_read_is_reported_when_the_page_cap_applies(self, sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fs_tools.extractors, "MAX_PDF_PAGES", 1)
        make_pdf(sandbox / "long.pdf", [["Page one"], ["Page two"]])

        result = fs_tools.read_file("long.pdf")

        assert result["metadata"]["page_count"] == 2 and result["metadata"]["pages_read"] == 1
        assert "Page two" not in result["content"] and "first 1 of 2 pages" in result["warning"]

    def test_slow_pdf_is_cut_off_instead_of_hanging(self, sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        make_pdf(sandbox / "bomb.pdf", [["Page one"], ["Page two"], ["Page three"]])
        monkeypatch.setattr(fs_tools.extractors, "PDF_TIME_LIMIT_SECONDS", 0.05)
        monkeypatch.setattr("pypdf.PageObject.extract_text", lambda *_args, **_kwargs: time.sleep(0.2) or "slow")

        started = time.monotonic()
        result = fs_tools.read_file("bomb.pdf")

        assert time.monotonic() - started < 1.0
        assert result["success"] is True and "took more than" in result["warning"]
        assert result["metadata"]["pages_read"] < 3


# ------------------------------------------------------------------------ sandbox


class TestSandbox:
    @pytest.mark.parametrize("path", ["../outside/secret.txt", "a/../../outside/secret.txt", "~/.ssh/id_rsa", "/etc/passwd"])
    def test_paths_outside_the_root_are_denied(self, sandbox: Path, path: str) -> None:
        result = fs_tools.read_file(path)

        assert_failed(result, "access denied")
        assert not leaks_secret(result)

    def test_symlinked_file_escape_is_denied(self, sandbox: Path) -> None:
        (sandbox / "innocent.txt").symlink_to(sandbox.parent / "outside" / "secret.txt")

        result = fs_tools.read_file("innocent.txt")

        assert_failed(result, "access denied")
        assert not leaks_secret(result)

    def test_symlinked_folder_escape_is_denied_for_every_tool(self, sandbox: Path) -> None:
        (sandbox / "linked").symlink_to(sandbox.parent / "outside", target_is_directory=True)

        results = [
            fs_tools.read_file("linked/secret.txt"),
            fs_tools.search_in_file("linked/secret.txt", "SECRET"),
            fs_tools.write_file("linked/new.txt", "x"),
        ]

        assert all(result["success"] is False for result in results)
        assert fs_tools.list_files("linked")[0]["error"].startswith("Access denied")
        assert not (sandbox.parent / "outside" / "new.txt").exists()
        assert not any(leaks_secret(result) for result in results)

    @pytest.mark.parametrize("path", [".env", ".git/config", "notes/.hidden.txt"])
    def test_hidden_files_are_denied(self, sandbox: Path, path: str) -> None:
        target = sandbox / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ANTHROPIC_API_KEY=fixture-secret-key", encoding="utf-8")

        result = fs_tools.read_file(path)

        assert_failed(result, "hidden")
        assert "fixture-secret-key" not in json.dumps(result)

    def test_symlink_to_hidden_file_is_denied(self, sandbox: Path) -> None:
        (sandbox / ".env").write_text("ANTHROPIC_API_KEY=fixture-secret-key", encoding="utf-8")
        (sandbox / "config.txt").symlink_to(sandbox / ".env")

        assert_failed(fs_tools.read_file("config.txt"), "hidden")

    def test_root_comes_from_the_environment_variable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "configured"
        root.mkdir()
        (root / "here.txt").write_text("configured root", encoding="utf-8")
        monkeypatch.setenv("FS_ASSISTANT_ROOT", str(root))
        monkeypatch.chdir(tmp_path)

        assert fs_tools.read_file("here.txt")["content"] == "configured root"
        assert fs_tools.sandbox_root() == os.path.realpath(root)


# --------------------------------------------------------------------- list_files


@pytest.fixture
def folder(sandbox: Path) -> Path:
    resumes = sandbox / "resumes"
    resumes.mkdir()
    for name in ("b_resume.PDF", "a_resume.pdf", "c_resume.docx", "notes.txt"):
        (resumes / name).write_bytes(b"x" * 3)
    return resumes


class TestListFiles:
    def test_lists_files_sorted_with_metadata(self, folder: Path) -> None:
        result = fs_tools.list_files("resumes")

        assert [item["name"] for item in result] == ["a_resume.pdf", "b_resume.PDF", "c_resume.docx", "notes.txt"]
        first = result[0]
        assert first["path"] == "resumes/a_resume.pdf"
        assert (first["extension"], first["size_bytes"]) == (".pdf", 3)
        assert datetime.fromisoformat(first["modified"])

    @pytest.mark.parametrize("extension", [".pdf", "pdf", ".PDF", "*.pdf", " .pdf "])
    def test_extension_filter_accepts_common_forms(self, folder: Path, extension: str) -> None:
        names = [item["name"] for item in fs_tools.list_files("resumes", extension)]

        assert names == ["a_resume.pdf", "b_resume.PDF"]

    def test_several_extensions_at_once(self, folder: Path) -> None:
        names = [item["name"] for item in fs_tools.list_files("resumes", ".pdf, docx")]

        assert names == ["a_resume.pdf", "b_resume.PDF", "c_resume.docx"]

    @pytest.mark.parametrize("extension", [None, "", "*", ".*", "None", "null", "all", "any", " ALL "])
    def test_empty_or_wildcard_filter_lists_everything(self, folder: Path, extension: str | None) -> None:
        assert len(fs_tools.list_files("resumes", extension)) == 4

    def test_skips_hidden_files_lock_files_and_folders(self, folder: Path) -> None:
        (folder / ".DS_Store").write_bytes(b"x")
        (folder / "~$c_resume.docx").write_bytes(b"x")
        (folder / "archive").mkdir()

        assert len(fs_tools.list_files("resumes")) == 4

    def test_empty_folder_gives_empty_list(self, sandbox: Path) -> None:
        (sandbox / "empty").mkdir()

        assert fs_tools.list_files("empty") == []

    def test_missing_folder_returns_error_list_with_suggestion(self, folder: Path) -> None:
        result = fs_tools.list_files("data/resumess")

        assert len(result) == 1 and set(result[0]) == {"error"}
        assert "Did you mean 'resumes'" in result[0]["error"]

    def test_file_instead_of_folder_returns_error(self, folder: Path) -> None:
        assert "not a folder" in fs_tools.list_files("resumes/notes.txt")[0]["error"]

    @pytest.mark.parametrize("directory", ["", ".", "./"])
    def test_root_can_be_listed(self, folder: Path, directory: str) -> None:
        (folder.parent / "top.txt").write_text("x", encoding="utf-8")

        assert [item["path"] for item in fs_tools.list_files(directory)] == ["top.txt"]

    def test_outside_folder_is_denied(self, sandbox: Path) -> None:
        result = fs_tools.list_files("../outside")

        assert result[0]["error"].startswith("Access denied") and not leaks_secret(result)

    def test_symlinks_that_escape_or_dangle_are_not_listed(self, folder: Path) -> None:
        (folder / "escape.txt").symlink_to(folder.parent.parent / "outside" / "secret.txt")
        (folder / "dangling.txt").symlink_to(folder / "missing.txt")
        (folder / "inside_link.txt").symlink_to(folder / "notes.txt")

        names = [item["name"] for item in fs_tools.list_files("resumes")]

        assert "escape.txt" not in names and "dangling.txt" not in names
        assert "inside_link.txt" in names

    @pytest.mark.parametrize("extension", [5, "../pdf"])
    def test_invalid_extension_filter_returns_error(self, folder: Path, extension: object) -> None:
        assert "error" in fs_tools.list_files("resumes", extension)[0]  # type: ignore[arg-type]

    def test_unexpected_errors_are_reported_not_raised(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fs_tools.os, "scandir", lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))

        assert "Unexpected error" in fs_tools.list_files("resumes")[0]["error"]


# --------------------------------------------------------------------- write_file


class TestWriteFile:
    def test_creates_missing_folders_and_counts_utf8_bytes(self, sandbox: Path) -> None:
        result = fs_tools.write_file("outputs/2026/summaries/john.txt", "Done ✓")

        assert result == {"success": True, "filepath": "outputs/2026/summaries/john.txt", "bytes_written": 8,
                          "overwritten": False, "warning": None, "error": None}
        assert (sandbox / "outputs/2026/summaries/john.txt").read_text(encoding="utf-8") == "Done ✓"

    def test_overwrites_existing_file_and_says_so(self, sandbox: Path) -> None:
        fs_tools.write_file("notes.md", "first")

        result = fs_tools.write_file("notes.md", "second")

        assert result["overwritten"] is True
        assert (sandbox / "notes.md").read_text(encoding="utf-8") == "second"

    @pytest.mark.parametrize("path", ["../outside/new.txt", "../../etc/passwd.txt", "/tmp/evil.txt", "a/../../escape.md"])
    def test_paths_outside_the_root_are_denied(self, sandbox: Path, path: str) -> None:
        before = sorted(os.listdir(sandbox.parent / "outside"))

        assert_failed(fs_tools.write_file(path, "pwned"), "access denied")
        assert sorted(os.listdir(sandbox.parent / "outside")) == before

    @pytest.mark.parametrize("path", [".env", ".git/hooks/pre-commit.txt", "notes/.secret.md"])
    def test_hidden_targets_are_denied(self, sandbox: Path, path: str) -> None:
        assert_failed(fs_tools.write_file(path, "x"), "hidden")
        assert not (sandbox / path).exists()

    @pytest.mark.parametrize("path", ["evil.py", "run.sh", "hook.pth", "summary.pdf", "Makefile", "page.html"])
    def test_only_plain_text_formats_can_be_written(self, sandbox: Path, path: str) -> None:
        assert_failed(fs_tools.write_file(path, "x"), "refusing")
        assert not (sandbox / path).exists()

    @pytest.mark.parametrize("path", ["summaries", "summaries/"])
    def test_folder_targets_are_rejected(self, sandbox: Path, path: str) -> None:
        (sandbox / "summaries").mkdir()

        assert_failed(fs_tools.write_file(path, "x"), "is a folder")

    def test_parent_that_is_a_file_is_reported(self, sandbox: Path) -> None:
        (sandbox / "notes.txt").write_text("x", encoding="utf-8")

        result = fs_tools.write_file("notes.txt/child.txt", "x")

        assert result["success"] is False and "notes.txt" in result["error"]

    @pytest.mark.parametrize("content", [None, 42, b"bytes", ["list"]])
    def test_content_must_be_a_string(self, sandbox: Path, content: object) -> None:
        assert_failed(fs_tools.write_file("x.txt", content), "content must be a string")  # type: ignore[arg-type]

    def test_lone_surrogates_are_replaced_instead_of_crashing(self, sandbox: Path) -> None:
        result = fs_tools.write_file("odd.txt", "bad \ud800 char")

        assert result["success"] is True
        assert (sandbox / "odd.txt").read_text(encoding="utf-8") == "bad ? char"

    def test_empty_content_creates_an_empty_file(self, sandbox: Path) -> None:
        assert fs_tools.write_file("empty.txt", "")["bytes_written"] == 0
        assert (sandbox / "empty.txt").read_bytes() == b""

    def test_content_over_limit_is_rejected(self, sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fs_tools, "MAX_WRITE_BYTES", 4)

        assert_failed(fs_tools.write_file("big.txt", "12345"), "too large")

    @pytest.mark.parametrize("path", ["requirements.txt", "README.md", "resumes/resume_jane.txt"])
    def test_files_it_did_not_create_are_never_overwritten(self, sandbox: Path, path: str) -> None:
        target = sandbox / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original", encoding="utf-8")

        result = fs_tools.write_file(path, "evil-package==6.6.6")

        assert_failed(result, "wasn't created by this assistant")
        assert "_new" in result["error"]  # suggests a fresh name the model can use instead
        assert target.read_text(encoding="utf-8") == "original"

    def test_files_it_created_can_be_replaced_in_a_later_session(self, sandbox: Path) -> None:
        fs_tools.write_file("summaries/jane.md", "v1")

        result = fs_tools.write_file("summaries/jane.md", "v2")

        assert result["success"] is True and result["overwritten"] is True
        assert json.loads((sandbox / fs_tools.WRITE_LOG_NAME).read_text())["files"] == ["summaries/jane.md"]

    def test_write_record_is_hidden_from_the_model(self, sandbox: Path) -> None:
        fs_tools.write_file("summaries/jane.md", "v1")

        assert_failed(fs_tools.read_file(fs_tools.WRITE_LOG_NAME), "hidden")
        assert_failed(fs_tools.write_file(fs_tools.WRITE_LOG_NAME, '{"files": ["README.md"]}'), "hidden")
        assert [item["name"] for item in fs_tools.list_files(".")] == []

    def test_damaged_write_record_protects_every_file(self, sandbox: Path) -> None:
        fs_tools.write_file("notes.md", "v1")
        (sandbox / fs_tools.WRITE_LOG_NAME).write_text("not json", encoding="utf-8")

        assert_failed(fs_tools.write_file("notes.md", "v2"), "wasn't created by this assistant")
        assert (sandbox / "notes.md").read_text(encoding="utf-8") == "v1"

    def test_missing_root_is_never_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("FS_ASSISTANT_ROOT", str(missing))

        assert_failed(fs_tools.write_file("a/b/c.txt", "x"), "doesn't exist")
        assert not missing.exists()

    def test_read_only_file_is_not_overwritten(self, sandbox: Path) -> None:
        target = sandbox / "locked.txt"
        fs_tools.write_file("locked.txt", "keep me")
        target.chmod(stat.S_IRUSR)
        try:
            assert_failed(fs_tools.write_file("locked.txt", "replaced"), "read-only")
            assert target.read_text(encoding="utf-8") == "keep me"
        finally:
            target.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_symlink_target_pointing_outside_is_denied(self, sandbox: Path) -> None:
        (sandbox / "link.txt").symlink_to(sandbox.parent / "outside" / "secret.txt")

        assert_failed(fs_tools.write_file("link.txt", "overwrite"), "access denied")
        assert (sandbox.parent / "outside" / "secret.txt").read_text(encoding="utf-8") == SECRET

    def test_write_is_atomic_and_uses_normal_permissions(self, sandbox: Path) -> None:
        fs_tools.write_file("report.md", "# Report")

        assert sorted(os.listdir(sandbox)) == [fs_tools.WRITE_LOG_NAME, "report.md"]  # no temp files left
        assert stat.S_IMODE((sandbox / "report.md").stat().st_mode) == 0o666 & ~_umask()

    def test_failed_write_keeps_the_old_file_and_cleans_up(self, sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fs_tools.write_file("report.md", "old")

        def fail(*_args: object, **_kwargs: object) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(fs_tools.os, "replace", fail)
        result = fs_tools.write_file("report.md", "new")

        assert_failed(result, "no space left")
        assert (sandbox / "report.md").read_text(encoding="utf-8") == "old"
        assert sorted(os.listdir(sandbox)) == [fs_tools.WRITE_LOG_NAME, "report.md"]


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


@pytest.mark.skipif(not fs_tools._WALK_BY_FD, reason="race-free opening needs dir_fd support")
class TestSymlinkSwapRace:
    """A symlink swapped in after the path check (a TOCTOU race) must still be refused.

    Pretending realpath saw the path before the swap reproduces the race deterministically.
    """

    @pytest.fixture
    def stale_realpath(self, sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(fs_tools.os.path, "realpath", os.path.abspath)
        return sandbox

    def test_swapped_parent_folder_is_refused_for_every_tool(self, stale_realpath: Path) -> None:
        outside = stale_realpath.parent / "outside"
        (stale_realpath / "linked").symlink_to(outside, target_is_directory=True)

        results = [fs_tools.read_file("linked/secret.txt"), fs_tools.write_file("linked/planted.txt", "x")]

        assert all(result["success"] is False for result in results)
        assert "error" in fs_tools.list_files("linked")[0]
        assert not (outside / "planted.txt").exists()
        assert not any(leaks_secret(result) for result in results)

    def test_swapped_file_is_refused(self, stale_realpath: Path) -> None:
        (stale_realpath / "innocent.txt").symlink_to(stale_realpath.parent / "outside" / "secret.txt")

        result = fs_tools.read_file("innocent.txt")

        assert_failed(result, "changed while it was being opened")
        assert not leaks_secret(result)


# ----------------------------------------------------------------- search_in_file


@pytest.fixture
def resume(sandbox: Path) -> Path:
    text = (
        "Jane Roe\nSenior Python developer with Django/Python and python 3 experience.\n"
        "Skills: JavaScript, TypeScript, C++, C#, .NET, (Python) scripting\n"
        "Also writes Java services.\n"
    )
    path = sandbox / "jane.txt"
    path.write_text(text, encoding="utf-8")
    return path


class TestSearchInFile:
    @pytest.mark.parametrize("keyword", ["python", "PYTHON", "PyThOn"])
    def test_search_is_case_insensitive(self, resume: Path, keyword: str) -> None:
        result = fs_tools.search_in_file("jane.txt", keyword)

        assert result["success"] is True and result["match_count"] == 4
        assert {match["match"].lower() for match in result["matches"]} == {"python"}

    def test_matches_include_line_and_trimmed_context(self, resume: Path) -> None:
        match = fs_tools.search_in_file("jane.txt", "services")["matches"][0]

        assert match["line"] == 4
        assert match["context"].startswith("...") and match["context"].endswith("services.")
        assert "\n" not in match["context"]

    def test_whole_word_tells_java_from_javascript(self, resume: Path) -> None:
        substring = fs_tools.search_in_file("jane.txt", "Java")
        whole_word = fs_tools.search_in_file("jane.txt", "Java", whole_word=True)

        assert substring["match_count"] == 2
        assert whole_word["match_count"] == 1 and "Java services" in whole_word["matches"][0]["context"]

    @pytest.mark.parametrize("keyword", ["C++", "C#", ".NET", "(Python)"])
    def test_regex_characters_are_matched_literally(self, resume: Path, keyword: str) -> None:
        assert fs_tools.search_in_file("jane.txt", keyword)["match_count"] == 1

    def test_phrase_matches_across_line_breaks(self, sandbox: Path) -> None:
        make_pdf(sandbox / "ml.pdf", [["Built machine", "learning pipelines"]])

        result = fs_tools.search_in_file("ml.pdf", "machine learning")

        assert result["match_count"] == 1 and result["matches"][0]["match"] == "machine learning"

    def test_searches_docx_files(self, sandbox: Path) -> None:
        make_docx(sandbox / "cv.docx", ["Kotlin and Swift"], table=[["Languages", "Go, PYTHON"]])

        assert fs_tools.search_in_file("cv.docx", "python")["match_count"] == 1

    def test_compatibility_characters_match_plain_text(self, sandbox: Path) -> None:
        (sandbox / "ligature.txt").write_text("Highly proﬁcient in Ｐｙｔｈｏｎ", encoding="utf-8")

        assert fs_tools.search_in_file("ligature.txt", "proficient")["match_count"] == 1
        assert fs_tools.search_in_file("ligature.txt", "python")["match_count"] == 1

    def test_match_list_is_capped_but_count_is_complete(self, sandbox: Path) -> None:
        (sandbox / "many.txt").write_text("go " * 120, encoding="utf-8")

        result = fs_tools.search_in_file("many.txt", "go")

        assert result["match_count"] == 120
        assert len(result["matches"]) == fs_tools.MAX_MATCHES and result["truncated"] is True

    def test_no_match_is_a_successful_empty_result(self, resume: Path) -> None:
        result = fs_tools.search_in_file("jane.txt", "Haskell")

        assert result["success"] is True and result["match_count"] == 0 and result["matches"] == []

    @pytest.mark.parametrize("keyword", ["", "   ", None, 5, "x" * 201])
    def test_invalid_keywords_return_errors(self, resume: Path, keyword: object) -> None:
        result = fs_tools.search_in_file("jane.txt", keyword)  # type: ignore[arg-type]

        assert result["success"] is False and result["error"]

    def test_whole_word_must_be_boolean(self, resume: Path) -> None:
        assert_failed(fs_tools.search_in_file("jane.txt", "java", whole_word="yes"), "whole_word")  # type: ignore[arg-type]

    def test_read_errors_are_passed_through(self, resume: Path) -> None:
        result = fs_tools.search_in_file("jan.txt", "python")

        assert_failed(result, "not found")
        assert "jane.txt" in result["error"]

    def test_unexpected_errors_are_reported_not_raised(self, resume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fs_tools, "_find_matches", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))

        assert_failed(fs_tools.search_in_file("jane.txt", "python"), "unexpected error")
