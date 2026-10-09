"""Integration tests: the sample resumes generate, read back cleanly and contain the planted cases."""

from __future__ import annotations

from pathlib import Path

import pytest

import fs_tools
import generate_resumes


@pytest.fixture
def samples(sandbox: Path) -> list[str]:
    generate_resumes.generate_all(sandbox / "resumes")
    return [entry["path"] for entry in fs_tools.list_files("resumes")]


def test_generates_three_pdf_three_docx_and_two_txt(samples: list[str]) -> None:
    assert len(samples) == 8
    assert sorted(Path(path).suffix for path in samples) == [".docx"] * 3 + [".pdf"] * 3 + [".txt"] * 2
    assert "resumes/resume_john_doe.pdf" in samples


def test_every_sample_reads_cleanly(samples: list[str]) -> None:
    for path in samples:
        result = fs_tools.read_file(path)
        assert result["success"] is True and result["warning"] is None, path
        assert result["metadata"]["word_count"] > 100, path
        assert "@example.com" in result["content"], path


def test_python_mentions_use_varied_spellings(samples: list[str]) -> None:
    mentions = {Path(path).stem for path in samples if fs_tools.search_in_file(path, "python")["match_count"]}

    assert mentions == {"resume_john_doe", "resume_priya_sharma", "resume_elena_rodriguez", "resume_david_kim",
                        "resume_sofia_novak"}  # Sofia's is only "Monty Python" - the model must spot that


def test_whole_word_java_excludes_javascript(samples: list[str]) -> None:
    java = {Path(path).stem for path in samples if fs_tools.search_in_file(path, "Java", whole_word=True)["match_count"]}

    assert java == {"resume_marcus_chen", "resume_aisha_bello"}


def test_layout_edge_cases_survive_extraction(samples: list[str]) -> None:
    assert fs_tools.read_file("resumes/resume_priya_sharma.pdf")["metadata"]["page_count"] == 2
    elena = fs_tools.read_file("resumes/resume_elena_rodriguez.docx")["content"]
    assert elena.splitlines()[0].startswith("Elena Rodriguez | elena.rodriguez@example.com")  # page header
    assert "Languages | Go, PYTHON, Bash" in elena  # skills table
