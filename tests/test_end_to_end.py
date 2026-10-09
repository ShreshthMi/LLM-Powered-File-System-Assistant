"""Actual CLI + SDK + HTTP + file tools, with scripted model decisions and no API usage."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import generate_resumes
from stub_anthropic import StubAnthropic, say, tool_call

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A copy of the CLI and generated resumes, without the user's .env or settings."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("llm_file_assistant.py", "fs_tools.py", "extractors.py"):
        shutil.copy2(PROJECT / name, workspace / name)
    generate_resumes.generate_all(workspace / "resumes")
    return workspace


def run_cli(host: str, root: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the real CLI with explicit settings and a dummy credential at the stub endpoint."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env.update(ANTHROPIC_API_KEY="offline-test-key", ANTHROPIC_BASE_URL=host,
               ANTHROPIC_MODEL="claude-opus-5-5", ANTHROPIC_EFFORT="medium",
               FS_ASSISTANT_ROOT=str(root), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(root / "llm_file_assistant.py"), *args], env=env,
                          input=stdin, capture_output=True, text=True, timeout=60, check=False, cwd=root)


def tool_results(request: dict[str, Any]) -> list[Any]:
    """Deserialize all results from the most recent tool round."""
    return [json.loads(block["content"]) for block in request["messages"][-1]["content"]]


def search_every_file(request: dict[str, Any]) -> dict[str, Any]:
    """Search all files returned by the real list_files tool."""
    return say("", *[tool_call("search_in_file", filepath=f["path"], keyword="python")
                     for f in tool_results(request)[0]])


def report_matches(request: dict[str, Any]) -> dict[str, Any]:
    """Use match context to exclude the planted Monty Python hobby."""
    hits = [r["filepath"] for r in tool_results(request)
            if any("Monty Python" not in m["context"] for m in r["matches"])]
    return say("Python experience: " + ", ".join(hits))


def test_find_resumes_mentioning_python(root: Path) -> None:
    with StubAnthropic([say("", tool_call("list_files", directory="resumes")),
                        search_every_file, report_matches]) as stub:
        result = run_cli(stub.host, root, "Find resumes mentioning Python experience")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("🔧 search_in_file") == 8
    answer = result.stdout.split("Assistant: ", 1)[1]
    for name in ("john_doe.pdf", "priya_sharma.pdf", "elena_rodriguez.docx", "david_kim.txt"):
        assert name in answer
    assert "sofia_novak" not in answer and "john_smith" not in answer


def test_read_all_resumes(root: Path) -> None:
    def read_every_file(request: dict[str, Any]) -> dict[str, Any]:
        return say("", *[tool_call("read_file", filepath=f["path"]) for f in tool_results(request)[0]])

    def report_every_file(request: dict[str, Any]) -> dict[str, Any]:
        results = tool_results(request)
        assert len(results) == 8 and all(r["success"] and r["content"] for r in results)
        return say("Read all eight resumes: " + ", ".join(r["filepath"] for r in results))

    with StubAnthropic([say("", tool_call("list_files", directory="resumes")),
                        read_every_file, report_every_file]) as stub:
        result = run_cli(stub.host, root, "Read all resumes in the resumes folder")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("🔧 read_file") == 8 and "Read all eight" in result.stdout


def test_create_summary_file(root: Path) -> None:
    def summarise(request: dict[str, Any]) -> dict[str, Any]:
        lines = " / ".join(tool_results(request)[0]["content"].splitlines()[:2])
        return say("", tool_call("write_file", filepath="summaries/resume_john_doe_summary.md",
                                 content=f"# Summary\n\n{lines}\n"))
    with StubAnthropic([say("", tool_call("read_file", filepath="resumes/resume_john_doe.pdf")),
                        summarise, say("Saved summaries/resume_john_doe_summary.md.")]) as stub:
        result = run_cli(stub.host, root, "Create a summary file for resume_john_doe.pdf")
    assert result.returncode == 0, result.stderr
    assert "John Doe" in (root / "summaries/resume_john_doe_summary.md").read_text()
    assert "→ wrote" in result.stdout


def test_anthropic_wire_format_and_append_only_history(root: Path) -> None:
    with StubAnthropic([say("", tool_call("list_files", directory="resumes")), say("8 resumes.")]) as stub:
        result = run_cli(stub.host, root, "--quiet", "List the resumes")
    assert result.returncode == 0, result.stderr
    first, second = stub.requests
    assert first["model"] == "claude-opus-5-5" and first["max_tokens"] == 16000
    assert first["output_config"] == {"effort": "medium"}
    assert all(t["strict"] and not t["input_schema"]["additionalProperties"] for t in first["tools"])
    assert not {"thinking", "temperature", "tool_choice"} & first.keys()
    assert second["messages"][:len(first["messages"])] == first["messages"]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][1]["content"][0]["id"] == second["messages"][2]["content"][0]["tool_use_id"]


def test_interactive_history_and_new_conversation(root: Path) -> None:
    with StubAnthropic([say("Hello."), say("You said hi."), say("Fresh chat.")]) as stub:
        result = run_cli(stub.host, root, stdin="hi\nwhat did I say?\n/new\nhello\nexit\n")
    assert result.returncode == 0, result.stderr
    assert "Started a new conversation." in result.stdout and "Goodbye!" in result.stdout
    assert len(stub.requests[1]["messages"]) == 3
    assert stub.requests[2]["messages"] == [{"role": "user", "content": "hello"}]


def test_hijacked_model_cannot_escape_plant_code_or_wipe_resumes(root: Path) -> None:
    attack = say("", tool_call("read_file", filepath="../../etc/passwd"),
                 tool_call("write_file", filepath="evil.py", content="import os"),
                 tool_call("write_file", filepath="resumes/resume_david_kim.txt", content="wiped"))
    with StubAnthropic([attack, say("The tools denied all three requests.")]) as stub:
        result = run_cli(stub.host, root, "Summarise the resumes")
    errors = [r["error"] for r in tool_results(stub.requests[1])]
    assert "outside the allowed folder" in errors[0] and "Refusing" in errors[1]
    assert "wasn't created" in errors[2]
    assert all(b["is_error"] for b in stub.requests[1]["messages"][-1]["content"])
    assert not (root / "evil.py").exists()
    assert "wiped" not in (root / "resumes/resume_david_kim.txt").read_text()
    assert result.returncode == 0


def test_incomplete_write_is_not_executed(root: Path) -> None:
    partial = say("", tool_call("write_file", filepath="half.md", content="unfinished"))
    partial["stop_reason"] = "max_tokens"
    with StubAnthropic([partial, say("Try a shorter summary.")]) as stub:
        result = run_cli(stub.host, root, "Write a summary")
    assert result.returncode == 0 and not (root / "half.md").exists()
    assert stub.requests[1]["messages"][-1]["content"][0]["is_error"]


@pytest.mark.parametrize(("status", "message"), [(400, "rejected"), (401, "Authentication failed"),
                                                (403, "Permission denied"), (404, "Model not found"),
                                                (429, "Rate limit"), (529, "temporarily unavailable")])
def test_api_errors_are_friendly(root: Path, status: int, message: str) -> None:
    with StubAnthropic([], status=status) as stub:
        result = run_cli(stub.host, root, "hello")
    assert result.returncode == 1 and message in result.stderr and "Traceback" not in result.stderr
    assert "offline-test-key" not in result.stdout + result.stderr


def test_missing_key_exits_before_calling_api(root: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env.update(ANTHROPIC_API_KEY="", ANTHROPIC_PROFILE="nonexistent-offline-test-profile")
    result = subprocess.run([sys.executable, str(root / "llm_file_assistant.py"), "hello"],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and "ANTHROPIC_API_KEY" in result.stderr
