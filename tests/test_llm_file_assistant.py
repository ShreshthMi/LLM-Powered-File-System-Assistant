"""Tests for the agent loop and CLI, using a scripted fake client (no network, no API key)."""

from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import anthropic
import httpx2
import pytest

import llm_file_assistant as app

SETTINGS = app.Settings(model="claude-opus-5-5", effort="medium")
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


# --------------------------------------------------------------------------- fakes


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(name: str, tool_input: dict[str, Any], block_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def response(*content: SimpleNamespace, stop: str = "end_turn", **extra: Any) -> SimpleNamespace:
    usage = SimpleNamespace(iterations=extra.pop("iterations", None))
    return SimpleNamespace(content=list(content), stop_reason=stop, model="claude-opus-5-5", usage=usage,
                           stop_details=extra.pop("stop_details", None))


class FakeClient:
    """Stands in for anthropic.Anthropic: returns scripted responses and records each request."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs))
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def status_error(cls: type[anthropic.APIStatusError], status: int, message: str = "boom") -> anthropic.APIStatusError:
    body = {"type": "error", "error": {"type": "error", "message": message}}
    return cls(message, response=httpx2.Response(status, request=REQUEST), body=body)


def run(client: FakeClient, question: str = "Hi", history: list | None = None) -> tuple[app.TurnOutcome, list[str]]:
    logs: list[str] = []
    outcome = app.run_turn(client, SETTINGS, history or [], question, logs.append)
    return outcome, logs


@pytest.fixture
def resumes(sandbox: Path) -> Path:
    folder = sandbox / "resumes"
    folder.mkdir()
    (folder / "resume_jane.txt").write_text("Jane - Python developer", encoding="utf-8")
    return folder


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real .env and API settings out of every test."""
    monkeypatch.setattr(app, "PROJECT_DIR", tmp_path)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL", "ANTHROPIC_EFFORT", "FS_ASSISTANT_ROOT",
                 "ANTHROPIC_BASE_URL", "ANTHROPIC_PROFILE", "ANTHROPIC_WORKSPACE_ID"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ the agent loop


class TestRunTurn:
    def test_general_question_is_answered_without_tools(self) -> None:
        client = FakeClient(response(text("Paris.")))

        outcome, logs = run(client, "What is the capital of France?")

        assert (outcome.answer, outcome.ok) == ("Paris.", True)
        assert [message["role"] for message in outcome.messages] == ["user", "assistant"]
        assert logs == []

    def test_runs_tool_calls_until_the_model_answers(self, resumes: Path) -> None:
        client = FakeClient(
            response(tool_use("list_files", {"directory": "resumes"}), stop="tool_use"),
            response(text("One resume: resume_jane.txt")),
        )

        outcome, logs = run(client, "List the resumes")

        assert outcome.answer == "One resume: resume_jane.txt"
        result_block = client.requests[1]["messages"][-1]["content"][0]
        assert result_block["tool_use_id"] == "toolu_1" and "is_error" not in result_block
        assert json.loads(result_block["content"])[0]["name"] == "resume_jane.txt"
        assert logs == ['🔧 list_files({"directory": "resumes"})', "   → 1 file"]

    def test_parallel_calls_return_every_result_in_one_message_in_order(self, resumes: Path) -> None:
        client = FakeClient(
            response(
                text("Checking both."),
                tool_use("search_in_file", {"filepath": "resumes/resume_jane.txt", "keyword": "python"}, "a"),
                tool_use("read_file", {"filepath": "resumes/missing.txt"}, "b"),
                tool_use("delete_everything", {}, "c"),
                tool_use("read_file", {"path": "resumes/resume_jane.txt"}, "d"),
                stop="tool_use",
            ),
            response(text("Done.")),
        )

        outcome, logs = run(client)

        results = client.requests[1]["messages"][-1]["content"]
        assert [block["tool_use_id"] for block in results] == ["a", "b", "c", "d"]
        assert [block.get("is_error", False) for block in results] == [False, True, True, True]
        assert "Unknown tool" in results[2]["content"] and "Invalid arguments" in results[3]["content"]
        assert logs[0] == "💬 Checking both." and outcome.ok

    def test_tool_call_cut_off_by_max_tokens_is_never_run(self, sandbox: Path) -> None:
        client = FakeClient(
            response(tool_use("write_file", {"filepath": "summary.md", "content": "half a summ"}), stop="max_tokens"),
            response(text("Retried with a shorter summary.")),
        )

        outcome, _ = run(client)

        assert not (sandbox / "summary.md").exists()
        result = client.requests[1]["messages"][-1]["content"][0]
        assert result["is_error"] is True and "cut off" in result["content"]
        assert outcome.ok

    def test_text_cut_off_by_max_tokens_is_kept_with_a_note(self) -> None:
        outcome, _ = run(FakeClient(response(text("A long answ"), stop="max_tokens")))

        assert outcome.answer.startswith("A long answ") and "cut off" in outcome.answer
        assert outcome.messages[-1]["role"] == "assistant"

    def test_max_tokens_before_any_text_drops_the_turn(self) -> None:
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": [text("ok")]}]

        outcome, _ = run(FakeClient(response(stop="max_tokens")), history=history)

        assert outcome.ok is False and outcome.messages == history

    def test_refusal_drops_the_turn_and_explains(self) -> None:
        refusal = response(stop="refusal", stop_details=SimpleNamespace(category="cyber"))

        outcome, _ = run(FakeClient(refusal), history=[])

        assert outcome.ok is False and outcome.messages == []
        assert "declined" in outcome.answer and "cyber" in outcome.answer

    def test_empty_reply_keeps_tool_results_but_adds_no_empty_assistant_turn(self, resumes: Path) -> None:
        client = FakeClient(
            response(tool_use("list_files", {"directory": "resumes"}), stop="tool_use"),
            response(stop="end_turn"),
        )

        outcome, _ = run(client)

        assert outcome.ok is False and "empty response" in outcome.answer
        assert [message["role"] for message in outcome.messages] == ["user", "assistant", "user"]

    def test_stops_after_the_round_limit(self, resumes: Path) -> None:
        looping = [response(tool_use("list_files", {"directory": "resumes"}), stop="tool_use")
                   for _ in range(app.MAX_TOOL_ROUNDS)]
        client = FakeClient(*looping)

        outcome, _ = run(client)

        assert outcome.ok is False and str(app.MAX_TOOL_ROUNDS) in outcome.answer
        assert len(client.requests) == app.MAX_TOOL_ROUNDS
        assert outcome.messages[-1]["content"][0]["type"] == "tool_result"

    def test_unknown_stop_reason_is_reported_with_the_text(self) -> None:
        outcome, _ = run(FakeClient(response(text("Partial"), stop="model_context_window_exceeded")))

        assert "Partial" in outcome.answer and "model_context_window_exceeded" in outcome.answer



class TestRequests:
    def test_request_follows_the_api_rules_for_this_model(self) -> None:
        client = FakeClient(response(text("Hi")))

        run(client)

        request = client.requests[0]
        assert request["model"] == "claude-opus-5-5" and request["max_tokens"] == app.MAX_TOKENS
        assert request["output_config"] == {"effort": "medium"}
        assert "betas" not in request and "fallbacks" not in request
        assert request["system"] == app.SYSTEM_PROMPT
        for forbidden in ("tool_choice", "thinking", "temperature", "top_p"):
            assert forbidden not in request
        assert request["messages"][-1]["role"] == "user"  # never an assistant prefill

    def test_every_tool_is_strict_and_matches_a_function(self) -> None:
        assert {tool["name"] for tool in app.TOOL_DEFINITIONS} == set(app.TOOLS)
        for tool in app.TOOL_DEFINITIONS:
            schema = tool["input_schema"]
            assert tool["strict"] is True and schema["additionalProperties"] is False
            assert set(schema["required"]) <= set(schema["properties"])

    def test_history_is_append_only_across_rounds_and_turns(self, resumes: Path) -> None:
        client = FakeClient(
            response(tool_use("list_files", {"directory": "resumes"}), stop="tool_use"),
            response(text("One resume.")),
            response(text("Jane writes Python.")),
        )
        first, _ = run(client, "List the resumes")

        run(client, "What does Jane do?", history=first.messages)

        sent = [request["messages"] for request in client.requests]
        for earlier, later in zip(sent, sent[1:]):
            assert later[: len(earlier)] == earlier
        assert all(request["system"] == app.SYSTEM_PROMPT and request["tools"] == app.TOOL_DEFINITIONS
                   for request in client.requests)

    def test_next_question_after_an_unfinished_turn_is_appended(self, resumes: Path) -> None:
        client = FakeClient(
            response(tool_use("list_files", {"directory": "resumes"}), stop="tool_use"),
            response(stop="end_turn"),
            response(text("Continuing.")),
        )
        unfinished, _ = run(client)

        run(client, "continue", history=unfinished.messages)

        assert client.requests[2]["messages"][: len(unfinished.messages)] == unfinished.messages
        assert client.requests[2]["messages"][-1] == {"role": "user", "content": "continue"}


# ------------------------------------------------------------------- error handling


@pytest.mark.parametrize(
    ("error", "fatal", "fragment"),
    [
        (status_error(anthropic.AuthenticationError, 401), True, "Authentication failed"),
        (status_error(anthropic.PermissionDeniedError, 403), True, "Permission denied"),
        (status_error(anthropic.NotFoundError, 404), True, "Model not found (404): 'claude-opus-5-5'"),
        (status_error(anthropic.RateLimitError, 429), False, "Rate limit"),
        (status_error(anthropic.RequestTooLargeError, 413), False, "too large"),
        (status_error(anthropic.BadRequestError, 400, "prompt is too long"), False, "prompt is too long"),
        (status_error(anthropic.UnprocessableEntityError, 422, "bad input"), False, "API error (422): bad input"),
        (status_error(anthropic.OverloadedError, 529), False, "temporarily unavailable (529)"),
        (status_error(anthropic.InternalServerError, 500), False, "temporarily unavailable (500)"),
        (anthropic.APITimeoutError(request=REQUEST), False, "timed out"),
        (anthropic.APIConnectionError(request=REQUEST), False, "Couldn't reach"),
        (RuntimeError("boom"), False, "Unexpected error: RuntimeError: boom"),
    ],
)
def test_api_errors_become_friendly_failures(error: Exception, fatal: bool, fragment: str) -> None:
    history = [{"role": "user", "content": "earlier"}]

    result = app.ask(FakeClient(error), SETTINGS, history, "Hi", lambda _line: None)

    assert isinstance(result, app.Failure)
    assert result.fatal is fatal and fragment in result.message
    assert history == [{"role": "user", "content": "earlier"}]


def test_error_midway_through_a_turn_leaves_history_untouched(resumes: Path) -> None:
    history: list = []
    client = FakeClient(
        response(tool_use("list_files", {"directory": "resumes"}), stop="tool_use"),
        status_error(anthropic.OverloadedError, 529),
    )

    result = app.ask(client, SETTINGS, history, "List the resumes", lambda _line: None)

    assert isinstance(result, app.Failure) and history == []


# ------------------------------------------------------------- tool-call plumbing


class TestToolPlumbing:
    def test_run_tool_rejects_non_object_arguments(self) -> None:
        result, is_error = app.run_tool("read_file", ["resumes/a.pdf"])

        assert is_error and "JSON object" in result["error"]

    def test_run_tool_survives_a_tool_that_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(app.TOOLS, "read_file", lambda filepath: 1 / 0)

        result, is_error = app.run_tool("read_file", {"filepath": "x"})

        assert is_error and "ZeroDivisionError" in result["error"]

    @pytest.mark.parametrize(
        ("result", "expected"),
        [({"success": False}, True), ({"success": True}, False), ([{"error": "x"}], True), ([], False), ("x", False)],
    )
    def test_is_error_result(self, result: Any, expected: bool) -> None:
        assert app.is_error_result(result) is expected

    def test_small_results_are_plain_json(self) -> None:
        assert app.serialize_result({"success": True, "name": "é"}) == '{"success": true, "name": "é"}'

    @pytest.mark.parametrize("content", ["x" * 50_000, 'line "quoted"\n' * 4_000, "é" * 30_000])
    def test_long_file_content_is_truncated_to_valid_json(self, content: str) -> None:
        result = {"success": True, "filepath": "big.txt", "content": content, "metadata": {"char_count": len(content)}}

        text_out = app.serialize_result(result)

        parsed = json.loads(text_out)
        assert len(text_out) <= app.MAX_TOOL_RESULT_CHARS
        assert parsed["truncated"] is True and "search_in_file" in parsed["truncation_note"]
        assert parsed["metadata"] == {"char_count": len(content)}

    def test_long_lists_keep_whole_entries_and_say_how_many_were_cut(self) -> None:
        entries = [{"name": f"resume_{index}.pdf", "path": f"resumes/resume_{index}.pdf"} for index in range(2_000)]

        parsed = json.loads(app.serialize_result(entries))

        assert "more entries not shown" in parsed[-1]["note"]
        assert all("name" in entry for entry in parsed[:-1]) and len(parsed) < 2_000

    def test_other_oversized_results_are_cut_with_a_marker(self) -> None:
        text_out = app.serialize_result({"success": False, "error": "e" * 30_000})

        assert text_out.endswith("more characters not shown]")

    def test_lone_surrogates_cannot_break_the_request(self) -> None:
        assert app.to_json({"name": "bad\udc80name"}).encode("utf-8")

    def test_log_line_shortens_long_arguments(self) -> None:
        line = app.describe_call("write_file", {"filepath": "a.md", "content": "y" * 500})

        assert '"filepath": "a.md"' in line and "(500 chars)" in line and len(line) < 200

    @pytest.mark.parametrize(
        ("name", "result", "expected"),
        [
            ("list_files", [{"name": "a"}, {"name": "b"}], "2 files"),
            ("list_files", [{"error": "Folder not found"}], "error: Folder not found"),
            ("read_file", {"success": True, "metadata": {"char_count": 1234, "page_count": 2}}, "read 1,234 characters, 2 pages"),
            ("read_file", {"success": True, "metadata": {"char_count": 0}, "warning": "empty"}, "read 0 characters (warning: empty)"),
            ("search_in_file", {"success": True, "match_count": 1}, "1 match"),
            ("search_in_file", {"success": True, "match_count": 3}, "3 matches"),
            ("write_file", {"success": True, "bytes_written": 5, "filepath": "x.md", "overwritten": True},
             "wrote 5 bytes to x.md (replaced the existing file)"),
            ("read_file", {"success": False, "error": "nope"}, "error: nope"),
            ("other", "plain", "done"),
        ],
    )
    def test_result_summaries(self, name: str, result: Any, expected: str) -> None:
        assert app.summarize_result(name, result) == expected


# --------------------------------------------------------------------- command line


def scripted_input(monkeypatch: pytest.MonkeyPatch, *lines: Any) -> None:
    """Feed lines to input(); an exception instance is raised instead of returned."""
    feed: Iterator[Any] = iter(lines)

    def fake_input(_prompt: str = "") -> str:
        item = next(feed)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(builtins, "input", fake_input)


def use_client(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    monkeypatch.setattr(app, "build_client", lambda: client)


class TestCommandLine:
    def test_help_exits_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            app.main(["--help"])

        assert exit_info.value.code == 0 and "interactive chat" in capsys.readouterr().out

    def test_empty_question_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert app.main([""]) == app.EXIT_USAGE
        assert "question is empty" in capsys.readouterr().err

    def test_missing_api_key_explains_how_to_fix_it(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(app.anthropic, "Anthropic", lambda: SimpleNamespace(api_key=None, auth_token=None, credentials=None))

        assert app.main(["List the resumes"]) == app.EXIT_ERROR
        assert "cp .env.example .env" in capsys.readouterr().err

    def test_invalid_effort_is_a_usage_error(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setenv("ANTHROPIC_EFFORT", "turbo")

        assert app.main(["hi"]) == app.EXIT_USAGE
        assert "ANTHROPIC_EFFORT" in capsys.readouterr().err

    def test_single_shot_joins_unquoted_words_and_prints_the_answer(self, monkeypatch: pytest.MonkeyPatch,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
        client = FakeClient(response(text("Here they are.")))
        use_client(monkeypatch, client)

        assert app.main(["List", "the", "resumes"]) == app.EXIT_OK
        assert client.requests[0]["messages"] == [{"role": "user", "content": "List the resumes"}]
        assert "Assistant: Here they are." in capsys.readouterr().out

    def test_quiet_mode_hides_the_tool_log(self, resumes: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
        use_client(monkeypatch, FakeClient(
            response(tool_use("list_files", {"directory": "resumes"}), stop="tool_use"), response(text("1 file."))))

        app.main(["--quiet", "List the resumes"])

        output = capsys.readouterr().out
        assert "🔧" not in output and "Assistant: 1 file." in output

    def test_single_shot_reports_api_errors_on_stderr(self, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
        use_client(monkeypatch, FakeClient(status_error(anthropic.AuthenticationError, 401)))

        assert app.main(["hi"]) == app.EXIT_ERROR
        captured = capsys.readouterr()
        assert "Authentication failed" in captured.err and "Traceback" not in captured.err

    def test_single_shot_refusal_exits_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_client(monkeypatch, FakeClient(response(stop="refusal")))

        assert app.main(["hi"]) == app.EXIT_ERROR

    def test_single_shot_ctrl_c_exits_130(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_client(monkeypatch, FakeClient(KeyboardInterrupt()))

        assert app.main(["hi"]) == app.EXIT_INTERRUPTED

    def test_chat_keeps_history_and_ignores_blank_lines(self, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
        client = FakeClient(response(text("Hello!")), response(text("You said hi.")))
        use_client(monkeypatch, client)
        scripted_input(monkeypatch, "", "hi", "what did I say?", "EXIT")

        assert app.main([]) == app.EXIT_OK

        output = capsys.readouterr().out
        assert "Assistant: Hello!" in output and "Assistant: You said hi." in output and "Goodbye!" in output
        assert [message["role"] for message in client.requests[1]["messages"]] == ["user", "assistant", "user"]

    @pytest.mark.parametrize("ending", [EOFError(), KeyboardInterrupt()])
    def test_chat_ends_on_ctrl_d_or_ctrl_c_at_the_prompt(self, monkeypatch: pytest.MonkeyPatch, ending: BaseException) -> None:
        use_client(monkeypatch, FakeClient())
        scripted_input(monkeypatch, ending)

        assert app.main([]) == app.EXIT_OK

    def test_ctrl_c_during_a_request_cancels_only_that_request(self, monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
        use_client(monkeypatch, FakeClient(KeyboardInterrupt(), response(text("Still here."))))
        scripted_input(monkeypatch, "slow question", "again", "quit")

        assert app.main([]) == app.EXIT_OK
        assert "Assistant: Still here." in capsys.readouterr().out

    def test_chat_survives_temporary_errors(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        client = FakeClient(status_error(anthropic.RateLimitError, 429), response(text("Recovered.")))
        use_client(monkeypatch, client)
        scripted_input(monkeypatch, "first", "second", "quit")

        assert app.main([]) == app.EXIT_OK
        assert client.requests[1]["messages"] == [{"role": "user", "content": "second"}]
        assert "Rate limit" in capsys.readouterr().err

    def test_chat_stops_on_fatal_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_client(monkeypatch, FakeClient(status_error(anthropic.AuthenticationError, 401)))
        scripted_input(monkeypatch, "hi", "never reached")

        assert app.main([]) == app.EXIT_ERROR


class TestConfiguration:
    def test_defaults(self) -> None:
        assert app.load_settings() == app.Settings(model="claude-opus-5-5", effort="medium", quiet=False)

    def test_environment_overrides_are_cleaned_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_MODEL", "  claude-sonnet-5-5 ")
        monkeypatch.setenv("ANTHROPIC_EFFORT", " LOW ")

        assert app.load_settings(quiet=True) == app.Settings(model="claude-sonnet-5-5", effort="low", quiet=True)

    def test_blank_model_falls_back_to_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_MODEL", "   ")

        assert app.load_settings().model == app.DEFAULT_MODEL

    def test_env_file_is_loaded_but_real_variables_win(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / ".env").write_text("ANTHROPIC_MODEL=from-file\nANTHROPIC_EFFORT=high\n", encoding="utf-8")
        monkeypatch.setenv("ANTHROPIC_MODEL", "from-environment")

        app.load_env_file()

        assert app.load_settings() == app.Settings(model="from-environment", effort="high")

    def test_blank_exported_variable_does_not_hide_the_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=test-key-from-file\n", encoding="utf-8")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")

        app.load_env_file()

        assert app.build_client().api_key == "test-key-from-file"

    def test_unreadable_env_file_is_a_warning_not_a_crash(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        (tmp_path / ".env").write_bytes(b"ANTHROPIC_API_KEY=\xff\xfe not utf-8")

        app.load_env_file()

        assert "couldn't read .env" in capsys.readouterr().err

    def test_api_key_whitespace_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  test-key \n")

        assert app.build_client().api_key == "test-key"

    def test_empty_api_key_does_not_shadow_other_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-token")

        client = app.build_client()

        assert client is not None and client.auth_token == "oauth-token"

    def test_no_credentials_at_all_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app.anthropic, "Anthropic", lambda: SimpleNamespace(api_key=None, auth_token=None, credentials=None))

        assert app.build_client() is None

    def test_stdio_setup_never_raises(self) -> None:
        app.configure_stdio()


class TestMigrationRegressions:
    def test_thinking_blocks_are_preserved_verbatim(self, resumes: Path) -> None:
        thinking = SimpleNamespace(type="thinking", thinking="", signature="opaque-signed-block")
        client = FakeClient(response(thinking, tool_use("list_files", {"directory": "resumes"}), stop="tool_use"),
                            response(text("Done.")))
        run(client)
        sent = client.requests[1]["messages"][1]["content"]
        assert sent[0] == thinking and sent[0].signature == "opaque-signed-block"

    def test_refusal_after_tools_rolls_back_only_the_new_turn(self, resumes: Path) -> None:
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": [text("ok")]}]
        client = FakeClient(response(tool_use("list_files", {"directory": "resumes"}), stop="tool_use"),
                            response(stop="refusal"))
        outcome, _ = run(client, history=history)
        assert outcome.messages == history and not outcome.ok

    def test_partial_answer_has_a_failure_exit_status(self) -> None:
        outcome, _ = run(FakeClient(response(text("Incomplete"), stop="max_tokens")))
        assert not outcome.ok and "cut off" in outcome.answer

    def test_tools_under_an_unexpected_stop_reason_are_not_executed(self, sandbox: Path) -> None:
        client = FakeClient(response(tool_use("write_file", {"filepath": "unsafe.md", "content": "x"}),
                                     stop="end_turn"), response(text("Skipped.")))
        run(client)
        assert not (sandbox / "unsafe.md").exists()
        assert client.requests[1]["messages"][-1]["content"][0]["is_error"]

    def test_broken_pipe_propagates_to_main(self) -> None:
        with pytest.raises(BrokenPipeError):
            app.ask(FakeClient(BrokenPipeError()), SETTINGS, [], "hi")

    def test_root_defaults_to_project_folder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path.parent)
        app.configure_root()
        assert app.fs_tools.sandbox_root() == str(tmp_path.resolve())

    def test_configured_root_is_respected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FS_ASSISTANT_ROOT", str(tmp_path.parent))
        app.configure_root()
        assert app.fs_tools.sandbox_root() == str(tmp_path.parent.resolve())

    def test_new_clears_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient(response(text("First.")), response(text("Fresh.")))
        use_client(monkeypatch, client)
        scripted_input(monkeypatch, "hi", "/new", "hello", "exit")
        assert app.main([]) == 0
        assert client.requests[1]["messages"] == [{"role": "user", "content": "hello"}]

    def test_undecodable_query_is_cleaned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient(response(text("OK")))
        use_client(monkeypatch, client)
        assert app.main(["bad\udcff"]) == 0
        assert client.requests[0]["messages"][0]["content"] == "bad?"

    @pytest.mark.parametrize("control", ["\x1b[2J", "\x00", "\x08", "\x9b"])
    def test_terminal_controls_are_neutralised(self, control: str, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
        use_client(monkeypatch, FakeClient(response(text(f"before{control}after"))))
        assert app.main(["hi"]) == 0
        output = capsys.readouterr().out
        assert control not in output and "before" in output and "after" in output

    def test_broken_stdout_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_client(monkeypatch, FakeClient(BrokenPipeError()))
        monkeypatch.setattr(app, "_discard_stdout", lambda: None)
        assert app.main(["hi"]) == 1

    def test_malformed_base_url_returns_friendly_startup_error(self, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
        def invalid() -> None:
            raise ValueError("Invalid API base URL")
        monkeypatch.setattr(app, "build_client", invalid)
        assert app.main(["hi"]) == 1
        assert "Invalid API base URL" in capsys.readouterr().err


def test_multi_workspace_key_sends_explicit_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "  wrkspc_test  ")
    client = FakeClient(response(text("OK")))
    run(client)
    assert client.requests[0]["workspace_id"] == "wrkspc_test"


def test_workspace_scoped_key_needs_no_extra_header() -> None:
    client = FakeClient(response(text("OK")))
    run(client)
    assert "workspace_id" not in client.requests[0]
