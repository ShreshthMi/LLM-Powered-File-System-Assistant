"""Scripted Anthropic Messages HTTP server for offline CLI integration tests."""
from __future__ import annotations

import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

Reply = dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]
_IDS = itertools.count(1)


def tool_call(name: str, **arguments: Any) -> dict[str, Any]:
    """Build a uniquely identified tool_use content block."""
    return {"type": "tool_use", "id": f"toolu_test_{next(_IDS)}", "name": name, "input": arguments}


def say(text: str, *calls: dict[str, Any]) -> dict[str, Any]:
    """Build a Messages response with optional text and tool calls."""
    content = ([{"type": "text", "text": text}] if text else []) + list(calls)
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 50}}


class StubAnthropic:
    """Serve scripted replies on localhost; never connects to the real API."""

    def __init__(self, replies: list[Reply], *, status: int = 200) -> None:
        self.replies = list(replies)
        self.status = status
        self.requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        """Base URL supplied to the real Anthropic SDK."""
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> StubAnthropic:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _answer(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if path.split("?")[0] != "/v1/messages":
            return 404, {"type": "error", "error": {"type": "not_found_error", "message": "Unknown endpoint"}}
        self.requests.append(body)
        if self.status != 200:
            return self.status, {"type": "error", "error": {"type": "api_error", "message": "Scripted API error"}}
        if not self.replies:
            return 500, {"type": "error", "error": {"type": "api_error", "message": "Script exhausted"}}
        reply = self.replies.pop(0)
        return 200, reply(body) if callable(reply) else reply

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - HTTP handler contract
                """Respond using the next scripted Messages reply."""
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                status, payload = stub._answer(self.path, body)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args: object) -> None:
                """Suppress access logs during tests."""

        return Handler
