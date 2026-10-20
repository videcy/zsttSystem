"""Serve the local demo page and proxy its API calls to the deployed system.

This is a lightweight UI preview for a checkout without local databases or
corpus data. It binds to loopback and only forwards known application routes.
"""
from __future__ import annotations

import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "templates" / "demo.html"
STATIC_ROOT = (ROOT / "src" / "static").resolve()
UPSTREAM = "https://zsttsystem.top"
HOST = "127.0.0.1"
PORT = 8765
MAX_BODY_BYTES = 1_000_000


def allowed_upstream(method: str, path: str) -> bool:
    if method == "POST":
        return path in {"/query", "/feedback"}
    if method != "GET":
        return False
    return (
        path in {
            "/health",
            "/overview",
            "/personas",
            "/programs",
            "/programs/graph",
            "/eval/latest",
            "/dependency",
        }
        or path.startswith("/courses/")
    )


class PreviewHandler(BaseHTTPRequestHandler):
    server_version = "zsttSystemLocalPreview/1.0"

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._serve_file(TEMPLATE, "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/static/"):
            relative = unquote(parsed.path.removeprefix("/static/"))
            target = (STATIC_ROOT / relative).resolve()
            if not target.is_relative_to(STATIC_ROOT) or not target.is_file():
                self.send_error(404)
                return
            self._serve_file(target, mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            return
        if allowed_upstream("GET", parsed.path):
            self._proxy("GET", self.path)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        if not allowed_upstream("POST", parsed.path):
            self.send_error(404)
            return
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{HOST}:{PORT}":
            self.send_error(403, "Cross-origin requests are not allowed")
            return
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            self.send_error(415, "JSON requests only")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length < 0 or content_length > MAX_BODY_BYTES:
            self.send_error(413, "Request body is too large")
            return
        self._proxy("POST", self.path, self.rfile.read(content_length))

    def _serve_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, method: str, path: str, body: bytes | None = None) -> None:
        headers = {"Accept": "application/json"}
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(f"{UPSTREAM}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=60) as response:
                self._send_upstream_response(response.status, response.headers.get("Content-Type"), response.read())
        except HTTPError as error:
            self._send_upstream_response(error.code, error.headers.get("Content-Type"), error.read())
        except (TimeoutError, URLError, OSError) as error:
            message = f"Upstream request failed: {error}".encode("utf-8", errors="replace")
            self._send_upstream_response(502, "text/plain; charset=utf-8", message)

    def _send_upstream_response(self, status: int, content_type: str | None, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        # Avoid writing query strings or request bodies to the local console.
        print(f"{self.address_string()} {self.command} {urlsplit(self.path).path}")


if __name__ == "__main__":
    print(f"Local UI: http://{HOST}:{PORT}/#qa")
    print(f"Query API: {UPSTREAM} (questions are sent to the deployed system)")
    ThreadingHTTPServer((HOST, PORT), PreviewHandler).serve_forever()
