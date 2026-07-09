"""Minimal health / readiness endpoint (task 0.3).

Stdlib-only (no FastAPI dependency for a two-route health probe):

* ``GET /health``        → 200 once the process is up (liveness).
* ``GET /health/ready``  → 200 when models are warm, 503 otherwise (readiness).

``readiness`` is a zero-arg callable so the server stays decoupled from the
pipeline — typically ``pipeline.ready.__get__`` style via a lambda in main.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_health_server(port: int, readiness: Callable[[], bool]) -> ThreadingHTTPServer:
    """Build (but do not start) a health server probing ``readiness``."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/health":
                self._respond(200, b"ok")
            elif self.path in ("/health/ready", "/health/readiness"):
                ready = readiness()
                self._respond(200 if ready else 503, b"ready" if ready else b"not-ready")
            else:
                self._respond(404, b"not found")

        def _respond(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:  # silence per-request logging
            pass

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def serve_in_background(port: int, readiness: Callable[[], bool]) -> ThreadingHTTPServer:
    """Start the health server on a daemon thread and return it."""
    server = make_health_server(port, readiness)
    thread = threading.Thread(target=server.serve_forever, name="argus-health", daemon=True)
    thread.start()
    return server
