"""Browser-based OAuth helpers shared across providers.

Two utilities here:

  - ``open_or_print_url(url)`` — try to open a URL in the user's default
    browser; fall back to printing it so headless / SSH sessions still work.

  - ``wait_for_oauth_callback(port, path, timeout)`` — spin up a one-shot
    localhost HTTP server, block until the OAuth provider redirects the
    browser back, and return the ``(code, state)`` pair from the query
    string. Used by ``BrowserOAuthMCPAuth`` and any future browser-based
    login flow whose redirect URI we control.

Stdlib only — no new dependencies. The callback server uses
``http.server.HTTPServer`` in a background thread so the asyncio caller can
``await`` on a future that resolves when the request arrives.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)


_HTML_OK = b"""<!doctype html>
<html><body style="font-family:sans-serif;text-align:center;padding:3em">
<h2>Authorization complete</h2>
<p>You can close this tab and return to the terminal.</p>
</body></html>
"""

_HTML_ERROR = b"""<!doctype html>
<html><body style="font-family:sans-serif;text-align:center;padding:3em">
<h2>Authorization failed</h2>
<p>%s</p>
<p>Check the terminal for details.</p>
</body></html>
"""


def open_or_print_url(url: str, *, prefix: str = "Open in browser:") -> None:
    """Try to open ``url`` in the default browser; always print it as a fallback."""
    print(f"{prefix} {url}")
    try:
        webbrowser.open(url, new=2)
    except Exception as e:  # noqa: BLE001 — best-effort UX nicety
        logger.debug("webbrowser.open failed: %s", e)


async def wait_for_oauth_callback(
    *,
    port: int = 0,
    path: str = "/callback",
    timeout: float = 300.0,
    bind_host: str = "127.0.0.1",
) -> tuple[str, str | None]:
    """Run a localhost HTTP server until a redirect with ``code`` arrives.

    Args:
        port: Port to bind. ``0`` lets the OS pick a free port — read it back
              from ``actual_port`` after construction if you need it (this
              helper does not return the bound port; callers that need it
              should use :func:`bind_callback_server` instead).
        path: Expected redirect path. Other paths return 404.
        timeout: Seconds to wait before raising :class:`TimeoutError`.
        bind_host: Address to bind on. Keep ``127.0.0.1`` for security —
                   anything else makes the auth code observable on the LAN.

    Returns:
        ``(code, state)`` from the query string. ``state`` is ``None`` when
        the provider does not echo it back.

    Raises:
        TimeoutError: No callback arrived within ``timeout`` seconds.
        RuntimeError: The redirect carried an ``error`` query parameter.
    """
    server, actual_port, future = bind_callback_server(port=port, path=path, bind_host=bind_host)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    finally:
        server.shutdown()


def bind_callback_server(
    *,
    port: int = 0,
    path: str = "/callback",
    bind_host: str = "127.0.0.1",
) -> tuple[HTTPServer, int, asyncio.Future[tuple[str, str | None]]]:
    """Start the callback server and return (server, port, future).

    Callers that need the bound port up front (to construct the redirect URI
    before opening the browser) use this and then ``await future``. Callers
    that already know the port should prefer :func:`wait_for_oauth_callback`.

    The server runs in a daemon thread and shuts down on the first valid
    callback or when ``server.shutdown()`` is called.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[tuple[str, str | None]] = loop.create_future()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != path:
                self.send_response(404)
                self.end_headers()
                return
            qs = urllib.parse.parse_qs(parsed.query)
            err = qs.get("error", [None])[0]
            if err:
                desc = qs.get("error_description", [""])[0]
                msg = f"{err}: {desc}".strip(": ")
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_HTML_ERROR % msg.encode("utf-8", "replace"))
                if not future.done():
                    loop.call_soon_threadsafe(
                        future.set_exception, RuntimeError(f"OAuth callback error: {msg}")
                    )
                return
            code = qs.get("code", [None])[0]
            state = qs.get("state", [None])[0]
            if not code:
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_HTML_OK)
            if not future.done():
                loop.call_soon_threadsafe(future.set_result, (code, state))

        def log_message(self, *_args: Any) -> None:  # silence stdlib's stderr noise
            return

    server = HTTPServer((bind_host, port), _Handler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port, future
