"""Local web viewer for BusEvent JSONL traces.

``serve(path, port)`` starts a stdlib HTTP server that renders the trace as
a vertical timeline. No external dependencies, no build step — the page is
a single embedded HTML document that loads the JSONL via ``fetch`` and
groups events by agent.

Usage::

    agent-harness view path/to/trace.jsonl

Or programmatically::

    from harness.trace_viewer import serve
    serve(\"trace.jsonl\", port=8765, open_browser=True)
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_VIEWER_HTML = """\
<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>agent-harness — trace viewer</title>
<style>
  :root {
    --bg: #0f1115; --card: #161a21; --line: #232a35; --text: #e6e8eb;
    --muted: #8a93a2; --accent: #5ab4ff; --warn: #ffb454; --err: #ff6b6b;
    --ok: #6cd97e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font: 13px/1.45 -apple-system, system-ui, sans-serif;
    background: var(--bg); color: var(--text);
  }
  header {
    padding: 14px 20px; border-bottom: 1px solid var(--line);
    display: flex; gap: 18px; align-items: center;
  }
  header h1 { font-size: 14px; margin: 0; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; }
  header input[type=text] {
    background: var(--card); color: var(--text); border: 1px solid var(--line);
    padding: 6px 10px; border-radius: 4px; width: 260px; font: inherit;
  }
  main { display: grid; grid-template-columns: 200px 1fr; min-height: 90vh; }
  aside {
    border-right: 1px solid var(--line); padding: 12px 16px;
    background: #11141a; position: sticky; top: 0; height: 100vh; overflow: auto;
  }
  aside h2 { font-size: 11px; text-transform: uppercase;
             letter-spacing: 0.06em; color: var(--muted); margin: 14px 0 6px; }
  aside label { display: flex; align-items: center; gap: 6px;
                padding: 3px 0; cursor: pointer; }
  aside label input { accent-color: var(--accent); }
  section.events { padding: 16px 24px; }
  .event {
    display: grid; grid-template-columns: 80px 110px 1fr;
    gap: 14px; padding: 8px 10px; border-bottom: 1px solid var(--line);
    align-items: start;
  }
  .event:hover { background: rgba(255,255,255,0.02); }
  .t { color: var(--muted); font-variant-numeric: tabular-nums; }
  .type {
    color: var(--accent); font-weight: 600; text-transform: uppercase;
    font-size: 11px; letter-spacing: 0.05em;
  }
  .type.error { color: var(--err); }
  .type.done, .type.task_done { color: var(--ok); }
  .type.replan { color: var(--warn); }
  .body { white-space: pre-wrap; word-break: break-word; }
  .agent { color: var(--muted); font-size: 11px; }
  details { margin-top: 4px; }
  details > summary { color: var(--muted); cursor: pointer; font-size: 11px; }
  pre {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 4px; padding: 8px 10px; margin: 6px 0 0; font-size: 12px;
    overflow-x: auto; max-height: 360px;
  }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>agent-harness · trace viewer</h1>
  <div class=\"meta\" id=\"meta\">loading…</div>
  <input id=\"q\" type=\"text\" placeholder=\"filter (type or text)…\">
</header>
<main>
  <aside>
    <h2>Agents</h2>
    <div id=\"agents\"></div>
    <h2>Event types</h2>
    <div id=\"types\"></div>
  </aside>
  <section class=\"events\" id=\"events\">
    <div class=\"empty\">Loading trace…</div>
  </section>
</main>
<script>
const SHORTEN_TYPES = new Set(['token']);
let events = [];
let filters = { agents: new Set(), types: new Set(), text: '' };

async function load() {
  const r = await fetch('/trace.jsonl');
  const text = await r.text();
  events = text.split('\\n').filter(Boolean).map(line => {
    try { return JSON.parse(line); } catch { return null; }
  }).filter(Boolean);
  buildSidebar();
  render();
  const start = events[0]?.timestamp ?? 0;
  const end = events[events.length - 1]?.timestamp ?? start;
  document.getElementById('meta').textContent =
    `${events.length} events · ${(end - start).toFixed(2)}s`;
}

function buildSidebar() {
  const agents = new Set(events.map(e => e.agent_id || '(orchestrator)'));
  const types = new Set(events.map(e => e.type));
  filters.agents = new Set(agents);
  filters.types = new Set(types);
  document.getElementById('agents').innerHTML = [...agents].sort().map(a =>
    `<label><input type=\"checkbox\" data-kind=\"agent\" data-name=\"${a}\" checked>${a}</label>`
  ).join('');
  document.getElementById('types').innerHTML = [...types].sort().map(t =>
    `<label><input type=\"checkbox\" data-kind=\"type\" data-name=\"${t}\" checked>${t}</label>`
  ).join('');
  for (const el of document.querySelectorAll('aside input[type=checkbox]')) {
    el.addEventListener('change', () => {
      const set = el.dataset.kind === 'agent' ? filters.agents : filters.types;
      el.checked ? set.add(el.dataset.name) : set.delete(el.dataset.name);
      render();
    });
  }
  document.getElementById('q').addEventListener('input', e => {
    filters.text = e.target.value.toLowerCase();
    render();
  });
}

function render() {
  const start = events[0]?.timestamp ?? 0;
  const visible = events.filter(e => {
    if (!filters.agents.has(e.agent_id || '(orchestrator)')) return false;
    if (!filters.types.has(e.type)) return false;
    if (filters.text) {
      const hay = (e.type + ' ' + (e.agent_id || '') + ' ' +
                   JSON.stringify(e.payload) + ' ' + (e.token || '') + ' ' +
                   (e.error || '')).toLowerCase();
      if (!hay.includes(filters.text)) return false;
    }
    return true;
  });
  const container = document.getElementById('events');
  if (visible.length === 0) {
    container.innerHTML = '<div class=\"empty\">No events match.</div>';
    return;
  }
  container.innerHTML = visible.map(e => {
    const dt = (e.timestamp - start).toFixed(3);
    const body = formatBody(e);
    const meta = e.agent_id ? `<span class=\"agent\">${escapeHTML(e.agent_id)}</span>` : '';
    return `<div class=\"event\">
      <div class=\"t\">+${dt}s</div>
      <div class=\"type ${e.type}\">${e.type}</div>
      <div>${meta}<div class=\"body\">${body}</div></div>
    </div>`;
  }).join('');
}

function formatBody(e) {
  if (e.token) return `<code>${escapeHTML(truncate(e.token, 200))}</code>`;
  if (e.error) return `<code style=\"color:var(--err)\">${escapeHTML(e.error)}</code>`;
  const payload = e.payload || {};
  const summary = summarizePayload(e.type, payload);
  const json = JSON.stringify(payload, null, 2);
  return `${summary}<details><summary>payload (${json.length} chars)</summary><pre>${escapeHTML(json)}</pre></details>`;
}

function summarizePayload(type, p) {
  if (type === 'thought') return escapeHTML(truncate(p.thought || '', 240));
  if (type === 'action') {
    const tool = p.tool || p.name || '?';
    const args = JSON.stringify(p.args || p.arguments || {});
    return `<code>${escapeHTML(tool)}</code> <span class=\"agent\">${escapeHTML(truncate(args, 200))}</span>`;
  }
  if (type === 'observation') return escapeHTML(truncate(JSON.stringify(p.observation ?? p), 240));
  if (type === 'route') return escapeHTML(`→ ${p.agent_id || '?'} (${p.rationale || ''})`);
  if (type === 'dispatch') return escapeHTML(`complexity=${p.complexity || '?'} path=${p.path || '?'}`);
  if (type === 'context') {
    const tokens = p.tokens || 0; const max = p.max_tokens || 0;
    const pct = max ? Math.round(100 * tokens / max) : 0;
    return escapeHTML(`${tokens.toLocaleString()} / ${max.toLocaleString()} tokens (${pct}%)`);
  }
  if (type === 'plan' && Array.isArray(p.tasks)) {
    return escapeHTML(`${p.tasks.length} task(s)`);
  }
  return '';
}

function truncate(s, n) {
  s = String(s);
  return s.length <= n ? s : s.slice(0, n) + '…';
}
function escapeHTML(s) {
  return String(s).replace(/[&<>\\\"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', \"'\": '&#39;'
  })[c]);
}

load().catch(e => {
  document.getElementById('events').innerHTML =
    `<div class=\"empty\">Failed to load trace: ${e.message}</div>`;
});
</script>
</body>
</html>
"""


class _TraceHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the embedded viewer + the raw trace file."""

    trace_path: Path

    def do_GET(self) -> None:  # noqa: N802 — stdlib API
        if self.path in ("/", "/index.html"):
            self._send(200, _VIEWER_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/trace.jsonl":
            try:
                data = self.trace_path.read_bytes()
            except OSError as e:
                self._send(404, f"Could not read trace: {e}".encode(), "text/plain")
                return
            self._send(200, data, "application/x-ndjson")
            return
        if self.path == "/manifest.json":
            payload = json.dumps(
                {"trace_path": str(self.trace_path), "exists": self.trace_path.exists()}
            ).encode()
            self._send(200, payload, "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:  # silence stderr
        return


def serve(
    path: str | Path,
    *,
    port: int = 8765,
    bind_host: str = "127.0.0.1",
    open_browser: bool = True,
    block: bool = True,
) -> HTTPServer:
    """Serve the trace viewer on ``http://<bind_host>:<port>``.

    Args:
        path: Path to a JSONL trace file produced by ``record_trace``.
        port: Port to bind (default 8765).
        bind_host: Address to bind. Keep ``127.0.0.1`` for local-only.
        open_browser: When True, open the default browser at the served URL.
        block: When True (default), serve forever in the calling thread.
               When False, start the server in a daemon thread and return
               immediately so the caller can run alongside.

    Returns the ``HTTPServer`` instance; call ``server.shutdown()`` to stop.
    """
    trace = Path(path).expanduser()

    # Subclass per-call so multiple servers on different traces don't
    # share a single class-level ``trace_path``.
    class _Handler(_TraceHandler):
        trace_path = trace

    server = HTTPServer((bind_host, port), _Handler)
    url = f"http://{bind_host}:{server.server_address[1]}/"
    print(f"Trace viewer: {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception as e:  # noqa: BLE001
            logger.debug("webbrowser.open failed: %s", e)
    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping viewer.")
            server.shutdown()
    else:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
