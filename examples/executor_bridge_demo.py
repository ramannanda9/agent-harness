"""
examples/executor_bridge_demo.py

Demonstrates ExecutorBridge — the controlled subprocess launcher for agent tools.

Two backends are shown:
  backend="none"   — routes calls through ah-executor (process isolation,
                     allowlist, timeout). Skipped if binary not installed.
  backend="docker" — runs each call in a fresh Docker container (real network/fs
                     isolation, memory + CPU limits). Skipped if Docker not found.

Run:
    python examples/executor_bridge_demo.py

Install ah-executor first (enables native backend):
    cargo install --path executor
"""
from __future__ import annotations

import asyncio
import shutil

from harness.executor_bridge import ExecutorBridge, ExecutorConfig, ExecutorTool, find_executor

_BINARY = find_executor()
_DOCKER = shutil.which("docker")

SEP = "─" * 56


def _header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _show(result: dict) -> None:
    status = "ok" if result["success"] else "FAIL"
    print(f"  status   : {status} (exit {result['exit_code']})")
    if result["stdout"]:
        print(f"  stdout   : {result['stdout'].strip()}")
    if result["stderr"]:
        print(f"  stderr   : {result['stderr'].strip()}")
    if result["error"]:
        print(f"  error    : {result['error']}")
    print(f"  duration : {result['duration_ms']} ms")


# ── Allowlist enforcement (both backends) ─────────────────────────────────────

async def demo_allowlist() -> None:
    """The allowlist is enforced before any subprocess is spawned."""
    _header("Allowlist enforcement")

    # Use docker backend so the demo runs without the Rust binary.
    if not _DOCKER:
        print("  (skipped — docker not on PATH)")
        return

    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        backend="docker",
        docker_image="alpine:3.20",
    ))

    print("  Calling allowed tool  : shell")
    result = await bridge.execute("shell", args={"cmd": "echo allowed"})
    _show(result)

    print("\n  Calling disallowed tool: rm")
    result = await bridge.execute("rm", args=["-rf", "/"])
    _show(result)


# ── Native backend (Rust executor) ────────────────────────────────────────────

async def demo_native() -> None:
    """
    backend="none": calls route through ah-executor (auto-discovered from PATH).
    Provides process isolation and a scrubbed environment.
    Does not provide fs/network namespacing — for that, use backend="docker".
    """
    _header("Native backend (ah-executor)")

    if not _BINARY:
        print("  (skipped — ah-executor not on PATH)")
        print("  Install: cargo install --path executor")
        return

    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        default_timeout_ms=5_000,
    ))

    # shell tool — dict-style args: {"cmd": "..."}
    shell = ExecutorTool("shell", "shell", bridge)

    print("  shell: uname -s")
    _show(await shell.execute(cmd="uname -s"))

    print("\n  shell: echo $SECRET  (env is scrubbed — should be empty)")
    import os
    os.environ["SECRET"] = "hunter2"
    result = await shell.execute(cmd="echo \"secret=${SECRET}\"")
    _show(result)
    # Rust executor drops everything except PATH, so SECRET won't appear.


# ── Docker backend ────────────────────────────────────────────────────────────

async def demo_docker() -> None:
    """
    backend="docker": each call runs in a fresh container.
    Real OS-level isolation: no network, read-only fs, memory + CPU limits.
    """
    _header("Docker backend")

    if not _DOCKER:
        print("  (skipped — docker not on PATH)")
        return

    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        backend="docker",
        docker_image="alpine:3.20",
        docker_network="none",
        docker_memory="64m",
        docker_cpus="0.5",
        docker_read_only=True,
    ))

    shell = ExecutorTool("shell", "shell", bridge)

    print("  shell: uname -sr  (runs inside alpine container)")
    _show(await shell.execute(cmd="uname -sr"))

    print("\n  shell: wget http://example.com  (network=none — should fail)")
    result = await shell.execute(cmd="wget -T 1 -q http://example.com -O - 2>&1 || echo BLOCKED")
    _show(result)

    print("\n  shell: write to /etc  (read-only fs — should fail)")
    result = await shell.execute(cmd="echo x > /etc/test 2>&1 || echo READ_ONLY")
    _show(result)


# ── Timeout enforcement ───────────────────────────────────────────────────────

async def demo_timeout() -> None:
    """Timeouts are enforced regardless of backend."""
    _header("Timeout enforcement")

    if not _DOCKER:
        print("  (skipped — docker not on PATH)")
        return

    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        backend="docker",
        docker_image="alpine:3.20",
        default_timeout_ms=500,
        outer_timeout_grace_seconds=1.0,
    ))

    shell = ExecutorTool("shell", "shell", bridge)

    print("  shell: sleep 30  (500 ms timeout — should be killed)")
    _show(await shell.execute(cmd="sleep 30"))


# ── Positional-arg tools (kubectl-style) ──────────────────────────────────────

async def demo_positional_tool() -> None:
    """
    For tools with positional args (kubectl, git, curl…) use arg_key="args".
    The LLM produces {"args": ["get", "pods"]} and ExecutorTool unwraps the list.

    This demo uses `ls` (always in alpine) to show the pattern without pulling
    a specialised image. In production swap the image and tool name:
        docker_image="bitnami/kubectl:latest", allowed_tools=("kubectl",)
    """
    _header("Positional-arg tool (ls via docker)")

    if not _DOCKER:
        print("  (skipped — docker not on PATH)")
        return

    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("ls",),
        backend="docker",
        docker_image="alpine:3.20",
        docker_network="none",
    ))

    ls = ExecutorTool("ls", "ls", bridge, arg_key="args")

    print("  ls -1 /usr/bin  (positional args: [\"-1\", \"/usr/bin\"])")
    _show(await ls.execute(args=["-1", "/usr/bin"]))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    await demo_allowlist()
    await demo_native()
    await demo_docker()
    await demo_timeout()
    await demo_positional_tool()
    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
