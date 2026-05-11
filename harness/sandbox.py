"""
Sandboxed tool execution via the Rust executor binary.

The executor (executor/src/main.rs) enforces:
  - Tool allowlist (EXECUTOR_ALLOW env var; the LLM cannot bypass it).
  - Wall-clock timeout per call.
  - Output size cap.
  - Subprocess isolation — a tool crash cannot reach the agent process.
  - Scrubbed environment: only PATH is forwarded; everything else is dropped.

What it does NOT enforce — for syscall / fs / network isolation deploy
the harness inside a container or VM:
  - seccomp / landlock filters
  - filesystem or network namespacing
  - rlimit-based CPU/memory caps

Build the executor before using:
    cd executor && cargo build --release
    # binary at executor/target/release/executor

Wire-up:
    sandbox = Sandbox(SandboxConfig(
        binary_path="executor/target/release/executor",
        allowed_tools=("kubectl", "curl"),
    ))
    tools.register(SandboxedTool(name="kubectl", executor_tool="kubectl",
                                 sandbox=sandbox, arg_key="args"))
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """Raised when the sandbox itself cannot run (binary missing, decode error)."""


@dataclass
class SandboxConfig:
    binary_path: str
    # Default deliberately excludes "shell": opt in explicitly per deployment.
    allowed_tools: tuple[str, ...] = ("kubectl", "curl")
    max_output_bytes: int = 1_000_000
    default_timeout_ms: int = 30_000
    # Outer Python-side guard against the binary itself hanging beyond its own
    # tokio timeout. Adds to default_timeout_ms.
    outer_timeout_grace_seconds: float = 5.0
    extra_env: dict[str, str] = field(default_factory=dict)


class Sandbox:
    """
    Per-call subprocess sandbox: spawns the Rust executor, sends a JSON
    request on stdin, and parses a single JSON response from stdout.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        if not config.allowed_tools:
            raise SandboxError("Sandbox requires a non-empty allowed_tools list")
        if not os.path.isfile(config.binary_path):
            raise SandboxError(
                f"executor binary not found at {config.binary_path}. "
                "Build it: cd executor && cargo build --release"
            )

    async def execute(
        self,
        tool: str,
        args: Any,
        timeout_ms: int | None = None,
    ) -> dict:
        # Front-line check: the binary also enforces this, but failing early
        # avoids the subprocess fork.
        if tool not in self._config.allowed_tools:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "duration_ms": 0,
                "error": f"tool '{tool}' not in sandbox allowlist",
            }

        request = json.dumps(
            {
                "tool": tool,
                "args": args,
                "timeout_ms": timeout_ms or self._config.default_timeout_ms,
            }
        )

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "EXECUTOR_ALLOW": ",".join(self._config.allowed_tools),
            "EXECUTOR_MAX_OUTPUT_BYTES": str(self._config.max_output_bytes),
            **self._config.extra_env,
        }

        proc = await asyncio.create_subprocess_exec(
            self._config.binary_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        outer_timeout = (
            (timeout_ms or self._config.default_timeout_ms) / 1000.0
            + self._config.outer_timeout_grace_seconds
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=(request + "\n").encode("utf-8")),
                timeout=outer_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "duration_ms": 0,
                "error": (
                    f"sandbox subprocess hung past outer timeout ({outer_timeout:.1f}s)"
                ),
            }

        if not stdout:
            err = stderr.decode("utf-8", errors="replace") if stderr else ""
            raise SandboxError(
                f"executor produced no stdout; stderr={err!r}; exit={proc.returncode}"
            )
        try:
            return json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise SandboxError(
                f"could not decode executor response: {e}; raw={stdout[:500]!r}"
            ) from e


class SandboxedTool:
    """
    Adapter implementing the in-process Tool interface (.name + async .execute).

    Args mapping:
      - For tools whose executor schema is positional (kubectl, curl), the LLM
        produces e.g. {"args": ["get", "pods"]} and you set arg_key="args" so
        kwargs["args"] is unwrapped into the request payload.
      - For tools with a dict schema (shell expects {"cmd": "..."}), leave
        arg_key=None and the kwargs dict is forwarded as-is.
    """

    def __init__(
        self,
        name: str,
        executor_tool: str,
        sandbox: Sandbox,
        *,
        arg_key: str | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.name = name
        self._executor_tool = executor_tool
        self._sandbox = sandbox
        self._arg_key = arg_key
        self._timeout_ms = timeout_ms

    async def execute(self, **kwargs: Any) -> dict:
        if self._arg_key is not None:
            payload = kwargs.get(self._arg_key, [])
        else:
            payload = kwargs
        return await self._sandbox.execute(
            tool=self._executor_tool,
            args=payload,
            timeout_ms=self._timeout_ms,
        )
