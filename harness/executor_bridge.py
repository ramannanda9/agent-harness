"""
Controlled subprocess execution for agent tools.

Two backends are supported:

  backend="none"  (default)
    Spawns the `ah-executor` binary as a one-shot subprocess per tool call.
    Enforces an allowlist, wall-clock timeout, and output size cap. Provides
    process-level isolation only — no syscall filtering, no filesystem or
    network namespacing.

    Install:  cargo install --path executor
    The binary is then on PATH as `ah-executor` and auto-discovered.

  backend="docker"
    Runs each tool call inside a fresh Docker container with configurable
    memory, CPU, network, and read-only filesystem constraints. This is
    real OS-level isolation. Requires Docker daemon on the host.
    The `ah-executor` binary is not used in this mode.

Wire-up example (native — binary auto-discovered from PATH):
    bridge = ExecutorBridge(ExecutorConfig(allowed_tools=("shell", "curl")))
    tools = {"shell": ExecutorTool("shell", "shell", bridge)}

Wire-up example (docker):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl",),
        backend="docker",
        docker_image="bitnami/kubectl:latest",
        docker_network="none",
    ))
    tools = {"kubectl": ExecutorTool("kubectl", "kubectl", bridge, arg_key="args")}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

EXECUTOR_BINARY = "ah-executor"


def find_executor() -> str:
    """Return the path to the ah-executor binary, or empty string if not found."""
    return shutil.which(EXECUTOR_BINARY) or ""


class ExecutorError(RuntimeError):
    """Raised when the executor itself cannot run (binary missing, decode error, etc.)."""


@dataclass
class ExecutorConfig:
    allowed_tools: tuple[str, ...]
    # --- native backend ---
    # Defaults to auto-discovery via shutil.which("ah-executor").
    # Override only when the binary is not on PATH.
    binary_path: str = field(default_factory=find_executor)
    max_output_bytes: int = 1_000_000
    default_timeout_ms: int = 30_000
    # Python-side guard against the binary hanging past its own tokio timeout.
    outer_timeout_grace_seconds: float = 5.0
    extra_env: dict[str, str] = field(default_factory=dict)
    # --- docker backend ---
    backend: Literal["none", "docker"] = "none"
    docker_image: str = "alpine:3.20"
    docker_memory: str = "256m"
    docker_cpus: str = "1.0"
    docker_network: str = "none"
    docker_read_only: bool = True


class ExecutorBridge:
    """
    Routes tool calls to either the native Rust executor or a Docker container.
    Enforces the allowlist before any subprocess is spawned.
    """

    def __init__(self, config: ExecutorConfig) -> None:
        if not config.allowed_tools:
            raise ExecutorError("ExecutorBridge requires a non-empty allowed_tools list")
        if config.backend == "none":
            if not os.path.isfile(config.binary_path):
                raise ExecutorError(
                    f"ah-executor binary not found (looked at: {config.binary_path!r}). "
                    "Install it: cargo install --path executor"
                )
        self._config = config

    async def execute(
        self,
        tool: str,
        args: Any,
        timeout_ms: int | None = None,
    ) -> dict:
        if tool not in self._config.allowed_tools:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "duration_ms": 0,
                "error": f"tool '{tool}' not in executor allowlist",
            }
        if self._config.backend == "docker":
            return await self._execute_docker(tool, args, timeout_ms)
        return await self._execute_native(tool, args, timeout_ms)

    # ── Native (Rust executor) ────────────────────────────────────────────────

    async def _execute_native(
        self, tool: str, args: Any, timeout_ms: int | None
    ) -> dict:
        cfg = self._config
        request = json.dumps({
            "tool": tool,
            "args": args,
            "timeout_ms": timeout_ms or cfg.default_timeout_ms,
        })

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "EXECUTOR_ALLOW": ",".join(cfg.allowed_tools),
            "EXECUTOR_MAX_OUTPUT_BYTES": str(cfg.max_output_bytes),
            **cfg.extra_env,
        }

        proc = await asyncio.create_subprocess_exec(
            cfg.binary_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        outer_timeout = (
            (timeout_ms or cfg.default_timeout_ms) / 1000.0
            + cfg.outer_timeout_grace_seconds
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
                "error": f"executor hung past outer timeout ({outer_timeout:.1f}s)",
            }

        if not stdout:
            err = stderr.decode("utf-8", errors="replace") if stderr else ""
            raise ExecutorError(
                f"executor produced no stdout; stderr={err!r}; exit={proc.returncode}"
            )
        try:
            return json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ExecutorError(
                f"could not decode executor response: {e}; raw={stdout[:500]!r}"
            ) from e

    # ── Docker backend ────────────────────────────────────────────────────────

    async def _execute_docker(
        self, tool: str, args: Any, timeout_ms: int | None
    ) -> dict:
        cfg = self._config
        timeout_sec = (timeout_ms or cfg.default_timeout_ms) / 1000.0

        docker_cmd = [
            "docker", "run", "--rm",
            "--network", cfg.docker_network,
            "--memory", cfg.docker_memory,
            "--cpus", cfg.docker_cpus,
        ]
        if cfg.docker_read_only:
            docker_cmd.append("--read-only")
        docker_cmd.append(cfg.docker_image)

        if tool == "shell":
            cmd_str = args.get("cmd", "") if isinstance(args, dict) else str(args)
            docker_cmd.extend(["sh", "-c", cmd_str])
        else:
            tool_args = args if isinstance(args, list) else []
            docker_cmd.extend([tool, *tool_args])

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            outer_timeout = timeout_sec + cfg.outer_timeout_grace_seconds
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=outer_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "exit_code": None,
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "error": f"docker execution timed out ({outer_timeout:.1f}s)",
                }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "duration_ms": int((time.monotonic() - start) * 1000),
                "error": f"docker spawn error: {e}",
            }

        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "success": proc.returncode == 0,
            "stdout": _truncate(stdout.decode("utf-8", errors="replace"), cfg.max_output_bytes),
            "stderr": _truncate(stderr.decode("utf-8", errors="replace"), cfg.max_output_bytes),
            "exit_code": proc.returncode,
            "duration_ms": duration_ms,
            "error": None,
        }


class ExecutorTool:
    """
    Adapter implementing the in-process Tool interface (.name + async .execute).

    Args mapping:
      - For positional-arg tools (kubectl, curl, git): the LLM produces
        {"args": ["get", "pods"]}; set arg_key="args" to unwrap the list.
      - For dict-arg tools (shell expects {"cmd": "..."}): leave arg_key=None
        and the kwargs dict is forwarded as-is.
    """

    def __init__(
        self,
        name: str,
        executor_tool: str,
        bridge: ExecutorBridge,
        *,
        arg_key: str | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.name = name
        self._executor_tool = executor_tool
        self._bridge = bridge
        self._arg_key = arg_key
        self._timeout_ms = timeout_ms

    async def execute(self, **kwargs: Any) -> dict:
        payload = kwargs.get(self._arg_key, []) if self._arg_key is not None else kwargs
        return await self._bridge.execute(
            tool=self._executor_tool,
            args=payload,
            timeout_ms=self._timeout_ms,
        )


def _truncate(s: str, max_bytes: int) -> str:
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    dropped = len(encoded) - max_bytes
    return f"{truncated}…[truncated {dropped} bytes]"
