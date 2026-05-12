"""
ExecutorBridge tests.

Unit tests mock asyncio.create_subprocess_exec so no real binary or Docker
daemon is required. Integration tests at the bottom are skipped unless the
Rust binary is built or Docker is available.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.executor_bridge import (
    ExecutorBridge,
    ExecutorConfig,
    ExecutorError,
    ExecutorTool,
)

# ── Test scaffolding ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    p = tmp_path / "executor"
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(0o755)
    return p


class FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False
        self.communicate_input: bytes | None = None
        self.env: dict | None = None
        self.argv: tuple = ()

    async def communicate(self, input: bytes | None = None):
        self.communicate_input = input
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def _patch_subprocess(proc: FakeProc):
    async def fake_create(*args, **kwargs):
        proc.env = kwargs.get("env")
        proc.argv = args
        return proc

    return patch("asyncio.create_subprocess_exec", side_effect=fake_create)


# ── Config validation ─────────────────────────────────────────────────────────


def test_missing_binary_raises(tmp_path: Path):
    with pytest.raises(ExecutorError, match="executor binary not found"):
        ExecutorBridge(ExecutorConfig(
            allowed_tools=("kubectl",),
            binary_path=str(tmp_path / "does-not-exist"),
        ))


def test_empty_allowlist_raises(fake_binary: Path):
    with pytest.raises(ExecutorError, match="non-empty allowed_tools"):
        ExecutorBridge(ExecutorConfig(allowed_tools=(), binary_path=str(fake_binary)))


def test_docker_backend_does_not_require_binary():
    # No binary_path needed when backend="docker".
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        backend="docker",
    ))
    assert bridge is not None


# ── Native backend: execute() ─────────────────────────────────────────────────


async def test_native_disallowed_tool_short_circuits(fake_binary: Path):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl",), binary_path=str(fake_binary)
    ))
    with patch("asyncio.create_subprocess_exec") as spawn:
        result = await bridge.execute("rm", args=["-rf", "/"])
    spawn.assert_not_called()
    assert result["success"] is False
    assert "not in executor allowlist" in result["error"]


async def test_native_passes_request_on_stdin(fake_binary: Path):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl",), binary_path=str(fake_binary)
    ))
    proc = FakeProc(stdout=json.dumps({
        "success": True, "stdout": "pods\n", "stderr": "",
        "exit_code": 0, "duration_ms": 12, "error": None,
    }).encode())
    with _patch_subprocess(proc):
        result = await bridge.execute("kubectl", args=["get", "pods"])

    assert result["stdout"] == "pods\n"
    sent = json.loads(proc.communicate_input.decode())
    assert sent["tool"] == "kubectl"
    assert sent["args"] == ["get", "pods"]
    assert sent["timeout_ms"] == 30_000


async def test_native_forwards_env_with_allowlist(fake_binary: Path):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl", "curl"),
        binary_path=str(fake_binary),
        max_output_bytes=2048,
        extra_env={"FOO": "bar"},
    ))
    proc = FakeProc(stdout=b'{"success":true,"stdout":"","stderr":"","exit_code":0,"duration_ms":0,"error":null}')
    with _patch_subprocess(proc):
        await bridge.execute("kubectl", args=["version"])

    assert proc.env["EXECUTOR_ALLOW"] == "kubectl,curl"
    assert proc.env["EXECUTOR_MAX_OUTPUT_BYTES"] == "2048"
    assert proc.env["FOO"] == "bar"
    assert "PATH" in proc.env


async def test_native_raises_on_garbage_output(fake_binary: Path):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl",), binary_path=str(fake_binary)
    ))
    proc = FakeProc(stdout=b"not json at all")
    with _patch_subprocess(proc), pytest.raises(ExecutorError, match="could not decode"):
        await bridge.execute("kubectl", args=["x"])


async def test_native_raises_on_empty_output(fake_binary: Path):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl",), binary_path=str(fake_binary)
    ))
    proc = FakeProc(stdout=b"", stderr=b"binary crashed", returncode=1)
    with _patch_subprocess(proc), pytest.raises(ExecutorError, match="no stdout"):
        await bridge.execute("kubectl", args=["x"])


async def test_native_outer_timeout_kills_subprocess(fake_binary: Path):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl",),
        binary_path=str(fake_binary),
        default_timeout_ms=10,
        outer_timeout_grace_seconds=0.0,
    ))

    class HangingProc(FakeProc):
        async def communicate(self, input=None):
            await asyncio.sleep(10)
            return b"", b""

    proc = HangingProc(stdout=b"")
    with _patch_subprocess(proc):
        result = await bridge.execute("kubectl", args=["x"])

    assert proc.killed is True
    assert result["success"] is False
    assert "outer timeout" in result["error"]


# ── Docker backend: execute() ─────────────────────────────────────────────────


async def test_docker_disallowed_tool_short_circuits():
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",), backend="docker"
    ))
    with patch("asyncio.create_subprocess_exec") as spawn:
        result = await bridge.execute("rm", args=["-rf", "/"])
    spawn.assert_not_called()
    assert result["success"] is False
    assert "not in executor allowlist" in result["error"]


async def test_docker_shell_tool_builds_correct_command():
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        backend="docker",
        docker_image="alpine:3.20",
        docker_memory="128m",
        docker_cpus="0.5",
        docker_network="none",
        docker_read_only=True,
    ))
    proc = FakeProc(stdout=b"hello\n", returncode=0)
    with _patch_subprocess(proc):
        result = await bridge.execute("shell", args={"cmd": "echo hello"})

    assert result["success"] is True
    assert result["stdout"] == "hello\n"
    argv = proc.argv
    assert "docker" in argv[0]
    assert "--network" in argv
    assert "none" in argv
    assert "--memory" in argv
    assert "128m" in argv
    assert "--read-only" in argv
    assert "alpine:3.20" in argv
    assert "sh" in argv
    assert "-c" in argv
    assert "echo hello" in argv


async def test_docker_positional_tool_builds_correct_command():
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl",),
        backend="docker",
        docker_image="bitnami/kubectl:latest",
        docker_read_only=False,
    ))
    proc = FakeProc(stdout=b"NAME\npod-1\n", returncode=0)
    with _patch_subprocess(proc):
        result = await bridge.execute("kubectl", args=["get", "pods"])

    assert result["success"] is True
    argv = proc.argv
    assert "kubectl" in argv
    assert "get" in argv
    assert "pods" in argv
    assert "--read-only" not in argv


async def test_docker_timeout_kills_container():
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        backend="docker",
        default_timeout_ms=10,
        outer_timeout_grace_seconds=0.0,
    ))

    class HangingProc(FakeProc):
        async def communicate(self, input=None):
            await asyncio.sleep(10)
            return b"", b""

    proc = HangingProc(stdout=b"")
    with _patch_subprocess(proc):
        result = await bridge.execute("shell", args={"cmd": "sleep 999"})

    assert proc.killed is True
    assert result["success"] is False
    assert "timed out" in result["error"]


async def test_docker_nonzero_exit_is_failure():
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",), backend="docker"
    ))
    proc = FakeProc(stdout=b"", stderr=b"command not found", returncode=127)
    with _patch_subprocess(proc):
        result = await bridge.execute("shell", args={"cmd": "bogus"})

    assert result["success"] is False
    assert result["exit_code"] == 127
    assert result["stderr"] == "command not found"


# ── ExecutorTool adapter ──────────────────────────────────────────────────────


async def test_executor_tool_dict_passthrough(fake_binary: Path):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",), binary_path=str(fake_binary)
    ))
    tool = ExecutorTool("shell", "shell", bridge)
    proc = FakeProc(stdout=b'{"success":true,"stdout":"hi\\n","stderr":"","exit_code":0,"duration_ms":1,"error":null}')
    with _patch_subprocess(proc):
        await tool.execute(cmd="echo hi")
    sent = json.loads(proc.communicate_input.decode())
    assert sent["tool"] == "shell"
    assert sent["args"] == {"cmd": "echo hi"}


async def test_executor_tool_arg_key_unwraps(fake_binary: Path):
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("kubectl",), binary_path=str(fake_binary)
    ))
    tool = ExecutorTool("kubectl", "kubectl", bridge, arg_key="args")
    proc = FakeProc(stdout=b'{"success":true,"stdout":"","stderr":"","exit_code":0,"duration_ms":0,"error":null}')
    with _patch_subprocess(proc):
        await tool.execute(args=["get", "pods", "-A"])
    sent = json.loads(proc.communicate_input.decode())
    assert sent["args"] == ["get", "pods", "-A"]


# ── Integration: real Rust binary ────────────────────────────────────────────

_BINARY = Path(__file__).parent.parent / "executor" / "target" / "release" / "executor"


@pytest.mark.skipif(
    not _BINARY.exists(),
    reason="rust executor not built; run: cd executor && cargo build --release",
)
async def test_integration_native_runs_echo():
    if not shutil.which("sh"):
        pytest.skip("sh not on PATH")
    bridge = ExecutorBridge(ExecutorConfig(
        binary_path=str(_BINARY), allowed_tools=("shell",)
    ))
    tool = ExecutorTool("shell", "shell", bridge)
    result = await tool.execute(cmd="printf 'hi'")
    assert result["success"] is True
    assert result["stdout"] == "hi"
    assert result["exit_code"] == 0


@pytest.mark.skipif(
    not _BINARY.exists(),
    reason="rust executor not built",
)
async def test_integration_native_rejects_disallowed_tool():
    bridge = ExecutorBridge(ExecutorConfig(
        binary_path=str(_BINARY), allowed_tools=("kubectl",)
    ))
    result = await bridge.execute("shell", args={"cmd": "rm -rf /"})
    assert result["success"] is False
    assert "not in executor allowlist" in result["error"]


# ── Integration: Docker backend ───────────────────────────────────────────────

_DOCKER = shutil.which("docker")


@pytest.mark.skipif(not _DOCKER, reason="docker not on PATH")
async def test_integration_docker_echo():
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        backend="docker",
        docker_image="alpine:3.20",
    ))
    result = await bridge.execute("shell", args={"cmd": "printf hi"})
    assert result["success"] is True
    assert result["stdout"] == "hi"


@pytest.mark.skipif(not _DOCKER, reason="docker not on PATH")
async def test_integration_docker_network_is_blocked():
    bridge = ExecutorBridge(ExecutorConfig(
        allowed_tools=("shell",),
        backend="docker",
        docker_image="alpine:3.20",
        docker_network="none",
    ))
    result = await bridge.execute("shell", args={"cmd": "wget -T2 -q http://example.com -O - || echo BLOCKED"})
    assert "BLOCKED" in result["stdout"] or result["exit_code"] != 0
