// Sandboxed tool executor for agent-harness.
//
// One-shot subprocess: reads a single JSON ToolRequest line on stdin,
// runs an allowlisted tool, and writes a single JSON ToolResult line to stdout.
// The Python harness/sandbox.py invokes this binary per tool call.
//
// What this enforces:
//   - Tool allowlist (EXECUTOR_ALLOW env var, comma-separated).
//   - Wall-clock timeout per call (from ToolRequest.timeout_ms).
//   - Output size cap (EXECUTOR_MAX_OUTPUT_BYTES, default 1 MiB).
//   - Subprocess isolation — a tool crash cannot reach the agent process.
//   - Scrubbed environment: only PATH is forwarded; everything else is dropped.
//
// What this does NOT enforce (deploy in a container/VM if you need it):
//   - Syscall-level filtering (seccomp / landlock).
//   - Filesystem or network namespacing.
//   - rlimit-based CPU / memory caps.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use tokio::process::Command;

const MAX_OUTPUT_DEFAULT: usize = 1_000_000;

#[derive(Deserialize)]
struct ToolRequest {
    tool: String,
    args: serde_json::Value,
    #[serde(default = "default_timeout")]
    timeout_ms: u64,
}

fn default_timeout() -> u64 {
    30_000
}

#[derive(Serialize)]
struct ToolResult {
    success: bool,
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    duration_ms: u64,
    error: Option<String>,
}

struct CmdOutput {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
}

#[tokio::main]
async fn main() {
    let allow = read_allowlist();
    let max_output = std::env::var("EXECUTOR_MAX_OUTPUT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_OUTPUT_DEFAULT);

    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_line(&mut input) {
        emit_err(&format!("failed to read request: {e}"));
        return;
    }

    let req: ToolRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            emit_err(&format!("invalid request JSON: {e}"));
            return;
        }
    };

    if !allow.contains(&req.tool) {
        emit_err(&format!(
            "tool '{}' not in EXECUTOR_ALLOW (allowed: {:?})",
            req.tool, allow
        ));
        return;
    }

    let result = execute_tool(req, max_output).await;
    println!("{}", serde_json::to_string(&result).unwrap());
}

fn read_allowlist() -> HashSet<String> {
    std::env::var("EXECUTOR_ALLOW")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn emit_err(msg: &str) {
    let res = ToolResult {
        success: false,
        stdout: String::new(),
        stderr: String::new(),
        exit_code: None,
        duration_ms: 0,
        error: Some(msg.to_string()),
    };
    println!("{}", serde_json::to_string(&res).unwrap());
}

fn truncate(s: String, max: usize) -> String {
    if s.len() <= max {
        s
    } else {
        // Truncate on a char boundary to avoid panicking on multibyte input.
        let mut cut = max;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        let dropped = s.len() - cut;
        format!("{}…[truncated {} bytes]", &s[..cut], dropped)
    }
}

async fn execute_tool(req: ToolRequest, max_output: usize) -> ToolResult {
    let start = Instant::now();
    let timeout = Duration::from_millis(req.timeout_ms);

    let outcome = match req.tool.as_str() {
        "shell" => run_shell(&req.args, timeout).await,
        bin => run_cmd(bin, &req.args, timeout).await,
    };

    let duration_ms = start.elapsed().as_millis() as u64;
    match outcome {
        Ok(out) => ToolResult {
            success: out.exit_code == Some(0),
            stdout: truncate(out.stdout, max_output),
            stderr: truncate(out.stderr, max_output),
            exit_code: out.exit_code,
            duration_ms,
            error: None,
        },
        Err(e) => ToolResult {
            success: false,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            duration_ms,
            error: Some(e),
        },
    }
}

fn build_command(bin: &str) -> Command {
    let mut cmd = Command::new(bin);
    cmd.env_clear();
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    } else {
        cmd.env("PATH", "/usr/bin:/bin");
    }
    cmd
}

async fn run_cmd(
    bin: &str,
    args: &serde_json::Value,
    timeout: Duration,
) -> Result<CmdOutput, String> {
    let cmd_args: Vec<String> = serde_json::from_value(args.clone())
        .map_err(|e| format!("expected JSON array of strings for args: {e}"))?;
    let mut cmd = build_command(bin);
    cmd.args(&cmd_args);
    finalize(cmd, timeout).await
}

async fn run_shell(args: &serde_json::Value, timeout: Duration) -> Result<CmdOutput, String> {
    let cmd_str = args
        .get("cmd")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "shell tool requires args.cmd (string)".to_string())?;
    let mut cmd = build_command("sh");
    cmd.args(["-c", cmd_str]);
    finalize(cmd, timeout).await
}

async fn finalize(mut cmd: Command, timeout: Duration) -> Result<CmdOutput, String> {
    let result = tokio::time::timeout(timeout, cmd.output()).await;
    match result {
        Err(_) => Err(format!("timeout after {}ms", timeout.as_millis())),
        Ok(Err(e)) => Err(format!("spawn error: {e}")),
        Ok(Ok(out)) => Ok(CmdOutput {
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
            exit_code: out.status.code(),
        }),
    }
}
