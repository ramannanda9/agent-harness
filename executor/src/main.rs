use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::process::Command;

#[derive(Deserialize)]
struct ToolRequest {
    tool: String,
    args: serde_json::Value,
    timeout_ms: u64,
}

#[derive(Serialize)]
struct ToolResult {
    success: bool,
    output: String,
    duration_ms: u64,
    error: Option<String>,
}

#[tokio::main]
async fn main() {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let req: ToolRequest = serde_json::from_str(&input).unwrap();
    let result = execute_tool(req).await;
    println!("{}", serde_json::to_string(&result).unwrap());
}

async fn execute_tool(req: ToolRequest) -> ToolResult {
    let start = Instant::now();
    let timeout = Duration::from_millis(req.timeout_ms);

    let outcome = match req.tool.as_str() {
        "kubectl" => run_cmd("kubectl", &req.args, timeout).await,
        "curl"    => run_cmd("curl",    &req.args, timeout).await,
        "shell"   => run_shell(&req.args, timeout).await,
        other     => Err(format!("Unknown tool: {other}")),
    };

    ToolResult {
        success: outcome.is_ok(),
        output:  outcome.unwrap_or_default(),
        duration_ms: start.elapsed().as_millis() as u64,
        error: None,
    }
}

async fn run_cmd(bin: &str, args: &serde_json::Value, timeout: Duration) -> Result<String, String> {
    let cmd_args: Vec<String> = serde_json::from_value(args.clone()).map_err(|e| e.to_string())?;
    let out = tokio::time::timeout(timeout, Command::new(bin).args(&cmd_args).output())
        .await.map_err(|_| "timeout".to_string())?
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

async fn run_shell(args: &serde_json::Value, timeout: Duration) -> Result<String, String> {
    let cmd = args["cmd"].as_str().ok_or("missing cmd")?;
    let out = tokio::time::timeout(timeout, Command::new("sh").args(["-c", cmd]).output())
        .await.map_err(|_| "timeout".to_string())?
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}
