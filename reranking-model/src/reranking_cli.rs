use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::ServerArgs;
use embedding_common::prelude::LogLevel;
use embedding_common::utils::tracting::setup_tracing;
use reranking_model::model_backend_config::ModelAppConfig;
use std::env;
use std::path::PathBuf;
use std::process::{Child, Command};
use tracing::error;

/// Given the name of another binary (as defined in Cargo.toml), find its path
/// relative to the current executable.
fn find_binary_path(binary_name: &str) -> PathBuf {
    // Get the current executable's directory.
    let mut exe_path = env::current_exe().expect("failed to get current exe");
    exe_path.pop(); // Remove the current executable's filename

    // Assume the other binaries are in the same directory.
    exe_path.push(binary_name);
    exe_path
}

/// Spawns another binary and returns its Child process handle.
fn spawn_process(
    binary_name: &str,
    config_file: &str,
    log_level: LogLevel,
) -> anyhow::Result<Child> {
    let binary_path = find_binary_path(binary_name);
    Command::new(binary_path)
        .args(&[
            "--config-file",
            config_file,
            "--log-level",
            &log_level.to_string(),
        ])
        // You can add arguments if needed, e.g. .arg("some_argument")
        .spawn()
        .map_err(|e| {
            error!("failed to spawn {}: {}", binary_name, e);
            e.into()
        })
}

fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();
    setup_tracing(args.log_level.to_tracing_level());
    let config = match ModelAppConfig::from_file(args.config_file.clone()) {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load config: {:?}", e);
            return Err(e);
        }
    };
    // Spawn multiple worker processes concurrently.
    let mut children = Vec::new();

    let broker = match spawn_process(
        "reranking_broker",
        args.config_file.as_str(),
        args.log_level,
    ) {
        Ok(broker) => broker,
        Err(e) => {
            error!("Failed to spawn model_broker: {:?}", e);
            return Err(e);
        }
    };
    children.push(broker);
    for _ in 0..config.zmq_config.num_workers {
        let worker = match spawn_process(
            "reranking_worker",
            args.config_file.as_str(),
            args.log_level,
        ) {
            Ok(worker) => worker,
            Err(e) => {
                error!("Failed to spawn model_worker: {:?}", e);
                return Err(e);
            }
        };
        children.push(worker);
    }
    // Optionally, wait for all children to finish.
    for mut child in children {
        let status = child.wait().expect("failed to wait on child");
        println!("Child exited with: {}", status);
    }
    Ok(())
}
