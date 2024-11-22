pub mod file_io;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Logging level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", value_enum)]
    pub log_level: tracing::Level,

    #[arg(long)]
    pub user_id: Option<String>,

    #[arg(long, default_value = "config.toml")]
    pub config_file: String,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Adds files to myapp
    Index {
        /// Path to the text file to process
        #[arg(long, default_value = "embedding-processing/tests/test_data/")]
        file_path: PathBuf,
    },
    Query {
        /// Path to the text file to process
        #[arg(long)]
        query: String,

        #[arg(long, default_value = "10")]
        top_k: usize,
    },
}

impl Cli {
    pub fn validate(&self) -> Result<(), String> {
        match self.command {
            Commands::Index { ref file_path } => {
                if file_path.is_dir() {
                    Ok(())
                } else {
                    Err(format!("{} is not a directory", file_path.display()))
                }
            }
            Commands::Query { .. } => Ok(()),
        }
    }
}
