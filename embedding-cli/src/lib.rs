pub mod file_io;

use clap::ValueEnum;
use std::path::PathBuf;


#[derive(clap::Parser, Debug, Clone)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// Number of parallel workers
    #[arg(long, default_value = "1")]
    pub np: usize,

    /// RocksDB directory path
    #[arg(long, default_value = "rocksdb_dir")]
    pub db_path: String,

    /// Max tokens in a split
    #[arg(long, default_value = "510")]
    pub max_tokens: usize,

    /// Dimensionality of the embeddings and hidden states.
    #[arg(long, default_value = "384")]
    pub n_embd: usize,

    /// Number of GPU layers
    #[arg(long, default_value = "1000")]
    pub ngl: usize,

    /// Minibatch size for document insertion
    #[arg(long, default_value = "8")]
    pub batch_size: usize,

    /// Merge level for the splits
    #[arg(long)]
    pub merge_level: Option<usize>,

    /// The path to the model
    #[arg(long, default_value = "models/all-minilm-l6-v2-q2_k.gguf")]
    pub model_path: String,

    /// Logging level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", value_enum)]
    pub log_level: LogLevel,

    /// Path to the text file to process
    #[arg(long, default_value = "embedding-processing/tests/test_data/")]
    pub file_path: PathBuf,

    #[arg(long)]
    pub user_id: Option<String>,

    #[arg(long, default_value = "index_dir")]
    pub index_dir: String,

    #[arg(long, default_value = "config.toml")]
    pub config_file: String,

}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn to_tracing_level(self) -> tracing::Level {
        match self {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

impl Args {
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_dir() {
            Ok(())
        } else {
            Err(format!("{} is not a directory", self.file_path.display()))
        }
    }
}
