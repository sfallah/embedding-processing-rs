use std::path::PathBuf;
use clap::ValueEnum;

pub mod zmq;
pub mod utils;
pub mod schema;

pub mod api;

#[derive(clap::Parser, Debug, Clone)]
pub struct ServerArgs {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    #[arg(long, default_value = "config.toml")]
    pub config_file: String,

    // Host of ZeroMQ server
    #[command()]
    #[arg(long, default_value = "127.0.0.1")]
    pub server_host: String,

    // Port of frontend ZeroMQ server
    #[command()]
    #[clap(long, default_value = "5556")]
    pub frontend_port: usize,

    // Port of backend ZeroMQ server
    #[command()]
    #[clap(long, default_value = "5560")]
    pub backend_port: usize,

    // Number of parallel workers
    #[command()]
    #[clap(long, default_value = "2")]
    pub np: usize,

    // RocksDB directory path
    #[arg(long)]
    pub db_path: Option<String>,

    /// max tokens in a split
    #[arg(long, default_value = "500")]
    pub max_tokens: usize,

    /// number of gpu layers
    #[arg(long, default_value = "1000")]
    pub ngl: usize,

    /// Batch size for document insertion
    #[arg(long, default_value = "512")]
    pub batch_size: usize,

    /// merge level for the splits
    #[arg(long)]
    pub merge_level: Option<usize>,

    /// The path to the model
    #[arg(long, default_value = "models/bge-base-en-v1.5.gguf")]
    pub model_path: String,

    /// Logging level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", value_enum)]
    pub log_level: LogLevel,

    /// Path to the text file to process
    #[arg(long, default_value = "embedding-processing/tests/test_data/superlinear.txt")]
    pub file_path: PathBuf,

    #[arg(long)]
    pub user_id: Option<String>,

    #[arg(long, default_value = "index_dir")]
    pub index_dir: Option<String>,
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