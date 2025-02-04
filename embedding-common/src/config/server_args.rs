use clap::ValueEnum;

#[derive(clap::Parser, Debug, Clone)]
pub struct ServerArgs {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    #[arg(long, default_value = "config.toml")]
    pub config_file: String,

    /// Logging level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", value_enum)]
    pub log_level: LogLevel,


    #[arg(long)]
    pub user_id: Option<String>,

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
