use clap::ValueEnum;

#[derive(clap::Parser, Debug, Clone)]
pub struct ServerArgs {
    #[arg(long)]
    pub config_file: String,
    /// Logging level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", value_enum)]
    pub log_level: LogLevel,
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
