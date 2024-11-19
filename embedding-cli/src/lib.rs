pub mod file_io;

use clap::ValueEnum;
use std::path::PathBuf;


#[derive(clap::Parser, Debug, Clone)]
pub struct Cli {

    /// Logging level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", value_enum)]
    pub log_level: tracing::Level,

    /// Path to the text file to process
    #[arg(long, default_value = "embedding-processing/tests/test_data/")]
    pub file_path: PathBuf,

    #[arg(long)]
    pub user_id: Option<String>,

    #[arg(long, default_value = "config.toml")]
    pub config_file: String,

}

impl Cli {
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_dir() {
            Ok(())
        } else {
            Err(format!("{} is not a directory", self.file_path.display()))
        }
    }
}
