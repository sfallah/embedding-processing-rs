use crate::config::*;
use anyhow::anyhow;
use config::Config;
use serde::Deserialize;
use std::path::Path;
use tracing::error;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct AppConfig {
    #[serde(rename = "index")]
    pub index_config: IndexConfig,
    #[serde(rename = "splitter")]
    pub splitter_config: SplitterConfig,
    #[serde(rename = "model")]
    pub model_config: ModelConfig,
    #[serde(rename = "database")]
    pub database_config: DatabaseConfig,
    #[serde(rename = "zmq")]
    pub zmq_config: ZmqConfig,
}

impl AppConfig {
    pub fn from_file(conf_file: String) -> anyhow::Result<Self> {
        let config_file_path = Path::new(conf_file.as_str());
        if !config_file_path.exists() {
            error!("Config file not found: {:?}", conf_file);
            return Err(anyhow!("Config file not found: {:?}", conf_file));
        } else if !config_file_path.is_file() {
            error!("Invalid config file: {:?}", conf_file);
            return Err(anyhow!("Invalid config file: {:?}", conf_file));
        } else {
            if let Some(ext) = config_file_path.extension() {
                if ext != "toml" {
                    error!(
                        "Invalid config file extension: {:?}, file: {:?}",
                        ext, conf_file
                    );
                    return Err(anyhow!("Invalid config file extension: {:?}", ext));
                }
            }
        }
        let conf = Config::builder()
            .add_source(config::File::with_name(conf_file.as_str()))
            .build()?;
        conf.try_deserialize()
            .map_err(|e| anyhow!("Failed to deserialize config: {:?}", e))
    }
    #[tracing::instrument]
    pub async fn from_file_async(conf_file: String) -> anyhow::Result<Self> {
        let conf_file = conf_file.clone();
        tokio::task::spawn_blocking(move || Self::from_file(conf_file)).await?
    }
}
