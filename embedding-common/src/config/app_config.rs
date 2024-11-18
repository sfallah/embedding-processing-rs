use anyhow::anyhow;
use config::Config;
use serde::Deserialize;
use crate::config::IndexConfig;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct AppConfig {
    pub index_config: IndexConfig,
}

impl AppConfig {
    pub fn from_file(conf_file: String) -> anyhow::Result<Self> {
        let conf = Config::builder()
            .add_source(config::File::with_name(conf_file.as_str()))
            .build()?;
        conf.try_deserialize().map_err(|e| anyhow!("Failed to deserialize config: {:?}", e))
    }
}