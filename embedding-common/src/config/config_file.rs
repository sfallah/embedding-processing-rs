use anyhow::anyhow;
use config::Config;
use serde::Deserialize;
use std::path::Path;
use tracing::{error, info};

pub trait ConfigFromFile
where
    Self: Sized + Send,
    Self: Deserialize<'static>,
{
    fn from_file(conf_file: String) -> anyhow::Result<Self> {
        info!("Loading config from file: {:?}", conf_file);
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
}
