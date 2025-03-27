use crate::config::config_file::ConfigFromFile;
use crate::config::*;
use serde::Deserialize;
use crate::config::endpoints::EndpointsConfig;

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
    #[serde(rename = "endpoints")]
    pub endpoints: EndpointsConfig,
}

impl ConfigFromFile for AppConfig {}
