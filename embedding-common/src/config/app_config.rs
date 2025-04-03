use crate::config::config_file::ConfigFromFile;
use crate::config::endpoints::EndpointsConfig;
use crate::config::model_info_config::ModelInfoConfig;
use crate::config::*;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct AppConfig {
    #[serde(rename = "index")]
    pub index_config: IndexConfig,
    #[serde(rename = "splitter")]
    pub splitter_config: SplitterConfig,
    #[serde(rename = "embedding_model")]
    pub embedding_model_info: ModelInfoConfig,
    #[serde(rename = "database")]
    pub database_config: DatabaseConfig,
    #[serde(rename = "zmq")]
    pub zmq_config: ZmqConfig,
    #[serde(rename = "endpoints")]
    pub endpoints: EndpointsConfig,
}

impl ConfigFromFile for AppConfig {}
