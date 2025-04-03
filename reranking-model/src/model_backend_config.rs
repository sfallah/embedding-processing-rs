use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::prelude::Serde;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct ModelAppConfig {
    #[serde(rename = "model")]
    pub model_config: ModelConfig,
    #[serde(rename = "zmq")]
    pub zmq_config: ModelZmqConfig,
}

impl ConfigFromFile for ModelAppConfig {}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub gguf_file: String,
    #[serde(default)]
    pub cpu: bool,
    #[serde(default = "default_ngl")]
    pub ngl: usize,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

impl ModelConfig {
    pub fn new(gguf_file: String, verbose: bool) -> Self {
        ModelConfig {
            gguf_file,
            cpu: false,
            ngl: 1000,
            verbose,
            max_tokens: None,
        }
    }
}

#[allow(unused)]
fn default_instances() -> usize {
    1
}
#[allow(unused)]
fn default_ngl() -> usize {
    1000
}

impl Serde for ModelConfig {}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct ModelZmqConfig {
    // Model Host
    pub host: String,
    // ROUTER-DEALER Proxy
    // Model request port
    // Receive Embedding and Health-check requests
    pub frontend_port: usize,
    pub backend_port: usize,
    // Number of workers
    pub num_workers: usize,
}
