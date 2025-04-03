use crate::config::config_file::ConfigFromFile;
use crate::config::ZmqConfig;
use crate::prelude::Serde;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct ModelBackendAppConfig {
    #[serde(rename = "model")]
    pub model_config: ModelConfig,
    #[serde(rename = "zmq")]
    pub zmq_config: ZmqConfig,
}

impl ConfigFromFile for ModelBackendAppConfig {}

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
