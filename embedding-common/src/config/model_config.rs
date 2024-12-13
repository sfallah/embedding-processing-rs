use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub gguf_file: String,
    #[serde(default)]
    pub cpu: bool,
    #[serde(default = "default_instances")]
    pub instances: usize,
    #[serde(default = "default_ngl")]
    pub ngl: usize,
    #[serde(default)]
    pub verbose: bool,
}

impl ModelConfig {
    pub fn new(gguf_file: String, instances: usize, verbose: bool) -> Self {
        ModelConfig {
            gguf_file,
            cpu: false,
            instances,
            ngl: 1000,
            verbose,
        }
    }
}

fn default_instances() -> usize {
    1
}
fn default_ngl() -> usize {
    1000
}
