use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
#[serde(deny_unknown_fields)]
pub struct SplitterConfig {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub hf_model_id: Option<String>,
    pub merge_level: Option<usize>,
    pub patterns: Option<Vec<Vec<String>>>,
}

fn default_max_tokens() -> usize {
    400
}
