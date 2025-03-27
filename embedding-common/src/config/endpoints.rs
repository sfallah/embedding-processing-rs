use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct EndpointsConfig {
    pub embedding_endpoint: String,
    pub reranking_endpoint: Option<String>,
}
