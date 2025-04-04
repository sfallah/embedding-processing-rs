use crate::prelude::Serde;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingsRequest {
    pub texts: Vec<String>,
}
impl EmbeddingsRequest {
    pub fn new(texts: Vec<String>) -> Self {
        EmbeddingsRequest { texts }
    }
}

impl Serde for EmbeddingsRequest {}

unsafe impl Sync for EmbeddingsRequest {}
unsafe impl Send for EmbeddingsRequest {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingsResponse {
    pub model_id: u64,
    pub embeddings: Vec<Vec<f32>>,
    pub error: Option<String>,
}
impl EmbeddingsResponse {
    pub fn new(model_id: u64, embeddings: Vec<Vec<f32>>, error: Option<String>) -> Self {
        EmbeddingsResponse {
            model_id,
            embeddings,
            error,
        }
    }
}

impl Serde for EmbeddingsResponse {}

unsafe impl Sync for EmbeddingsResponse {}
unsafe impl Send for EmbeddingsResponse {}
