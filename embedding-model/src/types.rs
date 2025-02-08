use embedding_common::prelude::Serde;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingsRequest {
    pub req_id: usize,
    pub seq_id: usize,
    pub n_embd: usize,
    pub texts: Vec<String>,
}
impl EmbeddingsRequest {
    pub fn new(req_id: usize, seq_id: usize, n_embd: usize, texts: Vec<String>) -> Self {
        EmbeddingsRequest {
            req_id,
            seq_id,
            n_embd,
            texts,
        }
    }
}

impl Serde for EmbeddingsRequest {}

unsafe impl Sync for EmbeddingsRequest {}
unsafe impl Send for EmbeddingsRequest {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingsResponse {
    pub req_id: usize,
    pub seq_id: usize,
    pub n_embd: usize,
    pub embeddings: Vec<Vec<f32>>,
}
impl EmbeddingsResponse {
    pub fn new(req_id: usize, seq_id: usize, n_embd: usize, embeddings: Vec<Vec<f32>>) -> Self {
        EmbeddingsResponse {
            req_id,
            seq_id,
            n_embd,
            embeddings,
        }
    }
}

impl Serde for EmbeddingsResponse {}

unsafe impl Sync for EmbeddingsResponse {}
unsafe impl Send for EmbeddingsResponse {}
