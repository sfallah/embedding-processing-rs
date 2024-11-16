use crate::models::embedding::{Embedding, EmbeddingDataType};

/// represents an individual embedding with metadata.
#[derive(Clone, Debug)]
pub struct EmbeddingDto {
    pub embedding_id: u64,
    /// actual data of the embedding as a vector of floats.
    pub embedding: Vec<f32>,
    pub model_id: u64,
}

impl EmbeddingDto {
    pub fn new(embedding_id: u64, embedding: Vec<f32>, model_id: u64) -> Self {
        EmbeddingDto {
            embedding_id,
            embedding,
            model_id,
        }
    }
    pub fn to_model(&self, data_id: u64, embedding_type: EmbeddingDataType) -> Embedding {
        Embedding::new(
            self.embedding_id,
            data_id,
            embedding_type,
            self.embedding.clone(),
            self.model_id,
        )
    }
}
