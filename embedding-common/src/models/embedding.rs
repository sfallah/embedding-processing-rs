use crate::prelude::Serde;
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

/// Represent the enum for data type of the embedding.
#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum EmbeddingDataType {
    Split = 1,
    Summary = 2,
}

/// represents an individual embedding with metadata.
#[derive(Serialize, Deserialize, Debug)]
pub struct Embedding {
    /// id of the embedding data.
    pub embedding_id: u64,
    /// data type of the embedding.
    pub embedding_type: EmbeddingDataType,
    /// actual data of the embedding as a vector of floats.
    pub embedding: Vec<f32>,
    pub model_id: u64,
}
impl Embedding {
    pub fn new(
        embedding_id: u64,
        embedding_type: EmbeddingDataType,
        embedding: Vec<f32>,
        model_id: u64,
    ) -> Self {
        Embedding {
            embedding_id,
            embedding_type,
            embedding,
            model_id,
        }
    }
}
impl Serde for Embedding {}
