use embedding_common::types::Serde;
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
    /// id of the data which the embedding belongs to.
    pub data_id: u64,
    /// data type of the embedding.
    pub embedding_type: EmbeddingDataType,
    /// actual data of the embedding as a vector of floats.
    pub embedding: Vec<f32>,
}
impl Embedding {
    pub fn new(
        embedding_id: u64,
        data_id: u64,
        embedding_type: EmbeddingDataType,
        embedding: Vec<f32>,
    ) -> Self {
        Embedding {
            embedding_id,
            data_id,
            embedding_type,
            embedding,
        }
    }
}
impl Serde for Embedding {}
