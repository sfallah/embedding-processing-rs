
/// represents an individual embedding with metadata.
#[derive(Clone, Debug)]
pub struct EmbeddingNewDto {
    pub embedding_id: u64,
    /// actual data of the embedding as a vector of floats.
    pub embedding: Vec<f32>,
}

impl EmbeddingNewDto {
    pub fn new(embedding_id: u64, embedding: Vec<f32>) -> Self {
        EmbeddingNewDto {
            embedding_id,
            embedding,
        }
    }
}