/// represents an individual embedding with metadata.
#[derive(Clone, Debug)]
pub struct EmbeddingDto {
    pub embedding_id: u64,
    /// actual data of the embedding as a vector of floats.
    pub embedding: Vec<f32>,
}

impl EmbeddingDto {
    pub fn new(embedding_id: u64, embedding: Vec<f32>) -> Self {
        EmbeddingDto {
            embedding_id,
            embedding,
        }
    }
}
