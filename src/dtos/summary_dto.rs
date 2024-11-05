use crate::dtos::embedding_dto::EmbeddingDto;

#[derive(Debug, Clone)]
pub struct SummaryDto {
    pub summary_id: u64,
    pub document_id: u64,
    pub split_id: u64,
    pub split_sequence_id: i32,
    pub text_content: String,
    pub token_len: usize,
    pub centrality: f32,
    pub embedding: Option<EmbeddingDto>,
}

impl SummaryDto {
    pub fn new(
        summary_id: u64,
        document_id: u64,
        split_id: u64,
        split_sequence_id: i32,
        text_content: &str,
        token_len: usize,
        centrality: f32,
        embedding: Option<EmbeddingDto>,
    ) -> Self {
        SummaryDto {
            summary_id,
            document_id,
            split_id,
            split_sequence_id,
            text_content: text_content.to_string(),
            token_len,
            centrality,
            embedding,
        }
    }
}
