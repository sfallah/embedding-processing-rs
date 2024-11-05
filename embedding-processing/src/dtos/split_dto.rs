use crate::dtos::embedding_dto::EmbeddingDto;
use crate::dtos::summary_dto::SummaryDto;

#[derive(Debug, Clone)]
pub struct SplitDto {
    pub split_id: u64,
    pub sequence_id: i32,
    pub doc_id: u64,
    pub text_content: String,
    pub token_len: usize,
    pub summaries: Vec<SummaryDto>,
    pub embedding: Option<EmbeddingDto>,
}

impl SplitDto {
    pub fn new(
        split_id: u64,
        sequence_id: i32,
        doc_id: u64,
        text_content: &str,
        token_len: usize,
        summaries: Vec<SummaryDto>,
        embedding: Option<EmbeddingDto>,
    ) -> Self {
        SplitDto {
            split_id,
            sequence_id,
            doc_id,
            text_content: text_content.to_string(),
            token_len,
            summaries,
            embedding,
        }
    }
}
