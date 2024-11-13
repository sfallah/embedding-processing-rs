use crate::dtos::embedding_dto::EmbeddingDto;
use crate::dtos::summary_dto::SummaryDto;
use crate::models::embedding::{Embedding, EmbeddingDataType};
use crate::models::split::Split;

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
    pub fn to_model(&self) -> Split {
        let embedding_id = self.embedding.as_ref().map_or(0, |e| e.embedding_id);

        let summary_ids = if self.summaries.is_empty() {
            None
        } else {
            Some(self.summaries.iter().map(|s| s.summary_id).collect())
        };

        Split {
            split_id: self.split_id,
            sequence_id: self.sequence_id,
            doc_id: self.doc_id,
            embedding_id,
            text_content: self.text_content.clone(),
            token_len: self.token_len,
            summary_ids,
        }
    }
    pub fn to_embedding_model(&self) -> Option<Embedding> {
        self.embedding.as_ref().map(|embedding_dto| {
            embedding_dto.to_model(
                self.split_id,
                EmbeddingDataType::Split,
            )
        })
    }
}
