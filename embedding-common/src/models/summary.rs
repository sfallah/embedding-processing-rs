use crate::prelude::{EmbeddingDto, Serde, SummaryDto};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub summary_id: u64,
    pub document_id: u64,
    pub split_id: u64,
    pub split_sequence_id: i32,
    pub text_content: String,
    pub token_len: usize,
    pub centrality: f32,
}

impl Summary {
    pub fn new(
        summary_id: u64,
        document_id: u64,
        split_id: u64,
        split_sequence_id: i32,
        text_content: &str,
        token_len: usize,
        centrality: f32,
    ) -> Self {
        Summary {
            summary_id,
            document_id,
            split_id,
            split_sequence_id,
            text_content: text_content.to_string(),
            token_len,
            centrality,
        }
    }

    pub fn to_dto(
        &self,
        embedding: Option<EmbeddingDto>,
        query_distance: Option<f32>,
    ) -> SummaryDto {
        SummaryDto::new(
            self.summary_id,
            self.document_id,
            self.split_id,
            self.split_sequence_id,
            &self.text_content,
            self.token_len,
            self.centrality,
            embedding,
            query_distance,
        )
    }
}

impl Serde for Summary {}
