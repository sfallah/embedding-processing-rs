use crate::prelude::{Serde, SummaryDto};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub summary_id: u64,
    pub document_id: u64,
    pub split_id: u64,
    pub split_sequence_id: i32,
    pub embedding_id: u64,
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
        embedding_id: u64,
        text_content: &str,
        token_len: usize,
        centrality: f32,
    ) -> Self {
        Summary {
            summary_id,
            document_id,
            split_id,
            split_sequence_id,
            embedding_id,
            text_content: text_content.to_string(),
            token_len,
            centrality,
        }
    }

    pub fn to_dto(&self) -> SummaryDto {
        SummaryDto {
            summary_id: self.summary_id,
            document_id: self.document_id,
            split_id: self.split_id,
            split_sequence_id: self.split_sequence_id,
            text_content: self.text_content.clone(),
            token_len: self.token_len,
            centrality: self.centrality,
            embedding: None,
        }
    }
}

impl Serde for Summary {}
