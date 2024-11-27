use crate::prelude::{Document, SplitDto, SummaryDto};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentDto {
    pub document_id: u64,
    pub document_url: String,
    pub splits: Vec<SplitDto>,
    pub summaries: Vec<SummaryDto>,
    pub top_query_distance: Option<f32>,
}

impl DocumentDto {
    pub fn new(
        document_id: u64,
        document_url: &str,
        splits: Vec<SplitDto>,
        summaries: Vec<SummaryDto>,
    ) -> Self {
        DocumentDto {
            document_id,
            document_url: document_url.to_string(),
            splits,
            summaries,
            top_query_distance: None,
        }
    }
    pub fn to_model(&self) -> Document {
        let split_ids = self.splits.iter().map(|s| s.split_id).collect();

        let summary_ids = if self.summaries.is_empty() {
            None
        } else {
            Some(self.summaries.iter().map(|s| s.summary_id).collect())
        };
        Document {
            document_id: self.document_id,
            document_url: self.document_url.clone(),
            split_ids,
            summary_ids,
        }
    }
}
