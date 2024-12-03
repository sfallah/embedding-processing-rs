use crate::prelude::{Document, SplitDto, SummaryDto};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentDto {
    pub document_id: u64,
    pub document_url: String,
    pub splits: Vec<SplitDto>,
    pub summaries: Option<Vec<SummaryDto>>,
}

impl DocumentDto {
    pub fn new(
        document_id: u64,
        document_url: &str,
        splits: Vec<SplitDto>,
        summaries: Option<Vec<SummaryDto>>,
    ) -> Self {
        DocumentDto {
            document_id,
            document_url: document_url.to_string(),
            splits,
            summaries,
        }
    }
    pub fn to_model(&self) -> Document {
        let split_ids = self.splits.iter().map(|s| s.split_id).collect();

        let summary_ids = self.summaries.clone().map(|summary_dto| summary_dto.iter().map(|s| s.summary_id).collect());
        Document {
            document_id: self.document_id,
            document_url: self.document_url.clone(),
            split_ids,
            summary_ids,
        }
    }
}
