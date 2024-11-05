use crate::dtos::split_dto::SplitDto;
use crate::dtos::summary_dto::SummaryDto;

#[derive(Debug, Clone)]
pub struct DocumentDto {
    pub document_id: u64,
    pub document_url: String,
    pub splits: Vec<SplitDto>,
    pub summaries: Vec<SummaryDto>,
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
        }
    }
}
