use crate::dtos::split_new::SplitDtoNew;
use crate::dtos::summary_new::SummaryDtoNew;

#[derive(Debug, Clone)]
pub struct DocumentDtoNew {
    pub document_id: u64,
    pub document_url: String,
    pub splits: Vec<SplitDtoNew>,
    pub summaries: Vec<SummaryDtoNew>,
}

impl DocumentDtoNew {
    pub fn new(
        document_id: u64,
        document_url: &str,
        splits: Vec<SplitDtoNew>,
        summaries: Vec<SummaryDtoNew>,
    ) -> Self {
        DocumentDtoNew {
            document_id,
            document_url: document_url.to_string(),
            splits,
            summaries,
        }
    }
}
