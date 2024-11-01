use crate::dtos::embedding_new::EmbeddingNewDto;
use crate::dtos::summary_new::SummaryDtoNew;

#[derive(Debug, Clone)]
pub struct SplitDtoNew {
    pub split_id: u64,
    pub sequence_id: i32,
    pub doc_id: u64,
    pub text_content: String,
    pub token_len: usize,
    pub summaries: Vec<SummaryDtoNew>,
    pub embedding: Option<EmbeddingNewDto>,
}

impl SplitDtoNew {
    pub fn new(
        split_id: u64,
        sequence_id: i32,
        doc_id: u64,
        text_content: &str,
        token_len: usize,
        summaries: Vec<SummaryDtoNew>,
        embedding: Option<EmbeddingNewDto>,
    ) -> Self {
        SplitDtoNew {
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
