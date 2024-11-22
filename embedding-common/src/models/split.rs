use crate::Serde;
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    pub split_id: u64,
    pub sequence_id: i32,
    pub doc_id: u64,
    pub embedding_id: u64,
    pub text_content: String,
    pub token_len: usize,
    pub summary_ids: Option<Vec<u64>>,
}

impl Split {
    pub fn new(
        split_id: u64,
        sequence_id: i32,
        doc_id: u64,
        embedding_id: u64,
        text_content: &str,
        token_len: usize,
        summary_ids: Option<Vec<u64>>,
    ) -> Self {
        Split {
            split_id,
            sequence_id,
            doc_id,
            embedding_id,
            text_content: text_content.to_string(),
            token_len,
            summary_ids,
        }
    }
}

impl Serde for Split {}
