use crate::prelude::Serde;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub document_id: u64,
    pub document_url: String,
    pub split_ids: Vec<u64>,
    pub summary_ids: Option<Vec<u64>>,
}

impl Document {
    pub fn new(url: &str, document_id: u64) -> Self {
        Document {
            document_id,
            document_url: url.to_string(),
            split_ids: Vec::new(),
            summary_ids: Some(Vec::new()),
        }
    }
}
impl Serde for Document {}
