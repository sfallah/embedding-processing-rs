use crate::prelude::Serde;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
pub struct EmbeddingFilterInfo {
    pub embed_id: u64,
    pub doc_id: u64,
    pub user_uuid: Uuid,
}

impl EmbeddingFilterInfo {
    pub fn new(embed_id: u64, doc_id: u64, user_uuid: Uuid) -> Self {
        EmbeddingFilterInfo {
            embed_id,
            doc_id,
            user_uuid,
        }
    }
}
impl Serde for EmbeddingFilterInfo {}
