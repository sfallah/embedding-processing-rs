use crate::prelude::Serde;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
pub struct EmbeddingUser {
    pub embed_id: u64,
    pub user_uuid: Uuid,
}

impl EmbeddingUser {
    pub fn new(embed_id: u64, user_uuid: Uuid) -> Self {
        EmbeddingUser {
            embed_id,
            user_uuid,
        }
    }
}
impl Serde for EmbeddingUser {}
