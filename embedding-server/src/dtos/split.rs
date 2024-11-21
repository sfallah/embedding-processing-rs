use serde::{Deserialize, Serialize};
use uuid::Uuid;
use embedding_common::dtos::embedding_dto::EmbeddingDto;
use embedding_common::dtos::split_dto::SplitDto;
use embedding_common::Serde;
use crate::dtos::document_status::RetrievalStatus;

/// request structure for Split retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct SplitRetrievalRequest {
    // split id to be retrieved
    pub split_id: u64,
    // optional UUID of the user making the request.
    pub user: Option<Uuid>,
    // optional boolean for detailed response
    pub verbose: Option<bool>,
}

impl Serde for SplitRetrievalRequest {}

/// response structure for Split retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct SplitRetrievalResponse {
    /// response status of the operation.
    pub status: RetrievalStatus,
    /// optional collection of split objects.
    pub split: Option<SplitDto>,
    /// optional collection of embedding objects.
    pub embeddings: Option<Vec<EmbeddingDto>>,
}
impl Serde for SplitRetrievalResponse {}