use serde::{Deserialize, Serialize};
use uuid::Uuid;
use embedding_common::dtos::summary_dto::SummaryDto;
use embedding_common::models::embedding::Embedding;
use embedding_common::Serde;
use crate::dtos::document_status::RetrievalStatus;

/// request structure for Summary retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct SummaryRetrievalRequest {
    // summary id to be retrieved
    pub summary_id: u64,
    // optional UUID of the user making the request.
    pub user: Option<Uuid>,
    // optional boolean for detailed response
    pub verbose: Option<bool>,
}

impl Serde for SummaryRetrievalRequest {}

/// response structure for Summary retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct SummaryRetrievalResponse {
    /// response status of the operation.
    pub status: RetrievalStatus,
    /// optional collection of summary object.
    pub summary: Option<SummaryDto>,
    /// optional collection of embedding objects.
    pub embeddings: Option<Vec<Embedding>>,
}
impl Serde for SummaryRetrievalResponse {}