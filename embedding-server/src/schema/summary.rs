use crate::schema::document_status::RetrievalStatus;
use embedding_common::prelude::{Embedding, Serde, SummaryDto};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
