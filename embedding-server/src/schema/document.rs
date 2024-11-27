use serde::{Deserialize, Serialize};
use embedding_common::dtos::embedding_dto::EmbeddingDto;
use embedding_common::dtos::split_dto::SplitDto;
use embedding_common::dtos::summary_dto::SummaryDto;
use uuid::Uuid;
use embedding_common::dtos::document_dto::DocumentDto;
use embedding_common::prelude::Serde;
use crate::schema::document_status::{DeletionStatus, RetrievalStatus};
/// request structure for Document retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentRetrievalRequest {
    // document id to be retrieved
    pub document_id: Option<u64>,
    // document url to be retrieved
    pub document_url: Option<String>,
    // optional UUID of the user making the request.
    pub user: Option<Uuid>,
    // optional boolean for detailed response
    pub verbose: Option<bool>,
}
impl Serde for DocumentRetrievalRequest {}

/// response structure for Document retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentRetrievalResponse {
    /// response status of the operation.
    pub status: RetrievalStatus,
    /// optional collection of document object.
    pub document: Option<DocumentDto>,
    /// optional collection of embedding objects.
    pub embeddings: Option<Vec<EmbeddingDto>>,
}
impl Serde for DocumentRetrievalResponse {}

/// request structure for Document deletion.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentDeletionRequest {
    // document id to be retrieved
    pub document_id: u64,
    // optional UUID of the user making the request.
    pub user: Option<Uuid>,
}
impl Serde for DocumentDeletionRequest {}

/// response structure for Document deletion.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentDeletionResponse {
    /// response status of the operation.
    pub status: DeletionStatus,
}
impl Serde for DocumentDeletionResponse {}

/// request structure for Document Splits retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentSplitsRetrievalRequest {
    // document id of the splits of documents to be retrieved
    pub document_id: u64,
    // optional UUID of the user making the request.
    pub user: Option<Uuid>,
    // optional boolean for detailed response
    pub verbose: Option<bool>,
}
impl Serde for DocumentSplitsRetrievalRequest {}

/// response structure for Document Splits retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentSplitsRetrievalResponse {
    /// response status of the operation.
    pub status: RetrievalStatus,
    // document id of the splits of documents to be retrieved
    pub document_id: u64,
    // document url of the splits of documents to be retrieved
    pub document_url: String,
    /// optional collection of split objects.
    pub splits: Option<Vec<SplitDto>>,
    /// optional collection of embedding objects.
    pub embeddings: Option<Vec<Vec<EmbeddingDto>>>,
}
impl Serde for DocumentSplitsRetrievalResponse {}

/// request structure for Document Summaries retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentSummariesRetrievalRequest {
    // document id of the summaries of documents to be retrieved
    pub document_id: u64,
    // optional UUID of the user making the request.
    pub user: Option<Uuid>,
    // optional boolean for detailed response
    pub verbose: Option<bool>,
}
impl Serde for DocumentSummariesRetrievalRequest {}

/// response structure for Document Summaries retrieval.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentSummariesRetrievalResponse {
    /// response status of the operation.
    pub status: RetrievalStatus,
    // document id of the summaries of documents to be retrieved
    pub document_id: u64,
    // document url of the splits of documents to be retrieved
    pub document_url: String,
    /// optional collection of summaries objects.
    pub summaries: Option<Vec<SummaryDto>>,
    /// optional collection of embedding objects.
    pub embeddings: Option<Vec<Vec<EmbeddingDto>>>,
}
impl Serde for DocumentSummariesRetrievalResponse {}
