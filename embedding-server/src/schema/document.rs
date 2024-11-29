use crate::schema::document_status::{DeletionStatus, DocumentInsertionStatus, RetrievalStatus};
use embedding_common::dtos::document_dto::DocumentDto;
use embedding_common::prelude::Serde;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// request structure for Document insertion.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentInsertionRequest {
    /// list of input data strings for which documents to be saved.
    pub input: Vec<String>,
    /// model name for generating the embeddings.
    pub model: String,
    /// optional UUID of the user making the request.
    pub user: Option<Uuid>,
    /// optional list of document urls data string to be saved.
    pub doc_urls: Option<Vec<String>>,
    /// optional boolean for detailed response
    pub verbose: Option<bool>,
}

impl Serde for DocumentInsertionRequest {}

/// response structure for Document insertion.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentInsertionResponse {
    /// response status of the operation.
    pub status: DocumentInsertionStatus,
    /// optional collection of document objects.
    pub documents: Option<Vec<DocumentDto>>,
}

impl Serde for DocumentInsertionResponse {}

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
