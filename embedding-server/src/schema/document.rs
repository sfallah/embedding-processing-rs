use crate::schema::document_status::{DeletionStatus, DocumentInsertionStatus, RetrievalStatus};
use crate::schema::search_mode::SearchModeType;
use embedding_common::dtos::document_dto::DocumentDto;
use embedding_common::prelude::Serde;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// request structure for Document insertion.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentInsertionRequest {
    /// input data string for document to be saved.
    pub input: String,
    /// model name for generating the embeddings.
    pub model: String,
    /// UUID of the user making the request.
    pub user: Uuid,
    /// document url data string to be saved.
    pub doc_url: String,
    /// optional boolean for detailed response
    pub verbose: Option<bool>,
}

impl Serde for DocumentInsertionRequest {}

/// response structure for Document insertion.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentInsertionResponse {
    /// response status of the operation.
    pub status: DocumentInsertionStatus,
    /// optional document object.
    pub document: Option<DocumentDto>,
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

/// request structure for Document querying
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentQueryRequest {
    /// string to query the index
    pub input: String,
    /// specifies the model for embeddings.
    pub model: String,
    /// optional search mode for querying the index.
    pub search_mode: Option<SearchModeType>,
    /// optional i32 to get k closest query result.
    pub top_k: Option<i32>,
    /// optional UUID of the user making the request.
    pub workspace_ids: Vec<Uuid>,
    pub doc_ids: Vec<u64>,
    /// optional
    pub rerank: Option<bool>,
    /// optional boolean for detailed response
    pub verbose: Option<bool>,
}

impl Serde for DocumentQueryRequest {}

/// response structure for Document querying.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentQueryResponse {
    /// collection of Document queries to be returned
    pub documents: Vec<DocumentDto>,
    /// optional model name for generating the embeddings.
    pub model: Option<String>,
    /// optional collection of query embeddings.
    pub query_embeddings: Option<Vec<f32>>,
}

impl Serde for DocumentQueryResponse {}
