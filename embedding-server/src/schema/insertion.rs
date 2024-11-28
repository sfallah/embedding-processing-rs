use crate::schema::document_status::DocumentInsertionStatus;
use crate::schema::embedding::EmbeddingUsageDto;
use embedding_common::dtos::DocumentDto;
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
    /// optional collection of embedding objects.
    //pub embeddings: Option<Vec<EmbeddingDto>>,
    /// optional usage statistics associated with the embedding request.
    pub usage: Option<EmbeddingUsageDto>,
}

impl Serde for DocumentInsertionResponse {}
