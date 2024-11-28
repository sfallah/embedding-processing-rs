use crate::schema::embedding::EmbeddingUsageDto;
use crate::schema::search_mode::SearchModeType;
use embedding_common::dtos::DocumentDto;
use embedding_common::prelude::Serde;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    /// optional format for encoding the output embeddings.
    pub encoding_format: Option<String>,
    /// optional dimensionality of the embeddings.
    pub dimensions: Option<i32>,
    /// optional UUID of the user making the request.
    pub user_ids: Vec<Uuid>,
    /// optional boolean for detailed response
    pub verbose: Option<bool>,
}

impl Serde for DocumentQueryRequest {}

/// response structure for Document querying.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentQueryResponse {
    /// collecton of Document queries to be returned
    pub documents: Vec<DocumentDto>,
    /// optional model name for generating the embeddings.
    pub model: Option<String>,
    /// optional collection of query embeddings.
    pub query_embeddings: Option<Vec<f32>>,
    /// optional usage statistics associated with the embedding request.
    pub usage: Option<EmbeddingUsageDto>,
}

impl Serde for DocumentQueryResponse {}
