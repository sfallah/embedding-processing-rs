use crate::prelude::{Embedding, EmbeddingDataType, EmbeddingDto, EmbeddingUser, Summary};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SummaryDto {
    pub summary_id: u64,
    pub document_id: u64,
    pub split_id: u64,
    pub split_sequence_id: i32,
    pub text_content: String,
    pub token_len: usize,
    pub centrality: f32,
    pub embedding: Option<EmbeddingDto>,
    pub query_distance: Option<f32>,
}

impl SummaryDto {
    pub fn new(
        summary_id: u64,
        document_id: u64,
        split_id: u64,
        split_sequence_id: i32,
        text_content: &str,
        token_len: usize,
        centrality: f32,
        embedding: Option<EmbeddingDto>,
        query_distance: Option<f32>,
    ) -> Self {
        SummaryDto {
            summary_id,
            document_id,
            split_id,
            split_sequence_id,
            text_content: text_content.to_string(),
            token_len,
            centrality,
            embedding,
            query_distance,
        }
    }
    pub fn to_model(&self) -> Summary {
        Summary::new(
            self.summary_id,
            self.document_id,
            self.split_id,
            self.split_sequence_id,
            &self.text_content,
            self.token_len,
            self.centrality,
        )
    }
    pub fn to_embedding_backend(&self) -> Option<Embedding> {
        self.embedding
            .as_ref()
            .map(|embedding_dto| embedding_dto.to_model(EmbeddingDataType::Summary))
    }

    pub fn to_embedding_user_model(&self, user_id: Uuid) -> Option<EmbeddingUser> {
        self.embedding
            .as_ref()
            .map(|embedding_dto| EmbeddingUser::new(embedding_dto.embedding_id, user_id))
    }
}
