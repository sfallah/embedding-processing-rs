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
        }
    }
    pub fn to_model(&self) -> Summary {
        let embedding_id = self.embedding.as_ref().map_or(0, |e| e.embedding_id);

        Summary {
            summary_id: self.summary_id,
            document_id: self.document_id,
            split_id: self.split_id,
            split_sequence_id: self.split_sequence_id,
            embedding_id,
            text_content: self.text_content.clone(),
            token_len: self.token_len,
            centrality: self.centrality,
        }
    }
    pub fn to_embedding_model(&self) -> Option<Embedding> {
        self.embedding.as_ref().map(|embedding_dto| {
            embedding_dto.to_model(self.summary_id, EmbeddingDataType::Summary)
        })
    }

    pub fn to_embedding_user_model(&self, user_id: Uuid) -> Option<EmbeddingUser> {
        self.embedding
            .as_ref()
            .map(|embedding_dto| EmbeddingUser::new(embedding_dto.embedding_id, user_id))
    }
}
