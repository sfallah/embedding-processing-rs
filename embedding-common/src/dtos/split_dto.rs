use crate::prelude::{
    Embedding, EmbeddingDataType, EmbeddingDto, EmbeddingFilterInfo, Rank, Split, SummaryDto,
};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SplitDto {
    pub split_id: u64,
    pub sequence_id: i32,
    pub doc_id: u64,
    pub text_content: String,
    pub token_len: usize,
    pub summaries: Vec<SummaryDto>,
    pub embedding: Option<EmbeddingDto>,
    pub query_distance: Option<f32>,
    pub rank: Option<Rank>,
}

impl Hash for SplitDto {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.split_id.hash(state);
    }
}

impl PartialEq for SplitDto {
    fn eq(&self, other: &Self) -> bool {
        self.split_id == other.split_id
    }
}

impl Eq for SplitDto {}

impl SplitDto {
    pub fn new(
        split_id: u64,
        sequence_id: i32,
        doc_id: u64,
        text_content: &str,
        token_len: usize,
        summaries: Vec<SummaryDto>,
        embedding: Option<EmbeddingDto>,
        query_distance: Option<f32>,
        rank: Option<Rank>,
    ) -> Self {
        SplitDto {
            split_id,
            sequence_id,
            doc_id,
            text_content: text_content.to_string(),
            token_len,
            summaries,
            embedding,
            query_distance,
            rank,
        }
    }
    pub fn to_model(&self) -> Split {
        let summary_ids = if self.summaries.is_empty() {
            None
        } else {
            Some(self.summaries.iter().map(|s| s.summary_id).collect())
        };

        Split::new(
            self.split_id,
            self.sequence_id,
            self.doc_id,
            &self.text_content,
            self.token_len,
            summary_ids,
        )
    }
    pub fn to_embedding_backend(&self) -> Option<Embedding> {
        self.embedding
            .as_ref()
            .map(|embedding_dto| embedding_dto.to_model(EmbeddingDataType::Split))
    }

    pub fn to_embedding_user_model(&self, user_id: Uuid) -> Option<EmbeddingFilterInfo> {
        self.embedding.as_ref().map(|embedding_dto| {
            EmbeddingFilterInfo::new(embedding_dto.embedding_id, self.doc_id, user_id)
        })
    }
}
