use embedding_common::prelude::Serde;
use serde::{Deserialize, Serialize};

/// holds some statistics for embeddings.
#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingUsageDto {
    /// number of tokens that have been processed for the query
    pub prompt_tokens: i32,
    /// total number of tokens if the return chunks would be tokenized (with embed model tokenizer)
    pub total_tokens: i32,
}

impl Serde for EmbeddingUsageDto {}
