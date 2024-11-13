use embedding_common::dtos::embedding_dto::EmbeddingDto;
use crate::services::embeddings::{async_get_embeddings, EmbeddingsRequest};
use async_channel::Sender;
use std::sync::Arc;
use tracing::trace;

#[tracing::instrument(skip(sender, sentences))]
pub async fn process_embedding(
    sender: Arc<Sender<EmbeddingsRequest>>,
    embed_id: u64,
    sentences: Vec<String>,
) -> anyhow::Result<EmbeddingDto> {
    trace!("Processing embedding...");
    let embedding = async_get_embeddings(sender.clone(), &sentences, 384).await?;
    trace!("Embedding processed");
    Ok(EmbeddingDto::new(embed_id, embedding))
}
