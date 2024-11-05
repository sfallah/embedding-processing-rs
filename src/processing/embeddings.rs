use crate::dtos::embedding_dto::EmbeddingDto;
use crate::services::embeddings::{async_get_embeddings, EmbeddingsRequest};
use std::sync::Arc;
use async_channel::Sender;

pub async fn process_embedding(
    sender: Arc<Sender<EmbeddingsRequest>>,
    embed_id: u64,
    sentences: Vec<String>,
) -> anyhow::Result<EmbeddingDto> {
    let embedding = async_get_embeddings(sender.clone(), &sentences, 384).await?;
    Ok(EmbeddingDto::new(embed_id, embedding))
}
