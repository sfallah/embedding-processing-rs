use crate::dtos::embedding_new::EmbeddingNewDto;
use crate::services::embeddings::{async_get_embeddings, EmbeddingsRequest};
use async_std::channel::Sender;
use std::sync::Arc;

pub async fn process_embedding(
    sender: Arc<Sender<EmbeddingsRequest>>,
    embed_id: u64,
    sentences: Vec<String>,
) -> anyhow::Result<EmbeddingNewDto> {
    let embedding = async_get_embeddings(sender.clone(), &sentences, 384).await?;
    Ok(EmbeddingNewDto::new(embed_id, embedding))
}
