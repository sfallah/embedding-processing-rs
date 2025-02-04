use crate::services::embeddings::{async_get_embeddings};
use async_channel::Sender;
use embedding_common::dtos::embedding_dto::EmbeddingDto;
use std::sync::Arc;
use tracing::trace;
use zeromq::ReqSocket;

#[tracing::instrument(skip(embedding_addr, sentences))]
pub async fn process_embedding(
    embedding_addr: String,
    embed_id: u64,
    sentences: Vec<String>,
    model_id: u64,
    n_embd: usize,
) -> anyhow::Result<EmbeddingDto> {
    trace!("Processing embedding...");
    //FIXME: n_embd is hardcoded to 384
    let embedding = async_get_embeddings(embedding_addr, &sentences, n_embd).await?;
    trace!("Embedding processed");
    Ok(EmbeddingDto::new(embed_id, embedding, model_id))
}
