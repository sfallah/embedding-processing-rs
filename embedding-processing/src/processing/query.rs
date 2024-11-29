use std::sync::Arc;
use async_channel::Sender;
use tracing::trace;
use crate::processing::context::ProcessingContext;
use crate::processing::splitter::split_text;
use crate::processing::utils;
use crate::services::embeddings::{async_get_embeddings, EmbeddingsRequest};

pub async fn process_query(
    ctx: Arc<ProcessingContext>,
    sender: Arc<Sender<EmbeddingsRequest>>,
    query: String,
) -> anyhow::Result<Vec<Vec<f32>>> {
    trace!("Processing embedding...");
    let query_splits = split_text(ctx.splitter.clone(), query.into_bytes()).await?;
    let query_split_texts = utils::splits_texts(&query_splits);
    let embedding = async_get_embeddings(sender.clone(), &query_split_texts, ctx.n_embd).await?;
    let mut embeddings = Vec::new();
    for i in 0..query_split_texts.len() {
        let i = i * ctx.n_embd;
        let j = (i + 1) * ctx.n_embd;
        embeddings.push(embedding[i..j].to_vec());
    }
    trace!("Embedding processed");
    Ok(embeddings)
}