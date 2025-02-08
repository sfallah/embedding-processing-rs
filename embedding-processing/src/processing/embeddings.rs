use crate::processing::utils::async_get_embeddings;
use embedding_common::dtos::embedding_dto::EmbeddingDto;
use std::sync::Arc;
use crate::processing::context::ProcessingContext;

#[tracing::instrument(skip(ctx, sentences))]
pub async fn process_embedding(
    ctx: Arc<ProcessingContext>,
    embed_id: u64,
    sentences: Vec<String>,
    model_id: u64,
    n_embd: usize,
) -> anyhow::Result<EmbeddingDto> {
    let embeddings =
        async_get_embeddings(ctx.zmq_context.clone(), ctx.model_endpoint.clone(), n_embd, sentences.clone()).await?;
    Ok(EmbeddingDto::new(embed_id, embeddings, model_id))
}
