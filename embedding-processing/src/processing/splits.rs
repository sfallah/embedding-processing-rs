use crate::processing::context::ProcessingContext;
use crate::processing::embeddings::process_embedding;
use crate::processing::summaries::process_summaries;
use crate::services::embeddings::EmbeddingsRequest;
use embedding_common::dtos::split_dto::SplitDto;
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use std::sync::Arc;
use tracing::trace;

#[tracing::instrument(skip(ctx, split_res, embed_sender))]
pub async fn process_split(
    ctx: Arc<ProcessingContext>,
    split_res: Arc<SplitResultLite>,
    embed_sender: Arc<async_channel::Sender<EmbeddingsRequest>>,
    doc_id: u64,
    seq_id: i32,
) -> anyhow::Result<SplitDto> {
    let split_id = ctx.hasher.hash(&format!("{}{}", doc_id, seq_id));
    trace!("Processing split: {}", split_id);

    let embedding = process_embedding(
        embed_sender.clone(),
        split_id,
        vec![split_res.split_string.clone()],
        ctx.model_id,
        ctx.n_embd,
    )
    .await?;
    let summaries = process_summaries(
        ctx.clone(),
        embed_sender.clone(),
        split_res.split_string.clone(),
        doc_id,
        split_id,
    )
    .await?;
    Ok(SplitDto::new(
        split_id,
        seq_id,
        doc_id,
        split_res.split_string.as_str(),
        split_res.tokens.len(),
        summaries,
        Some(embedding),
        None,
    ))
}
