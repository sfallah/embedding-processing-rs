use crate::processing::context::ProcessingContext;
use crate::processing::embeddings::process_embedding;
use crate::processing::summaries::process_summaries;
use embedding_common::dtos::split_dto::SplitDto;
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use std::rc::Rc;
use tracing::trace;

#[tracing::instrument(skip(ctx, split_res))]
pub fn process_split(
    ctx: Rc<ProcessingContext>,
    split_res: &SplitResultLite,
    doc_id: u64,
    seq_id: i32,
) -> anyhow::Result<SplitDto> {
    let split_id = ctx.hasher.hash(&format!("{}{}", doc_id, seq_id));
    trace!("Processing split: {}", split_id);

    let embedding = process_embedding(
        ctx.clone(),
        split_id,
        &[split_res.split_string.clone()],
        ctx.model_id,
        ctx.n_embd,
    )?;
    let summaries = process_summaries(
        ctx.clone(),
        split_res.split_string.clone(),
        doc_id,
        split_id,
    )?;
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
