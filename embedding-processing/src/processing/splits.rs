use crate::processing::context::ProcessingContext;
use crate::processing::embeddings::get_embeddings;
use crate::processing::summaries::{get_sentences, process_summaries};
use embedding_common::dtos::split_dto::SplitDto;
use embedding_common::dtos::EmbeddingDto;
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use std::sync::Arc;
use tracing::trace;

#[tracing::instrument(skip(ctx, split_res))]
pub fn process_split(
    ctx: Arc<ProcessingContext>,
    split_res: &SplitResultLite,
    doc_id: u64,
    seq_id: i32,
) -> anyhow::Result<SplitDto> {
    let split_id = ctx.hasher.hash(&format!("{}{}", doc_id, seq_id));
    trace!("Processing split: {}", split_id);

    let (sentences, sentences_no_tokens) =
        get_sentences(ctx.clone(), split_res.split_string.clone())?;

    let mut text_vec = Vec::new();
    text_vec.push(split_res.split_string.clone());
    text_vec.extend(sentences.clone());

    let embeddings = get_embeddings(
        ctx.zmq_context.clone(),
        ctx.embedding_endpoint.as_str(),
        &text_vec,
        split_id,
    )?;

    let split_embedding = embeddings.get(0).expect("Failed to get embedding");
    let embedding = EmbeddingDto::new(split_id, split_embedding.to_vec(), ctx.model_id);

    let sentences_embeddings: Vec<_> = embeddings.into_iter().skip(1).flatten().collect();

    let summaries = process_summaries(
        ctx.clone(),
        &sentences,
        &sentences_no_tokens,
        sentences_embeddings.as_slice(),
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
        None,
    ))
}
