use crate::processing::context::ProcessingContext;
use crate::processing::embeddings::get_embeddings;
use crate::processing::splitter::split_text;
use crate::processing::utils;
use std::rc::Rc;
use tracing::trace;

pub fn process_query(ctx: Rc<ProcessingContext>, query: String) -> anyhow::Result<Vec<Vec<f32>>> {
    trace!("Processing embedding...");
    let query_splits = split_text(ctx.splitter.clone(), query.clone().into_bytes())?;
    let query_split_texts = utils::splits_texts(&query_splits);
    let query_id = ctx.hasher.hash(&query);
    let embedding = get_embeddings(
        ctx.zmq_context.clone(),
        &ctx.model_endpoint,
        ctx.n_embd,
        &[query.clone()],
        query_id,
    )?;
    let mut embeddings = Vec::new();
    for i in 0..query_split_texts.len() {
        let i = i * ctx.n_embd;
        let j = (i + 1) * ctx.n_embd;
        embeddings.push(embedding[i..j].to_vec());
    }
    trace!("Embedding processed");
    Ok(embeddings[0].clone())
}
