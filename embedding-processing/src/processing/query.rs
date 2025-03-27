use crate::processing::context::ProcessingContext;
use crate::processing::embeddings::get_embeddings;
use std::sync::Arc;
use tracing::trace;

pub fn process_query(ctx: Arc<ProcessingContext>, query: String) -> anyhow::Result<Vec<f32>> {
    trace!("Processing embedding...");
    let query_id = ctx.hasher.hash(&query);
    let embedding = get_embeddings(
        ctx.zmq_context.clone(),
        &ctx.embedding_endpoint,
        ctx.n_embd,
        &[query.clone()],
        query_id,
    )?;
    trace!("Embedding processed");
    Ok(embedding[0].clone())
}
