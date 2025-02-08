use embedding_common::prelude::Serde;
use embedding_model::types::{EmbeddingsRequest, EmbeddingsResponse};
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use std::sync::Arc;
use tracing::trace;

pub fn splits_texts(splits: &Vec<SplitResultLite>) -> Vec<String> {
    splits
        .iter()
        .map(|sentence_split| sentence_split.split_string.clone())
        .collect()
}
pub fn get_embeddings(
    zmq_ctx: Arc<zmq::Context>,
    model_endpoint: String,
    n_embd: usize,
    texts: Vec<String>,
) -> anyhow::Result<Vec<f32>> {
    let socket = zmq_ctx.socket(zmq::DEALER)?;
    socket.connect(&model_endpoint).expect("Failed to connect");
    socket.set_linger(0).expect("Failed to set linger");
    socket
        .set_sndtimeo(1000)
        .expect("Failed to set send timeout");
    socket
        .set_rcvtimeo(30000)
        .expect("Failed to set receive timeout");
    // identity random uuid
    let identity = uuid::Uuid::new_v4();
    socket
        .set_identity(identity.as_bytes())
        .expect("Failed to set identity");
    trace!("Processing embedding...");
    //FIXME: n_embd is hardcoded to 384
    let request = EmbeddingsRequest::new(0, 0, n_embd, texts);
    let msg = request.pack().expect("Failed to pack");
    socket.send(msg, 0).expect("Failed to send");
    let rsp = socket.recv_bytes(0).expect("Failed to receive");
    let response: EmbeddingsResponse = EmbeddingsResponse::unpack(&rsp).expect("Failed to unpack");
    let embeddings = response.embeddings.into_iter().flatten().collect();
    Ok(embeddings)
}

pub async fn async_get_embeddings(
    zmq_ctx: Arc<zmq::Context>,
    model_endpoint: String,
    n_embd: usize,
    texts: Vec<String>,
) -> anyhow::Result<Vec<f32>> {
    tokio::task::spawn_blocking(move || {
        let embeddings = get_embeddings(zmq_ctx, model_endpoint, n_embd, texts.clone())?;
        Ok(embeddings)
    })
    .await?
}
