use embedding_common::prelude::{RerankRequest, RerankResponse, Serde};
use std::sync::Arc;
use tracing::trace;
use zmq::Context;

pub fn get_rerankings(
    zmq_ctx: Arc<Context>,
    endpoint: &str,
    query: String,
    texts: Vec<String>,
    rerank_id: u64,
) -> anyhow::Result<Vec<f32>> {
    let socket = zmq_ctx.socket(zmq::DEALER)?;
    socket.connect(endpoint).expect("Failed to connect");
    socket.set_linger(0).expect("Failed to set linger");
    socket
        .set_sndtimeo(1000)
        .expect("Failed to set send timeout");
    socket
        .set_rcvtimeo(30000)
        .expect("Failed to set receive timeout");
    socket
        .set_identity(rerank_id.to_string().as_bytes())
        .expect("Failed to set identity");
    trace!("Processing reranking...");
    let request = RerankRequest::new(None, query, texts.to_vec());
    let msg = request.pack().expect("Failed to pack");
    socket.send(msg, 0).expect("Failed to send");
    let rsp = socket.recv_bytes(0).expect("Failed to receive");
    let response: RerankResponse = RerankResponse::unpack(&rsp).expect("Failed to unpack");
    let result = response.ranks.iter().map(|rank| rank.score).collect();
    Ok(result)
}
