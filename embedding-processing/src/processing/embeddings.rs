use embedding_common::dtos::embedding_dto::EmbeddingDto;
use std::rc::Rc;
use tracing::trace;
use embedding_common::prelude::Serde;
use embedding_model::types::{EmbeddingsRequest, EmbeddingsResponse};
use crate::processing::context::ProcessingContext;

pub fn get_embeddings(
    zmq_ctx: Rc<zmq::Context>,
    model_endpoint: &str,
    n_embd: usize,
    texts: &[String],
    embed_id: u64,
) -> anyhow::Result<Vec<Vec<f32>>> {
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
    //let identity = uuid::Uuid::new_v4();
    socket
        .set_identity(embed_id.to_string().as_bytes())
        .expect("Failed to set identity");
    trace!("Processing embedding...");
    //FIXME: n_embd is hardcoded to 384
    let request = EmbeddingsRequest::new(0, 0, n_embd, texts.to_vec());
    let msg = request.pack().expect("Failed to pack");
    socket.send(msg, 0).expect("Failed to send");
    let rsp = socket.recv_bytes(0).expect("Failed to receive");
    let response: EmbeddingsResponse = EmbeddingsResponse::unpack(&rsp).expect("Failed to unpack");
    Ok(response.embeddings)
}