use anyhow::Context;
use embedding_common::prelude::{EmbeddingsRequest, EmbeddingsResponse, Serde};
use std::sync::Arc;
use tracing::{error};

pub fn get_embeddings(
    zmq_ctx: Arc<zmq::Context>,
    model_endpoint: &str,
    texts: &[String],
    embed_id: u64,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let socket = zmq_ctx
        .socket(zmq::DEALER)
        .with_context(|| "Failed to create zmq socket")?;
    if let Err(e) = socket.connect(&model_endpoint) {
        let error_message = format!("Failed to connect to Embedding Model Endpoint: {:?}", e);
        error!("{}", error_message);
        return Err(anyhow::Error::msg(error_message));
    };

    socket.set_linger(0)?;
    socket.set_sndtimeo(1000)?;
    socket.set_rcvtimeo(30000)?;
    socket
        .set_identity(embed_id.to_string().as_bytes())
        .with_context(|| "Failed to set identity")?;
    let request = EmbeddingsRequest::new(texts.to_vec());
    let msg = request
        .pack()
        .with_context(|| "Failed to pack EmbeddingsRequest")?;

    if let Err(e) = socket.send(msg, 0) {
        let error_message = format!("Failed to send Embedding request: {:?}", e);
        error!("{}", error_message);
        return Err(anyhow::Error::msg(error_message));
    }
    let rsp = match socket.recv_bytes(0) {
        Ok(rsp) => rsp,
        Err(e) => {
            let error_message = format!("Failed to receive Embedding response: {:?}", e);
            error!("{}", error_message);
            return Err(anyhow::Error::msg(error_message));
        }
    };
    let response: EmbeddingsResponse =
        EmbeddingsResponse::unpack(&rsp).with_context(|| "Failed to unpack response")?;
    Ok(response.embeddings)
}
