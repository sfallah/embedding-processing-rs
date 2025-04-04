use anyhow::Context;
use embedding_common::prelude::{Rank, RerankRequest, RerankResponse, Serde};
use std::sync::Arc;
use tracing::error;

pub fn get_rerankings(
    zmq_ctx: Arc<zmq::Context>,
    endpoint: &str,
    query: String,
    texts: Vec<String>,
    req_id: u64,
) -> anyhow::Result<Vec<Rank>> {
    let socket = zmq_ctx
        .socket(zmq::DEALER)
        .with_context(|| "Failed to create zmq socket")?;
    if let Err(e) = socket.connect(endpoint) {
        let error_message = format!("Failed to connect to endpoint: {}", e);
        return Err(anyhow::anyhow!(error_message));
    }
    socket.set_linger(0)?;
    socket.set_sndtimeo(1000)?;
    socket.set_rcvtimeo(30000)?;
    socket.set_identity(req_id.to_string().as_bytes())?;
    let request = RerankRequest::new(query, texts.to_vec(), false);
    let msg = match request.pack() {
        Ok(msg) => msg,
        Err(e) => {
            let error_message = format!("Failed to pack RerankRequest: {}", e);
            error!("{}", &error_message);
            return Err(anyhow::anyhow!(error_message));
        }
    };
    if let Err(e) = socket.send(msg, 0) {
        let error_message = format!("Failed to send RerankRequest: {}", e);
        error!("{}", &error_message);
        return Err(anyhow::anyhow!(error_message));
    }
    let rsp = match socket.recv_bytes(0) {
        Ok(rsp) => rsp,
        Err(e) => {
            let error_message = format!("Failed to receive RerankResponse: {}", e);
            error!("{}", &error_message);
            return Err(anyhow::anyhow!(error_message));
        }
    };
    match RerankResponse::unpack::<RerankResponse>(&rsp) {
        Ok(response) => {
            if !request.return_text {
                Ok(response.ranks)
            } else {
                let mut ranks = response.ranks;
                for rank in &mut ranks {
                    rank.text = Some(texts[rank.index].clone());
                }
                Ok(ranks)
            }
        }
        Err(e) => {
            let error_message = format!("Failed to unpack RerankResponse: {}", e);
            error!("{}", &error_message);
            Err(anyhow::anyhow!(error_message))
        }
    }
}
