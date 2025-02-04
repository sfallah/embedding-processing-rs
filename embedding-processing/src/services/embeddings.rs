use anyhow::Context;
use embedding_common::config::ModelConfig;
use embedding_common::prelude::Serde;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast::Receiver;
use tokio::sync::oneshot::Sender;
use tracing::{debug, error, info, trace};
use zeromq::{ReqSocket, Socket, SocketRecv, SocketSend, ZmqMessage};
use embedding_model::api::{EmbeddingsRequest, EmbeddingsResponse};

#[tracing::instrument(skip(embedding_addr, texts, n_embd))]
pub async fn async_get_embeddings(
    embedding_addr: String,
    texts: &Vec<String>,
    n_embd: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut socket = ReqSocket::new();

    socket
        .connect(embedding_addr.as_str())
        .await
        .expect("Failed to connect");

    trace!("Getting embeddings...");
    let embd_request = EmbeddingsRequest::new(0, n_embd, texts.clone());
    let embd_request_ser = embd_request.pack()?;
    if let Err(e) = socket.send(ZmqMessage::from(embd_request_ser)).await {
        error!(message = "Failed to send embeddings request", error = ?e);
        return Err(anyhow::anyhow!(
            "Failed to send embeddings request: {:?}",
            e
        ));
    }
    trace!("Embeddings request sent");
    match socket.recv().await {
        Ok(res) => {
            trace!("Embeddings received");
            let res: EmbeddingsResponse = EmbeddingsResponse::unpack(&res.get(0).unwrap())?;
            socket.close().await;
            let embeddings: Vec<_> = res.embeddings.into_iter().flatten().collect();
            Ok(embeddings.clone())
        }
        Err(e) => {
            socket.close().await;
            error!(message = "Failed to receive embeddings response", error = ?e);
            Err(anyhow::anyhow!(
                "Failed to receive embeddings response: {:?}",
                e
            ))
        }
    }
}

