use std::sync::{Arc, Mutex};
use llama_cxx_rs::LlamaContext;
use tokio::sync::broadcast::Receiver;
use tokio::sync::oneshot::Sender;
use tracing::{debug, error, info, trace};

#[derive(Debug)]
pub struct EmbeddingsRequest {
    pub seq_id: usize,
    pub n_embd: usize,
    pub texts: Vec<String>,
    pub sender: Sender<EmbeddingsResponse>,
}
impl EmbeddingsRequest {
    pub fn new(
        seq_id: usize,
        n_embd: usize,
        texts: Vec<String>,
        sender: Sender<EmbeddingsResponse>,
    ) -> Self {
        EmbeddingsRequest {
            seq_id,
            n_embd,
            texts,
            sender,
        }
    }
}

unsafe impl Sync for EmbeddingsRequest {}
unsafe impl Send for EmbeddingsRequest {}

#[derive(Debug, Clone)]
pub struct EmbeddingsResponse {
    pub seq_id: usize,
    pub n_embd: usize,
    pub embeddings: Vec<f32>,
}
impl EmbeddingsResponse {
    pub fn new(seq_id: usize, n_embd: usize, embeddings: Vec<f32>) -> Self {
        EmbeddingsResponse {
            seq_id,
            n_embd,
            embeddings,
        }
    }
    pub fn get_embeddings_vec(&self) -> Vec<Vec<f32>> {
        self.embeddings
            .chunks(self.n_embd)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

unsafe impl Sync for EmbeddingsResponse {}
unsafe impl Send for EmbeddingsResponse {}

#[tracing::instrument(skip(ctx, receiver, shutdown))]
pub async fn async_embeddings_routine(
    ctx: Arc<Mutex<LlamaContext>>,
    receiver: async_channel::Receiver<EmbeddingsRequest>,
    mut shutdown: Receiver<String>,
) {
    info!("Starting routine");
    loop {
        debug!("Waiting for requests.....");
        tokio::select! {
            _ = shutdown.recv() => {
                info!("Shutting down embeddings routine");
                break;
            }
            msg = receiver.recv() => {
                match msg {
                    Ok(EmbeddingsRequest {seq_id,n_embd, texts, sender}) => {
                        debug!("Received embeddings request");
                        let ctx = Arc::clone(&ctx);
                        let embeddings = match tokio::task::spawn_blocking(move || ctx.lock().unwrap().get_embeddings(&texts,false)).await {
                            Ok(embedding) => {
                                match embedding {
                                    Ok(embedding) => embedding,
                                    Err(e) => {
                                        error!("Failed to get embeddings: {:?}", e);
                                        continue;
                                    }
                                }
                            }
                            Err(_) => {
                                error!("Failed to get embeddings");
                            continue;
                            }
                        };
                        let response = EmbeddingsResponse::new(seq_id,n_embd, embeddings);
                        if let Err(e) = tokio::task::spawn_blocking(move || sender.send(response)).await {
                            error!("Failed to send embeddings response: {:?}", e);
                            continue;
                        }
                        debug!("Sent embeddings response");
                    },
                    Err(e) => {
                        error!("Failed to receive embeddings request: {:?}", e);
                        continue;
                    }
                }
            }
        }
    }
    info!("Embeddings routine exiting");
}

#[tracing::instrument(skip(req_sender, texts, n_embd))]
pub async fn async_get_embeddings(
    req_sender: Arc<async_channel::Sender<EmbeddingsRequest>>,
    texts: &Vec<String>,
    n_embd: usize,
) -> anyhow::Result<Vec<f32>> {
    trace!("Getting embeddings...");
    let (rep_sender, rep_receiver) = tokio::sync::oneshot::channel::<EmbeddingsResponse>();
    let embd_request = EmbeddingsRequest::new(0, n_embd, texts.clone(), rep_sender);
    if let Err(e) = req_sender.send(embd_request).await {
        error!(message = "Failed to send embeddings request", error = ?e);
        return Err(anyhow::anyhow!(
            "Failed to send embeddings request: {:?}",
            e
        ));
    }
    trace!("Embeddings request sent");
    match rep_receiver.await {
        Ok(res) => {
            trace!("Embeddings received");
            Ok(res.embeddings.clone())
        }
        Err(e) => {
            error!(message = "Failed to receive embeddings response", error = ?e);
            Err(anyhow::anyhow!(
                "Failed to receive embeddings response: {:?}",
                e
            ))
        }
    }
}
