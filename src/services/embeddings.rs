use crate::inference::llama_context::LlamaContext;
use std::sync::Arc;
use tokio::sync::broadcast::Receiver;
use tokio::sync::oneshot::Sender;

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
    pub fn new(seq_id: usize, n_embd:usize, embeddings: Vec<f32>) -> Self {
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

pub async fn async_embeddings_routine(
    ctx: Arc<LlamaContext>,
    receiver: async_channel::Receiver<EmbeddingsRequest>,
    mut shutdown: Receiver<String>,
) {
    eprintln!("Starting embeddings routine");
    loop {
        //eprintln!("Waiting for requests.....");
        tokio::select! {
            _ = shutdown.recv() => {
                eprintln!("Shutting down embeddings routine");
                break;
            }
            msg = receiver.recv() => {
                match msg {
                    Ok(EmbeddingsRequest {seq_id,n_embd, texts, sender}) => {
                        //eprintln!("Received embeddings request");
                        let ctx = Arc::clone(&ctx);
                        let embeddings = match tokio::task::spawn_blocking(move || ctx.get_embeddings_flat(&texts)).await {
                            Ok(embedding) => embedding,
                            Err(_) => {
                                eprintln!("Failed to get embeddings");
                            continue;
                            }
                        };
                        let response = EmbeddingsResponse::new(seq_id,n_embd, embeddings);
                        if let Err(e) = tokio::task::spawn_blocking(move || sender.send(response)).await {
                            eprintln!("Failed to send embeddings response: {:?}", e);
                            continue;
                        }
                        //eprintln!("Sent embeddings response");
                    },
                    Err(e) => {
                        eprintln!("Failed to receive embeddings request: {:?}", e);
                        continue;
                    }
                }
            }
        }
    }
    eprintln!("Embeddings routine exiting");
}

pub async fn async_get_embeddings(
    req_sender: Arc<async_channel::Sender<EmbeddingsRequest>>,
    texts: &Vec<String>,
    n_embd: usize,
) -> anyhow::Result<Vec<f32>> {
    let (rep_sender, rep_receiver) = tokio::sync::oneshot::channel::<EmbeddingsResponse>();
    let embd_request = EmbeddingsRequest::new(0, n_embd, texts.clone(), rep_sender);
    if let Err(e) = req_sender.send(embd_request).await {
        eprintln!("Failed to send embeddings request: {:?}", e);
        return Err(anyhow::anyhow!(
            "Failed to send embeddings request: {:?}",
            e
        ));
    }
    match rep_receiver.await {
        Ok(res) => Ok(res.embeddings.clone()),
        Err(e) => {
            eprintln!("Failed to receive embeddings response: {:?}", e);
            Err(anyhow::anyhow!(
                "Failed to receive embeddings response: {:?}",
                e
            ))
        }
    }
}
