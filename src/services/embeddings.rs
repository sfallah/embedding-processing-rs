use crate::inference::llama_context::LlamaContext;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use tokio::sync::broadcast::Receiver;
use tokio::sync::mpsc::UnboundedReceiver;

#[derive(Debug, Clone)]
pub struct EmbeddingsRequest {
    pub seq_id: usize,
    pub n_embd: usize,
    pub texts: Vec<String>,
    pub sender: Arc<Sender<EmbeddingsResponse>>,
}
impl EmbeddingsRequest {
    pub fn new(
        seq_id: usize,
        n_embd: usize,
        texts: Vec<String>,
        sender: Arc<Sender<EmbeddingsResponse>>,
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
    pub fn new(request: &EmbeddingsRequest, embeddings: Vec<f32>) -> Self {
        EmbeddingsResponse {
            seq_id: request.seq_id,
            n_embd: request.n_embd,
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
    mut receiver: UnboundedReceiver<EmbeddingsRequest>,
    mut shutdown: Receiver<String>,
) {
    eprintln!("Starting embeddings routine");
    loop {
        eprintln!("Waiting for requests.....");
        tokio::select! {
            _ = shutdown.recv() => {
                eprintln!("Shutting down embeddings routine");
                break;
            }
            msg = receiver.recv() => {
                match msg {
                    Some(request) => {
                        eprintln!("Received embeddings request");
                        let ctx = Arc::clone(&ctx);
                        let req = request.clone();
                        let embeddings = match tokio::task::spawn_blocking(move || ctx.get_embeddings_flat(&req.texts)).await {
                            Ok(embedding) => embedding,
                            Err(_) => {
                                eprintln!("Failed to get embeddings");
                            continue;
                            }
                        };
                        let response = EmbeddingsResponse::new(&request, embeddings);
                        let send_res = request.sender.send(response);
                        if let Err(e) = send_res {
                            eprintln!("Failed to send embeddings response: {:?}", e);
                        }
                        eprintln!("Sent embeddings response");
                    },
                    None => {
                        eprintln!("Failed to receive embeddings request");
                        continue;
                    }
                }
            }
        }
    }
    eprintln!("Embeddings routine exiting");
}

pub async fn async_get_embeddings(
    req_sender: Arc<tokio::sync::mpsc::UnboundedSender<EmbeddingsRequest>>,
    texts: &Vec<String>,
    n_embd: usize,
) -> anyhow::Result<Vec<f32>> {
    let (rep_sender, rep_receiver) = std::sync::mpsc::channel();
    let rep_sender = Arc::new(rep_sender);
    let embd_request = EmbeddingsRequest::new(0, n_embd, texts.clone(), rep_sender.clone());
    if let Err(e) = req_sender.clone().send(embd_request) {
        eprintln!("Failed to send embeddings request: {:?}", e);
        return Err(anyhow::anyhow!(
            "Failed to send embeddings request: {:?}",
            e
        ));
    }
    eprintln!("Sent embeddings request");
    let res = rep_receiver
        .recv()
        .expect("Failed to receive embeddings response");
    eprintln!("Received embeddings response");
    Ok(res.embeddings.clone())
}
