use crate::inference::llama_context::LlamaContext;
use async_std::channel::Receiver;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

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

pub fn embeddings_routine(
    ctx: Arc<LlamaContext>,
    receiver: Receiver<EmbeddingsRequest>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::SeqCst)
    /*&& !receiver.is_closed()*/
    {
        let recv_res = receiver.recv_blocking();
        match recv_res {
            Ok(request) => {
                //eprintln!("Received embeddings request: {:?}", request);
                let embeddings = ctx.get_embeddings_flat(&request.texts);
                let response = EmbeddingsResponse::new(&request, embeddings);
                let send_res = request.sender.send(response);
                if let Err(e) = send_res {
                    eprintln!("Failed to send embeddings response: {:?}", e);
                }
            }
            Err(_) => {
                eprintln!("Failed to receive embeddings request");
                break;
            }
        }
    }
    eprintln!("Embeddings routine exiting");
}

pub async fn async_embeddings_routine(
    ctx: Arc<LlamaContext>,
    receiver: Receiver<EmbeddingsRequest>,
) {
    while let Ok(request) = receiver.recv().await {
        //eprintln!("Received embeddings request: {:?}", request);
        let ctx = Arc::clone(&ctx);
        let req = request.clone();
        let embeddings =
            async_std::task::spawn(async move { ctx.get_embeddings_flat(&req.texts) }).await;
        let response = EmbeddingsResponse::new(&request, embeddings);
        let send_res = request.sender.send(response);
        if let Err(e) = send_res {
            eprintln!("Failed to send embeddings response: {:?}", e);
        }
    }
    eprintln!("Embeddings routine exiting");
}

pub async fn async_embeddings_routine_with_running(
    ctx: Arc<LlamaContext>,
    receiver: Receiver<EmbeddingsRequest>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::SeqCst) {
        let recv_res = receiver.recv().await;
        match recv_res {
            Ok(request) => {
                //eprintln!("Received embeddings request: {:?}", request);
                let embeddings = ctx.get_embeddings_flat(&request.texts);
                let response = EmbeddingsResponse::new(&request, embeddings);
                let send_res = request.sender.send(response);
                if let Err(e) = send_res {
                    eprintln!("Failed to send embeddings response: {:?}", e);
                }
            }
            Err(_) => {
                eprintln!("Failed to receive embeddings request");
                break;
            }
        }
    }
    eprintln!("Embeddings routine exiting");
}

pub fn channel_get_embeddings(
    embd_req_sender: &Arc<async_std::channel::Sender<EmbeddingsRequest>>,
    texts: &Vec<Vec<String>>,
    n_embd: usize,
) -> Vec<Vec<f32>> {
    let (embd_resp_sender, embd_resp_receiver) = std::sync::mpsc::channel();
    let embd_resp_sender = Arc::new(embd_resp_sender);
    texts.iter().enumerate().for_each(|(seq_id, text)| {
        let embd_request =
            EmbeddingsRequest::new(seq_id, n_embd, text.clone(), embd_resp_sender.clone());
        embd_req_sender
            .clone()
            .send_blocking(embd_request)
            .expect("Failed to send embeddings request");
    });
    let mut responses = vec![];
    for _ in 0..texts.len() {
        let embd_response = embd_resp_receiver
            .recv()
            .expect("Failed to receive embeddings response");
        responses.push(embd_response);
    }
    // Sort the responses by sequence id
    responses.sort_by(|a, b| a.seq_id.cmp(&b.seq_id));
    responses.iter().map(|r| r.embeddings.clone()).collect()
}

pub async fn async_get_embeddings(
    req_sender: Arc<async_std::channel::Sender<EmbeddingsRequest>>,
    texts: &Vec<String>,
    n_embd: usize,
) -> anyhow::Result<Vec<f32>> {
    let (rep_sender, rep_receiver) = std::sync::mpsc::channel();
    let rep_sender = Arc::new(rep_sender);
    let embd_request = EmbeddingsRequest::new(0, n_embd, texts.clone(), rep_sender.clone());
    req_sender
        .clone()
        .send_blocking(embd_request)
        .expect("Failed to send embeddings request");
    let res = rep_receiver
        .recv()
        .expect("Failed to receive embeddings response");
    Ok(res.embeddings.clone())
}
