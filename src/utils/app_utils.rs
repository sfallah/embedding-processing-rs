use crate::inference::llama_context::LlamaContext;
use crate::services::embeddings::{async_embeddings_routine, EmbeddingsRequest};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

pub async fn init() -> anyhow::Result<(Arc<mpsc::UnboundedSender<EmbeddingsRequest>>, Arc<broadcast::Sender<String>>, JoinHandle<()>)>
{
    let (embedding_sender, embedding_receiver) = mpsc::unbounded_channel::<EmbeddingsRequest>();

    let (shutdown_sender, shutdown_receiver) = broadcast::channel::<String>(1);

    let model_path = "models/all-minilm-l6-v2-q2_k.gguf";

    let model_instance = Arc::new(LlamaContext::new(model_path, 512, 1000));

    let embed_handle = tokio::spawn(async move {
        async_embeddings_routine(model_instance, embedding_receiver, shutdown_receiver).await;
    });
    let embed_sender = Arc::new(embedding_sender);
    let shutdown_sender = Arc::new(shutdown_sender);
    Ok((embed_sender, shutdown_sender, embed_handle))
}
