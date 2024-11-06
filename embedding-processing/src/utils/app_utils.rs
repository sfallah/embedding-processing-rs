use std::ops::Deref;
use crate::inference::llama_context::LlamaContext;
use crate::processing::context::ProcessingContext;
use crate::services::embeddings::{async_embeddings_routine, EmbeddingsRequest};
use embedding_common::utils::hashing::DeterministicAHasher;
use fast_text_splitter::config::SplitterLiteConfig;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

pub async fn init(model_path:&str, embed_workers: usize) -> anyhow::Result<(Arc<async_channel::Sender<EmbeddingsRequest>>, Arc<broadcast::Sender<String>>, Vec<JoinHandle<()>>)>
{
    let (embedding_sender, embedding_receiver) = async_channel::unbounded::<EmbeddingsRequest>();

    let (shutdown_sender, shutdown_receiver) = broadcast::channel::<String>(1);
    let shutdown_receiver = Arc::new(shutdown_receiver);
    let shutdown_sender = Arc::new(shutdown_sender);

    let mut embed_handles = Vec::new();
    for _ in 0..embed_workers {
        let model_instance = Arc::new(LlamaContext::new(model_path, 512, 1000));
        let embedding_receiver = embedding_receiver.clone();
        let shutdown_receiver = shutdown_receiver.clone();
        let embed_handle = tokio::spawn( async move {
                async_embeddings_routine(model_instance, embedding_receiver, shutdown_receiver.deref().resubscribe()).await;
        });
        embed_handles.push(embed_handle);
    }
    let embed_sender = Arc::new(embedding_sender);
    Ok((embed_sender, shutdown_sender, embed_handles))
}

pub async fn init_ctx() -> Arc<ProcessingContext> {
    let splitter_patterns = vec![
        vec!["\n\n".to_string()],
        vec!["\n".to_string()],
        vec![
            ".".to_string(),
            "!".to_string(),
            "?".to_string(),
            ". ".to_string(),
        ],
    ];
    let nw_splitter =
        SplitterLiteConfig::new_hf(splitter_patterns.clone(), Some(512), None, true, None);

    let sentence_splitter = SplitterLiteConfig::new_hf(
        splitter_patterns.clone(),
        Some(512),
        Some(splitter_patterns.len()),
        true,
        None,
    );

    let haser = DeterministicAHasher::new(None, None);
    Arc::new(ProcessingContext {
        splitter: Arc::new(nw_splitter),
        sentence_splitter: Arc::new(sentence_splitter),
        hasher: Arc::new(haser),
        n_embd: 384,
    })
}

pub fn setup_tracing(level: Level) {
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(level)
        // completes the builder.
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed")
}
