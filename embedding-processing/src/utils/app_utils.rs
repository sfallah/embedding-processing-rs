use crate::inference::llama_context::LlamaContext;
use crate::processing::context::ProcessingContext;
use crate::services::embeddings::{async_embeddings_routine, EmbeddingsRequest};
use embedding_common::prelude::Model;
use embedding_common::utils::hashing::DeterministicAHasher;
use fast_text_splitter::config::SplitterLiteConfig;
use std::ops::Deref;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

pub async fn init(
    model_path: &str,
    embed_workers: usize,
) -> anyhow::Result<(
    Arc<async_channel::Sender<EmbeddingsRequest>>,
    Arc<broadcast::Sender<String>>,
    Vec<JoinHandle<()>>,
    Arc<Model>,
)> {
    let (embedding_sender, embedding_receiver) = async_channel::unbounded::<EmbeddingsRequest>();

    let (shutdown_sender, shutdown_receiver) = broadcast::channel::<String>(1);
    let shutdown_receiver = Arc::new(shutdown_receiver);
    let shutdown_sender = Arc::new(shutdown_sender);

    let mut n_ctx = None;
    let mut n_embd = None;

    let mut embed_handles = Vec::new();
    for _ in 0..embed_workers {
        let model_instance = Arc::new(LlamaContext::new(model_path, 512, 1000));

        if n_ctx.is_none() {
            n_ctx = Some(model_instance.get_n_ctx());
            n_embd = Some(model_instance.get_n_embd());
        }

        let embedding_receiver = embedding_receiver.clone();
        let shutdown_receiver = shutdown_receiver.clone();
        let embed_handle = tokio::spawn(async move {
            async_embeddings_routine(
                model_instance,
                embedding_receiver,
                shutdown_receiver.deref().resubscribe(),
            )
            .await;
        });
        embed_handles.push(embed_handle);
    }
    let embed_sender = Arc::new(embedding_sender);
    let hasher = DeterministicAHasher::new(None, None);
    let model = Model::new(
        &hasher,
        model_path.to_string(),
        n_ctx.unwrap(),
        n_embd.unwrap(),
    );
    let model = Arc::new(model);
    Ok((embed_sender, shutdown_sender, embed_handles, model))
}

pub async fn init_ctx(
    max_tokens: usize,
    merge_level: Option<usize>,
    n_embd: usize,
    model_id: u64,
) -> Arc<ProcessingContext> {
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
    let nw_splitter = SplitterLiteConfig::new_hf(
        splitter_patterns.clone(),
        Some(max_tokens),
        merge_level,
        true,
        None,
    );

    let sentence_splitter = SplitterLiteConfig::new_hf(
        splitter_patterns.clone(),
        Some(max_tokens),
        Some(splitter_patterns.len()),
        true,
        None,
    );

    let hasher = DeterministicAHasher::new(None, None);
    Arc::new(ProcessingContext {
        splitter: Arc::new(nw_splitter),
        sentence_splitter: Arc::new(sentence_splitter),
        hasher: Arc::new(hasher),
        n_embd,
        model_id,
    })
}

pub fn setup_tracing(level: Level) {
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(level)
        // completes the builder.
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed")
}
