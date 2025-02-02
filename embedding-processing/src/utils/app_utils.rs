use crate::processing::context::ProcessingContext;
use crate::services::embeddings::{async_embeddings_routine, EmbeddingsRequest};
use anyhow::Context;
use embedding_common::config::ModelConfig;
use embedding_common::prelude::Model;
use embedding_common::utils::hashing::DeterministicAHasher;
use fast_text_splitter::config::SplitterLiteConfig;
use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::LlamaModel;
use llama_cxx_rs::LlamaContext;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

pub async fn init(
    model_config: ModelConfig,
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

    let mut embed_handles = Vec::new();
    for _ in 0..model_config.instances {
        let backend = LlamaBackend::init()?;
        // offload all layers to the gpu
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        let model_params = LlamaModelParams::default();

        let model_path: PathBuf = model_config.gguf_file.clone().into();

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| "unable to load model")?;

        let n_ctx = model.n_ctx_train() as usize;
        let n_embd = Some(model.n_embd());

        let embedding_receiver = embedding_receiver.clone();
        let shutdown_receiver = shutdown_receiver.clone();
        let embed_handle = tokio::spawn(async move {
            async_embeddings_routine(
                Arc::new(backend),
                Arc::new(model),
                model_config.clone(),
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
        model_config.gguf_file.clone(),
        n_ctx ,
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
