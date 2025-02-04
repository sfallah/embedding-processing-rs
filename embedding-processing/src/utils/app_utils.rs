use crate::processing::context::ProcessingContext;
use anyhow::Context;
use embedding_common::config::ModelConfig;
use embedding_common::prelude::Model;
use embedding_common::utils::hashing::DeterministicAHasher;
use fast_text_splitter::config::SplitterLiteConfig;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
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
