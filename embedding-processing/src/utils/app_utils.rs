use crate::processing::context::ProcessingContext;
use embedding_common::utils::hashing::DeterministicAHasher;
use fast_text_splitter::config::SplitterLiteConfig;
use std::sync::Arc;
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
    let zmq_context = zmq::Context::new();
    Arc::new(ProcessingContext {
        zmq_context: Arc::new(zmq_context),
        model_endpoint: "tcp://localhost:5559".to_string(),
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
