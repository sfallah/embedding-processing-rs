use criterion::{criterion_main, Criterion};
use fast_text_splitter::config::SplitterLiteConfig;
use std::fs;
use std::sync::Arc;

use criterion::async_executor::AsyncStdExecutor;
use embedding_processing_rs::inference::llama_context::LlamaContext;
use embedding_processing_rs::processing::context::ProcessingContext;
use embedding_processing_rs::processing::documents::process_document;
use embedding_processing_rs::services::embeddings::{async_embeddings_routine, EmbeddingsRequest};
use embedding_processing_rs::utils::hash_utils::DeterministicAHasher;

pub fn process_doc(c: &mut Criterion, doc: String, proc_ctx: Arc<ProcessingContext>) {
    c.bench_function("embeddings_splits_batch", |b| {
        b.to_async(AsyncStdExecutor).iter(|| process_document(proc_ctx.clone(), "url".to_string(), doc.clone().into_bytes()));
    });
}


pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
        .configure_from_args();


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

    let (embedding_sender, embedding_receiver) =
        async_std::channel::unbounded::<EmbeddingsRequest>();

    let embedding_sender = Arc::new(embedding_sender);
    let model_path = "models/all-minilm-l6-v2-q2_k.gguf";

    for _ in 0..4 {
        let model_instance = Arc::new(LlamaContext::new(model_path, 512, 1000));
        let receiver_clone = embedding_receiver.clone();
        async_std::task::spawn(async move {
            async_embeddings_routine(model_instance, receiver_clone).await;
        });
    }

    let hasher = DeterministicAHasher::new(None, None);
    let proc_ctx = Arc::new(ProcessingContext {
        splitter: Arc::new(nw_splitter),
        sentence_splitter: Arc::new(sentence_splitter),
        hasher: Arc::new(hasher),
        embeddings_sender: embedding_sender,
        n_embd: 384,
    });

    let file_path = "tests/test_data/superlinear.txt".to_string();
    // read the file
    let doc = fs::read_to_string(file_path).unwrap();
    process_doc(&mut criterion, doc, proc_ctx);
}
criterion_main!(benches);
