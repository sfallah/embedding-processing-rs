use criterion::{black_box, criterion_main, Criterion};
use embedding_processing::processing::documents::process_document;
use embedding_processing::utils::app_utils::init_ctx;
use fast_text_splitter::config::SplitterLiteConfig;
use std::fs;
use std::sync::Arc;
//use rayon::prelude::*;
use embedding_processing::processing::embeddings::get_embeddings;

pub fn process_doc(c: &mut Criterion, doc: String) {
    c.bench_function("embeddings_splits_batch", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let proc_ctx = rt.block_on(init_ctx(512, None, 384, 30600));
        b.to_async(rt).iter(|| {
            process_document(
                proc_ctx.clone(),
                "url".to_string(),
                doc.clone().into_bytes(),
            )
        });
    });
}

pub fn embedding_benchmark(c: &mut Criterion, doc: String) {
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
    let splitter =
        SplitterLiteConfig::new_hf(splitter_patterns.clone(), Some(512), None, true, None);
    let splitter = Arc::new(splitter);
    let splits = splitter.hf_splits(&doc.into_bytes());
    let splits = splits
        .into_iter()
        .map(|split| split.split_string)
        .collect::<Vec<String>>();
    c.bench_function("embeddings_splits", |b| {
        let zmq_ctx = Arc::new(zmq::Context::new());
        b.iter(|| {
            black_box(splits.clone())
                .iter()
                .enumerate()
                .for_each(|(idx, split)| {
                    let embedding = get_embeddings(
                        zmq_ctx.clone(),
                        "tcp://127.0.0.1:5559".to_string(),
                        384,
                        vec![split.clone()],
                        idx as u64,
                    )
                    .expect("Failed to get embeddings");
                    black_box(embedding);
                });
        });
    });
}

pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
        .configure_from_args();

    let file_path = "tests/test_data/superlinear.txt".to_string();
    // read the file
    let doc = fs::read_to_string(file_path).unwrap();
    //process_doc(&mut criterion, doc.clone());
    embedding_benchmark(&mut criterion, doc);
}
criterion_main!(benches);
