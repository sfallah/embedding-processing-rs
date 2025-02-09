use criterion::{criterion_main, Criterion};
use embedding_processing::processing::documents::process_document;
use embedding_processing::utils::app_utils::{init_ctx};
use std::fs;
use embedding_common::config::ModelConfig;

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

pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
        .configure_from_args();

    let file_path = "tests/test_data/superlinear.txt".to_string();
    // read the file
    let doc = fs::read_to_string(file_path).unwrap();
    process_doc(&mut criterion, doc);
}
criterion_main!(benches);
