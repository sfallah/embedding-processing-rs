use criterion::{criterion_main, Criterion};
use embedding_processing::processing::documents::process_document;
use embedding_processing::utils::app_utils::{init, init_ctx};
use std::fs;
use embedding_common::config::ModelConfig;

pub fn process_doc(c: &mut Criterion, name: &str, doc: String) {
    c.bench_function(format!("embeddings_splits_{}", name).as_str(), |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let model_config = ModelConfig::new("../models/all-minilm-l6-v2-q2_k.gguf".to_string(), 2, false);
        let (embed_sender, _shutdown, _handle, model) = rt
            .block_on(init(model_config))
            .unwrap();
        let proc_ctx = rt.block_on(init_ctx(400, None, 384, model.model_id));
        b.to_async(rt).iter(|| {
            process_document(
                proc_ctx.clone(),
                embed_sender.clone(),
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

    let superlinear_path = "tests/test_data/superlinear.txt".to_string();
    let superlinear = fs::read_to_string(superlinear_path).unwrap();
    process_doc(&mut criterion, "superlinear", superlinear);

    let superlinear_path = "tests/test_data/bert_paper.txt".to_string();
    let superlinear = fs::read_to_string(superlinear_path).unwrap();
    process_doc(&mut criterion, "bert_paper", superlinear);

    let paper_path = "tests/test_data/paper_arxiv_org__2108.07258v3.txt".to_string();
    let paper = fs::read_to_string(paper_path).unwrap();
    process_doc(&mut criterion, "paper", paper);


}
criterion_main!(benches);
