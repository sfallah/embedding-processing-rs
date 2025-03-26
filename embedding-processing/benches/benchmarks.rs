use criterion::{black_box, criterion_main, Criterion};
use embedding_common::prelude::{EmbeddingsRequest, EmbeddingsResponse, Serde};
use embedding_processing::processing::documents::process_document;
use embedding_processing::processing::embeddings::get_embeddings;
use embedding_processing::utils::app_utils::init_ctx;
use fast_text_splitter::config::SplitterLiteConfig;
use rand::distr::Uniform;
use rand::Rng;
use rayon::prelude::*;
use std::fs;
use std::sync::Arc;

fn generate_random_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    // Create a uniform distribution for f32 values between 0.0 and 1.0
    let distribution = Uniform::new(0.0, 1.0).expect("Failed to create distribution");

    // Initialize the matrix with random values
    let mut matrix = vec![vec![0.0; cols]; rows];

    // Fill the matrix with random values
    let mut rng = rand::rng();
    for i in 0..rows {
        for j in 0..cols {
            matrix[i][j] = rng.sample(distribution);
        }
    }

    matrix
}

pub fn process_doc(c: &mut Criterion, doc: String) {
    c.bench_function("process_doc", |b| {
        let proc_ctx = init_ctx(512, None, 384, 30600);
        b.iter(|| {
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
                .par_iter()
                .enumerate()
                .for_each(|(idx, split)| {
                    let embedding = get_embeddings(
                        zmq_ctx.clone(),
                        "tcp://127.0.0.1:5559",
                        384,
                        &[split.clone()],
                        idx as u64,
                    )
                    .expect("Failed to get embeddings");
                    black_box(embedding);
                });
        });
    });
}

pub fn embedding_msgpack_benchmark(c: &mut Criterion, doc: String) {
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

    //let fake_embeddings = generate_random_matrix(splits.len(), 384);
    c.bench_function("embedding_msgpack_benchmark", |b| {
        b.iter(|| {
            black_box(splits.clone())
                .iter()
                .enumerate()
                .for_each(|(_idx, split)| {
                    let request = EmbeddingsRequest::new(0, 0, 348, vec![split.clone()]);
                    let msg = request.pack().expect("Failed to pack");
                    black_box(msg);
                    let response = EmbeddingsResponse::new(
                        request.req_id,
                        request.seq_id,
                        request.n_embd,
                        generate_random_matrix(1, request.n_embd),
                    );
                    let response_packed = response.pack().expect("Failed to pack");
                    black_box(response_packed);
                });
        });
    });
}

pub fn embedding_benchmark_batch(c: &mut Criterion, doc: String) {
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
    c.bench_function("embeddings_splits_batch", |b| {
        let zmq_ctx = Arc::new(zmq::Context::new());
        b.iter(|| {
            let id_rnd = rand::random::<u64>();
            let embeddings = get_embeddings(
                zmq_ctx.clone(),
                "tcp://localhost:5559",
                384,
                &splits,
                id_rnd,
            )
            .expect("Failed to get embeddings");
            black_box(embeddings);
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
    process_doc(&mut criterion, doc.clone());
    //embedding_msgpack_benchmark(&mut criterion, doc.clone());
    embedding_benchmark_batch(&mut criterion, doc.clone());
    embedding_benchmark(&mut criterion, doc.clone());
}
criterion_main!(benches);
