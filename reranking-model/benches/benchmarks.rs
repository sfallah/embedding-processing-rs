use criterion::{black_box, criterion_main, Criterion};
use embedding_common::prelude::{RerankRequest, RerankResponse, Serde};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySummaries {
    pub query: String,
    pub summaries: Vec<String>,
}

pub fn benchmark_reranking_zmq(
    c: &mut Criterion,
    query: String,
    texts: Vec<String>,
    socket: &zmq::Socket,
) {
    c.bench_function("benchmark_reranking_zmq", |b| {
        b.iter(|| {
            let request = RerankRequest::new(query.clone(), texts.to_vec(), false);
            let msg = request.pack().expect("Failed to pack");
            socket.send(msg, 0).expect("Failed to send");
            let rsp = socket.recv_bytes(0).expect("Failed to receive");
            let response: RerankResponse = RerankResponse::unpack(&rsp).expect("Failed to unpack");
            black_box(response);
        });
    });
}

pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
        .configure_from_args();

    let data_path = "tests/test_data/bert_paper_query_summaries.json";
    let input_str = fs::read_to_string(data_path).expect("Failed to read");
    let query_summaries =
        serde_json::from_str::<QuerySummaries>(&input_str).expect("Failed to parse");
    let query = query_summaries.query;
    let texts = query_summaries.summaries;

    let context = zmq::Context::new();
    let socket = context
        .socket(zmq::DEALER)
        .expect("Failed to create socket");
    socket
        .connect("tcp://localhost:5557")
        .expect("Failed to connect");
    socket.set_linger(0).expect("Failed to set linger");
    socket
        .set_sndtimeo(1000)
        .expect("Failed to set send timeout");
    socket
        .set_rcvtimeo(30000)
        .expect("Failed to set receive timeout");
    // identity random uuid
    let identity = uuid::Uuid::new_v4();
    socket
        .set_identity(identity.as_bytes())
        .expect("Failed to set identity");

    benchmark_reranking_zmq(&mut criterion, query, texts, &socket);
}
criterion_main!(benches);
