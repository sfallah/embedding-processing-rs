use embedding_common::prelude::{RerankRequest, RerankResponse, Serde};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySummaries {
    pub query: String,
    pub summaries: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let data_path = "reranking-model/tests/test_data/bert_paper_query_summaries.json";
    let input_str = fs::read_to_string(data_path)?;
    let query_summaries = serde_json::from_str::<QuerySummaries>(&input_str)?;
    let query = query_summaries.query;
    let texts = query_summaries.summaries;

    let context = zmq::Context::new();
    let socket = context.socket(zmq::DEALER)?;
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

    // measure time
    let start = std::time::Instant::now();
    let request = RerankRequest::new(query, texts.to_vec(), false);
    let msg = request.pack().expect("Failed to pack");
    socket.send(msg, 0).expect("Failed to send");
    let rsp = socket.recv_bytes(0).expect("Failed to receive");
    let response: RerankResponse = RerankResponse::unpack(&rsp).expect("Failed to unpack");
    for ranking in response.ranks {
        println!("--------------- {} ---------------", ranking.index);
        println!("score: {:?}", ranking.score);
        println!("summary: {}", texts[ranking.index]);
    }
    let elapsed = start.elapsed();
    println!("Elapsed time: {:?}", elapsed);
    Ok(())
}
