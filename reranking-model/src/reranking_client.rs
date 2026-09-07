use embedding_common::prelude::{RerankRequest, RerankResponse, Serde};
use serde::{Deserialize, Serialize};
use serde_json::to_string;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySummaries {
    pub query: String,
    pub summaries: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    /*
    let data_path = "reranking-model/tests/test_data/bert_paper_query_summaries.json";
    let input_str = fs::read_to_string(data_path)?;
    let query_summaries = serde_json::from_str::<QuerySummaries>(&input_str)?;
    let query = query_summaries.query;
    let texts = query_summaries.summaries;
     */

    let query = "What is the capital of the United States?";
    let texts = [
        "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America.",
        "Capital punishment has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states. The federal government (including the United States military) also uses capital punishment.",
    ];

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
    let request = RerankRequest::new(
        query.to_string(),
        texts
            .to_vec()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        false,
    );
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
