use embedding_common::prelude::{EmbeddingsRequest, EmbeddingsResponse, Serde};

fn main() {
    let context = zmq::Context::new();
    let socket = context.socket(zmq::DEALER).unwrap();
    socket
        .connect("tcp://localhost:5559")
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

    let n_msgs = 2;
    for i in 0..n_msgs {
        let text = format!("Hello model! Please get the embedding for me, msg: {}.", i);
        let request = EmbeddingsRequest::new(vec![text]);
        let msg = request.pack().expect("Failed to pack");
        socket.send(msg, 0).expect("Failed to send");
        let rsp = socket.recv_bytes(0).expect("Failed to receive");
        let response: EmbeddingsResponse = EmbeddingsResponse::unpack(&rsp).expect("Failed to unpack");
        println!("Received response dimensions: {:?}", response.embeddings[0].len());
        let json = EmbeddingsResponse::to_json(&response);
        println!("Received: {:?}", json);
    }
}
