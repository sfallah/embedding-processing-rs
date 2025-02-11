use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::ServerArgs;
use embedding_common::prelude::Serde;
use embedding_common::utils::tracting::setup_tracing;
use embedding_model::config::ModelAppConfig;
use embedding_model::types::{EmbeddingsRequest, EmbeddingsResponse};
use std::thread;
use std::time::Duration;
use tracing::{debug, error, info};

fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();
    //let uuid = uuid::Uuid::new_v4();
    //let worker_id = uuid.to_string();


    setup_tracing(args.log_level.to_tracing_level());

    let config = match ModelAppConfig::from_file(args.config_file) {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load config: {:?}", e);
            return Err(e);
        }
    };
    debug!("Config loaded: {:?}", config);

    //  Prepare our context and socket
    let context = zmq::Context::new();
    let socket = match context.socket(zmq::DEALER) {
        Ok(socket) => socket,
        Err(e) => {
            error!("Failed to create socket: {:?}", e);
            return Err(e.into());
        }
    };

    if let Err(e) = socket.set_linger(0) {
        error!("Failed to set linger: {:?}", e);
        return Err(e.into());
    }

    let backend_endpoint = format!(
        "tcp://{}:{}",
        config.zmq_config.host, config.zmq_config.backend_port
    );
    info!("Connecting to model host: {}", backend_endpoint);
    if let Err(e) = socket.connect(&backend_endpoint) {
        error!("Failed to connect to model host: {:?}", e);
        return Err(e.into());
    }
    loop {
        let messages = match socket.recv_multipart(0) {
            Ok(messages) => messages,
            Err(e) => {
                error!("Failed to receive: {:?}", e);
                break;
            }
        };
        //debug!("Worker {} received messages", worker_id);
        let identity = messages[0].clone();
        let request = match EmbeddingsRequest::unpack::<EmbeddingsRequest>(&messages[1]) {
            Ok(request) => request,
            Err(e) => {
                error!("Failed to parse request: {:?}", e);
                continue;
            }
        };

        thread::sleep(Duration::from_millis(30));
        let n_texts = request.texts.len();

        let mut output = Vec::new();
        for _ in  0..n_texts {
            let embeddings = vec![0.0; request.n_embd];
            output.push(embeddings);
        }

        let response = EmbeddingsResponse::new(
            request.req_id,
            request.seq_id,
            request.n_embd,
            output.to_vec(),
        );
        let response_bytes = response.pack()?;
        if let Err(e) = socket.send_multipart(vec![identity, response_bytes.into()], 0) {
            error!("Failed to send response: {:?}", e);
            continue;
        }
    }
    info!("Worker Shutting down");
    Ok(())
}
