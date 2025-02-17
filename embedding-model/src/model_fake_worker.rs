use std::thread;
use std::time::Duration;
use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::ServerArgs;
use embedding_common::prelude::Serde;
use embedding_common::utils::tracting::setup_tracing;
use embedding_model::config::ModelAppConfig;
use embedding_model::types::{EmbeddingsRequest, EmbeddingsResponse};
use rand::distr::Uniform;
use rand::Rng;
use tracing::{debug, error, info};

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

        //thread::sleep(Duration::from_millis(30));

        let response = EmbeddingsResponse::new(
            request.req_id,
            request.seq_id,
            request.n_embd,
            generate_random_matrix(request.texts.len(), request.n_embd),
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
