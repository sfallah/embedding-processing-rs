use anyhow::Context;
use clap::Parser;
use embedding_model::config::ModelAppConfig;
use embedding_model::types::{EmbeddingsRequest, EmbeddingsResponse};
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::ServerArgs;
use embedding_common::prelude::Serde;
use embedding_common::utils::tracting::setup_tracing;
use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::context::LlamaContext;
use llama_cpp::ggml_time_us;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use std::path::PathBuf;
use tracing::{debug, error, info, warn};

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

    let mut backend = LlamaBackend::init()?;
    if !config.model_config.verbose {
        backend.void_logs();
    }

    // offload all layers to the gpu
    let model_params = LlamaModelParams::default();

    let model_path: PathBuf = config.model_config.gguf_file.try_into()?;

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model")?;

    // initialize the context
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    let n_ctx = ctx.n_ctx() as usize;
    let _n_ctx_train = model.n_ctx_train();
    let mut batch = LlamaBatch::new(n_ctx, 1);

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
        let tokens_lines_list = request
            .texts
            .iter()
            .filter_map(|text| {
                let res = model.str_to_token(text, AddBos::Always);
                match res {
                    Ok(tokens) => {
                        if tokens.len() > 0 {
                            if tokens.len() > n_ctx {
                                warn!("Token sequence exceeds context window, truncating");
                                Some(tokens[..n_ctx].to_vec())
                            } else {
                                Some(tokens)
                            }
                        } else {
                            None
                        }
                    }
                    Err(e) => {
                        error!("Failed to tokenize: {}", e);
                        None
                    }
                }
            })
            .collect::<Vec<_>>();

        let mut max_seq_id_batch = 0;
        let mut output = Vec::with_capacity(tokens_lines_list.len());

        let _t_main_start = ggml_time_us();

        for tokens in &tokens_lines_list {
            // Flush the batch if the next prompt would exceed our batch size
            if (batch.n_tokens() as usize + tokens.len()) > n_ctx {
                batch_decode(&mut ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
                max_seq_id_batch = 0;
            }

            batch.add_sequence(tokens, max_seq_id_batch, false)?;
            max_seq_id_batch += 1;
        }
        // Handle final batch
        batch_decode(&mut ctx, &mut batch, max_seq_id_batch, &mut output, true)?;

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
fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> anyhow::Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();

    Ok(())
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}
