use anyhow::Context;
use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::model_backend_config::ModelBackendAppConfig;
use embedding_common::config::ServerArgs;
use embedding_common::prelude::{
    DeterministicAHasher, EmbeddingsRequest, EmbeddingsResponse, Model, Serde,
};
use embedding_common::utils::tracting::setup_tracing;
use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::context::LlamaContext;
use llama_cpp::ggml_time_us;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use std::num::NonZero;
use std::path::PathBuf;
use tracing::{debug, error, info, warn};
use zmq::Socket;

fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();
    setup_tracing(args.log_level.to_tracing_level());

    let config = match ModelBackendAppConfig::from_file(args.config_file) {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load config: {:?}", e);
            return Err(e);
        }
    };
    debug!("Config loaded: {:?}", config);

    let mut backend = match LlamaBackend::init() {
        Ok(backend) => backend,
        Err(e) => {
            error!("Failed to initialize backend: {:?}", e);
            return Err(e.into());
        }
    };

    if !config.model_config.verbose {
        backend.void_logs();
    }

    // offload all layers to the gpu
    let model_params = if cfg!(feature = "cuda") {
        LlamaModelParams::default().with_n_gpu_layers(config.model_config.ngl as u32)
    } else {
        LlamaModelParams::default()
    };

    let model_path: PathBuf = config.model_config.gguf_file.clone().try_into()?;

    let model = match LlamaModel::load_from_file(&backend, model_path, &model_params) {
        Ok(model) => model,
        Err(e) => {
            error!("Failed to load model: {:?}", e);
            return Err(e.into());
        }
    };
    let n_embd = model.n_embd();
    let default_hasher = DeterministicAHasher::default_hasher();
    let model_id = Model::model_id(&default_hasher, &config.model_config.gguf_file, n_embd);

    let n_ctx = config
        .model_config
        .max_tokens
        .unwrap_or(model.n_ctx_train());

    let n_ctx = if n_ctx > model.n_ctx_train() {
        model.n_ctx_train()
    } else {
        n_ctx
    };

    // initialize the context
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_n_ctx(Some(NonZero::new(n_ctx).unwrap()))
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ctx)
        .with_kv_unified(true)
        .with_embeddings(true);

    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            error!("Failed to create context: {:?}", e);
            return Err(e.into());
        }
    };

    let mut batch = LlamaBatch::new(n_ctx as usize, 0, 1);

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
        let identity = &messages[0].clone();
        let request = match EmbeddingsRequest::unpack::<EmbeddingsRequest>(&messages[1]) {
            Ok(request) => request,
            Err(e) => {
                let error_msg = format!("Failed to unpack Rerank request: {:?}", e);
                error!("{}", e);
                send_error(&socket, identity, model_id, &error_msg);
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
                            if tokens.len() > n_ctx as usize {
                                warn!("Token sequence exceeds context window, truncating");
                                Some(tokens[..n_ctx as usize].to_vec())
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

        let mut encoding_failed = false;

        for tokens in &tokens_lines_list {
            // Flush the batch if the next prompt would exceed our batch size
            if (batch.n_tokens() as usize + tokens.len()) > n_ctx as usize {
                if let Err(e) =
                    batch_decode(&mut ctx, &mut batch, max_seq_id_batch, &mut output, true)
                {
                    let error_message = format!("Failed to decode batch: {:?}", e);
                    error!("{}", e);
                    send_error(&socket, identity, model_id, &error_message);
                    encoding_failed = true;
                    break;
                }
                max_seq_id_batch = 0;
            }

            if let Err(e) = batch.add_sequence(tokens, max_seq_id_batch, false) {
                let error_message = format!("Failed to add sequence: {:?}", e);
                error!("{}", e);
                send_error(&socket, identity, model_id, &error_message);
                encoding_failed = true;
                break;
            }
            max_seq_id_batch += 1;
        }
        // Handle final batch
        if encoding_failed {
            batch.clear();
            error!("Failed to create batch: {:?}", batch);
            continue;
        }

        if let Err(e) = batch_decode(&mut ctx, &mut batch, max_seq_id_batch, &mut output, true) {
            let error_message = format!("Failed to decode batch: {:?}", e);
            error!("{}", e);
            send_error(&socket, identity, model_id, &error_message);
            continue;
        }
        let response = EmbeddingsResponse::new(model_id, output, None);
        let response_bytes = response.pack()?;
        if let Err(e) = socket.send_multipart(vec![identity, &response_bytes], 0) {
            error!("Failed to send response: {:?}", e);
            continue;
        }
    }
    info!("Worker Shutting down");
    Ok(())
}

fn send_error(socket: &Socket, identity: &Vec<u8>, model_id: u64, error_msg: &str) {
    let response = EmbeddingsResponse::new(model_id, vec![], Some(error_msg.to_string()));
    match response.pack() {
        Ok(response_bytes) => {
            if let Err(e) = socket.send_multipart(vec![identity, &response_bytes], 0) {
                error!("Failed to send error response: {:?}", e);
            }
        }
        Err(e) => {
            error!("Failed to pack error response: {:?}", e);
        }
    }
}

fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> anyhow::Result<()> {
    // The batch is cleared on every path, not just the successful one. A failed decode that left
    // its tokens in the batch would have the next request's sequences added on top of them, and
    // `embeddings_seq_ith(i)` would then return embeddings belonging to the previous request --
    // wrong vectors, silently, with no error anywhere.
    let result = (|| -> anyhow::Result<()> {
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
        Ok(())
    })();

    batch.clear();
    result
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}
