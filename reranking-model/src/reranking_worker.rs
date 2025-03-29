use anyhow::Context;
use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::ServerArgs;
use embedding_common::prelude::{Rank, RerankRequest, RerankResponse, Serde};
use embedding_common::utils::tracting::setup_tracing;
use llama_cpp::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp::context::LlamaContext;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use reranking_model::config::ModelAppConfig;
use std::num::NonZero;
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
    let model_params = if cfg!(feature = "cuda") {
        LlamaModelParams::default().with_n_gpu_layers(config.model_config.ngl as u32)
    } else {
        LlamaModelParams::default()
    };

    let model_path: PathBuf = config.model_config.gguf_file.try_into()?;

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model")?;

    let n_ctx = config
        .model_config
        .max_tokens
        .unwrap_or(model.n_ctx_train());

    error!("model n_ctx_train: {}", n_ctx);
    let pooling_type = LlamaPoolingType::Rank;

    // initialize the context
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_pooling_type(pooling_type)
        .with_n_ctx(Some(NonZero::new(n_ctx).unwrap()))
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ctx)
        .with_embeddings(true);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);

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

    let eos = "</s>";
    let sep = "</s>";
    let bos = "<s>";

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
        let request = match RerankRequest::unpack::<RerankRequest>(&messages[1]) {
            Ok(request) => request,
            Err(e) => {
                error!("Failed to parse request: {:?}", e);
                continue;
            }
        };
        let req_id = request.req_id;
        let query = request.query;
        let prompt_lines = {
            let mut lines = Vec::new();
            for doc in &request.texts {
                // Todo!  update to get eos and sep from model instead of hardcoding
                lines.push(format!("{bos}{query}{eos}{sep}{doc}{eos}"));
            }
            lines
        };

        let tokens_lines_list = prompt_lines
            .iter()
            .filter_map(|text| {
                let res = model.str_to_token(text, AddBos::Never);
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

        for tokens in &tokens_lines_list {
            // Flush the batch if the next prompt would exceed our batch size
            if (batch.n_tokens() as usize + tokens.len()) > n_ctx as usize {
                batch_decode(
                    &mut ctx,
                    &mut batch,
                    max_seq_id_batch,
                    &mut output,
                    true,
                    "rank".to_string(),
                )?;
                max_seq_id_batch = 0;
                batch.clear();
            }

            batch.add_sequence(tokens, max_seq_id_batch, false)?;
            max_seq_id_batch += 1;
        }
        // Handle final batch
        // Handle final batch
        batch_decode(
            &mut ctx,
            &mut batch,
            max_seq_id_batch,
            &mut output,
            true,
            "rank".to_string(),
        )?;

        let scores: Vec<f32> = output.iter().map(|embeddings| embeddings[0]).collect();
        let mut scores_indexed: Vec<(usize, &f32)> = scores.iter().enumerate().collect();
        scores_indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let ranks: Vec<Rank> = scores_indexed
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, score))| Rank::new(idx, Some(rank), Some(*score), None))
            .collect();
        let response = RerankResponse::new(ranks, req_id, None);
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
    pooling: String,
) -> anyhow::Result<()> {
    eprintln!(
        "{}: n_tokens = {}, n_seq = {}",
        stringify!(batch_decode),
        batch.n_tokens(),
        s_batch
    );

    // Clear previous kv_cache values
    ctx.clear_kv_cache();

    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embeddings = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get sequence embeddings")?;
        let normalized = if normalise {
            if pooling == "rank" {
                normalize_embeddings(&embeddings, -1)
            } else {
                normalize_embeddings(&embeddings, 2)
            }
        } else {
            embeddings.to_vec()
        };
        output.push(normalized);
    }
    for out in output.clone().iter() {
        debug!("Output: {}", out[0].to_string());
    }
    batch.clear();

    Ok(())
}

/// Normalizes embeddings based on different normalization strategies
fn normalize_embeddings(input: &[f32], embd_norm: i32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0; n];

    let sum = match embd_norm {
        -1 => 1.0, // no normalization
        0 => {
            // max absolute
            let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / 32760.0;
            max_abs as f64
        }
        2 => {
            // euclidean norm
            input
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        }
        p => {
            // p-norm
            let sum = input.iter().map(|x| (x.abs() as f64).powi(p)).sum::<f64>();
            sum.powf(1.0 / p as f64)
        }
    };

    let norm = if sum > 0.0 { 1.0 / sum } else { 0.0 };

    for i in 0..n {
        output[i] = (input[i] as f64 * norm) as f32;
    }

    output
}
