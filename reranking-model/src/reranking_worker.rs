use anyhow::Context;
use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::ServerArgs;
use embedding_common::prelude::{
    DeterministicAHasher, Model, Rank, RerankRequest, RerankResponse, Serde,
};
use embedding_common::utils::tracting::setup_tracing;
use indexmap::IndexMap;
use llama_cpp::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp::context::LlamaContext;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use reranking_model::model_backend_config::ModelAppConfig;
use std::num::NonZero;
use std::path::PathBuf;
use tracing::{debug, error, info};
use zmq::Socket;

fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();

    setup_tracing(args.log_level.to_tracing_level());

    let config = match ModelAppConfig::from_file(args.config_file) {
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
    let pooling_type = LlamaPoolingType::Rank;

    // initialize the context
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_pooling_type(pooling_type)
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

    let mut batch = LlamaBatch::new(n_ctx as usize,0, 1);

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

    let bos_token = model.token_bos();
    let eos_token = model.token_eos();
    let sep_token = model.token_sep();

    loop {
        let messages = match socket.recv_multipart(0) {
            Ok(messages) => messages,
            Err(e) => {
                error!("Failed to receive: {:?}", e);
                // can we send error message without identity?
                break;
            }
        };
        //debug!("Worker {} received messages", worker_id);
        let identity = &messages[0];
        let request = match RerankRequest::unpack::<RerankRequest>(&messages[1]) {
            Ok(request) => request,
            Err(e) => {
                let error_msg = format!("Failed to unpack Rerank request: {:?}", e);
                error!("{}", e);
                send_error(&socket, identity, model_id, &error_msg);
                continue;
            }
        };
        let query = request.query;

        if query.is_empty() {
            let error_msg = "Query is empty";
            error!("{}", error_msg);
            send_error(&socket, identity, model_id, &error_msg);
            continue;
        }
        let query_tokens = match model.str_to_token(&query, AddBos::Never) {
            Ok(tokens) => tokens,
            Err(e) => {
                let error_msg = format!("Failed to tokenize query: {:?}", e);
                error!("{}", error_msg);
                send_error(&socket, identity, model_id, &error_msg);
                continue;
            }
        };

        let query_no_tokens = query_tokens.len();

        if query_no_tokens + 2 > n_ctx as usize {
            let error_msg = format!(
                "Query no_tokens: {} exceeds n_ctx: {}",
                query_no_tokens, n_ctx
            );
            error!("{}", error_msg);
            send_error(&socket, identity, model_id, &error_msg);
            continue;
        }

        let mut sequence_pairs_map = IndexMap::new();
        for (idx, text) in request.texts.iter().enumerate() {
            let text_tokens = match model.str_to_token(text, AddBos::Never) {
                Ok(tokens) => tokens,
                Err(e) => {
                    let error_msg = format!(
                        "Failed to tokenize seq: {}\n, text: {:?}\n, error: {:?}",
                        idx, text, e
                    );
                    error!("{}", error_msg);
                    continue;
                }
            };
            let text_no_tokens = text_tokens.len();
            if text_no_tokens + query_no_tokens + 4 > n_ctx as usize {
                let error_msg = format!(
                    "Sequence Pair no_tokens exceeds n_ctx. Query: {}, text: {}, n_ctx: {}",
                    query_no_tokens, text_no_tokens, n_ctx
                );
                error!("{}", error_msg);
                continue;
            }
            //"{bos}{query}{eos}{sep}{doc}{eos}"
            let mut sequence_pairs_tokens = query_tokens.clone();
            sequence_pairs_tokens.insert(0, bos_token);
            sequence_pairs_tokens.push(eos_token);
            sequence_pairs_tokens.push(sep_token);
            sequence_pairs_tokens.append(&mut text_tokens.clone());
            sequence_pairs_tokens.push(eos_token);
            sequence_pairs_map.insert(idx, sequence_pairs_tokens);
        }

        let mut max_seq_id_batch = 0;
        let mut output = Vec::with_capacity(sequence_pairs_map.len());

        let mut decode_failed = false;

        for tokens in sequence_pairs_map.values().into_iter() {
            // Flush the batch if the next prompt would exceed our batch size
            if (batch.n_tokens() as usize + tokens.len()) > n_ctx as usize {
                if let Err(e) = batch_decode(
                    &mut ctx,
                    &mut batch,
                    max_seq_id_batch,
                    &mut output,
                    true,
                    //FIXME: this should be a parameter
                    "rank".to_string(),
                ) {
                    let error_message = format!("Failed to decode batch: {:?}", e);
                    error!("{}", error_message);
                    send_error(&socket, identity, model_id, &error_message);
                    decode_failed = true;
                    break;
                }
                max_seq_id_batch = 0;
                batch.clear();
            }
            batch.add_sequence(tokens, max_seq_id_batch, false)?;
            max_seq_id_batch += 1;
        }
        if decode_failed {
            continue;
        }
        // Handle final batch
        if let Err(e) = batch_decode(
            &mut ctx,
            &mut batch,
            max_seq_id_batch,
            &mut output,
            true,
            //FIXME: this should be a parameter
            "rank".to_string(),
        ) {
            let error_message = format!("Failed to decode batch: {:?}", e);
            error!("{}", error_message);
            send_error(&socket, identity, model_id, &error_message);
            continue;
        }

        let scores: Vec<f32> = output.iter().map(|embeddings| embeddings[0]).collect();
        let mut scores_indexed: Vec<(&usize, &f32)> = sequence_pairs_map
            .keys()
            .into_iter()
            .zip(scores.iter())
            .collect();
        scores_indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let ranks: Vec<Rank> = scores_indexed
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, score))| Rank::new(*idx, Some(rank), Some(*score), None))
            .collect();
        let response = RerankResponse::new(Some(model_id), ranks, None);
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
    let response = RerankResponse::new(Some(model_id), vec![], Some(error_msg.to_string()));
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
    pooling: String,
) -> anyhow::Result<()> {
    eprintln!(
        "{}: n_tokens = {}, n_seq = {}",
        stringify!(batch_decode),
        batch.n_tokens(),
        s_batch
    );

    // The batch is cleared on every path, not just the successful one. A failed decode that left
    // its tokens in the batch would have the next request's pairs added on top of them, and
    // `embeddings_seq_ith(i)` would then score the previous request's pairs -- wrong ranks,
    // silently, with no error anywhere.
    let result = (|| -> anyhow::Result<()> {
        // Clear previous kv_cache values
        ctx.clear_kv_cache();

        ctx.decode(batch).with_context(|| "llama_decode() failed")?;

        for i in 0..s_batch {
            let embeddings = ctx
                .embeddings_seq_ith(i)
                .with_context(|| "Failed to get sequence embeddings")?;
            println!(
                "embeddings: {:?}",
                embeddings.iter().take(20).collect::<Vec<_>>()
            );
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
        Ok(())
    })();
    batch.clear();
    result?;

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
