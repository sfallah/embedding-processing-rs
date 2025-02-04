use anyhow::{bail, Context};
use clap::Parser;
use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::context::LlamaContext;
use llama_cpp::ggml_time_us;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel, Special};
use llama_cpp::LLamaCppError::EmbeddingError;
use std::path::PathBuf;
use std::time::Duration;
use tracing::error;
use zeromq::{Socket, SocketRecv, SocketSend, ZmqMessage};
use embedding_common::config::{AppConfig, ServerArgs};
use embedding_common::prelude::Serde;
use embedding_model::api::{EmbeddingsRequest, EmbeddingsResponse};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    error!("Starting up");
    let args = ServerArgs::parse();

    let app_config = AppConfig::from_file_async(args.config_file).await?;

    // init LLM
    let backend = LlamaBackend::init()?;

    // offload all layers to the gpu
    let mut model_params = LlamaModelParams::default();

    let model_path: PathBuf = app_config.model_config.gguf_file.try_into()?;

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
    let n_ctx_train = model.n_ctx_train();

    let mut socket = zeromq::RepSocket::new();

    socket
        .bind("tcp://0.0.0.0:5560")
        .await
        .expect("Failed to connect");

    // create a llama_batch with the size of the context
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(n_ctx, 1);

    loop {
        if let Ok(msg) = socket.recv().await {
            let request: EmbeddingsRequest = EmbeddingsRequest::unpack(&msg.get(0).unwrap())?;

            // tokenize the prompt
            let tokens_lines_list = request
                .texts
                .clone()
                .into_iter()
                .map(|line| model.str_to_token(line.as_str(), AddBos::Always))
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| format!("failed to tokenize {:?}", request.texts))?;

            if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
                bail!("One of the provided prompts exceeds the size of the context window");
            }

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

            let response = EmbeddingsResponse::new(request.seq_id, request.n_embd, output.to_vec());

            let response_bytes = response.pack()?;
            socket.send(ZmqMessage::from(response_bytes)).await?;
        }
    }

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
