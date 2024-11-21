use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use clap::Parser;
use tracing::{error, info};
use zeromq::Socket;
use embedding_common::models::embedding::EmbeddingDataType;
use embedding_common::utils::helpers::{create_directory, get_db_dir};
use embedding_database::db::rocksdb_impl::RocksDB;
use embedding_index::hnsw_index::HnswIndex;
use embedding_database::dao::embedding_dao::get_all_embeddings;
use embedding_processing::inference::llama_context::LlamaContext;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::utils::app_utils;
use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};
use embedding_server::{zmq, ServerArgs};
use embedding_server::utils::index_utils::initialize_index_from_db;
use embedding_server::zmq::server_params::ZmqParams;
use embedding_server::zmq::server_task::ServerTask;
use embedding_server::zmq::server_worker::{worker_routine, ServerWorker};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error>{
    // Parse command line arguments
    let args = ServerArgs::parse();

    setup_tracing(args.log_level.to_tracing_level());

    info!("Starting up");

    // Get available workers
    let max_cores = num_cpus::get();
    let parallel_workers = args.np.max(1).min(max_cores);
    let gpu_layers = args.ngl;

    info!("Number of workers: {}", args.np);
    let n_workers: usize = parallel_workers;

    // Initialize ZeroMQ parameters
    let zmq_params = Arc::new(ZmqParams {
        hostname: "127.0.0.1".to_string(),
        frontend_port: args.frontend_port,
        backend_port: args.backend_port,
        workers: n_workers,
    });

    let no_model_instances = 2 * n_workers;
    eprintln!("Number of model instances: {}", no_model_instances);
    let model_instances = (0..no_model_instances)
        .map(|_| Arc::new(LlamaContext::new(&args.model_path, 600, gpu_layers)))
        .collect::<Vec<_>>();

    let n_embd = model_instances[0].get_n_embd() as usize;


    let db_path_binding = get_db_dir(Some(&args.db_path.unwrap_or_else(|| "rocks_db_dir".to_string())))?;
    let db_path = db_path_binding.to_str().unwrap();
    create_directory(db_path).expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path).await?;
    let db = Arc::new(rocksdb);

    let (embed, shutdown, handles, model) = app_utils::init(&args.model_path, args.np).await?;

    let processing_ctx = init_ctx(args.max_tokens, args.merge_level, n_embd, model.model_id).await;

    let clients = ServerTask::init(&args.server_host, args.frontend_port, args.backend_port).await;

    // Set up signal handling
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        info!("Signal received. Shutting down...");
    })
        .expect("Error setting Ctrl-C handler");

    // Create an Arc reference to the database
    let db = Arc::new(db);

    // Create an HNSW index and initialize it from the database
    let split_index = Arc::new(HnswIndex::new(n_embd)?);

    let summary_index = Arc::new(HnswIndex::new(n_embd)?);

    initialize_index_from_db(&db, &split_index, &summary_index).await;

    let mut worker_handles = Vec::new();

    // Initialize separate workers for each thread
    let mut workers = Vec::new();
    for i in 0..n_workers {
        workers.push(ServerWorker::init(&args.server_host, args.backend_port).await);
    }

    // Start worker threads
    for (i, mut worker) in workers.into_iter().enumerate() {
        let running_clone = running.clone();
        let processing_ctx = Arc::clone(&processing_ctx);
        let zmq_params_clone = Arc::clone(&zmq_params);
        let model_clone = Arc::clone(&model);
        let split_index_clone = Arc::clone(&split_index);
        let summary_index_clone = Arc::clone(&summary_index);
        let db_clone = Arc::clone(&db);
        let sender_clone = Arc::clone(&embed);

        let handle = tokio::task::spawn(async move {
            worker_routine(
                running_clone,
                &mut worker,
                &zmq_params_clone,
                processing_ctx,
                model_clone,
                &split_index_clone,
                &summary_index_clone,
                &db_clone,
                sender_clone,
                n_embd,
            )
                .await;
        });

        worker_handles.push(handle);
    }

    // Start the ZMQ proxy
    zeromq::proxy(clients.frontend, clients.backend, None).await?;

    // Shut down
    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles).await;
    futures::future::join_all(worker_handles).await;

    Ok(())

}