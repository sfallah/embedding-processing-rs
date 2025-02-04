use clap::Parser;
use embedding_common::config::{AppConfig, ServerArgs};
use embedding_common::utils::helpers::{create_directory, get_db_dir};
use embedding_database::prelude::{put_model, RocksDB};
use embedding_index::hnsw_index::HnswIndex;
use embedding_index::initialize_index_from_db;
//use embedding_index::save_index;
use embedding_processing::utils::app_utils;
use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};
use embedding_server::zmq::server_task::ServerTask;
use embedding_server::zmq::server_worker::{worker_routine, ServerWorker};
use std::sync::Arc;
use tokio::select;
use tokio::sync::broadcast;
use tracing::{error, info};
use embedding_common::prelude::{DeterministicAHasher, Model};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    error!("Starting up");
    // Parse command line arguments
    let args = ServerArgs::parse();

    let app_config = AppConfig::from_file_async(args.config_file).await?;

    let model_config = app_config.model_config;
    let splitter_config = app_config.splitter_config;
    let db_config = app_config.database_config;

    setup_tracing(args.log_level.to_tracing_level());

    info!("Starting up");

    let zmq_config = Arc::new(app_config.zmq_config);

    // Get available workers
    let max_cores = num_cpus::get();
    let parallel_workers = zmq_config.zmq_num_workers.max(1).min(max_cores);

    info!("Number of workers: {}", parallel_workers);

    let db_path_binding = get_db_dir(Some(&db_config.rocksdb_dir))?;
    let db_path = db_path_binding.to_str().unwrap();
    create_directory(db_path).expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path).await?;
    let db = Arc::new(rocksdb);

    let n_ctx = 512;
    let n_embd = 384;

    let hasher = DeterministicAHasher::new(None, None);
    let model = Model::new(
        &hasher,
        model_config.gguf_file.clone(),
        n_ctx ,
        n_embd,
    );

    put_model(&db, &model).await?;

    let splitter_max_tokens = if model.n_ctx - 2 <= splitter_config.max_tokens as i32 {
        info!(
            "Model context size ({}) does not match max tokens ({})",
            model.n_ctx,
            splitter_config.max_tokens - 2
        );
        let new_max_tokens = ((model.n_ctx - 2) as f32 * 0.8) as usize;
        info!("Setting max tokens to {}", new_max_tokens);
        new_max_tokens
    } else {
        splitter_config.max_tokens
    };

    let processing_ctx = init_ctx(
        splitter_max_tokens,
        splitter_config.merge_level,
        model.n_embd as usize,
        model.model_id,
    )
    .await;
    let clients = ServerTask::init(
        &zmq_config.zmq_host,
        zmq_config.zmq_frontend_port,
        zmq_config.zmq_backend_port,
    )
    .await;

    // Set up signal handling

    let (shutdown_sender, shutdown_receiver) = broadcast::channel::<String>(1);

    let shutdown_clone = shutdown_sender.clone();
    let shutdown_handle = tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!("Failed to listen for Ctrl+C event: {}", e);
        }
        if let Err(e) = shutdown_clone.send("shutdown".to_string()) {
            error!("Failed to send shutdown signal: {}", e);
        }
        info!("Sent shutdown signal");
    });

    // Create an Arc reference to the database
    let db = Arc::new(db);

    // Create an HNSW index and initialize it from the database
    let split_index = Arc::new(
        HnswIndex::async_create_index("splits".to_string(), app_config.index_config.clone())
            .await?,
    );

    let summary_index = Arc::new(
        HnswIndex::async_create_index("summaries".to_string(), app_config.index_config).await?,
    );

    initialize_index_from_db(&db, &split_index, &summary_index).await?;

    let mut worker_handles = Vec::new();

    let model_addr = format!("tcp://{}:{}", zmq_config.zmq_host, "5560");

    // Initialize separate workers for each thread
    for _ in 0..parallel_workers {
        let mut worker = ServerWorker::init(zmq_config.clone()).await;
        let processing_ctx = Arc::clone(&processing_ctx);
        let split_index_clone = Arc::clone(&split_index);
        let summary_index_clone = Arc::clone(&summary_index);
        let db_clone = Arc::clone(&db);
        let shutdown_receiver = shutdown_sender.subscribe();
        let model_addr = model_addr.clone();
        let handle = tokio::task::spawn(async move {
            worker_routine(
                shutdown_receiver,
                &mut worker,
                processing_ctx,
                &split_index_clone,
                &summary_index_clone,
                &db_clone,
                model_addr,
            )
            .await;
        });

        worker_handles.push(handle);
    }

    // Start the ZMQ proxy
    let mut shutdown = shutdown_sender.subscribe();
    select! {
        _ =  shutdown.recv() => {
            //save_index(split_index, summary_index).await?;
            info!("Shutting down");
        }
        _ = zeromq::proxy(clients.frontend, clients.backend, None) => {
            info!("Proxy exited");
        }
    }

    // Shut down
    info!("embedding routines handles joined");
    futures::future::join_all(worker_handles).await;
    info!("worker handles joined");
    shutdown_handle.await?;

    Ok(())
}
