use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::AppConfig;
use embedding_common::prelude::{DeterministicAHasher, Model};
use embedding_common::utils::helpers::{create_directory, get_db_dir};
use embedding_database::prelude::{put_model, RocksDB};
use embedding_index::hnsw_index::HnswIndex;
use embedding_index::initialize_index_from_db;
use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};
use embedding_server::zmq::server_worker::worker_routine;
use embedding_server::ServerArgs;
use std::sync::Arc;
use std::thread;
use tracing::{error, info};

fn main() -> Result<(), anyhow::Error> {
    error!("Starting up");
    // Parse command line arguments
    let args = ServerArgs::parse();

    let app_config = AppConfig::from_file(args.config_file)?;

    let embedding_model_info = app_config.embedding_model_info;
    let splitter_config = app_config.splitter_config;
    let db_config = app_config.database_config;

    setup_tracing(args.log_level.to_tracing_level());

    info!("Starting up");

    let zmq_config = Arc::new(app_config.zmq_config);

    // Get available workers
    let max_cores = num_cpus::get();
    let parallel_workers = zmq_config.num_workers.max(1).min(max_cores);

    info!("Number of workers: {}", parallel_workers);

    let db_path_binding = get_db_dir(Some(&db_config.rocksdb_dir))?;
    let db_path = db_path_binding.to_str().unwrap();
    create_directory(db_path).expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path)?;
    let db = Arc::new(rocksdb);

    let default_hasher = DeterministicAHasher::default_hasher();
    let model_id = Model::model_id(
        &default_hasher,
        &embedding_model_info.path,
        embedding_model_info.n_embd,
    );

    let model = Model::new(
        model_id,
        embedding_model_info.path.clone(),
        embedding_model_info.n_embd,
    );

    put_model(&db, &model)?;

    let splitter_max_tokens = splitter_config.max_tokens;

    let processing_ctx = init_ctx(
        splitter_max_tokens,
        splitter_config.merge_level,
        model.n_embd as usize,
        model.model_id,
        app_config.endpoints.embedding_endpoint.clone(),
        app_config.endpoints.reranking_endpoint.clone(),
    );

    let context = zmq::Context::new();
    let frontend = match context.socket(zmq::ROUTER) {
        Ok(socket) => socket,
        Err(e) => {
            error!("Failed to create frontend socket: {:?}", e);
            return Err(e.into());
        }
    };

    let backend = match context.socket(zmq::DEALER) {
        Ok(socket) => socket,
        Err(e) => {
            error!("Failed to create backend socket: {:?}", e);
            return Err(e.into());
        }
    };

    let frontend_endpoint = format!("tcp://{}:{:?}", &zmq_config.host, zmq_config.frontend_port);
    if let Err(e) = frontend.bind(&frontend_endpoint) {
        error!("Failed to bind frontend socket: {:?}", e);
        return Err(e.into());
    }

    let backend_endpoint = format!("tcp://{}:{:?}", &zmq_config.host, zmq_config.backend_port);

    if let Err(e) = backend.bind(&backend_endpoint) {
        error!("Failed to bind backend socket: {:?}", e);
        return Err(e.into());
    }

    // Set up signal handling

    /*
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
     */

    // Create an Arc reference to the database
    let db = Arc::new(db);

    // Create an HNSW index and initialize it from the database
    let split_index = Arc::new(HnswIndex::create_index(
        "splits".to_string(),
        app_config.index_config.clone(),
    )?);

    let summary_index = Arc::new(HnswIndex::create_index(
        "summaries".to_string(),
        app_config.index_config,
    )?);

    initialize_index_from_db(&db, &split_index, &summary_index)?;

    // Initialize separate workers for each thread
    let mut worker_thread_pool = Vec::new();
    for _ in 0..parallel_workers {
        let processing_ctx = Arc::clone(&processing_ctx);
        let split_index_clone = Arc::clone(&split_index);
        let summary_index_clone = Arc::clone(&summary_index);
        let db_clone = Arc::clone(&db);
        let zmq_config = Arc::clone(&zmq_config);
        let context = context.clone();
        let worker_handler = thread::spawn(move || {
            worker_routine(
                zmq_config,
                &context,
                processing_ctx,
                &split_index_clone,
                &summary_index_clone,
                &db_clone,
            )
        });
        worker_thread_pool.push(worker_handler);
    }

    if let Err(e) = zmq::proxy(&frontend, &backend) {
        error!("Failed to proxy: {:?}", e);
        return Err(e.into());
    }

    info!("Model broker shutting down");
    for worker in worker_thread_pool {
        worker.join().unwrap();
    }
    Ok(())
}
