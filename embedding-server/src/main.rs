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
use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use tracing::{error, info};

fn main() -> Result<(), anyhow::Error> {
    error!("Starting up");
    // Parse command line arguments
    let args = ServerArgs::parse();

    let app_config = AppConfig::from_file(args.config_file)?;

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
    let rocksdb = RocksDB::open(&db_path)?;
    let db = Arc::new(rocksdb);

    let model_config =
        embedding_model::config::ModelConfig::new(model_config.gguf_file, model_config.verbose);

    let model = Model::new(
        &DeterministicAHasher::new(None, None),
        model_config.gguf_file,
        512,
        384,
    );

    put_model(&db, &model)?;

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
    );

    let zmq_ctx = zmq::Context::new();
    if let Err(e) = zmq_ctx.set_io_threads(parallel_workers as i32) {
        error!("Failed to set IO threads: {:?}", e);
        return Err(e.into());
    }

    let frontend = match zmq_ctx.socket(zmq::ROUTER) {
        Ok(socket) => socket,
        Err(e) => {
            error!("Failed to create frontend socket: {:?}", e);
            return Err(e.into());
        }
    };

    frontend.bind(&format!(
        "tcp://{}:{}",
        zmq_config.zmq_host, zmq_config.zmq_frontend_port
    ))?;

    let backend = match zmq_ctx.socket(zmq::DEALER) {
        Ok(socket) => socket,
        Err(e) => {
            error!("Failed to create backend socket: {:?}", e);
            return Err(e.into());
        }
    };
    backend.bind(&format!(
        "tcp://{}:{}",
        zmq_config.zmq_host, zmq_config.zmq_backend_port
    ))?;

    let controller = zmq_ctx.socket(zmq::PUB)?;
    controller
        .bind(format!("tcp://{}:{}", zmq_config.zmq_host, zmq_config.zmq_control_port).as_str())
        .expect("failed connecting controller");

    // Create an atomic boolean to signal shutdown
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = Arc::clone(&shutdown_flag);

    // Spawn a thread to listen for graceful shutdown signals
    let mut signals = Signals::new(&[SIGINT, SIGTERM])?;
    let shutdown_handler = thread::spawn(move || {
        for signal in signals.forever() {
            match signal {
                SIGINT | SIGTERM => {
                    info!("Received signal: {:?}", signal);
                    shutdown_flag_clone.store(true, Ordering::Relaxed);
                    controller.send("SHUTDOWN".as_bytes(), 0).unwrap();
                    info!("Shutting down control message sent");
                    break;
                }
                _ => {}
            }
        }
    });

    // Create an Arc reference to the database
    let db = Arc::new(db);

    // Create an HNSW index and initialize it from the database
    let split_index = Arc::new(HnswIndex::async_create_index(
        "splits".to_string(),
        app_config.index_config.clone(),
    )?);

    let summary_index = Arc::new(HnswIndex::async_create_index(
        "summaries".to_string(),
        app_config.index_config,
    )?);

    initialize_index_from_db(&db, &split_index, &summary_index)?;

    let zmq_ctx = Arc::new(zmq_ctx);

    // Initialize separate workers for each thread
    for _ in 0..parallel_workers {
        let processing_ctx = Arc::clone(&processing_ctx);
        let split_index_clone = Arc::clone(&split_index);
        let summary_index_clone = Arc::clone(&summary_index);
        let db_clone = Arc::clone(&db);
        let zmq_ctx = Arc::clone(&zmq_ctx);
        let zmq_config = Arc::clone(&zmq_config);
        thread::spawn(move || {
            worker_routine(
                zmq_ctx,
                zmq_config,
                processing_ctx,
                &split_index_clone,
                &summary_index_clone,
                &db_clone,
            );
        });
    }

    // Start the ZMQ proxy
    if let Err(e) = zmq::proxy(&frontend, &backend) {
        error!("Failed to start proxy: {:?}", e);
    }
    //FIXME: This is a blocking call, we need to handle timeout
    if shutdown_flag.load(Ordering::Relaxed) {
        info!("Shutting down gracefully...");
        shutdown_handler.join().unwrap();
    }

    Ok(())
}
