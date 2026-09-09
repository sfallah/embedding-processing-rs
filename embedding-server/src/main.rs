use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::AppConfig;
use embedding_common::prelude::{DeterministicAHasher, Model};
use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};
use embedding_server::maintenance::{self, MaintenancePolicy};
use embedding_server::zmq::server_worker::worker_routine;
use embedding_server::ServerArgs;
use embedding_store::prelude::{ShardOptions, ShardPool};
use std::sync::Arc;
use std::thread;
use tracing::{error, info};

fn main() -> Result<(), anyhow::Error> {
    // Parse command line arguments
    let args = ServerArgs::parse();

    let app_config = AppConfig::from_file(args.config_file)?;

    let embedding_model_info = app_config.embedding_model_info;
    let splitter_config = app_config.splitter_config;
    let storage_config = app_config.storage_config;
    storage_config.validate()?;

    setup_tracing(args.log_level.to_tracing_level());

    info!("Starting up");

    let zmq_config = Arc::new(app_config.zmq_config);

    // Get available workers
    let max_cores = num_cpus::get();
    let parallel_workers = zmq_config.num_workers.max(1).min(max_cores);

    info!("Number of workers: {}", parallel_workers);

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

    // One shard per workspace, each holding its own log and its own two indexes. The model id
    // goes into every shard's manifest, which is what refuses a shard written by another model
    // (decision D5).
    let mut shard_options = ShardOptions::new(model.model_id, app_config.index_config.clone());
    shard_options.segment_max_bytes = storage_config.segment_max_bytes();

    let storage_dir = storage_config.storage_dir()?;
    let pool = Arc::new(
        ShardPool::new(&storage_dir, shard_options)?
            .with_memory_budget_mb(storage_config.memory_budget_mb),
    );

    // Nothing is loaded until a request asks for a workspace, so startup no longer pays for the
    // whole corpus: at 500k splits the old rebuild took minutes and 11.9 GB of peak memory.
    let known_workspaces = pool.list_workspaces()?;
    info!(
        "Storage at {}: {} workspace(s) on disk, {} MB budget",
        storage_dir.display(),
        known_workspaces.len(),
        storage_config.memory_budget_mb
    );

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

    // Housekeeping. Without these two the shards are only put on disk when the pool evicts one,
    // so a kill would leave the whole log to replay on the next start.
    let policy = MaintenancePolicy::from_config(&storage_config);
    maintenance::install_shutdown_handler(Arc::clone(&pool))?;
    maintenance::spawn(Arc::clone(&pool), policy);

    // With `fsync_interval_ms = 0` there is no timer to sync the log, so a write is synced before
    // it is acknowledged (decision D4).
    let sync_on_write = policy.fsync_interval.is_none();
    if sync_on_write {
        info!("fsync policy: every write is synced before it is acknowledged");
    }

    // Initialize separate workers for each thread
    let mut worker_thread_pool = Vec::new();
    for _ in 0..parallel_workers {
        let processing_ctx = Arc::clone(&processing_ctx);
        let pool = Arc::clone(&pool);
        let zmq_config = Arc::clone(&zmq_config);
        let context = context.clone();
        let worker_handler = thread::spawn(move || {
            worker_routine(zmq_config, &context, processing_ctx, pool, sync_on_write)
        });
        worker_thread_pool.push(worker_handler);
    }

    if let Err(e) = zmq::proxy(&frontend, &backend) {
        // A signal interrupts the proxy here while the shutdown handler snapshots on its own
        // thread. Returning would end the process mid-snapshot, so wait for it instead, and do
        // not call an ordinary shutdown a failure.
        if e == zmq::Error::EINTR && maintenance::await_shutdown() {
            info!("shut down on signal");
            return Ok(());
        }
        error!("Failed to proxy: {:?}", e);
        return Err(e.into());
    }

    info!("Model broker shutting down");
    for worker in worker_thread_pool {
        worker.join().unwrap();
    }
    Ok(())
}
