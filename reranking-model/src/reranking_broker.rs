use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::ServerArgs;
use embedding_common::utils::tracting::setup_tracing;
use reranking_model::model_backend_config::ModelAppConfig;
use tracing::{debug, error, info};

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

    let context = zmq::Context::new();
    if let Err(e) = context.set_io_threads(config.zmq_config.num_workers as i32) {
        error!("Failed to set IO threads: {:?}", e);
        return Err(e.into());
    }
    let io_threads = context.get_io_threads()?;
    info!("ZMQ IO threads: {}", io_threads);
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

    let frontend_endpoint = format!("tcp://*:{:?}", config.zmq_config.frontend_port);
    if let Err(e) = frontend.bind(&frontend_endpoint) {
        error!("Failed to bind frontend socket: {:?}", e);
        return Err(e.into());
    }

    let backend_endpoint = format!("tcp://*:{:?}", config.zmq_config.backend_port);

    if let Err(e) = backend.bind(&backend_endpoint) {
        error!("Failed to bind backend socket: {:?}", e);
        return Err(e.into());
    }
    if let Err(e) = zmq::proxy(&frontend, &backend) {
        error!("Failed to proxy: {:?}", e);
        return Err(e.into());
    }

    info!("Model broker shutting down");
    Ok(())
}
