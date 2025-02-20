use crate::api::delete::process_document_deletion_request;
use crate::api::health::process_health_check;
use crate::api::insertion::process_document_insertion_request;
use crate::api::query::process_document_query_request;
use crate::api::retrieval::process_document_retrieval_request;
use crate::schema::zmq_message_header::{ZmqMessageHeader, ZmqMessageType};
use crate::utils::zmq_utils::handle_error_and_respond;
use embedding_common::config::ZmqConfig;
use embedding_common::prelude::{DeterministicAHasher, Serde};
use embedding_database::prelude::RocksDB;
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use std::sync::Arc;
use tracing::{error, info};

pub fn worker_routine(
    zmq_ctx: Arc<zmq::Context>,
    zmq_config: Arc<ZmqConfig>,
    processing_context: Arc<ProcessingContext>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    db: &Arc<RocksDB>,
) {
    let socket = match zmq_ctx.socket(zmq::DEALER) {
        Ok(socket) => socket,
        Err(e) => {
            error!("Failed to create socket: {:?}", e);
            panic!("Failed to create socket: {:?}", e);
        }
    };
    if let Err(e) = socket.set_linger(0) {
        error!("Failed to set linger: {:?}", e);
        panic!("Failed to set linger: {:?}", e);
    }

    let backend_endpoint = format!(
        "tcp://{}:{}",
        zmq_config.zmq_host, zmq_config.zmq_backend_port
    );
    info!("Connecting to model host: {}", backend_endpoint);
    if let Err(e) = socket.connect(&backend_endpoint) {
        error!("Failed to connect to model host: {:?}", e);
        panic!("Failed to connect to model host: {:?}", e);
    }

    let controller = zmq_ctx.socket(zmq::SUB).unwrap();
    controller
        .connect(
            format!(
                "tcp://{}:{}",
                zmq_config.zmq_host, zmq_config.zmq_control_port
            )
            .as_str(),
        )
        .expect("failed connecting controller");
    controller.set_subscribe(b"").expect("failed subscribing");

    let hasher = Arc::new(DeterministicAHasher::new(None, None));
    let zmq_config = zmq_config.clone();
    let socket = Arc::new(socket);
    loop {
        let mut items = [
            controller.as_poll_item(zmq::POLLIN),
            socket.as_poll_item(zmq::POLLIN),
        ];
        //FIXME: This is a blocking call, we need to handle shutdown signals
        zmq::poll(&mut items, -1).expect("failed polling");

        if items[0].is_readable() {
            info!("Received shutdown control signal");
            break;
        } else if items[1].is_readable() {
            let messages = match socket.recv_multipart(0) {
                Ok(messages) => messages,
                Err(e) => {
                    let error_message = format!("Error receiving message: {}", e);
                    error!("{}", &error_message);
                    continue;
                }
            };

            let identity = messages[0].clone();

            let messages = messages.into_iter().skip(1).collect::<Vec<_>>();
            let mut message_header: ZmqMessageHeader =
                match ZmqMessageHeader::unpack(&messages.get(0).unwrap()) {
                    Ok(header) => header,
                    Err(e) => {
                        let error_message = format!("Error unpacking message header: {}", e);
                        handle_error_and_respond(
                            socket.clone(),
                            &identity,
                            &error_message,
                            ZmqMessageType::Unknown,
                        );
                        continue;
                    }
                };

            if messages.len() < 2 && message_header.message_type != ZmqMessageType::HealthCheck {
                let error_message = format!(
                    "Request Body received for message type: {:?}",
                    message_header.message_type
                );
                handle_error_and_respond(
                    socket.clone(),
                    &identity,
                    &error_message,
                    message_header.message_type,
                );
                continue;
            }

            match message_header.message_type {
                ZmqMessageType::DocumentInsertion => {
                    process_document_insertion_request(
                        socket.clone(),
                        db,
                        processing_context.clone(),
                        split_index,
                        summary_index,
                        &mut message_header,
                        &identity,
                        &messages.get(1).unwrap().to_vec(),
                    );
                }
                ZmqMessageType::DocumentQuery => {
                    process_document_query_request(
                        socket.clone(),
                        db,
                        processing_context.clone(),
                        split_index,
                        summary_index,
                        &mut message_header,
                        &identity,
                        &messages.get(1).unwrap().to_vec(),
                    );
                }
                ZmqMessageType::DocumentRetrieval => {
                    process_document_retrieval_request(
                        socket.clone(),
                        hasher.clone(),
                        db,
                        &mut message_header,
                        &identity,
                        &messages.get(1).unwrap().to_vec(),
                    );
                }
                ZmqMessageType::DocumentDeletion => {
                    process_document_deletion_request(
                        socket.clone(),
                        split_index,
                        summary_index,
                        db,
                        &mut message_header,
                        &identity,
                        &messages.get(1).unwrap().to_vec(),
                    );
                }
                ZmqMessageType::HealthCheck => {
                    process_health_check(
                        socket.clone(),
                        &identity,
                        &mut message_header,
                        zmq_config.clone(),
                    );
                }
                _ => {
                    let error_message =
                        format!("Unknown message type: {:?}", message_header.message_type);
                    handle_error_and_respond(
                        socket.clone(),
                        &identity,
                        &error_message,
                        message_header.message_type,
                    );
                }
            }
        }
    }
}
