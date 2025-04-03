use crate::api::delete::process_document_deletion_request;
use crate::api::health::process_health_check;
use crate::api::insertion::process_document_insertion_request;
use crate::api::query::process_document_query_request;
use crate::api::rerank::process_rerank;
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
    zmq_config: Arc<ZmqConfig>,
    context: &zmq::Context,
    processing_context: Arc<ProcessingContext>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    db: &Arc<RocksDB>,
) {
    let hasher = Arc::new(DeterministicAHasher::new(None, None));

    //  Prepare our context and socket
    let socket = match context.socket(zmq::DEALER) {
        Ok(socket) => socket,
        Err(e) => {
            error!("Failed to create socket: {:?}", e);
            return;
        }
    };

    if let Err(e) = socket.set_linger(0) {
        error!("Failed to set linger: {:?}", e);
        return;
    }

    let backend_endpoint = format!(
        "tcp://{}:{}",
        zmq_config.host, zmq_config.backend_port
    );
    info!("Connecting to model host: {}", backend_endpoint);
    if let Err(e) = socket.connect(&backend_endpoint) {
        error!("Failed to connect to model host: {:?}", e);
        return;
    }
    info!(
        "Embedding-Server worker started on endpoint: {}",
        backend_endpoint
    );
    loop {
        let messages = match socket.recv_multipart(0) {
            Ok(messages) => messages,
            Err(e) => {
                error!("Error receiving message: {}", e);
                break;
            }
        };
        let identity = messages[0].clone();

        let mut message_header: ZmqMessageHeader =
            match ZmqMessageHeader::unpack(&messages.get(1).unwrap()) {
                Ok(header) => header,
                Err(e) => {
                    let error_message = format!("Error unpacking message header: {}", e);
                    handle_error_and_respond(
                        &socket,
                        &error_message,
                        ZmqMessageType::Unknown,
                        &identity,
                    );
                    continue;
                }
            };

        if message_header.message_type == ZmqMessageType::HealthCheck {
            process_health_check(&socket, &mut message_header, zmq_config.clone(), &identity);
            continue;
        }

        if messages.len() < 3 {
            let error_message = format!(
                "Request Body received for message type: {:?}",
                message_header.message_type
            );
            handle_error_and_respond(
                &socket,
                &error_message,
                message_header.message_type,
                &identity,
            );
            continue;
        }

        let message_body = match messages.get(2) {
            Some(body) => body,
            None => {
                let error_message = format!(
                    "No Request Body for message type: {:?}",
                    message_header.message_type
                );
                handle_error_and_respond(
                    &socket,
                    &error_message,
                    message_header.message_type,
                    &identity,
                );
                continue;
            }
        };
        match message_header.message_type {
            ZmqMessageType::DocumentInsertion => {
                process_document_insertion_request(
                    &socket,
                    db,
                    processing_context.clone(),
                    split_index,
                    summary_index,
                    &mut message_header,
                    &message_body,
                    &identity,
                );
            }
            ZmqMessageType::DocumentQuery => {
                process_document_query_request(
                    &socket,
                    db,
                    processing_context.clone(),
                    split_index,
                    summary_index,
                    &mut message_header,
                    &message_body,
                    &identity,
                );
            }
            ZmqMessageType::DocumentRetrieval => {
                process_document_retrieval_request(
                    &socket,
                    hasher.clone(),
                    db,
                    &mut message_header,
                    &message_body,
                    &identity,
                );
            }
            ZmqMessageType::DocumentDeletion => {
                process_document_deletion_request(
                    &socket,
                    split_index,
                    summary_index,
                    db,
                    &mut message_header,
                    &message_body,
                    &identity,
                );
            }
            ZmqMessageType::Rerank => {
                process_rerank(
                    &socket,
                    processing_context.clone(),
                    &mut message_header,
                    &message_body,
                    &identity,
                );
            }
            _ => {
                let error_message = format!(
                    "Unrecognized message type: {:?}",
                    message_header.message_type
                );
                handle_error_and_respond(
                    &socket,
                    &error_message,
                    message_header.message_type,
                    &identity,
                );
            }
        }
    }
    info!("Worker shutting down");
}
