use crate::api::delete::process_document_deletion_request;
use crate::api::insertion::{process_document_insertion_request};
use crate::api::query::process_document_query_request;
use crate::api::retrieval::process_document_retrieval_request;
use crate::schema::zmq_message_header::{ZmqMessageHeader, ZmqMessageType};
use crate::utils::zmq_utils::{handle_error_and_respond};
use embedding_database::prelude::RocksDB;
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use std::sync::Arc;
use async_channel::Sender;
use tokio::sync::broadcast::Receiver;
use tokio::{select};
use tracing::info;
use zeromq::{RepSocket, Socket, SocketRecv};
use embedding_common::config::ZmqConfig;
use embedding_common::prelude::{DeterministicAHasher, Serde};
use embedding_processing::services::embeddings::EmbeddingsRequest;
use crate::api::health::process_health_check;

pub struct ServerWorker {
    pub worker: RepSocket,
    pub zmq_config: Arc<ZmqConfig>,
    pub hasher: Arc<DeterministicAHasher>,
}

impl ServerWorker {
    pub async fn init(zmq_config: Arc<ZmqConfig>) -> Self {
        let mut worker = RepSocket::new();
        let endpoint = format!("tcp://{}:{}", zmq_config.zmq_host, zmq_config.zmq_backend_port);
        worker
            .connect(&endpoint)
            .await
            .expect("Worker failed to connect to backend");
        let hasher = Arc::new(DeterministicAHasher::new(None, None));
        let zmq_config = zmq_config.clone();
        ServerWorker { worker, zmq_config, hasher }
    }
}

pub async fn worker_routine(
    mut shutdown: Receiver<String>,
    worker_socket: &mut ServerWorker,
    processing_context: Arc<ProcessingContext>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    db: &Arc<RocksDB>,
    embedding_sender: Arc<Sender<EmbeddingsRequest>>,
) {
    loop {
        select! {
          msg  = worker_socket.worker.recv() => {
                let messages = match msg {
                    Ok(messages) => messages,
                    Err(e) => {
                        let error_message = format!("Error receiving message: {}", e);
                        handle_error_and_respond(
                            &mut worker_socket.worker,
                            &error_message,
                            ZmqMessageType::Unknown,
                        )
                            .await;
                        continue;
                    }
                };

        let mut message_header: ZmqMessageHeader =
            match ZmqMessageHeader::unpack(&messages.get(0).unwrap()) {
                Ok(header) => header,
                Err(e) => {
                    let error_message = format!("Error unpacking message header: {}", e);
                    handle_error_and_respond(
                        &mut worker_socket.worker,
                        &error_message,
                        ZmqMessageType::Unknown,
                    )
                        .await;
                    continue;
                }
            };

        if messages.len() < 2 && message_header.message_type != ZmqMessageType::HealthCheck {
            let error_message = format!(
                "Request Body received for message type: {:?}",
                message_header.message_type
            );
            handle_error_and_respond(
                &mut worker_socket.worker,
                &error_message,
                message_header.message_type,
            )
                .await;
            continue;
        }

        match message_header.message_type {
            ZmqMessageType::DocumentInsertion => {
                process_document_insertion_request(
                    &mut worker_socket.worker,
                    db,
                    processing_context.clone(),
                    split_index,
                    summary_index,
                    embedding_sender.clone(),
                    &mut message_header,
                    &messages.get(1).unwrap().to_vec(),
                )
                    .await;
            }
            ZmqMessageType::DocumentQuery => {
                process_document_query_request(
                    &mut worker_socket.worker,
                    db,
                    processing_context.clone(),
                    split_index,
                    summary_index,
                    embedding_sender.clone(),
                    &mut message_header,
                    &messages.get(1).unwrap().to_vec(),
                )
                    .await;
            }
            ZmqMessageType::DocumentRetrieval => {
                        process_document_retrieval_request(
                            &mut worker_socket.worker,
                            worker_socket.hasher.clone(),
                            db,
                            &mut message_header,
                            &messages.get(1).unwrap().to_vec(),
                        )
                            .await;
                    }
            ZmqMessageType::DocumentDeletion => {
                        process_document_deletion_request(
                            &mut worker_socket.worker,
                            split_index,
                            summary_index,
                            db,
                            &mut message_header,
                            &messages.get(1).unwrap().to_vec(),
                        )
                            .await;
                    }
                    ZmqMessageType::HealthCheck => {
                process_health_check(&mut worker_socket.worker, &mut message_header, worker_socket.zmq_config.clone()).await;
            }
            _ => {
                let error_message = format!(
                    "Unrecognized message type: {:?}",
                    message_header.message_type
                );
                handle_error_and_respond(
                    &mut worker_socket.worker,
                    &error_message,
                    message_header.message_type,
                )
                    .await;
            }
        }
        },
            _kill = shutdown.recv() => {
                info!("Shutting down worker routine");
                break;
        }
        }
    }
    info!("Worker shutting down");
}
