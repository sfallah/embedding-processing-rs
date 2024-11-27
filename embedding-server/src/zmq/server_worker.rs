use crate::api::insertion::{process_document_insertion_request, send_document_insertion_response};
use crate::api::query::process_document_query_request;
use crate::schema::insertion::DocumentInsertionRequest;
use crate::schema::zmq_message_header::{ZmqMessageHeader, ZmqMessageType};
use crate::utils::zmq_utils::{handle_error_and_respond, send_exception_response};
use crate::zmq::server_params::ZmqParams;
use crate::ServerArgs;
use anyhow::Result;
use async_channel::Sender;
use embedding_common::dtos::document_dto::DocumentDto;
use embedding_common::prelude::*;
use embedding_database::prelude::RocksDB;
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::documents::process_document;
use embedding_processing::services::embeddings::EmbeddingsRequest;
use futures::FutureExt;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast::Receiver;
use tokio::{select, task};
use tracing::info;
use uuid::Uuid;
use zeromq::{RepSocket, Socket, SocketRecv};

pub struct ServerWorker {
    pub worker: RepSocket,
}

impl ServerWorker {
    pub async fn init(host: &str, port: usize) -> Self {
        let mut worker = RepSocket::new();
        let endpoint = format!("tcp://{}:{}", host, port);
        worker
            .connect(&endpoint)
            .await
            .expect("Worker failed to connect to backend");
        ServerWorker { worker }
    }
}

pub async fn worker_routine(
    mut shutdown: Receiver<String>,
    worker_socket: &mut ServerWorker,
    processing_context: Arc<ProcessingContext>,
    model_clone: Arc<Model>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    db: &Arc<RocksDB>,
    embedding_sender: Arc<Sender<EmbeddingsRequest>>,
) {
    loop {
        select! {
          msg  = worker_socket.worker.recv().fuse() => {
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
