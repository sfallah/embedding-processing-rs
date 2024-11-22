use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use async_channel::Sender;
use anyhow::Result;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use tokio::task;
use tracing::{info};
use uuid::Uuid;
use zeromq::{RepSocket, Socket, SocketRecv};
use embedding_common::dtos::document_dto::DocumentDto;
use embedding_common::models::document::Document;
use embedding_common::models::embedding::{Embedding, EmbeddingUser};
use embedding_common::models::model::Model;
use embedding_common::models::split::Split;
use embedding_common::models::summary::Summary;
use embedding_common::Serde;
use embedding_common::utils::helpers::{from_epoch_micros, time_millis};
use embedding_database::db::rocksdb_impl::RocksDB;
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::documents::process_document;
use embedding_processing::services::embeddings::EmbeddingsRequest;
use crate::dtos::document::DocumentInsertionRequest;
use crate::dtos::zmq_message_header::{ZmqMessageHeader, ZmqMessageType};
use crate::ServerArgs;
use crate::utils::db_utils::save_models_to_db;
use crate::utils::zmq_utils::{handle_error_and_respond, send_document_insertion_response, send_exception_response};
use crate::zmq::server_params::ZmqParams;

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
    running: Arc<AtomicBool>,
    worker_socket: &mut ServerWorker,
    zmq_params: &ZmqParams,
    processing_context: Arc<ProcessingContext>,
    model_clone: Arc<Model>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    db: &Arc<RocksDB>,
    embedding_sender: Arc<Sender<EmbeddingsRequest>>,
    n_embd: usize,
) {

    while running.load(Ordering::SeqCst) {
        let ts = Instant::now();
        let messages = worker_socket.worker
            .recv()
            .await
            .expect("Failed to receive message");

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
        let req_ts = from_epoch_micros(message_header.request_ts);
        let _ = time_millis(&req_ts, "Request receive time");

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
                    processing_context.clone(),
                    model_clone.clone(),
                    split_index,
                    summary_index,
                    db,
                    &mut message_header,
                    &messages.get(1).unwrap().to_vec(),
                    embedding_sender.clone(),
                    n_embd,
                )
                    .await;
                time_millis(&ts, "Document Insertion");
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
    }
    info!("Worker shutting down");
}

pub async fn process_document_insertion_request(
    worker_socket: &mut RepSocket,
    processing_context: Arc<ProcessingContext>,
    model: Arc<Model>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    db: &Arc<RocksDB>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
    embd_req_sender: Arc<Sender<EmbeddingsRequest>>,
    n_embd: usize,
) {
    let request: DocumentInsertionRequest;
    match DocumentInsertionRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentInsertionRequest: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    }

    let docs_input: Vec<String> = request.input;
    let doc_urls: Vec<String> = request.doc_urls.unwrap_or(Vec::new());

    let document_dto = process_document(
        processing_context,
        embd_req_sender,
        doc_urls[0].to_string(),
        docs_input[0].clone().into_bytes().to_vec()
    ).await.unwrap();

    let (document, splits, summaries, embeddings, embedding_users) =
        process_document_dto(document_dto.clone(), request.user.unwrap_or(Uuid::new_v4()).clone()).await.unwrap();

    save_models_to_db(db, &document, &splits, &summaries, &embeddings, &embedding_users, model)
        .await
        .expect("Failed to write models to DB");

    send_document_insertion_response(
        worker_socket,
        message_header,
        &document_dto,
        request.verbose.unwrap_or(false),
    )
        .await;

    info!("Document processed and response sent")
}

async fn process_document_dto(
    res: DocumentDto,
    user_id: Uuid,
) -> Result<(
    Document,
    Vec<Split>,
    Vec<Summary>,
    Vec<Embedding>,
    Vec<EmbeddingUser>,
)> {
    let (document, splits, summaries, embeddings, embedding_users) = task::spawn_blocking(move || {
        let document = res.to_model();

        let splits: Vec<_> = res.splits.par_iter().map(|split| split.to_model()).collect();
        let summaries: Vec<_> = res.summaries.par_iter().map(|summary| summary.to_model()).collect();

        let embeddings: Vec<Embedding> = res.splits.par_iter()
            .filter_map(|split_dto| split_dto.to_embedding_model())
            .chain(
                res.summaries.par_iter()
                    .filter_map(|summary_dto| summary_dto.to_embedding_model())
            )
            .collect();

        let embedding_users = embeddings.iter().map(|embedding| {
            let embed_id = embedding.embedding_id;
            EmbeddingUser::new(embed_id, user_id)
        }).collect::<Vec<_>>();

        (document, splits, summaries, embeddings, embedding_users)
    }).await?;

    Ok((document, splits, summaries, embeddings, embedding_users))
}

