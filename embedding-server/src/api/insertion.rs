use crate::dtos::document::DocumentInsertionRequest;
use crate::dtos::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_document_insertion_response, send_exception_response};
use async_channel::Sender;
use embedding_common::prelude::*;
use embedding_index::add_to_indices;
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::documents::process_document;
use embedding_processing::services::embeddings::EmbeddingsRequest;
use std::sync::Arc;
use tokio::task;
use tracing::info;
use uuid::Uuid;
use zeromq::RepSocket;
use embedding_database::prelude::{save_doc, RocksDB};

pub async fn process_document_insertion_request(
    worker_socket: &mut RepSocket,
    db: &Arc<RocksDB>,
    processing_context: Arc<ProcessingContext>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    embd_req_sender: Arc<Sender<EmbeddingsRequest>>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
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
        docs_input[0].clone().into_bytes().to_vec(),
    )
    .await
    .unwrap();

    let user_id = request.user.unwrap_or(Uuid::new_v4());

    save_doc(db, &document_dto, user_id)
        .await
        .expect("Failed to write models to DB");

    add_to_indices(split_index.clone(), summary_index.clone(), &document_dto)
        .await
        .expect("Failed to add to indices");

    send_document_insertion_response(
        worker_socket,
        message_header,
        &document_dto,
        request.verbose.unwrap_or(false),
    )
    .await;

    info!("Document processed and response sent")
}
