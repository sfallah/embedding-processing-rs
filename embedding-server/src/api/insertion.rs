use std::rc::Rc;
use crate::schema::document::{DocumentInsertionRequest, DocumentInsertionResponse};
use crate::schema::document_status::DocumentInsertionStatus;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils;
use crate::utils::zmq_utils::send_exception_response;
use embedding_common::prelude::*;
use embedding_database::prelude::{save_doc, RocksDB};
use embedding_index::add_to_indices;
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::documents::process_document;
use std::sync::Arc;
use tracing::{debug, error, info};
use zeromq::RepSocket;

pub async fn process_document_insertion_request(
    worker_socket: &mut RepSocket,
    db: &Arc<RocksDB>,
    processing_context: Rc<ProcessingContext>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
) {
    let request: DocumentInsertionRequest;
    match DocumentInsertionRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentInsertionRequest: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    }

    debug!("insertion request: {:?}", request);

    let document_dto = tokio::task::spawn_blocking(move || {

        process_document(
        processing_context,
        request.doc_url.to_string(),
        request.input.clone().into_bytes().to_vec(),
    ).expect("Failed to process document")
    })
    .await
    .unwrap();

    let user_id = request.user;
    debug!("User ID: {}", user_id);

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

// Responses
pub async fn send_document_insertion_response(
    socket: &mut RepSocket,
    message_header: &mut ZmqMessageHeader,
    document_dto: &DocumentDto,
    verbose: bool,
) {
    let response = DocumentInsertionResponse {
        status: DocumentInsertionStatus::Success,
        document: if verbose {
            Some(document_dto.clone())
        } else {
            None
        },
    };
    zmq_utils::send_success_response(socket, response, message_header).await;
}
