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
use tracing::error;
use zmq::Socket;

pub fn process_document_insertion_request(
    worker_socket: &Socket,
    db: &Arc<RocksDB>,
    processing_context: Arc<ProcessingContext>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
    identity: &Vec<u8>,
) {
    let request: DocumentInsertionRequest;
    match DocumentInsertionRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentInsertionRequest: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    }

    let document_dto = match process_document(
        processing_context,
        request.doc_url.to_string(),
        request.input.clone().into_bytes().to_vec(),
    ) {
        Ok(dto) => dto,
        Err(e) => {
            let error_message = format!("Error processing document: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    };

    let user_id = request.workspace_id;

    if let Err(e) = save_doc(db, &document_dto, user_id) {
        let error_message = format!("Error saving document to DB: {:?}", e);
        error!("{}", &error_message);
        send_exception_response(worker_socket, &error_message, message_header, identity);
        return;
    }

    if let Err(e) = add_to_indices(split_index.clone(), summary_index.clone(), &document_dto) {
        let error_message = format!("Error adding document to indices: {:?}", e);
        error!("{}", &error_message);
        send_exception_response(worker_socket, &error_message, message_header, identity);
        return;
    }

    send_document_insertion_response(
        worker_socket,
        message_header,
        &document_dto,
        request.verbose.unwrap_or(false),
        identity,
    );
}

// Responses
pub fn send_document_insertion_response(
    socket: &Socket,
    message_header: &mut ZmqMessageHeader,
    document_dto: &DocumentDto,
    verbose: bool,
    identity: &Vec<u8>,
) {
    let response = DocumentInsertionResponse {
        status: DocumentInsertionStatus::Success,
        document: if verbose {
            Some(document_dto.clone())
        } else {
            None
        },
    };
    zmq_utils::send_success_response(socket, response, message_header, identity);
}
