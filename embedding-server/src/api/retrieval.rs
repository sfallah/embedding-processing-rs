use crate::schema::document::{DocumentRetrievalRequest, DocumentRetrievalResponse};
use crate::schema::document_status::RetrievalStatus;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_database::prelude::*;
use std::sync::Arc;
use tracing::error;

pub fn process_document_retrieval_request(
    worker_socket: Arc<zmq::Socket>,
    hasher: Arc<DeterministicAHasher>,
    db: &Arc<RocksDB>,
    message_header: &mut ZmqMessageHeader,
    identity: &Vec<u8>,
    body_message: &Vec<u8>,
) {
    // unpack into DocumentRetrievalRequest
    let request: DocumentRetrievalRequest;
    match DocumentRetrievalRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentRetrievalRequest: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, identity, &error_message, message_header);
            return;
        }
    }

    // assert that the request has a valid document_id OR document_url
    if request.document_id.is_none() && request.document_url.is_none() {
        let error_message =
            "DocumentRetrievalRequest must have either a document_id or document_url";
        error!("{}", error_message);
        send_exception_response(
            worker_socket.clone(),
            identity,
            error_message,
            message_header,
        );
        return;
    }

    // get the document_id or document_url from the request
    let mut document_id = request.document_id.unwrap_or(0);

    // if there is no document_id, get the document_url and hash it to get the document_id
    if document_id == 0 {
        let document_url = request.document_url.unwrap_or("".to_string());
        document_id = hasher.hash(&document_url);
    }

    match get_full_doc(db, document_id, request.verbose.unwrap_or(false), None) {
        Ok(doc_dto) => {
            send_document_retrieval_response(worker_socket, &doc_dto, identity, message_header)
        }
        Err(e) => {
            let error_message = format!("Error retrieving document: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, identity, &error_message, message_header);
            return;
        }
    }
}
fn send_document_retrieval_response(
    socket: Arc<zmq::Socket>,
    document: &Option<DocumentDto>,
    identity: &Vec<u8>,
    message_header: &mut ZmqMessageHeader,
) {
    let response = match document {
        Some(doc) => DocumentRetrievalResponse {
            status: RetrievalStatus::Success,
            document: Some(doc.clone()),
        },
        None => DocumentRetrievalResponse {
            status: RetrievalStatus::Error,
            document: None,
        },
    };

    send_success_response(socket.clone(), identity, response, message_header)
}
