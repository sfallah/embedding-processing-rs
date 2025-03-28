use crate::schema::document::{DocumentDeletionRequest, DocumentDeletionResponse};
use crate::schema::document_status::DeletionStatus;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_database::prelude::{delete_doc_full, RocksDB};
use embedding_index::hnsw_index::HnswIndex;
use std::sync::Arc;
use tracing::error;
use zmq::Socket;

pub fn process_document_deletion_request(
    worker_socket: &Socket,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    db: &Arc<RocksDB>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
    identity: &Vec<u8>,
) {
    let request: DocumentDeletionRequest = match DocumentDeletionRequest::unpack(&body_message) {
        Ok(req) => req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentDeletionRequest: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    };
    match delete_doc_full(db, request.document_id) {
        Ok(Some(doc)) => {
            if let Err(e) = split_index.delete(&doc.split_ids) {
                let error_message = format!("Error deleting split from index: {:?}", e);
                error!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header, identity);
            }
            if let Err(e) = summary_index.delete(&doc.summary_ids.unwrap_or_default()) {
                let error_message = format!("Error deleting summary from index: {:?}", e);
                error!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header, identity);
            }
            send_document_deletion_response(worker_socket, true, message_header, identity);
        }

        Ok(None) => {
            send_document_deletion_response(worker_socket, false, message_header, identity);
        }
        Err(e) => {
            let error_message = format!("Error deleting document: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
        }
    }
}

fn send_document_deletion_response(
    socket: &Socket,
    removed: bool,
    message_header: &mut ZmqMessageHeader,
    identity: &Vec<u8>,
) {
    if removed {
        send_success_response(
            socket,
            DocumentDeletionResponse {
                status: DeletionStatus::Success,
            },
            message_header,
            identity,
        )
    } else {
        send_success_response(
            socket,
            DocumentDeletionResponse {
                status: DeletionStatus::NotFound,
            },
            message_header,
            identity,
        )
    }
}
