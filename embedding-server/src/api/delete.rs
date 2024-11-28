use crate::schema::document::{DocumentDeletionRequest, DocumentDeletionResponse};
use crate::schema::document_status::DeletionStatus;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_database::prelude::{delete_doc_full, get_document, RocksDB};
use embedding_index::hnsw_index::HnswIndex;
use std::sync::Arc;
use zeromq::RepSocket;

async fn process_document_deletion_request(
    worker_socket: &mut RepSocket,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    db: &Arc<RocksDB>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
) {
    let request: DocumentDeletionRequest = match DocumentDeletionRequest::unpack(&body_message) {
        Ok(req) => req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentDeletionRequest: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    };
    match delete_doc_full(db, request.document_id).await {
        Ok(removed) => {
            send_document_deletion_response(worker_socket, removed, message_header).await;
        }
        Err(e) => {
            let error_message = format!("Error deleting document: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
        }
    }
}

async fn send_document_deletion_response(
    socket: &mut RepSocket,
    removed: bool,
    message_header: &mut ZmqMessageHeader,
) {
    if removed {
        send_success_response(
            socket,
            DocumentDeletionResponse {
                status: DeletionStatus::Success,
            },
            message_header,
        )
        .await
    } else {
        send_success_response(
            socket,
            DocumentDeletionResponse {
                status: DeletionStatus::NotFound,
            },
            message_header,
        )
        .await
    }
}
