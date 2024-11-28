use crate::schema::document::{DocumentDeletionRequest, DocumentDeletionResponse};
use crate::schema::document_status::DeletionStatus;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_database::prelude::{delete_doc_full, RocksDB};
use embedding_index::hnsw_index::HnswIndex;
use std::sync::Arc;
use tracing::error;
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
        Ok(Some(doc)) => {
            for split_id in doc.split_ids {
                if let Err(e) = split_index.delete(split_id).await {
                    let error_message = format!("Error deleting split from index: {:?}", e);
                    error!("{}", &error_message);
                }
            }
            for summary_id in doc.summary_ids.unwrap_or_default() {
                if let Err(e) = summary_index.delete(summary_id).await {
                    let error_message = format!("Error deleting summary from index: {:?}", e);
                    error!("{}", &error_message);
                }
            }
            send_document_deletion_response(worker_socket, true, message_header).await;
        }

        Ok(None) => {
            send_document_deletion_response(worker_socket, false, message_header).await;
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
