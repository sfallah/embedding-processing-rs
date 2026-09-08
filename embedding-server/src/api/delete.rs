use crate::schema::document::{DocumentDeletionRequest, DocumentDeletionResponse};
use crate::schema::document_status::DeletionStatus;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_store::prelude::ShardPool;
use std::sync::Arc;
use tracing::error;
use zmq::Socket;

pub fn process_document_deletion_request(
    worker_socket: &Socket,
    pool: &Arc<ShardPool>,
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

    // As for retrieval: without the workspace there is no shard to look in (gap G4).
    let workspace_id = match request.user {
        Some(workspace_id) => workspace_id,
        None => {
            let error_message = "DocumentDeletionRequest must have a workspace id";
            error!("{}", error_message);
            send_exception_response(worker_socket, error_message, message_header, identity);
            return;
        }
    };

    let shard = match pool.get_existing(workspace_id) {
        Ok(None) => {
            send_document_deletion_response(worker_socket, false, message_header, identity);
            return;
        }
        Ok(Some(shard)) => shard,
        Err(e) => {
            let error_message = format!(
                "Error opening the shard for workspace {}: {:?}",
                workspace_id, e
            );
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    };

    // The log and both indexes go together, so there is one outcome to report rather than the
    // error-then-success pair the two index deletes used to be able to send (gap G8).
    match shard.delete(request.document_id) {
        Ok(removed) => {
            send_document_deletion_response(worker_socket, removed, message_header, identity)
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
