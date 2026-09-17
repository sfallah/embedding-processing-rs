use crate::schema::document::{DocumentRetrievalRequest, DocumentRetrievalResponse};
use crate::schema::document_status::RetrievalStatus;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_store::prelude::ShardPool;
use std::sync::Arc;
use tracing::error;
use zmq::Socket;

pub fn process_document_retrieval_request(
    worker_socket: &Socket,
    hasher: Arc<DeterministicAHasher>,
    pool: &Arc<ShardPool>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
    identity: &Vec<u8>,
) {
    // unpack into DocumentRetrievalRequest
    let request: DocumentRetrievalRequest;
    match DocumentRetrievalRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentRetrievalRequest: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    }

    // A document lives in exactly one workspace's shard and there is no index from document to
    // workspace, so the request has to say which one.
    let workspace_id = match request.user {
        Some(workspace_id) => workspace_id,
        None => {
            let error_message = "DocumentRetrievalRequest must have a workspace id";
            error!("{}", error_message);
            send_exception_response(worker_socket, error_message, message_header, identity);
            return;
        }
    };

    // assert that the request has a valid document_id OR document_url
    if request.document_id.is_none() && request.document_url.is_none() {
        let error_message =
            "DocumentRetrievalRequest must have either a document_id or document_url";
        error!("{}", error_message);
        send_exception_response(worker_socket, error_message, message_header, identity);
        return;
    }

    // get the document_id or document_url from the request
    let mut document_id = request.document_id.unwrap_or(0);

    // if there is no document_id, get the document_url and hash it to get the document_id
    if document_id == 0 {
        //FIXME: send error response if document_url is None
        let document_url = request.document_url.unwrap_or("".to_string());
        document_id = hasher.hash(&document_url);
    }

    let shard = match pool.get_existing(workspace_id) {
        // A workspace nothing was ever written to holds no documents; that is a miss, not an
        // error, and asking must not create the shard.
        Ok(None) => {
            send_document_retrieval_response(worker_socket, &None, message_header, identity);
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

    match shard.get_doc(document_id, request.verbose.unwrap_or(false)) {
        Ok(doc_dto) => {
            send_document_retrieval_response(worker_socket, &doc_dto, message_header, identity)
        }
        Err(e) => {
            let error_message = format!("Error retrieving document: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
        }
    }
}

fn send_document_retrieval_response(
    socket: &Socket,
    document: &Option<DocumentDto>,
    message_header: &mut ZmqMessageHeader,
    identity: &Vec<u8>,
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

    send_success_response(socket, response, message_header, identity)
}
