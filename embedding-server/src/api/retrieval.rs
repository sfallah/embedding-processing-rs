use crate::schema::document::{DocumentRetrievalRequest, DocumentRetrievalResponse};
use crate::schema::document_status::RetrievalStatus;
use crate::schema::split::{SplitRetrievalRequest, SplitRetrievalResponse};
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_database::prelude::*;
use std::sync::Arc;
use zeromq::RepSocket;
use crate::schema::summary::{SummaryRetrievalRequest, SummaryRetrievalResponse};

async fn process_document_retrieval_request(
    worker_socket: &mut RepSocket,
    hasher: &DeterministicAHasher,
    db: &Arc<RocksDB>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
) {
    // unpack into DocumentRetrievalRequest
    let request: DocumentRetrievalRequest;
    match DocumentRetrievalRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentRetrievalRequest: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    }

    // assert that the request has a valid document_id OR document_url
    if request.document_id.is_none() && request.document_url.is_none() {
        let error_message =
            "DocumentRetrievalRequest must have either a document_id or document_url";
        eprintln!("{}", error_message);
        send_exception_response(worker_socket, error_message, message_header).await;
        return;
    }

    // get the document_id or document_url from the request
    let mut document_id = request.document_id.unwrap_or(0);

    // if there is no document_id, get the document_url and hash it to get the document_id
    if document_id == 0 {
        let document_url = request.document_url.unwrap_or("".to_string());
        document_id = hasher.hash(&document_url);
    }

    match get_full_doc(db, document_id).await {
        Ok(doc_dto) => {
            send_document_retrieval_response(worker_socket, &doc_dto, message_header).await
        }
        Err(e) => {
            let error_message = format!("Error retrieving document: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    }
}

async fn send_document_retrieval_response(
    socket: &mut RepSocket,
    document: &Option<DocumentDto>,
    message_header: &mut ZmqMessageHeader,
) {
    let response = match document {
        Some(doc) => DocumentRetrievalResponse {
            status: RetrievalStatus::Success,
            document: Some(doc.clone()),
            embeddings: None,
        },
        None => DocumentRetrievalResponse {
            status: RetrievalStatus::Error,
            document: None,
            embeddings: None,
        },
    };

    send_success_response(socket, response, message_header).await
}

async fn process_split_retrieval_request(
    worker_socket: &mut RepSocket,
    db: &Arc<RocksDB>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
) {
    let request: SplitRetrievalRequest;
    match SplitRetrievalRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking SplitRetrievalRequest: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    }
    match get_split_full(db, request.split_id).await {
        Ok(split) => send_split_retrieval_response(worker_socket, &split, message_header).await,
        Err(e) => {
            let error_message = format!("Error retrieving split: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    }
}

async fn send_split_retrieval_response(
    socket: &mut RepSocket,
    split: &Option<SplitDto>,
    message_header: &mut ZmqMessageHeader,
) {
    let response = match split {
        Some(doc) => SplitRetrievalResponse {
            status: RetrievalStatus::Success,
            split: Some(doc.clone()),
            embeddings: None,
        },
        None => SplitRetrievalResponse {
            status: RetrievalStatus::Error,
            split: None,
            embeddings: None,
        },
    };

    send_success_response(socket, response, message_header).await
}

async fn process_summary_retrieval_request(
    worker_socket: &mut RepSocket,
    db: &Arc<RocksDB>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
) {
    let request: SummaryRetrievalRequest = match SummaryRetrievalRequest::unpack(&body_message) {
        Ok(req) => req,
        Err(e) => {
            let error_message = format!("Error unpacking SummaryRetrievalRequest: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    };

    match get_summary_full(db, &request.summary_id).await {
        Ok(summary) => send_summary_retrieval_response(worker_socket, &summary, message_header).await,
        Err(e) => {
            let error_message = format!("Error occurred during summary retrieval: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    }
}

async fn send_summary_retrieval_response(
    socket: &mut RepSocket,
    summary: &Option<SummaryDto>,
    message_header: &mut ZmqMessageHeader,
) {
    let response = match summary {
        Some(summary) => SummaryRetrievalResponse {
            status: RetrievalStatus::Success,
            summary: Some(summary.clone()),
            embeddings: None,
        },
        None => SummaryRetrievalResponse {
            status: RetrievalStatus::NotFound,
            summary: None,
            embeddings: None,
        },
    };

    send_success_response(socket, response, message_header).await
}
