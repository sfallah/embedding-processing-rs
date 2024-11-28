use crate::schema::document_status::DocumentInsertionStatus;
use crate::schema::embedding::EmbeddingUsageDto;
use crate::schema::insertion::DocumentInsertionRequest;
use crate::schema::insertion::DocumentInsertionResponse;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils;
use crate::utils::zmq_utils::send_exception_response;
use async_channel::Sender;
use embedding_common::prelude::*;
use embedding_database::prelude::{save_doc, RocksDB};
use embedding_index::add_to_indices;
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::documents::process_document;
use embedding_processing::services::embeddings::EmbeddingsRequest;
use std::sync::Arc;
use tokio::task;
use tracing::{debug, info};
use uuid::Uuid;
use zeromq::RepSocket;

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

    debug!("insertion request: {:?}", request);

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
    document_dtos: &DocumentDto,
    verbose: bool,
) {
    let mut embeddings_data = Vec::new();

    for (i, _) in document_dtos.splits.iter().enumerate() {
        let data = EmbeddingDto {
            embedding_id: 0,   // Placeholder
            embedding: vec![], // Placeholder
            model_id: 0,       // Placeholder
        };
        embeddings_data.push(data);
    }

    let total_tokens = 0;

    let response = DocumentInsertionResponse {
        status: DocumentInsertionStatus::Success,
        documents: if verbose {
            Some(vec![document_dtos.clone()])
        } else {
            None
        },
        usage: Some(EmbeddingUsageDto {
            prompt_tokens: total_tokens,
            total_tokens,
        }),
        //embeddings: if verbose { Some(embeddings_data) } else { None },
    };
    zmq_utils::send_success_response(socket, response, message_header).await;
}
