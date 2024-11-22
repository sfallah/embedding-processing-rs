use std::collections::VecDeque;
use tracing::error;
use zeromq::{RepSocket, SocketSend, ZmqMessage};
use embedding_common::dtos::document_dto::DocumentDto;
use embedding_common::dtos::embedding_dto::EmbeddingDto;
use embedding_common::Serde;
use crate::dtos::document::DocumentInsertionResponse;
use crate::dtos::document_status::DocumentInsertionStatus;
use crate::dtos::embedding::EmbeddingUsageDto;
use crate::dtos::zmq_message_header::{ZmqMessageHeader, ZmqMessageStatus, ZmqMessageType};


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
            embedding_id: 0, // Placeholder
            embedding: vec![], // Placeholder
            model_id: 0, // Placeholder
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
        embeddings: if verbose { Some(embeddings_data) } else { None },
    };
    send_success_response(socket, response, message_header).await;
}


// Handling responses, successes, errors and exceptions
pub async fn handle_error_and_respond(
    worker_socket: &mut RepSocket,
    error_message: &str,
    message_type: ZmqMessageType,
) {
    let mut error_header = ZmqMessageHeader::new(message_type);
    error_header.set_error(Some(error_message.to_string()), None);
    send_exception_response(worker_socket, error_message, &mut error_header).await;
}

pub async fn send_exception_response(
    socket: &mut RepSocket,
    error_message: &str,
    message_header: &mut ZmqMessageHeader,
) {
    let buf = message_header
        .pack()
        .expect("Failed to pack message header");
    socket
        .send(buf.into())
        .await
        .expect("Failed to send message");

    error!(
        "Error response sent: error = {}, response = {:?}",
        error_message,
        message_header.to_json()
    );
}

pub async fn send_success_response<T: Serde + serde::Serialize>(
    socket: &mut RepSocket,
    response: T,
    message_header: &mut ZmqMessageHeader,
) {
    message_header.set_status(ZmqMessageStatus::Success);
    message_header.set_response_ts();

    let serialized_header = message_header
        .pack()
        .expect("Failed to pack message header");
    let serialized_body = response.pack().expect("Failed to pack response body");
    let mut frames = VecDeque::with_capacity(2);
    frames.push_back(serialized_header.into());
    frames.push_back(serialized_body.into());
    let m = ZmqMessage::try_from(frames).unwrap();
    if let Err(e) = socket.send(m).await {
        error!("Error sending success response: {:?}", e);
    }
}