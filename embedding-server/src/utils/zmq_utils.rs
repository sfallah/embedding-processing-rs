use std::collections::VecDeque;
use crate::schema::zmq_message_header::{ZmqMessageHeader, ZmqMessageStatus, ZmqMessageType};
use embedding_common::prelude::Serde;
use tracing::error;
use zeromq::{RepSocket, SocketSend, ZmqMessage};

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
    message_header.set_error(Some(error_message.to_string()), None);
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
