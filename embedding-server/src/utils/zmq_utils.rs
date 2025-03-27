use crate::schema::zmq_message_header::{ZmqMessageHeader, ZmqMessageStatus, ZmqMessageType};
use embedding_common::prelude::Serde;
use tracing::error;
use zmq::Socket;

// Handling responses, successes, errors and exceptions
pub fn handle_error_and_respond(
    worker_socket: &Socket,
    error_message: &str,
    message_type: ZmqMessageType,
    identity: &Vec<u8>,
) {
    let mut error_header = ZmqMessageHeader::new(message_type);
    error_header.set_error(Some(error_message.to_string()), None);
    send_exception_response(worker_socket, error_message, &mut error_header, identity);
}

pub fn send_exception_response(
    socket: &Socket,
    error_message: &str,
    message_header: &mut ZmqMessageHeader,
    identity: &Vec<u8>,
) {
    message_header.set_error(Some(error_message.to_string()), None);
    let buf = message_header
        .pack()
        .expect("Failed to pack message header");

    if let Err(e) = socket.send_multipart(vec![identity, &buf], 0) {
        error!("Failed to send exception response: {:?}", e);
    }

    error!(
        "Error response sent: error = {}, response = {:?}",
        error_message,
        message_header.to_json()
    );
}

pub fn send_success_response<T: Serde + serde::Serialize>(
    socket: &Socket,
    response: T,
    message_header: &mut ZmqMessageHeader,
    identity: &Vec<u8>,
) {
    message_header.set_status(ZmqMessageStatus::Success);
    message_header.set_response_ts();

    let serialized_header = message_header
        .pack()
        .expect("Failed to pack message header");
    let serialized_body = response.pack().expect("Failed to pack response body");

    if let Err(e) = socket.send_multipart(
        vec![identity, &serialized_header, &serialized_body],
        0,
    ) {
        error!("Failed to send response: {:?}", e);
    }
}
