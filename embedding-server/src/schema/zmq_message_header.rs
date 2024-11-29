use chrono::Utc;
use embedding_common::prelude::Serde;
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use std::fmt;
use tracing::error;
use uuid::Uuid;

/// Represents the type of messages that can be processed by the ZMQ server.
#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum ZmqMessageType {
    /// Message type for embedding operations.
    DocumentInsertion = 1,
    /// Message type for query operations.
    DocumentQuery = 2,
    /// Message type for retrieval operations for documents.
    DocumentRetrieval = 3,
    /// Message type for retrieval operations for splits of document.
    DocumentSplitsRetrieval = 4,
    /// Message type for retrieval operations for summaries of document.
    DocumentSummariesRetrieval = 5,
    /// Message type for retrieval operations for summaries of document.
    DocumentDeletion = 6,
    /// Message type for retrieval operations for splits.
    SplitRetrieval = 7,
    /// Message type for retrieval operations for splits.
    SummaryRetrieval = 8,
    /// Message type for health check operations, used to verify the operational status of the server.
    HealthCheck = 9,
    /// Represents an undefined or unrecognized message type, used as a fallback or error state.
    Unknown = 10,
}

/// Converts a `ZmqMessageType` enum to its corresponding string representation.
impl fmt::Display for ZmqMessageType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            ZmqMessageType::DocumentInsertion => "DocumentInsertion",
            ZmqMessageType::DocumentQuery => "DocumentQuery",
            ZmqMessageType::DocumentRetrieval => "DocumentRetrieval",
            ZmqMessageType::DocumentSplitsRetrieval => "DocumentSplitsRetrieval",
            ZmqMessageType::DocumentSummariesRetrieval => "DocumentSummariesRetrieval",
            ZmqMessageType::DocumentDeletion => "DocumentDeletion",
            ZmqMessageType::SplitRetrieval => "SplitRetrieval",
            ZmqMessageType::SummaryRetrieval => "SummaryRetrieval",
            ZmqMessageType::HealthCheck => "HealthCheck",
            ZmqMessageType::Unknown => "UNKNOWN",
        };
        write!(f, "{}", s)
    }
}

/// Defines the possible statuses of a message processed by the ZMQ server, indicating success or failure.
#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum ZmqMessageStatus {
    /// Indicates that the message was processed successfully without any errors.
    Success = 1,
    /// Indicates that an error occurred during the processing of the message.
    Error = 2,
}

/// Converts a `ZmqMessageStatus` enum to its corresponding string representation.
impl fmt::Display for ZmqMessageStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            ZmqMessageStatus::Success => "Success",
            ZmqMessageStatus::Error => "Error",
        };
        write!(f, "{}", s)
    }
}

/// Header information for messages sent and received via ZMQ.
#[derive(Serialize, Deserialize, Debug)]
pub struct ZmqMessageHeader {
    /// Unique identifier for the message.
    pub zmq_message_id: Uuid,
    /// Type of the message.
    pub message_type: ZmqMessageType,
    /// Status of the message, which may include success or error states.
    pub status: Option<ZmqMessageStatus>,
    /// Optional error code associated with the message if an error occurred.
    pub error_code: Option<i32>,
    /// Optional detailed message describing the error if one occurred.
    pub error_message: Option<String>,
    /// Timestamp when the request was made.
    pub request_ts: u64,
    /// Optional timestamp when the response was sent.
    pub response_ts: Option<u64>,
}

impl ZmqMessageHeader {
    /// Constructs a new `ZmqMessageHeader` with the specified message type and current timestamp.
    pub fn new(message_type: ZmqMessageType) -> Self {
        let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or_else(|| {
            error!("Failed to get timestamp!");
            0
        }) as u64;

        ZmqMessageHeader {
            zmq_message_id: Uuid::new_v4(),
            message_type,
            status: None,
            error_code: None,
            error_message: None,
            request_ts: timestamp,
            response_ts: None,
        }
    }

    /// Sets the message type of the header.
    pub fn set_message_type(&mut self, new_type: ZmqMessageType) {
        self.message_type = new_type;
    }

    /// Updates the message ID to a new UUID.
    pub fn set_message_id(&mut self, new_id: Uuid) {
        self.zmq_message_id = new_id;
    }

    /// Sets the status of the message header.
    pub fn set_status(&mut self, new_status: ZmqMessageStatus) {
        self.status = Some(new_status);
    }

    /// Sets an error state for the message header, including an optional message and code.
    pub fn set_error(&mut self, err_message: Option<String>, err_code: Option<i32>) {
        self.set_status(ZmqMessageStatus::Error);
        self.error_message = err_message;
        self.error_code = err_code;
    }

    /// Updates the response timestamp to the current time.
    pub fn set_response_ts(&mut self) {
        let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or_else(|| {
            error!("Failed to obtain a valid timestamp");
            0
        }) as u64;

        self.response_ts = Some(timestamp);
    }
}

impl Serde for ZmqMessageHeader {}
