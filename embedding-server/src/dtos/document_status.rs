use serde_repr::{Deserialize_repr, Serialize_repr};
use std::fmt;

/// Defines the possible statuses of a document processed by the server, indicating success or failure.
#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum DocumentInsertionStatus {
    /// Indicates that the message was processed successfully without any errors.
    Success = 1,
    /// Indicates that an error occurred during the processing of the message.
    Error = 2,
}

/// Converts a `DocumentInsertionStatus` enum to its corresponding string representation.
impl fmt::Display for DocumentInsertionStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            DocumentInsertionStatus::Success => "Success",
            DocumentInsertionStatus::Error => "Error",
        };
        write!(f, "{}", s)
    }
}

/// Defines the possible statuses of a document processed by the server, indicating success or failure.
#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum RetrievalStatus {
    /// Indicates that the message was processed successfully without any errors.
    Success = 1,
    /// Indicates that the related entity is not found.
    NotFound = 2,
    /// Indicates that an error occurred during the processing of the message.
    Error = 3,
}

/// Converts a `RetrievalStatus` enum to its corresponding string representation.
impl fmt::Display for RetrievalStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            RetrievalStatus::Success => "Success",
            RetrievalStatus::NotFound => "NotFound",
            RetrievalStatus::Error => "Error",
        };
        write!(f, "{}", s)
    }
}

/// Defines the possible statuses of a document processed by the server, indicating success or failure.
#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum DeletionStatus {
    /// Indicates that the message was processed successfully without any errors.
    Success = 1,
    /// Indicates that the related entity is not found.
    NotFound = 2,
    /// Indicates that an error occurred during the processing of the message.
    Error = 3,
}

/// Converts a `DeletionStatus` enum to its corresponding string representation.
impl fmt::Display for DeletionStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            DeletionStatus::Success => "Success",
            DeletionStatus::NotFound => "NotFound",
            DeletionStatus::Error => "Error",
        };
        write!(f, "{}", s)
    }
}
