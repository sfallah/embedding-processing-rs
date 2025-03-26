use embedding_common::prelude::Serde;
use serde::{Deserialize, Serialize};

/// represents the response data for a health check of the server.
#[derive(Serialize, Deserialize, Debug)]
pub struct HealthCheckResponse {
    /// the status of the health check, typically indicating whether the server is up and running.
    pub status: String,
    /// the hostname of the server.
    pub host: String,
    /// the count of active workers.
    pub worker_count: usize,
}

impl Serde for HealthCheckResponse {}
