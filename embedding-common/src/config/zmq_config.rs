use crate::prelude::Serde;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct ZmqConfig {
    // Model Host
    pub host: String,
    // ROUTER-DEALER Proxy
    // Model request port
    // Receive Embedding and Health-check requests
    pub frontend_port: usize,
    pub backend_port: usize,
    // Number of workers
    pub num_workers: usize,
}

impl Serde for ZmqConfig {}
