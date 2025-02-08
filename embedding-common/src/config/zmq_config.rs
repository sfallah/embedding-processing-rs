use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct ZmqConfig {
    pub zmq_host: String,
    pub zmq_frontend_port: usize,
    pub zmq_backend_port: usize,
    pub zmq_num_workers: usize,
}
