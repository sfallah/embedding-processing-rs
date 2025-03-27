use crate::schema::health_check::HealthCheckResponse;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::send_success_response;
use embedding_common::config::ZmqConfig;
use std::sync::Arc;
use zmq::Socket;

pub fn process_health_check(
    socket: &Socket,
    message_header: &mut ZmqMessageHeader,
    zmq_config: Arc<ZmqConfig>,
    identity: &Vec<u8>,
) {
    let response = HealthCheckResponse {
        status: "OK".to_string(),
        host: format!("{}:{}", zmq_config.zmq_host, zmq_config.zmq_frontend_port),
        worker_count: zmq_config.zmq_num_workers,
    };

    send_success_response(socket, response, message_header, identity);
}
