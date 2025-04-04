use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::{RerankRequest, RerankResponse, Serde};
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::rerankings::get_rerankings;
use std::sync::Arc;
use tracing::error;
use zmq::Socket;

pub fn process_rerank(
    socket: &Socket,
    processing_context: Arc<ProcessingContext>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
    identity: &Vec<u8>,
) {
    let request: RerankRequest;
    match RerankRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking RerankRequest: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(socket, &error_message, message_header, identity);
            return;
        }
    }

    let id = uuid::Uuid::new_v4();
    let req_id = processing_context.hasher.hash(&id.to_string());

    let response = match get_rerankings(
        processing_context.zmq_context.clone(),
        &processing_context.reranking_endpoint.clone().unwrap(),
        request.query,
        request.texts.to_vec(),
        req_id,
    ) {
        Ok(mut ranks) => {
            if request.return_text {
                for rank in &mut ranks {
                    rank.text = Some(request.texts[rank.index].clone());
                }
            }
            RerankResponse::new(None, ranks, None)
        }
        Err(e) => {
            let error_message = format!("Error processing rerankings: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(socket, &error_message, message_header, identity);
            return;
        }
    };

    send_success_response(socket, response, message_header, identity);
}
