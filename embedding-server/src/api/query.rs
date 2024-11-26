use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use async_channel::Sender;
use rayon::iter::IntoParallelRefMutIterator;
use zeromq::RepSocket;
use embedding_database::prelude::{get_all_summaries, get_document, get_embedding, get_split, get_summaries_of_document, get_summaries_of_split, RocksDB};
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::services::embeddings::EmbeddingsRequest;
use crate::dtos::document::{DocumentQueryRequest, DocumentQueryResponse};
use crate::dtos::zmq_message_header::ZmqMessageHeader;
use embedding_common::prelude::*;
use embedding_processing::processing::query::process_query;
use crate::dtos::embedding::EmbeddingUsageDto;
use crate::dtos::search_mode::SearchModeType;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};

// Requests calling embedding model
pub async fn process_document_query_request(
    worker_socket: &mut RepSocket,
    db: &Arc<RocksDB>,
    processing_context: Arc<ProcessingContext>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    embd_req_sender: Arc<Sender<EmbeddingsRequest>>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
) {
    let request: DocumentQueryRequest;
    match DocumentQueryRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentQueryRequest: {:?}", e);
            eprintln!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header).await;
            return;
        }
    }

    let search_mode: SearchModeType = request
        .search_mode
        .unwrap_or(SearchModeType::SplitAndSummary);

    let query_embeddings = process_query(processing_context.clone(), embd_req_sender.clone(), request.input.clone()).await.unwrap();
    let query_embeddings = &query_embeddings[0];


    // 1. Search for the query in the indexes
    // Search for summary indexes in summary index
    let mut summaries: Vec<Summary> = Vec::new();
    if search_mode == SearchModeType::SummaryOnly || search_mode == SearchModeType::SplitAndSummary
    {
        let (summary_ids,summary_distances) = summary_index.query_filter(
            db,
            &request.user_ids,
            &query_embeddings,
            request.top_k.unwrap_or(5) as usize,
        ).await.unwrap();


        summaries = get_all_summaries(db, &summary_ids).await.unwrap();
    }

    // Search for split indexes in hnswlib index
    let mut splits: Vec<Split> = Vec::new();
    let mut split_dists: Vec<f32> = Vec::new();
    let mut split_ids: Vec<u64> = Vec::new();
    if search_mode == SearchModeType::SplitOnly || search_mode == SearchModeType::SplitAndSummary {

        let res = split_index.query_filter(
            db,
            &request.user_ids,
            &query_embeddings,
            request.top_k.unwrap_or(5) as usize,
        ).await.unwrap();

        split_ids = res.0;
        split_dists = res.1;


        // Load splits from rocks db
        for split_id in split_ids.iter() {
            let split = match get_split(db, split_id).await {
                Ok(Some(split)) => split,
                Ok(None) => continue,
                Err(e) => {
                    let error_message = format!("Error getting split: {:?}", e);
                    eprintln!("{}", &error_message);
                    send_exception_response(worker_socket, &error_message, message_header).await;
                    return;
                }
            };
            splits.push(split);
        }
    }

    // 2. Load complementary splits of summaries not found in the split index query

    let mut splits_complement: Vec<Split> = Vec::new();

    // Add splits of summaries if id are not in the retrieved split_ids
    for summary in summaries.clone() {
        let split_id = summary.split_id;
        let split_info = match get_split(db, &split_id).await {
            Ok(Some(split_info)) => split_info,
            Ok(None) => continue,
            Err(e) => {
                let error_message = format!("Error getting split: {:?}", e);
                eprintln!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header).await;
                return;
            }
        };
        if !split_ids.contains(&split_info.split_id) {
            splits_complement.push(split_info);
        }
    }
    let _split_complement_ids = splits_complement
        .iter()
        .map(|x| x.split_id)
        .collect::<Vec<u64>>();

    // Add complementary splits to splits at the beginning
    splits.extend(splits_complement.clone());
    split_dists.extend(splits_complement.iter().map(|_| 0.0));


    // Combine the results from the split and summary indexes TODO
    let mut docs: Vec<DocumentDto> = Vec::with_capacity(splits.len());
    let mut doc_map: HashMap<u64, usize> = HashMap::new();
    for chunk in splits {
        let split_summaries: Vec<Summary> = match get_summaries_of_split(db, &chunk.split_id).await {
            Ok(summaries) => summaries.unwrap(),
            Err(e) => {
                let error_message = format!("Error getting summaries of split: {:?}", e);
                eprintln!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header).await;
                return;
            }
        };
        let document = match get_document(db, &chunk.doc_id).await {
            Ok(info) => info.unwrap(),
            Err(e) => {
                let error_message = format!("Error getting document: {:?}", e);
                eprintln!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header).await;
                return;
            }
        };

        let chunk_dto = chunk.to_dto(split_summaries.into_iter().map(|x| x.to_dto()).collect());

        if let Some(&index) = doc_map.get(&document.document_id) {
            docs[index].splits.push(chunk_dto);
        } else {
            let summaries = match get_summaries_of_document(db, &document.document_id).await {
                Ok(summaries) => summaries.unwrap(),
                Err(e) => {
                    let error_message = format!("Error getting summaries of document: {:?}", e);
                    eprintln!("{}", &error_message);
                    send_exception_response(worker_socket, &error_message, message_header).await;
                    return;
                }
            };

            let mut new_doc_query = DocumentDto::new(document.document_id, &document.document_url, Vec::new(), Vec::new());
            new_doc_query.splits.push(chunk_dto);
            new_doc_query.summaries = summaries
                .into_iter()
                .map(|summary| summary.to_dto())
                .collect();

            docs.push(new_doc_query);
            doc_map.insert(document.document_id, docs.len() - 1);
        }
    }
    send_document_query_response(
        worker_socket,
        &request.model,
        message_header,
        &docs,
        &query_embeddings,
        0,
        0,
        request.verbose.unwrap_or(false),
    )
        .await;

}

async fn send_document_query_response(
    socket: &mut RepSocket,
    model: &str,
    message_header: &mut ZmqMessageHeader,
    documents: &Vec<DocumentDto>,
    query_embeddings: &Vec<f32>,
    tokens_query: usize,
    tokens_answer_and_query: usize,
    verbose: bool,
) {
    let response = DocumentQueryResponse {
        documents: documents.clone(),
        model: Some(model.to_string()),
        query_embeddings: if verbose {
            Some(query_embeddings.clone())
        } else {
            None
        },
        usage: Some(EmbeddingUsageDto {
            prompt_tokens: tokens_query as i32,
            total_tokens: tokens_answer_and_query as i32,
        }),
    };
    send_success_response(socket, response, message_header).await
}