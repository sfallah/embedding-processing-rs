use crate::schema::embedding::EmbeddingUsageDto;
use crate::schema::query::DocumentQueryRequest;
use crate::schema::query::DocumentQueryResponse;
use crate::schema::search_mode::SearchModeType;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use async_channel::Sender;
use embedding_common::prelude::*;
use embedding_database::prelude::{
    get_all_summaries, get_document, get_split, get_summaries_of_document, RocksDB,
};
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::query::process_query;
use embedding_processing::services::embeddings::EmbeddingsRequest;
use indexmap::{IndexMap, IndexSet};
use std::sync::Arc;
use tracing::{debug, info};
use zeromq::RepSocket;

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
    debug!("query request: {:?}", request);

    let search_mode: SearchModeType = request
        .search_mode
        .unwrap_or(SearchModeType::SplitAndSummary);

    let query_embeddings = process_query(
        processing_context.clone(),
        embd_req_sender.clone(),
        request.input.clone(),
    )
    .await
    .unwrap();
    let query_embeddings = &query_embeddings[0];

    // 1. Search for the query in the indexes
    // Search for summary indexes in summary index
    let mut summaries_query_res: IndexMap<u64, f32> = IndexMap::new();
    let mut split_summary_map: IndexMap<u64, Vec<SummaryDto>> = IndexMap::new();
    if search_mode == SearchModeType::SummaryOnly || search_mode == SearchModeType::SplitAndSummary
    {
        summaries_query_res = summary_index
            .query_filter(
                db,
                &request.user_ids,
                &query_embeddings,
                request.top_k.unwrap_or(5) as usize,
            )
            .await
            .unwrap();
        debug!("Summary query results: {:?}", summaries_query_res);

        let summary_ids: Vec<u64> = summaries_query_res.keys().map(|x| *x).collect();

        let summaries = get_all_summaries(db, &summary_ids).await.unwrap();

        for summary in summaries.iter() {
            let summary_distance = summaries_query_res.get(&summary.summary_id).unwrap();
            let summary_dto = summary.to_dto(Some(*summary_distance));
            if split_summary_map.contains_key(&summary.split_id) {
                split_summary_map
                    .get_mut(&summary.split_id)
                    .unwrap()
                    .push(summary_dto);
            } else {
                let mut summary_set = Vec::new();
                summary_set.push(summary_dto);
                split_summary_map.insert(summary.split_id, summary_set);
            }
        }
    }

    // Search for split indexes in hnswlib index
    let mut split_query_res = IndexMap::new();
    if search_mode == SearchModeType::SplitOnly || search_mode == SearchModeType::SplitAndSummary {
        split_query_res = split_index
            .query_filter(
                db,
                &request.user_ids,
                &query_embeddings,
                request.top_k.unwrap_or(5) as usize,
            )
            .await
            .unwrap();
        debug!("Split query results: {:?}", split_query_res);
    }

    let splits_min_distances = split_summary_map.iter().map(|(k, v)| {
        let distances = v
            .iter()
            .map(|x| x.query_distance.unwrap())
            .reduce(|a, b| a.min(b))
            .unwrap();
        (*k, distances)
    });
    let mut splits_min_distances: IndexMap<u64, f32> = IndexMap::from_iter(splits_min_distances);
    splits_min_distances.extend(split_query_res.clone());
    splits_min_distances.sort_by(|_, v1, _, v2| v1.partial_cmp(v2).unwrap());

    let final_split_ids: IndexSet<u64> = splits_min_distances.keys().map(|x| *x).collect();

    // Load splits from rocks db
    let mut doc_split_map: IndexMap<u64, Vec<SplitDto>> = IndexMap::new();
    for split_id in final_split_ids.iter() {
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
        let mut split_dto = if let Some(summary_dtos) = split_summary_map.get(&split.split_id) {
            split.to_dto_full(summary_dtos.clone())
        } else {
            split.to_dto_full(vec![])
        };
        if split_query_res.contains_key(&split.split_id) {
            split_dto.query_distance = Some(*split_query_res.get(&split.split_id).unwrap());
        }

        if doc_split_map.contains_key(&split.doc_id) {
            doc_split_map
                .get_mut(&split.doc_id)
                .unwrap()
                .push(split_dto);
        } else {
            let mut split_set = Vec::new();
            split_set.push(split_dto);
            doc_split_map.insert(split.doc_id, split_set);
        }
    }

    // Load documents from rocks db
    let mut docs: Vec<DocumentDto> = Vec::new();
    for (doc_id, doc_splits) in doc_split_map.into_iter() {
        let document = match get_document(db, &doc_id).await {
            Ok(Some(document)) => document,
            Ok(None) => continue,
            Err(e) => {
                let error_message = format!("Error getting document: {:?}", e);
                eprintln!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header).await;
                return;
            }
        };
        let doc_summaries = get_summaries_of_document(db, &doc_id).await.unwrap();
        let doc_summaries_dto = doc_summaries
            .iter()
            .map(|x| match summaries_query_res.get(&x.summary_id) {
                Some(distance) => x.to_dto(Some(*distance)),
                None => x.to_dto(None),
            })
            .collect();

        let new_doc_query = DocumentDto::new(
            document.document_id,
            &document.document_url,
            doc_splits,
            doc_summaries_dto,
        );
        docs.push(new_doc_query);
    }
    info!("Sending response for document query");

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
