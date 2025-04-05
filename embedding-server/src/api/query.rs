use crate::schema::document::DocumentQueryRequest;
use crate::schema::document::DocumentQueryResponse;
use crate::schema::search_mode::SearchModeType;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_database::prelude::{get_doc_splits_map, get_full_doc, get_summaries_full, RocksDB};
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::query::process_query;
use embedding_processing::processing::rerankings::get_rerankings;
use indexmap::IndexMap;
use std::sync::Arc;
use tracing::{debug, error, info};
use zmq::Socket;

// Requests calling embedding model
pub fn process_document_query_request(
    worker_socket: &Socket,
    db: &Arc<RocksDB>,
    processing_context: Arc<ProcessingContext>,
    split_index: &Arc<HnswIndex>,
    summary_index: &Arc<HnswIndex>,
    message_header: &mut ZmqMessageHeader,
    body_message: &Vec<u8>,
    identity: &Vec<u8>,
) {
    let request: DocumentQueryRequest;
    match DocumentQueryRequest::unpack(&body_message) {
        Ok(req) => request = req,
        Err(e) => {
            let error_message = format!("Error unpacking DocumentQueryRequest: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    }
    debug!("query request: {:?}", request);

    let with_embeddings = request.verbose.unwrap_or(false);

    let search_mode: SearchModeType = request
        .search_mode
        .unwrap_or(SearchModeType::SplitAndSummary);

    let query_embeddings = match process_query(processing_context.clone(), request.input.clone()) {
        Ok(embeddings) => embeddings,
        Err(e) => {
            let error_message = format!("Error processing query: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    };
    // 1. Search for the query in the indexes
    // Search for summary indexes in summary index
    let top_k = request.top_k.unwrap_or(5) as usize;
    info!("Searching for query in indexes, top_k: {}", top_k);
    let mut split_summary_map: IndexMap<u64, Vec<SummaryDto>> = IndexMap::new();
    if search_mode == SearchModeType::SummaryOnly || search_mode == SearchModeType::SplitAndSummary
    {
        let summaries_query_res = match summary_index.query_filter(
            db,
            &request.workspace_ids,
            &request.doc_ids,
            &query_embeddings,
            top_k,
        ) {
            Ok(res) => res,
            Err(e) => {
                let error_message = format!("Failed to query summaries: {:?}", e);
                error!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header, identity);
                return;
            }
        };
        debug!("Summary query results: {:?}", summaries_query_res);

        let summary_ids: Vec<u64> = summaries_query_res.keys().map(|x| *x).collect();
        let all_summaries = match get_summaries_full(
            db,
            summary_ids.as_slice(),
            Some(&summaries_query_res),
            with_embeddings,
        ) {
            Ok(summaries) => summaries,
            Err(e) => {
                let error_message = format!("Failed to get summaries: {:?}", e);
                error!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header, identity);
                return;
            }
        };

        let all_summaries = match request.rerank {
            Some(rerank) => {
                if rerank && !all_summaries.is_empty() {
                    match rerank_summaries(&processing_context, &request, &all_summaries) {
                        Ok(ranked_summaries) => ranked_summaries,
                        Err(e) => {
                            error!("{}", &e);
                            send_exception_response(
                                worker_socket,
                                &e.to_string(),
                                message_header,
                                identity,
                            );
                            return;
                        }
                    }
                } else {
                    all_summaries
                }
            }
            None => all_summaries,
        };

        for summary_dto in all_summaries {
            let split_id = summary_dto.split_id;
            if split_summary_map.contains_key(&split_id) {
                split_summary_map
                    .get_mut(&split_id)
                    .unwrap()
                    .push(summary_dto);
            } else {
                split_summary_map.insert(split_id, vec![summary_dto]);
            }
        }
    }

    // Search for split indexes in hnswlib index
    let mut split_query_res = IndexMap::new();
    if search_mode == SearchModeType::SplitOnly || search_mode == SearchModeType::SplitAndSummary {
        split_query_res = match split_index.query_filter(
            db,
            &request.workspace_ids,
            &request.doc_ids,
            &query_embeddings,
            top_k,
        ) {
            Ok(res) => res,
            Err(e) => {
                let error_message = format!("Failed to query splits: {:?}", e);
                error!("{}", &error_message);
                send_exception_response(worker_socket, &error_message, message_header, identity);
                return;
            }
        };
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

    let final_split_ids: Vec<_> = splits_min_distances.keys().map(|x| *x).collect();

    // Load splits from rocks db
    let doc_split_map: IndexMap<u64, Vec<SplitDto>> = match get_doc_splits_map(
        db,
        final_split_ids.as_slice(),
        Some(&split_query_res),
        Some(&split_summary_map),
        with_embeddings,
    ) {
        Ok(splits) => splits,
        Err(e) => {
            let error_message = format!("Failed to get splits: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    };

    // Load documents from rocks db
    let mut docs: Vec<DocumentDto> = Vec::new();
    for (doc_id, doc_splits) in doc_split_map.into_iter() {
        let doc_dto = get_full_doc(db, doc_id, with_embeddings, Some(doc_splits));
        match doc_dto {
            Ok(Some(doc)) => {
                let mut doc_clone = doc.clone();
                match doc.summaries {
                    Some(summaries) => {
                        let doc_summaries = match request.rerank {
                            Some(rerank) => {
                                if rerank && !summaries.is_empty() {
                                    match rerank_summaries(
                                        &processing_context,
                                        &request,
                                        &summaries,
                                    ) {
                                        Ok(ranked_summaries) => ranked_summaries,
                                        Err(e) => {
                                            error!("{}", &e);
                                            send_exception_response(
                                                worker_socket,
                                                &e.to_string(),
                                                message_header,
                                                identity,
                                            );
                                            return;
                                        }
                                    }
                                } else {
                                    summaries
                                }
                            }
                            None => summaries,
                        };
                        doc_clone.summaries = Some(doc_summaries);
                    }
                    None => {}
                }
                docs.push(doc_clone)
            }
            Ok(None) => {
                let error_message = format!("Document not found for id: {}", doc_id);
                error!("{}", error_message);
                send_exception_response(worker_socket, &error_message, message_header, identity);
            }
            Err(e) => {
                let error_message = format!(
                    "Error retrieving document, id: {:?}, error: {:?}",
                    doc_id, e
                );
                error!("{}", error_message);
                send_exception_response(worker_socket, &error_message, message_header, identity);
            }
        }
    }
    info!("Sending response for document query");

    send_document_query_response(
        worker_socket,
        &request.model,
        message_header,
        &docs,
        &query_embeddings,
        with_embeddings,
        identity,
    )
}

fn rerank_summaries(
    processing_context: &Arc<ProcessingContext>,
    request: &DocumentQueryRequest,
    all_summaries: &Vec<SummaryDto>,
) -> anyhow::Result<Vec<SummaryDto>> {
    let id = uuid::Uuid::new_v4();
    let req_id = processing_context.hasher.hash(&id.to_string());

    let texts: Vec<_> = all_summaries
        .iter()
        .map(|summary| summary.text_content.clone())
        .collect();

    let ranks = match get_rerankings(
        processing_context.zmq_context.clone(),
        &processing_context.reranking_endpoint.clone().unwrap(),
        request.input.clone(),
        texts,
        req_id,
    ) {
        Ok(ranks) => ranks,
        Err(e) => {
            return Err(anyhow::anyhow!("Error getting reranking scores: {:?}", e));
        }
    };

    let mut rerank_map = IndexMap::new();
    for rank in ranks.iter() {
        rerank_map.insert(rank.index, rank.clone());
    }
    let mut ranked_summaries = Vec::new();
    for (idx, summary) in all_summaries.iter().enumerate() {
        if let Some(rank) = rerank_map.get(&idx) {
            let mut new_summary = summary.clone();
            new_summary.rank = Some(rank.clone());
            ranked_summaries.push(new_summary);
        }
    }
    Ok(ranked_summaries)
}

fn send_document_query_response(
    socket: &Socket,
    model: &str,
    message_header: &mut ZmqMessageHeader,
    documents: &Vec<DocumentDto>,
    query_embeddings: &Vec<f32>,
    verbose: bool,
    identity: &Vec<u8>,
) {
    let response = DocumentQueryResponse {
        documents: documents.clone(),
        model: Some(model.to_string()),
        query_embeddings: if verbose {
            Some(query_embeddings.clone())
        } else {
            None
        },
    };
    send_success_response(socket, response, message_header, identity);
}
