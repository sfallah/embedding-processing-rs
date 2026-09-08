use crate::schema::document::DocumentQueryRequest;
use crate::schema::document::DocumentQueryResponse;
use crate::schema::search_mode::SearchModeType;
use crate::schema::zmq_message_header::ZmqMessageHeader;
use crate::utils::zmq_utils::{send_exception_response, send_success_response};
use embedding_common::prelude::*;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::processing::query::process_query;
use embedding_processing::processing::rerankings::get_rerankings;
use embedding_store::prelude::{Shard, ShardPool};
use indexmap::IndexMap;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use zmq::Socket;

// Requests calling embedding model
pub fn process_document_query_request(
    worker_socket: &Socket,
    pool: &Arc<ShardPool>,
    processing_context: Arc<ProcessingContext>,
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

    let query_embeddings = match process_query(processing_context.clone(), request.input.clone()) {
        Ok(embeddings) => embeddings,
        Err(e) => {
            let error_message = format!("Error processing query: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    };

    let documents = match run_query(pool, &processing_context, &request, &query_embeddings) {
        Ok(documents) => documents,
        Err(e) => {
            let error_message = format!("Failed to query: {:?}", e);
            error!("{}", &error_message);
            send_exception_response(worker_socket, &error_message, message_header, identity);
            return;
        }
    };

    info!("Sending response for document query");

    send_document_query_response(
        worker_socket,
        &request.model,
        message_header,
        &documents,
        &query_embeddings,
        request.verbose.unwrap_or(false),
        identity,
    )
}

/// Search every requested workspace and return the documents in distance order.
///
/// Each shard searches its own two indexes, the hit lists are cut back to the globally nearest
/// `top_k` per index, and everything after that — the summary-to-split merge, reranking, loading —
/// runs per shard, because only that shard's log holds the records. The documents of all shards
/// are then ordered by their best split. With one workspace, which is the common request, the cut
/// is a no-op and this is what the RocksDB build did.
fn run_query(
    pool: &Arc<ShardPool>,
    processing_context: &Arc<ProcessingContext>,
    request: &DocumentQueryRequest,
    query_embeddings: &[f32],
) -> anyhow::Result<Vec<DocumentDto>> {
    let with_embeddings = request.verbose.unwrap_or(false);
    let search_mode: SearchModeType = request
        .search_mode
        .unwrap_or(SearchModeType::SplitAndSummary);
    let top_k = request.top_k.unwrap_or(5) as usize;
    info!("Searching for query in indexes, top_k: {}", top_k);

    // An empty document list means no restriction; an empty *filter* would match nothing.
    let doc_filter: Option<HashSet<u64>> = (!request.doc_ids.is_empty())
        .then(|| request.doc_ids.iter().copied().collect::<HashSet<u64>>());

    let mut shards: Vec<(Uuid, Arc<Shard>)> = Vec::new();
    for workspace_id in &request.workspace_ids {
        // A workspace nothing was ever written to has nothing to contribute, and a query must not
        // bring its shard into being.
        if let Some(shard) = pool.get_existing(*workspace_id)? {
            shards.push((*workspace_id, shard));
        }
    }

    let mut summary_hits: Vec<IndexMap<u64, f32>> = vec![IndexMap::new(); shards.len()];
    if search_mode == SearchModeType::SummaryOnly || search_mode == SearchModeType::SplitAndSummary
    {
        for (i, (_, shard)) in shards.iter().enumerate() {
            summary_hits[i] =
                shard.search_summaries(query_embeddings, top_k, doc_filter.as_ref())?;
        }
        keep_nearest(&mut summary_hits, top_k);
        debug!("Summary query results: {:?}", summary_hits);
    }

    let mut split_hits: Vec<IndexMap<u64, f32>> = vec![IndexMap::new(); shards.len()];
    if search_mode == SearchModeType::SplitOnly || search_mode == SearchModeType::SplitAndSummary {
        for (i, (_, shard)) in shards.iter().enumerate() {
            split_hits[i] = shard.search_splits(query_embeddings, top_k, doc_filter.as_ref())?;
        }
        keep_nearest(&mut split_hits, top_k);
        debug!("Split query results: {:?}", split_hits);
    }

    // Each document with the distance of its best split, so shards can be merged afterwards.
    let mut answers: Vec<(f32, DocumentDto)> = Vec::new();

    for (i, (workspace_id, shard)) in shards.iter().enumerate() {
        let summary_hits = &summary_hits[i];
        let split_hits = &split_hits[i];

        // Summary hits, reranked as one batch, then grouped by the split they came from.
        let summary_ids: Vec<u64> = summary_hits.keys().copied().collect();
        let all_summaries =
            shard.load_summaries(&summary_ids, Some(summary_hits), with_embeddings)?;
        let all_summaries = if request.rerank.unwrap_or(false) && !all_summaries.is_empty() {
            rerank_summaries(processing_context, request, &all_summaries)?
        } else {
            all_summaries
        };

        let mut split_summary_map: IndexMap<u64, Vec<SummaryDto>> = IndexMap::new();
        for summary_dto in all_summaries {
            split_summary_map
                .entry(summary_dto.split_id)
                .or_default()
                .push(summary_dto);
        }

        // A summary-derived split takes the least of its summaries' distances; a direct hit takes
        // its own and overrides the derived value, keeping the position it was first inserted at.
        let mut splits_min_distances: IndexMap<u64, f32> = split_summary_map
            .iter()
            .map(|(split_id, summaries)| {
                let best = summaries
                    .iter()
                    .filter_map(|summary| summary.query_distance)
                    .fold(f32::INFINITY, f32::min);
                (*split_id, best)
            })
            .collect();
        splits_min_distances.extend(split_hits.iter().map(|(id, d)| (*id, *d)));
        splits_min_distances.sort_by(|_, v1, _, v2| v1.partial_cmp(v2).unwrap_or(Ordering::Equal));

        let final_split_ids: Vec<u64> = splits_min_distances.keys().copied().collect();
        let splits = shard.load_splits(
            &final_split_ids,
            Some(split_hits),
            Some(&split_summary_map),
            with_embeddings,
        )?;

        // Splits arrive nearest first, so a document's first split is its best one.
        let mut doc_splits: IndexMap<u64, (f32, Vec<SplitDto>)> = IndexMap::new();
        for split in splits {
            let distance = splits_min_distances
                .get(&split.split_id)
                .copied()
                .unwrap_or(f32::INFINITY);
            let entry = doc_splits.entry(split.doc_id).or_insert((distance, vec![]));
            entry.1.push(split);
        }

        for (doc_id, (distance, hit_splits)) in doc_splits {
            let Some(mut document) = shard.load_doc(doc_id, hit_splits, with_embeddings)? else {
                // Only reachable if the document was deleted between the search and the load, so
                // it is dropped rather than failing a query the rest of which is good.
                warn!(
                    "workspace {}: document {} vanished between search and load",
                    workspace_id, doc_id
                );
                continue;
            };
            if let Some(summaries) = document.summaries.take() {
                document.summaries = Some(
                    if request.rerank.unwrap_or(false) && !summaries.is_empty() {
                        rerank_summaries(processing_context, request, &summaries)?
                    } else {
                        summaries
                    },
                );
            }
            answers.push((distance, document));
        }
    }

    // Stable, so documents from one shard keep the order that shard put them in.
    answers.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(answers.into_iter().map(|(_, document)| document).collect())
}

/// Cut per-shard hit lists down to the globally nearest `top_k`, leaving each shard's own list in
/// ascending order. A single shard cannot hold more than `top_k` hits, so it is untouched.
fn keep_nearest(per_shard: &mut [IndexMap<u64, f32>], top_k: usize) {
    let total: usize = per_shard.iter().map(|hits| hits.len()).sum();
    if total <= top_k {
        return;
    }

    let mut all: Vec<(usize, u64, f32)> = per_shard
        .iter()
        .enumerate()
        .flat_map(|(i, hits)| hits.iter().map(move |(id, d)| (i, *id, *d)))
        .collect();
    all.sort_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    all.truncate(top_k);

    let mut kept: Vec<HashSet<u64>> = vec![HashSet::new(); per_shard.len()];
    for (i, id, _) in all {
        kept[i].insert(id);
    }
    for (i, hits) in per_shard.iter_mut().enumerate() {
        hits.retain(|id, _| kept[i].contains(id));
    }
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

#[cfg(test)]
mod tests {
    use super::keep_nearest;
    use indexmap::IndexMap;

    fn hits(pairs: &[(u64, f32)]) -> IndexMap<u64, f32> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn one_shard_is_left_alone() {
        // A search returns at most top_k, so the single-workspace request every client makes
        // reaches this with nothing to cut and must come out byte for byte the same.
        let mut per_shard = vec![hits(&[(1, 0.1), (2, 0.2), (3, 0.3)])];
        keep_nearest(&mut per_shard, 3);
        assert_eq!(
            per_shard[0]
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect::<Vec<_>>(),
            vec![(1, 0.1), (2, 0.2), (3, 0.3)]
        );
    }

    #[test]
    fn the_nearest_hits_win_across_shards() {
        let mut per_shard = vec![
            hits(&[(1, 0.1), (2, 0.5), (3, 0.9)]),
            hits(&[(4, 0.2), (5, 0.3), (6, 0.8)]),
        ];
        keep_nearest(&mut per_shard, 3);

        // 0.1, 0.2 and 0.3 survive wherever they came from; each shard keeps ascending order.
        assert_eq!(per_shard[0].keys().copied().collect::<Vec<_>>(), vec![1]);
        assert_eq!(per_shard[1].keys().copied().collect::<Vec<_>>(), vec![4, 5]);
    }

    #[test]
    fn a_shard_can_lose_every_hit() {
        let mut per_shard = vec![hits(&[(1, 0.1), (2, 0.2)]), hits(&[(3, 0.7), (4, 0.8)])];
        keep_nearest(&mut per_shard, 2);
        assert_eq!(per_shard[0].len(), 2);
        assert!(per_shard[1].is_empty());
    }

    #[test]
    fn fewer_hits_than_top_k_are_all_kept() {
        let mut per_shard = vec![hits(&[(1, 0.4)]), hits(&[(2, 0.1)])];
        keep_nearest(&mut per_shard, 10);
        assert_eq!(per_shard[0].len(), 1);
        assert_eq!(per_shard[1].len(), 1);
    }
}
