use crate::dao::embedding_dao::delete_all_embeddings;
use crate::dao::embedding_user_dao::delete_all_embedding_users;
use crate::dao::summary_dao::delete_all_summaries;
use crate::prelude::{get_all_summaries, put_embedding, put_embedding_user, put_summary, RocksDB};
use crate::services::embedding_service::get_embeddings_map;
use anyhow::Result;
use embedding_common::prelude::*;
use indexmap::IndexMap;
use std::sync::Arc;
use uuid::Uuid;

pub(crate) async fn save_summary(db: &Arc<RocksDB>, dto: &SummaryDto, user_id: Uuid) -> Result<()> {
    let summary = dto.to_model();
    let embedding = dto.to_embedding_model();
    let embedding_user = dto.to_embedding_user_model(user_id);
    put_summary(db, &summary).await?;
    if let Some(embedding) = embedding {
        put_embedding(db, &embedding).await?;
    }
    if let Some(embedding_user) = embedding_user {
        put_embedding_user(db, &embedding_user).await?;
    }
    Ok(())
}

pub async fn get_summaries_full(
    db: &Arc<RocksDB>,
    summary_ids: &[u64],
    summaries_query_res: Option<&IndexMap<u64, f32>>,
    with_embeddings: bool,
) -> Result<Vec<SummaryDto>> {
    let summaries = get_all_summaries(db, summary_ids).await?;
    let summary_ids: Vec<_> = summaries.iter().map(|s| s.summary_id).collect();
    let embeddings_map = if with_embeddings {
        get_embeddings_map(db, summary_ids.as_slice()).await?
    } else {
        IndexMap::new()
    };

    let summary_dtos: Vec<_> = summaries
        .iter()
        .map(|summary| {
            let embedding = embeddings_map
                .get(&summary.summary_id)
                .map(|embd| embd.clone());
            let query_score = summaries_query_res.and_then(|qr| qr.get(&summary.summary_id).cloned());
            summary.to_dto(embedding, query_score)
        })
        .collect();
    Ok(summary_dtos)
}

pub async fn get_split_summaries_map(
    db: &Arc<RocksDB>,
    summary_ids: &[u64],
    query_res: Option<&IndexMap<u64, f32>>,
    with_embeddings: bool,
) -> Result<IndexMap<u64, Vec<SummaryDto>>> {
    let summary_dtos = get_summaries_full(db, summary_ids, query_res, with_embeddings).await?;
    let mut summary_map: IndexMap<u64, Vec<SummaryDto>> = IndexMap::new();
    for summary_dto in summary_dtos {
        let split_id = summary_dto.split_id;
        if summary_map.contains_key(&split_id) {
            summary_map.get_mut(&split_id).unwrap().push(summary_dto);
        } else {
            summary_map.insert(split_id, vec![summary_dto]);
        }
    }
    Ok(summary_map)
}

pub async fn delete_summaries_full(db: &Arc<RocksDB>, split_ids: &Vec<u64>) -> Result<()> {
    delete_all_summaries(db, split_ids).await?;
    delete_all_embeddings(db, split_ids).await?;
    delete_all_embedding_users(db, split_ids).await
}
