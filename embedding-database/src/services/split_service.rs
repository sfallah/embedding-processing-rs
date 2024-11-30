use crate::dao::embedding_dao::{delete_all_embeddings, put_embedding};
use crate::dao::embedding_user_dao::{delete_all_embedding_users, put_embedding_user};
use crate::dao::split_dao::{delete_all_splits, get_all_splits, put_split};
use crate::prelude::*;
use crate::services::embedding_service::get_embeddings_map;
use anyhow::Result;
use embedding_common::prelude::*;
use indexmap::IndexMap;
use std::sync::Arc;
use uuid::Uuid;
use crate::db::rocksdb_impl::RocksDB;

pub(crate) async fn save_split(
    db: &Arc<RocksDB>,
    split_dto: &SplitDto,
    user_id: Uuid,
) -> Result<()> {
    let split = split_dto.to_model();
    let embedding = split_dto.to_embedding_model();
    let embedding_user = split_dto.to_embedding_user_model(user_id);
    put_split(db, &split).await?;
    if let Some(embedding) = embedding {
        put_embedding(db, &embedding).await?;
    }
    if let Some(embedding_user) = embedding_user {
        put_embedding_user(db, &embedding_user).await?;
    }
    Ok(())
}

pub async fn delete_splits_full(db: &Arc<RocksDB>, split_ids: &Vec<u64>) -> Result<()> {
    delete_all_splits(db, split_ids).await?;
    delete_all_embeddings(db, split_ids).await?;
    delete_all_embedding_users(db, split_ids).await
}
pub async fn get_splits_full(
    db: &Arc<RocksDB>,
    split_ids: &[u64],
    splits_query_res: Option<&IndexMap<u64, f32>>,
    summary_map: Option<&IndexMap<u64, Vec<SummaryDto>>>,
    with_embeddings: bool,
) -> Result<Vec<SplitDto>> {
    let splits = get_all_splits(&db, split_ids).await?;

    let split_ids: Vec<_> = splits.iter().map(|s| s.split_id).collect();
    let embeddings_map = if with_embeddings {
        get_embeddings_map(db, split_ids.as_slice()).await?
    } else {
        IndexMap::new()
    };

    let summary_map = match summary_map {
        Some(summary_map) => summary_map,
        None => {
            let summary_ids: Vec<_> = splits.iter().flat_map(|split| split.summary_ids.clone().unwrap_or_default()).collect();
            &get_split_summaries_map(db, summary_ids.as_slice(), None, with_embeddings).await?
        }
    };
    let mut splits_dtos = Vec::new();
    for split in splits {
        let embedding = embeddings_map
            .get(&split.split_id)
            .map(|embd| embd.clone());
        let summary_dtos = summary_map
            .get(&split.split_id).map_or(Vec::new(), |summary_dtos| summary_dtos.clone());
        let query_score = splits_query_res.and_then(|qr| qr.get(&split.split_id).cloned());
        let split_dto = split.to_dto_full(summary_dtos, embedding, query_score);
        splits_dtos.push(split_dto);
    }
    Ok(splits_dtos)
}

pub async fn get_doc_splits_map(
    db: &Arc<RocksDB>,
    split_ids: &[u64],
    splits_query_res: Option<&IndexMap<u64, f32>>,
    summary_map: Option<&IndexMap<u64, Vec<SummaryDto>>>,
    with_embeddings: bool,
) -> Result<IndexMap<u64,Vec<SplitDto>>> {
    let splits_dtos = get_splits_full(db, split_ids, splits_query_res, summary_map, with_embeddings).await?;
    let mut split_map:IndexMap<u64,Vec<SplitDto>> = IndexMap::new();
    for split_dto in splits_dtos {
        let doc_id = split_dto.doc_id;
        if split_map.contains_key(&doc_id) {
            split_map.get_mut(&doc_id).unwrap().push(split_dto);
        } else {
            split_map.insert(doc_id, vec![split_dto]);
        }
    }
    Ok(split_map)
}
