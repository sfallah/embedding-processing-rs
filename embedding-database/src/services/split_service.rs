use crate::dao::split_dao::{get_all_splits, get_split};
use crate::db::column_families::ColumnFamilyType;
use crate::db::db_record::{DbRecordKey, DbRecordValue};
use crate::db::rocksdb_impl::RocksDB;
use crate::prelude::*;
use crate::services::embedding_service::{get_embeddings_map, to_embedding_record};
use crate::services::embedding_user_service::to_embedding_user_record;
use crate::services::summary_service::{delete_summaries_aux, save_summary_aux};
use anyhow::Result;
use embedding_common::prelude::*;
use indexmap::IndexMap;
use std::sync::Arc;
use uuid::Uuid;

pub(crate) async fn save_split(
    db: &Arc<RocksDB>,
    split_dto: &SplitDto,
    user_id: Uuid,
) -> Result<()> {
    let mut db_records = Vec::new();
    let split = split_dto.to_model();
    let embedding = split_dto.to_embedding_backend();
    let embedding_user = split_dto.to_embedding_user_model(user_id);
    if let Some(embedding) = embedding {
        let embedding_record = to_embedding_record(&embedding).await?;
        db_records.push(embedding_record);
    }
    if let Some(embedding_user) = embedding_user {
        let embedding_user_record = to_embedding_user_record(&embedding_user).await?;
        db_records.push(embedding_user_record);
    }
    db_records.push(DbRecordValue::new(
        ColumnFamilyType::Splits,
        split.split_id,
        split.pack()?,
    ));
    save_summary_aux(&split_dto.summaries, user_id, &mut db_records).await?;
    db.save_records(Arc::new(db_records)).await?;
    Ok(())
}

pub async fn delete_split_full(db: &Arc<RocksDB>, split_id: &u64) -> anyhow::Result<Option<Split>> {
    let split = get_split(db, *split_id).await?;
    match split {
        None => Ok(None),
        Some(split) => {
            let mut key_records = Vec::new();
            key_records.push(DbRecordKey::new(ColumnFamilyType::Splits, *split_id));
            let summary_ids = split.summary_ids.clone().unwrap_or_default();
            delete_summaries_aux(&summary_ids, &mut key_records).await?;
            delete_split_aux(*split_id, &mut key_records).await?;
            match db.delete_records(Arc::new(key_records)).await {
                Ok(_) => Ok(Some(split)),
                Err(e) => Err(e),
            }
        }
    }
}

pub async fn delete_split_aux(split_id: u64, db_records: &mut Vec<DbRecordKey>) -> Result<()> {
    let split_key = DbRecordKey::new(ColumnFamilyType::Splits, split_id);
    db_records.push(split_key);
    let embedding_key = DbRecordKey::new(ColumnFamilyType::Embeddings, split_id);
    db_records.push(embedding_key);
    let embedding_user_key = DbRecordKey::new(ColumnFamilyType::EmbeddingUsers, split_id);
    db_records.push(embedding_user_key);
    Ok(())
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
            let summary_ids: Vec<_> = splits
                .iter()
                .flat_map(|split| split.summary_ids.clone().unwrap_or_default())
                .collect();
            &get_split_summaries_map(db, summary_ids.as_slice(), None, with_embeddings).await?
        }
    };
    let mut splits_dtos = Vec::new();
    for split in splits {
        let embedding = embeddings_map.get(&split.split_id).map(|embd| embd.clone());
        let summary_dtos = summary_map
            .get(&split.split_id)
            .map_or(Vec::new(), |summary_dtos| summary_dtos.clone());
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
) -> Result<IndexMap<u64, Vec<SplitDto>>> {
    let splits_dtos = get_splits_full(
        db,
        split_ids,
        splits_query_res,
        summary_map,
        with_embeddings,
    )
    .await?;
    let mut split_map: IndexMap<u64, Vec<SplitDto>> = IndexMap::new();
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
