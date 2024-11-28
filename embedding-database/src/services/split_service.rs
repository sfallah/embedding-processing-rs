use crate::dao::embedding_dao::delete_all_embeddings;
use crate::dao::embedding_user_dao::delete_all_embedding_users;
use crate::dao::split_dao::delete_all_splits;
use crate::prelude::*;
use anyhow::{Result};
use embedding_common::prelude::*;
use std::sync::Arc;
use uuid::Uuid;
use crate::services::summary_service::get_all_summaries_full;

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

pub async fn get_split_full(db: &Arc<RocksDB>, split_id: u64) -> Result<Option<SplitDto>> {
    match get_split(&db, &split_id).await? {
        Some(split) => {
            let summary_ids = split.summary_ids.clone().unwrap_or_else(|| Vec::new());
            let summaries = get_all_summaries(&db, &summary_ids).await?;
            let summary_dtos: Vec<_> = summaries.iter().map(|s| s.to_dto(None)).collect();
            let split_dto = split.to_dto_full(summary_dtos);
            Ok(Some(split_dto))
        }
        None => Ok(None),
    }
}

pub async fn delete_splits_full(db: &Arc<RocksDB>, split_ids: &Vec<u64>) -> Result<()> {
    delete_all_splits(db, split_ids).await?;
    delete_all_embeddings(db, split_ids).await?;
    delete_all_embedding_users(db, split_ids).await
}

pub async fn get_all_splits_full(db: &Arc<RocksDB>, split_ids: &Vec<u64>) -> Result<Vec<SplitDto>> {
    let splits = get_all_splits(&db, split_ids).await?;
    let mut splits_dtos = Vec::new();
    for split in splits {
        let summary_ids = split.summary_ids.clone().unwrap_or_else(|| Vec::new());
        let summary_dtos = get_all_summaries_full(&db, &summary_ids).await?;
        let split_dto = split.to_dto_full(summary_dtos);
        splits_dtos.push(split_dto);
    }
    Ok(splits_dtos)
}
