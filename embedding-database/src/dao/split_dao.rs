use crate::dao::summary_dao::get_all_summaries;
use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;

/// Stores a `Split` in the database.
pub async fn put_split(db: &Arc<RocksDB>, split: &Split) -> anyhow::Result<()> {
    let split_id = &split.split_id;
    let data = split
        .pack()
        .map_err(|e| anyhow!("Failed to pack split: {}", e))?;
    db.put(ColumnFamilyType::Splits, split_id, &data).await?;
    Ok(())
}

/// Retrieves a `Split` by its ID.
pub async fn get_split(db: &Arc<RocksDB>, split_id: &u64) -> anyhow::Result<Option<Split>> {
    match db.get(ColumnFamilyType::Splits, split_id).await? {
        Some(data) => {
            let split =
                Split::unpack(&data).map_err(|e| anyhow!("Failed to unpack split: {}", e))?;
            Ok(Some(split))
        }
        None => Ok(None),
    }
}

/// Retrieves all `Summaries` associated with a `Split`.
pub async fn get_summaries_of_split(
    db: &Arc<RocksDB>,
    split_id: &u64,
) -> anyhow::Result<Option<Vec<Summary>>> {
    let split_info = match get_split(&db, split_id).await? {
        Some(split) => split,
        None => return Ok(None),
    };

    let summary_ids = match &split_info.summary_ids {
        Some(ids) => ids,
        None => return Ok(None),
    };
    let summaries = get_all_summaries(&db, summary_ids).await?;

    Ok(Some(summaries))
}

pub async fn get_all_splits(db: &Arc<RocksDB>, split_ids: &Vec<u64>) -> anyhow::Result<Vec<Split>> {
    let split_bytes = db.multi_get(ColumnFamilyType::Splits, split_ids).await?;
    let mut splits = Vec::new();
    for option_bytes in split_bytes {
        if let Some(bytes) = option_bytes {
            let split =
                Split::unpack(&bytes).map_err(|e| anyhow!("Failed to unpack summary: {}", e))?;
            splits.push(split);
        }
    }
    Ok(splits)
}

pub async fn get_all_splits_full(
    db: &Arc<RocksDB>,
    split_ids: &Vec<u64>,
) -> anyhow::Result<Vec<SplitDto>> {
    let split_bytes = db.multi_get(ColumnFamilyType::Splits, split_ids).await?;
    let mut splits_dtos = Vec::new();
    for option_bytes in split_bytes {
        if let Some(bytes) = option_bytes {
            let split: Split =
                Split::unpack(&bytes).map_err(|e| anyhow!("Failed to unpack summary: {}", e))?;
            let summary_ids = split.summary_ids.clone().unwrap_or_else(|| Vec::new());
            let summaries = get_all_summaries(&db, &summary_ids).await?;
            let summary_dtos: Vec<_> = summaries.iter().map(|s| s.to_dto(None)).collect();
            let split_dto = split.to_dto_full(summary_dtos);
            splits_dtos.push(split_dto);
        }
    }
    Ok(splits_dtos)
}
