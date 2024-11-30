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


pub async fn delete_all_splits(db: &Arc<RocksDB>, split_ids: &Vec<u64>) -> anyhow::Result<()> {
    db.multi_delete(ColumnFamilyType::Splits, split_ids.as_slice())
        .await
}

pub async fn get_all_splits(db: &Arc<RocksDB>, split_ids: &[u64]) -> anyhow::Result<Vec<Split>> {
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

