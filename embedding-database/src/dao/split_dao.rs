use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;
use tracing::error;

pub fn get_all_splits(db: &Arc<RocksDB>, split_ids: &[u64]) -> anyhow::Result<Vec<Split>> {
    let split_bytes = db.multi_get(ColumnFamilyType::Splits, split_ids)?;
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

pub fn get_split(db: &Arc<RocksDB>, split_id: u64) -> anyhow::Result<Option<Split>> {
    let split_bytes = db.get(ColumnFamilyType::Splits, &split_id)?;
    match split_bytes {
        None => Ok(None),
        Some(bytes) => match Split::unpack(&bytes) {
            Ok(split) => Ok(Some(split)),
            Err(e) => {
                let error_message = format!("Failed to unpack split: {}", e);
                error!("{}", error_message);
                Err(anyhow!(error_message))
            }
        },
    }
}
