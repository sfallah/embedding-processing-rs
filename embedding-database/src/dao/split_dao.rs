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

    let summary_bytes = db
        .multi_get(ColumnFamilyType::Summaries, summary_ids)
        .await?;

    let mut summaries = Vec::with_capacity(summary_bytes.len());
    for option_bytes in summary_bytes {
        if let Some(bytes) = option_bytes {
            let summary =
                Summary::unpack(&bytes).map_err(|e| anyhow!("Failed to unpack summary: {}", e))?;
            summaries.push(summary);
        }
    }

    Ok(Some(summaries))
}
