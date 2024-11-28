use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;

/// Stores a `Summary` in the database.
pub async fn put_summary(db: &Arc<RocksDB>, summary: &Summary) -> anyhow::Result<()> {
    let summary_id = &summary.summary_id;
    let data = summary
        .pack()
        .map_err(|e| anyhow!("Failed to pack summary: {}", e))?;
    db.put(ColumnFamilyType::Summaries, summary_id, &data)
        .await?;
    Ok(())
}

/// Retrieves a `Summary` by its ID.
pub async fn get_summary(db: &Arc<RocksDB>, summary_id: &u64) -> anyhow::Result<Option<Summary>> {
    match db.get(ColumnFamilyType::Summaries, summary_id).await? {
        Some(data) => {
            let summary =
                Summary::unpack(&data).map_err(|e| anyhow!("Failed to unpack summary: {}", e))?;
            Ok(Some(summary))
        }
        None => Ok(None),
    }
}

pub async fn get_all_summaries(
    db: &Arc<RocksDB>,
    summary_ids: &[u64],
) -> anyhow::Result<Vec<Summary>> {
    let summary_bytes = db
        .multi_get(ColumnFamilyType::Summaries, summary_ids)
        .await?;
    let mut summaries = Vec::new();
    for option_bytes in summary_bytes {
        if let Some(bytes) = option_bytes {
            let summary =
                Summary::unpack(&bytes).map_err(|e| anyhow!("Failed to unpack summary: {}", e))?;
            summaries.push(summary);
        }
    }
    Ok(summaries)
}

pub async fn delete_all_summaries(db: &Arc<RocksDB>, summary_ids: &[u64]) -> anyhow::Result<()> {
    db.multi_delete(ColumnFamilyType::Summaries, summary_ids).await
}