use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;

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