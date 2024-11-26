use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;

/// Stores a `Document` in the database.
pub async fn put_document(db: &Arc<RocksDB>, document: &Document) -> anyhow::Result<()> {
    let document_id = &document.document_id;
    let data = document
        .pack()
        .map_err(|e| anyhow!("Failed to pack document: {}", e))?;
    db.put(ColumnFamilyType::Documents, document_id, &data)
        .await?;
    Ok(())
}

/// Retrieves a `Document` by its ID.
pub async fn get_document(
    db: &Arc<RocksDB>,
    document_id: &u64,
) -> anyhow::Result<Option<Document>> {
    match db.get(ColumnFamilyType::Documents, document_id).await? {
        Some(data) => {
            let document =
                Document::unpack(&data).map_err(|e| anyhow!("Failed to unpack document: {}", e))?;
            Ok(Some(document))
        }
        None => Ok(None),
    }
}

/// Retrieves all `Splits` associated with a `Document`.
pub async fn get_splits_of_document(
    db: &Arc<RocksDB>,
    document_id: &u64,
) -> anyhow::Result<Option<Vec<Split>>> {
    let document_info = match get_document(&db, document_id).await? {
        Some(doc) => doc,
        None => return Ok(None),
    };

    let split_bytes = db
        .multi_get(ColumnFamilyType::Splits, &document_info.split_ids)
        .await?;

    let mut splits = Vec::with_capacity(split_bytes.len());
    for option_bytes in split_bytes {
        if let Some(bytes) = option_bytes {
            let split =
                Split::unpack(&bytes).map_err(|e| anyhow!("Failed to unpack split: {}", e))?;
            splits.push(split);
        }
    }

    Ok(Some(splits))
}

/// Retrieves all `Summaries` associated with a `Document`.
pub async fn get_summaries_of_document(
    db: &Arc<RocksDB>,
    document_id: &u64,
) -> anyhow::Result<Option<Vec<Summary>>> {
    let document_info = match get_document(&db, document_id).await? {
        Some(doc) => doc,
        None => return Ok(None),
    };

    let summary_ids = match &document_info.summary_ids {
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
