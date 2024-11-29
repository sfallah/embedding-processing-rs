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

pub async fn delete_document(db: &Arc<RocksDB>, document_id: &u64) -> anyhow::Result<()> {
    db.delete(ColumnFamilyType::Documents, document_id).await
}

/// Retrieves all `Summaries` associated with a `Document`.
pub async fn get_summaries_of_document(
    db: &Arc<RocksDB>,
    document_id: &u64,
) -> anyhow::Result<Vec<Summary>> {
    let doc = match get_document(&db, document_id).await? {
        Some(doc) => doc,
        None => return Ok(Vec::new()),
    };

    let summary_ids = match doc.summary_ids {
        Some(ids) => ids,
        None => return Ok(Vec::new()),
    };

    let summary_bytes = db
        .multi_get(ColumnFamilyType::Summaries, &summary_ids)
        .await?;

    let mut summaries = Vec::with_capacity(summary_bytes.len());
    for option_bytes in summary_bytes {
        if let Some(bytes) = option_bytes {
            let summary =
                Summary::unpack(&bytes).map_err(|e| anyhow!("Failed to unpack summary: {}", e))?;
            summaries.push(summary);
        }
    }

    Ok(summaries)
}
