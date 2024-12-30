use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::{anyhow, Ok};
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

/// Retrieves a `Document` by its ID.
pub async fn document_exists(
    db: &Arc<RocksDB>,
    document_id: &u64,
) -> anyhow::Result<bool> {
    let doc_exists = db.key_may_exist(ColumnFamilyType::Documents, document_id).await?;
    Ok(doc_exists)
}

pub async fn delete_document(db: &Arc<RocksDB>, document_id: &u64) -> anyhow::Result<()> {
    db.delete(ColumnFamilyType::Documents, document_id).await
}

