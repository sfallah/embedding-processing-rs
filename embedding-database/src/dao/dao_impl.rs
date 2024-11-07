use crate::db::rocksdb_impl::RocksDB;
use crate::models::document::Document;
use crate::models::embedding::Embedding;
use crate::models::split::Split;
use crate::models::summary::Summary;
use crate::db::column_families::ColumnFamilyType;
use anyhow::{anyhow, Result};
use embedding_common::Serde;
use std::sync::Arc;

/// Retrieves a `Document` by its ID.
pub async fn get_document(db: Arc<RocksDB>, document_id: &u64) -> Result<Option<Document>> {
    match db.get(ColumnFamilyType::Documents, document_id).await? {
        Some(data) => {
            let document =
                Document::unpack(&data).map_err(|e| anyhow!("Failed to unpack document: {}", e))?;
            Ok(Some(document))
        }
        None => Ok(None),
    }
}

/// Retrieves a `Split` by its ID.
pub async fn get_split(db: Arc<RocksDB>, split_id: &u64) -> Result<Option<Split>> {
    match db.get(ColumnFamilyType::Splits, split_id).await? {
        Some(data) => {
            let split =
                Split::unpack(&data).map_err(|e| anyhow!("Failed to unpack split: {}", e))?;
            Ok(Some(split))
        }
        None => Ok(None),
    }
}

/// Retrieves a `Summary` by its ID.
pub async fn get_summary(db: Arc<RocksDB>, summary_id: &u64) -> Result<Option<Summary>> {
    match db.get(ColumnFamilyType::Summaries, summary_id).await? {
        Some(data) => {
            let summary =
                Summary::unpack(&data).map_err(|e| anyhow!("Failed to unpack summary: {}", e))?;
            Ok(Some(summary))
        }
        None => Ok(None),
    }
}

/// Retrieves an `Embedding` by its ID.
pub async fn get_embedding(db: Arc<RocksDB>, embedding_id: &u64) -> Result<Option<Embedding>> {
    match db.get(ColumnFamilyType::Embeddings, embedding_id).await? {
        Some(data) => {
            let embedding =
                Embedding::unpack(&data).map_err(|e| anyhow!("Failed to unpack embedding: {}", e))?;
            Ok(Some(embedding))
        }
        None => Ok(None),
    }
}

/// Retrieves all `Splits` associated with a `Document`.
pub async fn get_splits_of_document(
    db: Arc<RocksDB>,
    document_id: &u64,
) -> Result<Option<Vec<Split>>> {
    let document_info = match get_document(db.clone(), document_id).await? {
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
    db: Arc<RocksDB>,
    document_id: &u64,
) -> Result<Option<Vec<Summary>>> {
    let document_info = match get_document(db.clone(), document_id).await? {
        Some(doc) => doc,
        None => return Ok(None),
    };

    let summary_ids = match &document_info.summary_ids {
        Some(ids) => ids,
        None => return Ok(None),
    };

    let summary_bytes = db.multi_get(ColumnFamilyType::Summaries, summary_ids).await?;

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

/// Retrieves all `Summaries` associated with a `Split`.
pub async fn get_summaries_of_split(
    db: Arc<RocksDB>,
    split_id: &u64,
) -> Result<Option<Vec<Summary>>> {
    let split_info = match get_split(db.clone(), split_id).await? {
        Some(split) => split,
        None => return Ok(None),
    };

    let summary_ids = match &split_info.summary_ids {
        Some(ids) => ids,
        None => return Ok(None),
    };

    let summary_bytes = db.multi_get(ColumnFamilyType::Summaries, summary_ids).await?;

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
