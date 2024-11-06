use std::sync::Arc;
use anyhow::anyhow;
use embedding_common::Serde;
use crate::db::interface::{Database, Table};
use crate::db::rocksdb_impl::RocksDB;
use crate::models::document::Document;
use crate::models::embedding::Embedding;
use crate::models::split::Split;
use crate::models::summary::Summary;

// Get
fn get_document(db: &Arc<RocksDB>, document_id: &u64) -> anyhow::Result<Option<Document>> {
    let document_bytes = match db.get(Table::Documents, document_id) {
        Ok(Some(data)) => Some(data),
        Ok(None) => return Ok(None),
        Err(e) => return Err(anyhow!(format!("Failed to get document: {}", e)).into()),
    };

    let document_info = match document_bytes {
        Some(bytes) => match Document::unpack(&bytes) {
            Ok(document) => Some(document),
            Err(e) => return Err(anyhow!(format!("Failed to unpack document: {}", e)).into()),
        },
        None => None,
    };

    Ok(document_info)
}

fn get_split(db: &Arc<RocksDB>, split_id: &u64) -> anyhow::Result<Option<Split>> {
    let split_bytes = match db.get(Table::Splits, split_id) {
        Ok(Some(data)) => Some(data),
        Ok(None) => return Ok(None),
        Err(e) => return Err(anyhow!(format!("Failed to get split: {}", e)).into()),
    };

    let split_info = match split_bytes {
        Some(bytes) => match Split::unpack(&bytes) {
            Ok(split) => Some(split),
            Err(e) => return Err(anyhow!(format!("Failed to unpack split: {}", e)).into()).into(),
        },
        None => None,
    };

    Ok(split_info)
}

fn get_summary(db: &Arc<RocksDB>, summary_id: &u64) -> anyhow::Result<Option<Summary>> {
    let summary_bytes = match db.get(Table::Summaries, summary_id) {
        Ok(Some(data)) => Some(data),
        Ok(None) => return Ok(None),
        Err(e) => return Err(anyhow!(format!("Failed to get summary: {}", e))),
    };

    let summary_info = match summary_bytes {
        Some(bytes) => match Summary::unpack(&bytes) {
            Ok(summary) => Some(summary),
            Err(e) => return Err(anyhow!(format!("Failed to unpack summary: {}", e))),
        },
        None => None,
    };

    Ok(summary_info)
}

fn get_embeddings(db: &Arc<RocksDB>, embedding_id: &u64) -> anyhow::Result<Option<Embedding>> {
    let embedding_bytes = match db.get(Table::Embeddings, embedding_id) {
        Ok(Some(data)) => Some(data),
        Ok(None) => return Ok(None),
        Err(e) => return Err(anyhow!(format!("Failed to get embedding: {}", e))),
    };

    let embedding_info = match embedding_bytes {
        Some(bytes) => match Embedding::unpack(&bytes) {
            Ok(embedding) => Some(embedding),
            Err(e) => return Err(anyhow!(format!("Failed to unpack embedding: {}", e))),
        },
        None => None,
    };

    Ok(embedding_info)
}

fn get_splits_of_document(
    db: &Arc<RocksDB>,
    document_id: &u64,
) -> anyhow::Result<Option<Vec<Split>>> {
    let document_info = match get_document(db, document_id)? {
        Some(doc) => doc,
        None => return Ok(None),
    };

    let split_bytes = match db.multi_get(Table::Splits, &document_info.split_ids) {
        Ok(data) => data,
        Err(e) => return Err(anyhow!(format!("Failed to get splits: {}", e))),
    };

    let mut splits = Vec::with_capacity(split_bytes.len());
    for option_bytes in split_bytes {
        if let Some(bytes) = option_bytes {
            match Split::unpack(&bytes) {
                Ok(split) => splits.push(split),
                Err(e) => return Err(anyhow!(format!("Failed to unpack split: {}", e))),
            }
        }
    }

    Ok(Some(splits))
}

fn get_summaries_of_document(
    db: &Arc<RocksDB>,
    document_id: &u64,
) -> anyhow::Result<Option<Vec<Summary>>> {
    let document_info = match get_document(db, document_id)? {
        Some(doc) => doc,
        None => return Ok(None),
    };

    let summary_bytes = match db.multi_get(Table::Summaries, &document_info.summary_ids.unwrap()) {
        Ok(data) => data,
        Err(e) => return Err(anyhow!(format!("Failed to get summaries: {}", e))),
    };

    let mut summaries = Vec::with_capacity(summary_bytes.len());
    for option_bytes in summary_bytes {
        if let Some(bytes) = option_bytes {
            match Summary::unpack(&bytes) {
                Ok(summary) => summaries.push(summary),
                Err(e) => return Err(anyhow!(format!("Failed to unpack summary: {}", e))),
            }
        }
    }

    Ok(Some(summaries))
}

fn get_summaries_of_split(
    db: &Arc<RocksDB>,
    split_id: &u64,
) -> anyhow::Result<Option<Vec<Summary>>> {
    let split_info = match get_split(db, split_id)? {
        Some(split_info) => split_info,
        None => return Ok(None),
    };

    let summary_bytes = match db.multi_get(Table::Summaries, &split_info.summary_ids.unwrap()) {
        Ok(data) => data,
        Err(e) => return Err(anyhow!(format!("Failed to get summaries: {}", e))),
    };

    let mut summaries = Vec::with_capacity(summary_bytes.len());
    for option_bytes in summary_bytes {
        if let Some(bytes) = option_bytes {
            match Summary::unpack(&bytes) {
                Ok(summary) => summaries.push(summary),
                Err(e) => return Err(anyhow!(format!("Failed to unpack summary: {}", e))),
            }
        }
    }

    Ok(Some(summaries))
}