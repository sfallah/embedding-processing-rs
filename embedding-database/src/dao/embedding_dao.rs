use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::{anyhow, Context};
use embedding_common::prelude::*;
use std::sync::Arc;

pub fn get_all_embeddings(db: &Arc<RocksDB>) -> anyhow::Result<Vec<Embedding>> {
    let embedding_data_bytes_vec = db
        .get_all(ColumnFamilyType::Embeddings)

        .context("Failed to get all embeddings")?;

    let embeddings_data =  embedding_data_bytes_vec
        .into_iter()
        .map(|embedding_data_bytes| {
            Embedding::unpack(&embedding_data_bytes).expect("Failed to unpack embedding")
        }).collect::<Vec<_>>();


    Ok(embeddings_data)
}

pub fn get_embeddings(
    db: &Arc<RocksDB>,
    embedding_ids: &[u64],
) -> anyhow::Result<Vec<Embedding>> {
    let embedding_data_bytes_vec = db
        .multi_get(ColumnFamilyType::Embeddings, embedding_ids)

        .context("Failed to get all embeddings")?;
    let mut embeddings = Vec::new();
    for option_bytes in embedding_data_bytes_vec {
        if let Some(bytes) = option_bytes {
            let embedding = Embedding::unpack(&bytes)
                .map_err(|e| anyhow!("Failed to unpack embedding: {}", e))?;
            embeddings.push(embedding);
        }
    }
    Ok(embeddings)
}

/// Retrieves an `Embedding` by its ID.
pub fn get_embedding(
    db: &Arc<RocksDB>,
    embedding_id: &u64,
) -> anyhow::Result<Option<Embedding>> {
    match db.get(ColumnFamilyType::Embeddings, embedding_id)? {
        Some(data) => {
            let embedding = Embedding::unpack(&data)
                .map_err(|e| anyhow!("Failed to unpack embedding: {}", e))?;
            Ok(Some(embedding))
        }
        None => Ok(None),
    }
}
