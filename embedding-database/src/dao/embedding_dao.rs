use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::{anyhow, Context};
use embedding_common::prelude::*;
use futures::future::try_join_all;
use std::sync::Arc;

/// Stores an `Embedding` in the database.
pub async fn put_embedding(db: &Arc<RocksDB>, embedding: &Embedding) -> anyhow::Result<()> {
    let embedding_id = &embedding.embedding_id;
    let data = embedding
        .pack()
        .map_err(|e| anyhow!("Failed to pack embedding: {}", e))?;
    db.put(ColumnFamilyType::Embeddings, embedding_id, &data)
        .await?;
    Ok(())
}

pub async fn put_embeddings(db: &Arc<RocksDB>, embeddings: Vec<Embedding>) -> anyhow::Result<()> {
    let futures = embeddings
        .iter()
        .map(|embedding| put_embedding(db, embedding));
    futures::future::join_all(futures)
        .await
        .into_iter()
        .collect::<anyhow::Result<()>>()?;
    Ok(())
}

pub async fn get_all_embeddings(db: &Arc<RocksDB>) -> anyhow::Result<Vec<Embedding>> {
    let embedding_data_bytes_vec = db
        .get_all(ColumnFamilyType::Embeddings)
        .await
        .context("Failed to get all embeddings")?;

    let futures = embedding_data_bytes_vec
        .into_iter()
        .map(|embedding_data_bytes| async move {
            Embedding::unpack(&embedding_data_bytes).context("Failed to unpack embedding")
        });

    let embeddings_data = try_join_all(futures).await?;

    Ok(embeddings_data)
}

pub async fn get_splits_by_embedding_ids(
    db: &Arc<RocksDB>,
    embedding_ids: Vec<u64>,
) -> anyhow::Result<Vec<Split>> {
    let embeddings = db
        .multi_get(ColumnFamilyType::Embeddings, &embedding_ids.to_vec())
        .await?;
    let mut split_ids = Vec::new();
    for embed_opt in embeddings.iter() {
        if let Some(embedding) = embed_opt {
            let embedding: Embedding = Embedding::unpack(embedding)?;
            split_ids.push(embedding.data_id);
        }
    }

    let raw_splits = db
        .multi_get(ColumnFamilyType::Splits, &split_ids.to_vec())
        .await?;

    let mut splits = Vec::new();
    for split_opt in raw_splits.iter() {
        if let Some(split) = split_opt {
            let split: Split = Split::unpack(split)?;
            splits.push(split);
        }
    }
    Ok(splits)
}

/// Retrieves an `Embedding` by its ID.
pub async fn get_embedding(
    db: &Arc<RocksDB>,
    embedding_id: &u64,
) -> anyhow::Result<Option<Embedding>> {
    match db.get(ColumnFamilyType::Embeddings, embedding_id).await? {
        Some(data) => {
            let embedding = Embedding::unpack(&data)
                .map_err(|e| anyhow!("Failed to unpack embedding: {}", e))?;
            Ok(Some(embedding))
        }
        None => Ok(None),
    }
}
