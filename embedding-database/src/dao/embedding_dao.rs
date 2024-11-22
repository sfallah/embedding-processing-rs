use std::sync::Arc;
use anyhow::Context;
use futures::future::try_join_all;
use embedding_common::models::embedding::Embedding;
use embedding_common::models::split::Split;
use embedding_common::Serde;
use crate::dao::dao_impl::put_embedding;
use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;

pub async fn put_embeddings(
    db: &Arc<RocksDB>,
    embeddings: Vec<Embedding>,
) -> anyhow::Result<()> {
    let futures = embeddings.iter().map(|embedding| put_embedding(db, embedding));
    futures::future::join_all(futures).await.into_iter().collect::<anyhow::Result<()>>()?;
    Ok(())
}

pub async fn get_all_embeddings(db: &Arc<RocksDB>) -> anyhow::Result<Vec<Embedding>> {
    let embedding_data_bytes_vec = db
        .get_all(ColumnFamilyType::Embeddings)
        .await
        .context("Failed to get all embeddings")?;

    let futures = embedding_data_bytes_vec.into_iter().map(|embedding_data_bytes| {
        async move {
            Embedding::unpack(&embedding_data_bytes)
                .context("Failed to unpack embedding")
        }
    });

    let embeddings_data = try_join_all(futures).await?;

    Ok(embeddings_data)
}


pub async fn get_splits_by_embedding_ids(
    db: &Arc<RocksDB>,
    embedding_ids: Vec<u64>,
) -> anyhow::Result<Vec<Split>> {
    let embeddings = db.multi_get(ColumnFamilyType::Embeddings, &embedding_ids.to_vec())
        .await?;
    let mut split_ids = Vec::new();
    for embed_opt in embeddings.iter() {
        if let Some(embedding) = embed_opt {
            let embedding: Embedding = Embedding::unpack(embedding)?;
            split_ids.push(embedding.data_id);
        }
    }

    let raw_splits = db.multi_get(ColumnFamilyType::Splits, &split_ids.to_vec()).await?;

    let mut splits = Vec::new();
    for split_opt in raw_splits.iter() {
        if let Some(split) = split_opt {
            let split: Split = Split::unpack(split)?;
            splits.push(split);
        }
    }
    Ok(splits)
}