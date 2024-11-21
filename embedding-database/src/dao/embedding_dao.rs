use std::sync::Arc;
use anyhow::anyhow;
use embedding_common::models::embedding::Embedding;
use embedding_common::Serde;
use crate::dao::dao_impl::put_embedding;
use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;

pub async fn put_embeddings(
    db: &Arc<RocksDB>,
    embeddings: &Vec<Embedding>,
) -> anyhow::Result<()> {
    let futures = embeddings.into_iter().map(|embedding| put_embedding(db, embedding));
    futures::future::join_all(futures).await.into_iter().collect::<Result<(), _>>()?;
    Ok(())
}

pub async fn get_all_embeddings(db: &Arc<RocksDB>) -> Result<Vec<Embedding>, anyhow::Error> {
    let embedding_data_bytes_vec = match db.get_all(ColumnFamilyType::Embeddings).await {
        Ok(data) => data,
        Err(e) => return Err(anyhow!(format!("Failed to get all embeddings: {}", e))),
    };

    let mut embeddings_data = Vec::with_capacity(embedding_data_bytes_vec.len());
    for embedding_data_bytes in embedding_data_bytes_vec {
        match Embedding::unpack(&embedding_data_bytes) {
            Ok(embedding) => embeddings_data.push(embedding),
            Err(e) => return Err(anyhow!(format!("Failed to unpack embedding: {}", e))),
        }
    }

    Ok(embeddings_data)
}