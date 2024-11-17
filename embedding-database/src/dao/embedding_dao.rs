use std::sync::Arc;
use embedding_common::models::embedding::Embedding;
use crate::dao::dao_impl::put_embedding;
use crate::db::rocksdb_impl::RocksDB;

pub async fn put_embeddings(
    db: &Arc<RocksDB>,
    embeddings: &Vec<Embedding>,
) -> anyhow::Result<()> {
    let futures = embeddings.into_iter().map(|embedding| put_embedding(db, embedding));
    futures::future::join_all(futures).await.into_iter().collect::<Result<(), _>>()?;
    Ok(())
}